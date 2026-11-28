// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using Iced.Intel;
using SharpEmu.Core.Cpu.Emulation;
using SharpEmu.HLE;
using SharpEmu.HLE.GpuMemory;

namespace SharpEmu.Core.Cpu.Native;

// Recover supported instructions only after an illegal-instruction fault.
// Leave native instructions and unrelated register values unchanged.
public sealed partial class DirectExecutionBackend
{
    // Byte offset of Xmm0 within the Win64 CONTEXT record: FltSave (the XMM_SAVE_AREA32/FXSAVE
    // image) starts right after Rip at offset 256, and XmmRegisters[0] sits 160 bytes into that
    // area (32-byte header + 8 legacy x87/MMX slots x 16 bytes). 256 + 160 = 416 (0x1A0). Cross-
    // checked against this file's own Win64ContextSize (0x4D0): rebuilding the whole CONTEXT
    // layout field-by-field from offset 0 lands on the same 0x4D0 total, which would not happen
    // if this offset (or anything before it) were wrong.
    private const int Win64ContextXmm0Offset = 0x1A0;
    private const int ClzeroStubRegionSize = 256 * 1024;
    private const int ClzeroHelper64Offset = 0;
    private const int ClzeroHelper32Offset = 64;
    private const int ClzeroFirstTrampolineOffset = 128;

    private static readonly bool _hostSupportsClzero = DetectHostClzeroSupport();

    private static int _sse4aSoftwareFallbackAnnounced;
    private static long _sse4aInstructionsEmulated;
    private static int _monitorxSoftwareFallbackAnnounced;
    private static long _monitorxInstructionsEmulated;
    private static int _clzeroSoftwareFallbackAnnounced;
    private static long _clzeroInstructionsEmulated;

    private nint _clzeroStubRegion;
    private int _clzeroStubOffset;
    private Dictionary<ClzeroTrampolineKey, nint>? _clzeroTrampolines;

    private readonly record struct ClzeroTrampolineKey(
        bool Is32Bit,
        byte Length,
        ulong LowBytes,
        ulong HighBytes);

    private readonly record struct ClzeroPendingCandidate(
        ulong Ip,
        int Length,
        bool Is32Bit);

    private readonly record struct ClzeroPatchCandidate(
        ulong ClzeroIp,
        int ClzeroLength,
        bool Is32Bit,
        ulong NextIp,
        int NextLength);

    internal static bool HostSupportsClzero => _hostSupportsClzero;

    private static bool DetectHostClzeroSupport()
    {
        if (!X86Base.IsSupported)
        {
            return false;
        }

        var maxExtended = unchecked((uint)X86Base.CpuId(unchecked((int)0x80000000), 0).Eax);
        if (maxExtended < 0x80000008u)
        {
            return false;
        }

        var extendedInfo = X86Base.CpuId(unchecked((int)0x80000008), 0);
        return (extendedInfo.Ebx & 1) != 0;
    }

    private static bool IsPatchableClzeroInstruction(in Instruction instruction, out bool is32Bit)
    {
        if (instruction.Mnemonic == Mnemonic.Clzero &&
            instruction.SegmentPrefix is not (Register.FS or Register.GS) &&
            instruction.Code is (Code.Clzeroq or Code.Clzerod))
        {
            is32Bit = instruction.Code == Code.Clzerod;
            return true;
        }

        is32Bit = false;
        return false;
    }

    private static bool CanFuseClzeroSuccessor(in Instruction instruction, int clzeroLength)
    {
        if (instruction.Code == Code.INVALID ||
            instruction.FlowControl != FlowControl.Next ||
            instruction.IsIPRelativeMemoryOperand ||
            instruction.SegmentPrefix is (Register.FS or Register.GS) ||
            instruction.IsStackInstruction ||
            instruction.StackPointerIncrement != 0)
        {
            return false;
        }

        var totalLength = clzeroLength + instruction.Length;
        if (totalLength is < 5 or > 15)
        {
            return false;
        }

        if (instruction.MemoryBase is (Register.RSP or Register.ESP or Register.SP) ||
            instruction.MemoryIndex is (Register.RSP or Register.ESP or Register.SP))
        {
            return false;
        }

        for (var i = 0; i < instruction.OpCount; i++)
        {
            if (instruction.GetOpKind(i) == OpKind.Register &&
                instruction.GetOpRegister(i) is (Register.RSP or Register.ESP or Register.SP or Register.SPL))
            {
                return false;
            }
        }

        // Never relocate another emulated instruction verbatim into a trampoline.
        return instruction.Mnemonic is not (
            Mnemonic.Clzero or
            Mnemonic.Monitorx or
            Mnemonic.Mwaitx or
            Mnemonic.Extrq or
            Mnemonic.Insertq or
            Mnemonic.Sha1msg1 or
            Mnemonic.Sha1msg2 or
            Mnemonic.Sha1nexte or
            Mnemonic.Sha1rnds4 or
            Mnemonic.Sha256msg1 or
            Mnemonic.Sha256msg2 or
            Mnemonic.Sha256rnds2);
    }

    private unsafe bool EnsureClzeroStubRegionLocked(ulong nearAddress)
    {
        if (_clzeroStubRegion != 0)
        {
            return true;
        }

        nint region = 0;
        if (_entryPoint != 0)
        {
            region = (nint)TryAllocateNearEntry(ClzeroStubRegionSize);
        }

        if (region == 0 && nearAddress != 0)
        {
            region = (nint)TryAllocateNearAddress(nearAddress, ClzeroStubRegionSize);
        }

        if (region == 0)
        {
            region = (nint)VirtualAlloc(null, ClzeroStubRegionSize, 12288u, PAGE_EXECUTE_READWRITE);
        }

        if (region == 0)
        {
            return false;
        }

        // Preserves every GPR, XMM/YMM register, and RFLAGS; zeroes the 64-byte cache line
        // containing RAX (or zero-extended EAX for the 32-bit address-size form).
        ReadOnlySpan<byte> helper64 =
        [
            0x9C,                         // pushfq
            0x51,                         // push rcx
            0x52,                         // push rdx
            0x48, 0x89, 0xC1,             // mov rcx, rax
            0x48, 0x83, 0xE1, 0xC0,       // and rcx, -64
            0x31, 0xD2,                   // xor edx, edx
            0x48, 0x89, 0x11,             // mov [rcx], rdx
            0x48, 0x89, 0x51, 0x08,       // mov [rcx+0x08], rdx
            0x48, 0x89, 0x51, 0x10,       // mov [rcx+0x10], rdx
            0x48, 0x89, 0x51, 0x18,       // mov [rcx+0x18], rdx
            0x48, 0x89, 0x51, 0x20,       // mov [rcx+0x20], rdx
            0x48, 0x89, 0x51, 0x28,       // mov [rcx+0x28], rdx
            0x48, 0x89, 0x51, 0x30,       // mov [rcx+0x30], rdx
            0x48, 0x89, 0x51, 0x38,       // mov [rcx+0x38], rdx
            0x5A,                         // pop rdx
            0x59,                         // pop rcx
            0x9D,                         // popfq
            0xC3,                         // ret
        ];

        ReadOnlySpan<byte> helper32 =
        [
            0x9C,                         // pushfq
            0x51,                         // push rcx
            0x52,                         // push rdx
            0x89, 0xC1,                   // mov ecx, eax (zero-extends to rcx)
            0x48, 0x83, 0xE1, 0xC0,       // and rcx, -64
            0x31, 0xD2,                   // xor edx, edx
            0x48, 0x89, 0x11,             // mov [rcx], rdx
            0x48, 0x89, 0x51, 0x08,       // mov [rcx+0x08], rdx
            0x48, 0x89, 0x51, 0x10,       // mov [rcx+0x10], rdx
            0x48, 0x89, 0x51, 0x18,       // mov [rcx+0x18], rdx
            0x48, 0x89, 0x51, 0x20,       // mov [rcx+0x20], rdx
            0x48, 0x89, 0x51, 0x28,       // mov [rcx+0x28], rdx
            0x48, 0x89, 0x51, 0x30,       // mov [rcx+0x30], rdx
            0x48, 0x89, 0x51, 0x38,       // mov [rcx+0x38], rdx
            0x5A,                         // pop rdx
            0x59,                         // pop rcx
            0x9D,                         // popfq
            0xC3,                         // ret
        ];

        new Span<byte>((void*)region, ClzeroFirstTrampolineOffset).Fill(0x90);
        helper64.CopyTo(new Span<byte>((byte*)region + ClzeroHelper64Offset, helper64.Length));
        helper32.CopyTo(new Span<byte>((byte*)region + ClzeroHelper32Offset, helper32.Length));

        uint previousProtection = 0;
        VirtualProtect((void*)region, ClzeroStubRegionSize, PAGE_EXECUTE_READ, &previousProtection);
        FlushInstructionCache(GetCurrentProcess(), (void*)region, ClzeroFirstTrampolineOffset);

        _clzeroStubRegion = region;
        _clzeroStubOffset = ClzeroFirstTrampolineOffset;
        _clzeroTrampolines ??= new Dictionary<ClzeroTrampolineKey, nint>();
        return true;
    }

    private static ClzeroTrampolineKey BuildClzeroTrampolineKey(bool is32Bit, ReadOnlySpan<byte> successorBytes)
    {
        Span<byte> padded = stackalloc byte[16];
        padded.Clear();
        successorBytes.CopyTo(padded);
        return new ClzeroTrampolineKey(
            is32Bit,
            (byte)successorBytes.Length,
            BinaryPrimitives.ReadUInt64LittleEndian(padded),
            BinaryPrimitives.ReadUInt64LittleEndian(padded[8..]));
    }

    private unsafe nint GetOrCreateClzeroTrampolineLocked(
        bool is32Bit,
        ReadOnlySpan<byte> successorBytes,
        ulong nearAddress)
    {
        if (successorBytes.Length is < 2 or > 12)
        {
            return 0;
        }

        if (!EnsureClzeroStubRegionLocked(nearAddress))
        {
            return 0;
        }

        var key = BuildClzeroTrampolineKey(is32Bit, successorBytes);
        if (_clzeroTrampolines!.TryGetValue(key, out var existing))
        {
            return existing;
        }

        var stubLength = 5 + successorBytes.Length + 1;
        var alignedSize = (stubLength + 15) & ~15;
        if (_clzeroStubOffset + alignedSize > ClzeroStubRegionSize)
        {
            return 0;
        }

        var trampoline = _clzeroStubRegion + _clzeroStubOffset;
        var helperAddress = _clzeroStubRegion + (is32Bit ? ClzeroHelper32Offset : ClzeroHelper64Offset);
        var helperDisplacement = (long)helperAddress - ((long)trampoline + 5);
        if (helperDisplacement < int.MinValue || helperDisplacement > int.MaxValue)
        {
            return 0;
        }

        uint previousProtection = 0;
        if (!VirtualProtect((void*)trampoline, (nuint)alignedSize, PAGE_EXECUTE_READWRITE, &previousProtection))
        {
            return 0;
        }

        try
        {
            var destination = new Span<byte>((void*)trampoline, alignedSize);
            destination.Fill(0x90);
            destination[0] = 0xE8; // call rel32 clzero helper
            BinaryPrimitives.WriteInt32LittleEndian(destination[1..], (int)helperDisplacement);
            successorBytes.CopyTo(destination[5..]);
            destination[5 + successorBytes.Length] = 0xC3; // ret
        }
        finally
        {
            uint ignoredProtection = 0;
            VirtualProtect((void*)trampoline, (nuint)alignedSize, PAGE_EXECUTE_READ, &ignoredProtection);
            FlushInstructionCache(GetCurrentProcess(), (void*)trampoline, (nuint)alignedSize);
        }

        _clzeroStubOffset += alignedSize;
        _clzeroTrampolines[key] = trampoline;
        return trampoline;
    }

    private unsafe int PatchClzeroCandidatesLocked(
        List<ClzeroPatchCandidate> candidates,
        HashSet<ulong>? branchTargets)
    {
        var patched = 0;
        Span<byte> replacementBuffer = stackalloc byte[15];
        for (var i = 0; i < candidates.Count; i++)
        {
            var candidate = candidates[i];
            if (branchTargets is not null && branchTargets.Contains(candidate.NextIp))
            {
                continue;
            }

            var totalLength = candidate.ClzeroLength + candidate.NextLength;
            if (totalLength is < 5 or > 15)
            {
                continue;
            }

            var successorBytes = new ReadOnlySpan<byte>((void*)candidate.NextIp, candidate.NextLength);
            var trampoline = GetOrCreateClzeroTrampolineLocked(candidate.Is32Bit, successorBytes, candidate.ClzeroIp);
            if (trampoline == 0)
            {
                continue;
            }

            var displacement = (long)trampoline - ((long)candidate.ClzeroIp + 5);
            if (displacement < int.MinValue || displacement > int.MaxValue)
            {
                continue;
            }

            var replacement = replacementBuffer[..totalLength];
            replacement.Fill(0x90);
            replacement[0] = 0xE8; // call rel32 trampoline
            BinaryPrimitives.WriteInt32LittleEndian(replacement[1..], (int)displacement);
            if (WriteTlsInstruction((nint)candidate.ClzeroIp, replacement))
            {
                patched++;
            }
        }

        return patched;
    }

    private unsafe void ReleaseClzeroStubRegion()
    {
        if (_clzeroStubRegion != 0)
        {
            VirtualFree((void*)_clzeroStubRegion, 0u, 32768u);
            _clzeroStubRegion = 0;
            _clzeroStubOffset = 0;
        }

        _clzeroTrampolines?.Clear();
    }

    private unsafe bool TryRecoverAmdCompatInstruction(void* contextRecord, ulong rip)
    {
        if (TryRecoverMonitorxMwaitx(contextRecord, rip) ||
            TryRecoverClzero(contextRecord, rip))
        {
            return true;
        }

        // EXTRQ and INSERTQ need the current vector register values.
        // Use recovery only when the signal context contains these values.
        return (OperatingSystem.IsWindows() || _posixXmmContextBridged) &&
            TryRecoverSse4aExtractInsert(contextRecord, rip);
    }

    private unsafe bool TryRecoverClzero(void* contextRecord, ulong rip)
    {
        if (!TryReadFaultingInstruction(rip, out var instruction) ||
            instruction.Mnemonic != Mnemonic.Clzero ||
            instruction.SegmentPrefix is Register.FS or Register.GS ||
            instruction.Code is not (Code.Clzeroq or Code.Clzerod))
        {
            return false;
        }

        var rawAddress = ReadCtxU64(contextRecord, CTX_RAX);
        var address = instruction.Code == Code.Clzerod
            ? rawAddress & 0xFFFF_FFFFUL
            : rawAddress;
        var cacheLineAddress = address & ~63UL;

        if (!TryEnsureClzeroTargetWritable(contextRecord, rip, cacheLineAddress))
        {
            return false;
        }

        new Span<byte>((void*)cacheLineAddress, 64).Clear();
        WriteCtxU64(contextRecord, CTX_RIP, rip + (ulong)instruction.Length);

        Interlocked.Increment(ref _clzeroInstructionsEmulated);
        if (Interlocked.Exchange(ref _clzeroSoftwareFallbackAnnounced, 1) == 0)
        {
            Console.Error.WriteLine(
                "[LOADER][INFO] Host lacks AMD CLZERO used by the guest; " +
                "emulating those instructions in software.");
        }

        return true;
    }

    private unsafe bool TryEnsureClzeroTargetWritable(void* contextRecord, ulong rip, ulong cacheLineAddress)
    {
        if (!IsCanonicalUserAddress(cacheLineAddress) ||
            cacheLineAddress > 0x0000_8000_0000_0000UL - 64)
        {
            return false;
        }

        // On POSIX, GuestImageWriteTracker arms PROT_READ directly via mprotect without updating
        // HostMemory's shadow region table, so resolve any armed image write watch before relying
        // on IsWritableHostRange.
        if (!OperatingSystem.IsWindows())
        {
            _ = GuestImageWriteTracker.TryHandleWriteFault(cacheLineAddress);
        }

        if (IsWritableHostRange(cacheLineAddress, 64))
        {
            return true;
        }

        // Emulate the write-fault resolution chain that a hardware CLZERO store would trigger:
        // image write tracking, lazy page commit, and GPU page-guard invalidation. A page can
        // require multiple steps (e.g. lazy commit followed by GPU watch release), so re-check
        // host writability after each step and continue if the page is still not writable.
        if (OperatingSystem.IsWindows() &&
            GuestImageWriteTracker.TryHandleWriteFault(cacheLineAddress) &&
            IsWritableHostRange(cacheLineAddress, 64))
        {
            return true;
        }

        var rsp = ReadCtxU64(contextRecord, CTX_RSP);
        if (TryHandleLazyCommittedPage(1uL, cacheLineAddress, rip, rsp) &&
            IsWritableHostRange(cacheLineAddress, 64))
        {
            return true;
        }

        if (GuestGpuMemoryHook.TryResolveFault(FaultKind.Write, cacheLineAddress) &&
            IsWritableHostRange(cacheLineAddress, 64))
        {
            return true;
        }

        return IsWritableHostRange(cacheLineAddress, 64);
    }

    private unsafe bool TryRecoverMonitorxMwaitx(void* contextRecord, ulong rip)
    {
        // MONITORX (0F 01 FA) and MWAITX (0F 01 FB) are fixed 3-byte encodings with no
        // ModRM/SIB/displacement/immediate, so a raw byte compare is sufficient and unambiguous.
        var opcode = new byte[3];
        if (!TryReadExecutableBytes(rip, opcode) ||
            opcode[0] != 0x0F || opcode[1] != 0x01 || (opcode[2] != 0xFA && opcode[2] != 0xFB))
        {
            return false;
        }

        // PS5 titles use this pair in idle/wait loops: MONITORX arms a monitor on a cache line
        // and MWAITX blocks until that line is written (or a timeout elapses). Hosts without
        // the extension raise #UD on either one. We do not model the monitor itself, only its
        // observable effect on guest forward progress: MONITORX becomes a no-op (arming a
        // watch we never honour has no side effect of its own) and MWAITX becomes a plain
        // thread yield, i.e. treat the awaited condition as already satisfied so the guest
        // loop keeps making progress instead of executing an illegal opcode forever.
        if (opcode[2] == 0xFB)
        {
            Thread.Yield();
        }

        WriteCtxU64(contextRecord, CTX_RIP, rip + 3);

        Interlocked.Increment(ref _monitorxInstructionsEmulated);
        if (Interlocked.Exchange(ref _monitorxSoftwareFallbackAnnounced, 1) == 0)
        {
            Console.Error.WriteLine(
                "[LOADER][INFO] Host lacks AMD MONITORX/MWAITX used by the guest; " +
                "emulating those instructions in software.");
        }

        return true;
    }

    private unsafe bool TryRecoverSse4aExtractInsert(void* contextRecord, ulong rip)
    {
        if (!OperatingSystem.IsWindows() && !_posixXmmContextBridged ||
            !TryReadFaultingInstruction(rip, out var instruction))
        {
            return false;
        }

        var isExtrq = instruction.Mnemonic == Mnemonic.Extrq;
        var isInsertq = instruction.Mnemonic == Mnemonic.Insertq;
        if (!isExtrq && !isInsertq)
        {
            return false;
        }

        var isImmediateExtrq = isExtrq && instruction.OpCount == 3;
        var isRegisterExtrq = isExtrq && instruction.OpCount == 2;
        var isImmediateInsertq = isInsertq && instruction.OpCount == 4;
        if (!isImmediateExtrq && !isRegisterExtrq && !isImmediateInsertq)
        {
            return false;
        }

        if (instruction.GetOpKind(0) != OpKind.Register ||
            !TryGetXmmOffset(instruction.GetOpRegister(0), out var destOffset))
        {
            return false;
        }

        var destLow = ReadCtxU64(contextRecord, destOffset);
        if (isExtrq)
        {
            int length;
            int index;
            if (isRegisterExtrq)
            {
                if (instruction.GetOpKind(1) != OpKind.Register ||
                    !TryGetXmmOffset(instruction.GetOpRegister(1), out var controlOffset))
                {
                    return false;
                }

                // AMD defines the register form's field length in xmm2[5:0] and its start
                // index in xmm2[13:8]. Other control bits do not affect the instruction.
                var control = ReadCtxU64(contextRecord, controlOffset);
                length = (int)(control & 0x3F);
                index = (int)((control >> 8) & 0x3F);
            }
            else
            {
                length = (int)instruction.GetImmediate(1);
                index = (int)instruction.GetImmediate(2);
            }

            WriteCtxU64(contextRecord, destOffset, Sse4aBitFieldEmulator.ExtractBitField(destLow, length, index));
            WriteCtxU64(contextRecord, destOffset + 8, 0);
        }
        else
        {
            if (instruction.GetOpKind(1) != OpKind.Register ||
                !TryGetXmmOffset(instruction.GetOpRegister(1), out var srcOffset))
            {
                return false;
            }

            var length = (int)instruction.GetImmediate(2);
            var index = (int)instruction.GetImmediate(3);
            WriteCtxU64(contextRecord, destOffset, Sse4aBitFieldEmulator.InsertBitField(
                destLow, ReadCtxU64(contextRecord, srcOffset), length, index));
            WriteCtxU64(contextRecord, destOffset + 8, 0);
        }

        WriteCtxU64(contextRecord, CTX_RIP, rip + (ulong)instruction.Length);

        Interlocked.Increment(ref _sse4aInstructionsEmulated);
        if (Interlocked.Exchange(ref _sse4aSoftwareFallbackAnnounced, 1) == 0)
        {
            Console.Error.WriteLine(
                "[LOADER][INFO] Host lacks SSE4a EXTRQ/INSERTQ used by the guest; " +
                "emulating those instructions in software.");
        }

        return true;
    }

    // Maps an Iced XMM register to its byte offset in the Win64 CONTEXT record. Written as an
    // explicit switch (rather than arithmetic on the Register enum) to match the style already
    // used by TryGetGprSlot/TryGetGpr64Offset in DirectExecutionBackend.IllegalInstruction.cs.
    private static bool TryGetXmmOffset(Register register, out int offset)
    {
        switch (register)
        {
            case Register.XMM0: offset = Win64ContextXmm0Offset + 16 * 0; return true;
            case Register.XMM1: offset = Win64ContextXmm0Offset + 16 * 1; return true;
            case Register.XMM2: offset = Win64ContextXmm0Offset + 16 * 2; return true;
            case Register.XMM3: offset = Win64ContextXmm0Offset + 16 * 3; return true;
            case Register.XMM4: offset = Win64ContextXmm0Offset + 16 * 4; return true;
            case Register.XMM5: offset = Win64ContextXmm0Offset + 16 * 5; return true;
            case Register.XMM6: offset = Win64ContextXmm0Offset + 16 * 6; return true;
            case Register.XMM7: offset = Win64ContextXmm0Offset + 16 * 7; return true;
            case Register.XMM8: offset = Win64ContextXmm0Offset + 16 * 8; return true;
            case Register.XMM9: offset = Win64ContextXmm0Offset + 16 * 9; return true;
            case Register.XMM10: offset = Win64ContextXmm0Offset + 16 * 10; return true;
            case Register.XMM11: offset = Win64ContextXmm0Offset + 16 * 11; return true;
            case Register.XMM12: offset = Win64ContextXmm0Offset + 16 * 12; return true;
            case Register.XMM13: offset = Win64ContextXmm0Offset + 16 * 13; return true;
            case Register.XMM14: offset = Win64ContextXmm0Offset + 16 * 14; return true;
            case Register.XMM15: offset = Win64ContextXmm0Offset + 16 * 15; return true;
            default:
                offset = 0;
                return false;
        }
    }
}
