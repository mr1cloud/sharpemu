// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpEmu.Core.Cpu.Native;
using SharpEmu.HLE;
using Xunit;

namespace SharpEmu.Libs.Tests.Cpu;

public sealed unsafe class ClzeroWindowsExceptionRecoveryTests
{
    private const int Win64ContextRaxOffset = 0x78;
    private const int Win64ContextRipOffset = 0xF8;

    private static readonly MethodInfo TryRecoverAmdCompat = typeof(DirectExecutionBackend).GetMethod(
        "TryRecoverAmdCompatInstruction",
        BindingFlags.Instance | BindingFlags.NonPublic)!;

    private static readonly MethodInfo PatchTlsPatternsInRangeWithClzero = typeof(DirectExecutionBackend).GetMethod(
        "PatchTlsPatternsInRange",
        BindingFlags.Instance | BindingFlags.NonPublic,
        binder: null,
        [typeof(ulong), typeof(ulong), typeof(bool)],
        modifiers: null)!;

    private static readonly MethodInfo ReleaseClzeroStubRegionMethod = typeof(DirectExecutionBackend).GetMethod(
        "ReleaseClzeroStubRegion",
        BindingFlags.Instance | BindingFlags.NonPublic)!;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ClzeroZerosAlignedCacheLineAndAdvancesRip(bool executeOnlyCode)
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        byte[] instruction = [0x0F, 0x01, 0xFC];
        var code = AllocatePage(HostMemory.PAGE_READWRITE);
        var data = AllocatePage(HostMemory.PAGE_READWRITE);
        try
        {
            instruction.CopyTo(new Span<byte>((void*)code, instruction.Length));
            if (executeOnlyCode)
            {
                Assert.True(HostMemory.Protect((void*)code, (nuint)Environment.SystemPageSize,
                    HostMemory.PAGE_EXECUTE, out _));
            }

            new Span<byte>((void*)data, 128).Fill(0xA5);

            var contextBytes = new byte[0x4D0];
            contextBytes.AsSpan().Fill(0x5A);
            var targetAddress = (ulong)data + 19;
            fixed (byte* context = contextBytes)
            {
                *(ulong*)(context + Win64ContextRaxOffset) = targetAddress;
                *(ulong*)(context + Win64ContextRipOffset) = (ulong)code;
            }

            var expected = (byte[])contextBytes.Clone();
            fixed (byte* expectedContext = expected)
            {
                *(ulong*)(expectedContext + Win64ContextRipOffset) = (ulong)code + (ulong)instruction.Length;
            }

            var backend = RuntimeHelpers.GetUninitializedObject(typeof(DirectExecutionBackend));
            fixed (byte* context = contextBytes)
            {
                Assert.True((bool)TryRecoverAmdCompat.Invoke(
                    backend,
                    [Pointer.Box(context, typeof(void*)), (ulong)code])!);
            }

            Assert.Equal(expected, contextBytes);
            var span = new ReadOnlySpan<byte>((void*)data, 128);
            for (var i = 0; i < 64; i++)
            {
                Assert.Equal(0, span[i]);
            }

            for (var i = 64; i < 128; i++)
            {
                Assert.Equal(0xA5, span[i]);
            }
        }
        finally
        {
            Assert.True(HostMemory.Free((void*)code, 0, HostMemory.MEM_RELEASE));
            Assert.True(HostMemory.Free((void*)data, 0, HostMemory.MEM_RELEASE));
        }
    }

    [Fact]
    public void ClzeroReturnsFalseWhenTargetIsReadOnly()
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        byte[] instruction = [0x0F, 0x01, 0xFC];
        var code = AllocatePage(HostMemory.PAGE_READWRITE);
        var data = AllocatePage(HostMemory.PAGE_READONLY);
        try
        {
            instruction.CopyTo(new Span<byte>((void*)code, instruction.Length));

            var context = stackalloc byte[0x4D0];
            new Span<byte>(context, 0x4D0).Clear();
            *(ulong*)(context + Win64ContextRaxOffset) = (ulong)data;
            *(ulong*)(context + Win64ContextRipOffset) = (ulong)code;

            var backend = RuntimeHelpers.GetUninitializedObject(typeof(DirectExecutionBackend));
            Assert.False((bool)TryRecoverAmdCompat.Invoke(
                backend,
                [Pointer.Box(context, typeof(void*)), (ulong)code])!);
            Assert.Equal((ulong)code, *(ulong*)(context + Win64ContextRipOffset));
        }
        finally
        {
            Assert.True(HostMemory.Free((void*)code, 0, HostMemory.MEM_RELEASE));
            Assert.True(HostMemory.Free((void*)data, 0, HostMemory.MEM_RELEASE));
        }
    }

    [Fact]
    public void ClzeroReturnsFalseWhenTargetIsUnmapped()
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        byte[] instruction = [0x0F, 0x01, 0xFC];
        var code = AllocatePage(HostMemory.PAGE_READWRITE);
        try
        {
            instruction.CopyTo(new Span<byte>((void*)code, instruction.Length));

            var context = stackalloc byte[0x4D0];
            new Span<byte>(context, 0x4D0).Clear();
            *(ulong*)(context + Win64ContextRaxOffset) = 0x1234;
            *(ulong*)(context + Win64ContextRipOffset) = (ulong)code;

            var backend = RuntimeHelpers.GetUninitializedObject(typeof(DirectExecutionBackend));
            Assert.False((bool)TryRecoverAmdCompat.Invoke(
                backend,
                [Pointer.Box(context, typeof(void*)), (ulong)code])!);
            Assert.Equal((ulong)code, *(ulong*)(context + Win64ContextRipOffset));
        }
        finally
        {
            Assert.True(HostMemory.Free((void*)code, 0, HostMemory.MEM_RELEASE));
        }
    }

    [Fact]
    public void ClzeroBinaryPatchExecutesLoopNativelyAndPreservesRegistersAndFlags()
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        // Win64 ABI: rcx = start, rdx = end (returns rcx ^ rdx in rax to check register preservation).
        //   mov rax, rcx         (48 89 C8)
        //   lea r8, [rcx + 0x10] (4C 8D 41 10)  -- canary register
        //   cmp rdx, rcx         (48 39 CA)     -- sets flags checked after loop? (overwritten by cmp in loop)
        // loop:
        //   clzero               (0F 01 FC)     -- offset 7
        //   add rax, 0x40        (48 83 C0 40)  -- offset 10
        //   cmp rax, rdx         (48 39 D0)     -- offset 14
        //   jb loop              (72 F4)        -- offset 17 (target = 19 - 12 = 7)
        //   clzero               (0F 01 FC)     -- offset 19 (zeroes one more line at rax, unaligned test)
        //   mov r9, 0x1234       (49 C7 C1 34 12 00 00) -- offset 22 (7 bytes, 3+7=10 <= 15)
        //   xor rax, rax         (48 31 C0)
        //   cmp r8, rcx          (49 39 C8)     -- verify rcx/r8 untouched (r8 = rcx + 0x10)
        //   sub r8, rcx          (49 29 C8)
        //   add r8, r9           (4D 01 C8)     -- should be 0x1244
        //   mov rax, r8          (4C 89 C0)
        //   ret                  (C3)
        byte[] guestCode =
        [
            0x48, 0x89, 0xC8,
            0x4C, 0x8D, 0x41, 0x10,
            0x0F, 0x01, 0xFC,
            0x48, 0x83, 0xC0, 0x40,
            0x48, 0x39, 0xD0,
            0x72, 0xF4,
            0x0F, 0x01, 0xFC,
            0x49, 0xC7, 0xC1, 0x34, 0x12, 0x00, 0x00,
            0x49, 0x29, 0xC8,
            0x4D, 0x01, 0xC8,
            0x4C, 0x89, 0xC0,
            0xC3,
        ];

        var code = AllocatePage(HostMemory.PAGE_EXECUTE_READWRITE);
        var data = AllocatePage(HostMemory.PAGE_READWRITE);
        var backend = RuntimeHelpers.GetUninitializedObject(typeof(DirectExecutionBackend));
        try
        {
            guestCode.CopyTo(new Span<byte>((void*)code, guestCode.Length));
            Assert.True(HostMemory.Protect((void*)code, (nuint)Environment.SystemPageSize,
                HostMemory.PAGE_EXECUTE_READ, out _));

            var counts = PatchTlsPatternsInRangeWithClzero.Invoke(
                backend,
                [(ulong)code, (ulong)code + (ulong)guestCode.Length, true])!;
            var clzeroCount = (int)counts.GetType().GetProperty("Clzeros")!.GetValue(counts)!;
            Assert.Equal(2, clzeroCount);

            // Verify both sites were replaced by CALL rel32 (0xE8)
            Assert.Equal(0xE8, Marshal.ReadByte(code, 7));
            Assert.Equal(0xE8, Marshal.ReadByte(code, 19));

            new Span<byte>((void*)data, 512).Fill(0xCC);

            var fn = (delegate* unmanaged[Stdcall]<ulong, ulong, ulong>)code;
            // Start at data + 64 + 13 (unaligned within line 1), loop until data + 256 + 13 (lines 1, 2, 3),
            // then final clzero clears line 4 (data + 256..319). Guard lines 0 and 5 must remain 0xCC.
            var start = (ulong)data + 64 + 13;
            var end = (ulong)data + 256 + 13;
            var ret = fn(start, end);
            Assert.Equal(0x1244UL, ret);

            var span = new ReadOnlySpan<byte>((void*)data, 512);
            for (var i = 0; i < 64; i++)
            {
                Assert.Equal(0xCC, span[i]);
            }

            for (var i = 64; i < 320; i++)
            {
                Assert.Equal(0x00, span[i]);
            }

            for (var i = 320; i < 512; i++)
            {
                Assert.Equal(0xCC, span[i]);
            }
        }
        finally
        {
            ReleaseClzeroStubRegionMethod.Invoke(backend, null);
            Assert.True(HostMemory.Free((void*)code, 0, HostMemory.MEM_RELEASE));
            Assert.True(HostMemory.Free((void*)data, 0, HostMemory.MEM_RELEASE));
        }
    }

    [Fact]
    public void ClzeroBinaryPatchSkipsSiteWhenBranchTargetsSuccessor()
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        //   jz target          (74 03)          -- jumps directly to the instruction after clzero!
        //   clzero             (0F 01 FC)       -- offset 2
        // target:
        //   add rax, 0x40      (48 83 C0 40)    -- offset 5
        //   ret                (C3)
        byte[] guestCode =
        [
            0x74, 0x03,
            0x0F, 0x01, 0xFC,
            0x48, 0x83, 0xC0, 0x40,
            0xC3,
        ];

        var code = AllocatePage(HostMemory.PAGE_EXECUTE_READWRITE);
        var backend = RuntimeHelpers.GetUninitializedObject(typeof(DirectExecutionBackend));
        try
        {
            guestCode.CopyTo(new Span<byte>((void*)code, guestCode.Length));
            var counts = PatchTlsPatternsInRangeWithClzero.Invoke(
                backend,
                [(ulong)code, (ulong)code + (ulong)guestCode.Length, true])!;
            var clzeroCount = (int)counts.GetType().GetProperty("Clzeros")!.GetValue(counts)!;
            Assert.Equal(0, clzeroCount);
            Assert.Equal(0x0F, Marshal.ReadByte(code, 2));
        }
        finally
        {
            ReleaseClzeroStubRegionMethod.Invoke(backend, null);
            Assert.True(HostMemory.Free((void*)code, 0, HostMemory.MEM_RELEASE));
        }
    }

    [Fact]
    public void ClzeroBinaryPatchSkipsStackAndRipRelativeSuccessors()
    {
        if (!OperatingSystem.IsWindows() || RuntimeInformation.ProcessArchitecture != Architecture.X64)
        {
            return;
        }

        // Site 1: clzero; add rsp, 8        (touches rsp -> skip)
        // Site 2: clzero; mov rax, [rip+0]  (RIP-relative -> skip)
        // Site 3: clzero; add rax, 0x40     (safe -> patch)
        // Site 4: clzero; add rax, 0x40     (safe -> patch, reuses trampoline!)
        byte[] guestCode =
        [
            0x0F, 0x01, 0xFC,
            0x48, 0x83, 0xC4, 0x08,
            0x0F, 0x01, 0xFC,
            0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00,
            0x0F, 0x01, 0xFC,
            0x48, 0x83, 0xC0, 0x40,
            0x0F, 0x01, 0xFC,
            0x48, 0x83, 0xC0, 0x40,
            0xC3,
        ];

        var code = AllocatePage(HostMemory.PAGE_EXECUTE_READWRITE);
        var backend = RuntimeHelpers.GetUninitializedObject(typeof(DirectExecutionBackend));
        try
        {
            guestCode.CopyTo(new Span<byte>((void*)code, guestCode.Length));
            var counts = PatchTlsPatternsInRangeWithClzero.Invoke(
                backend,
                [(ulong)code, (ulong)code + (ulong)guestCode.Length, true])!;
            var clzeroCount = (int)counts.GetType().GetProperty("Clzeros")!.GetValue(counts)!;
            Assert.Equal(2, clzeroCount);
            Assert.Equal(0x0F, Marshal.ReadByte(code, 0));
            Assert.Equal(0x0F, Marshal.ReadByte(code, 7));
            Assert.Equal(0xE8, Marshal.ReadByte(code, 17));
            Assert.Equal(0xE8, Marshal.ReadByte(code, 24));
        }
        finally
        {
            ReleaseClzeroStubRegionMethod.Invoke(backend, null);
            Assert.True(HostMemory.Free((void*)code, 0, HostMemory.MEM_RELEASE));
        }
    }

    private static nint AllocatePage(uint protection)
    {
        var size = checked((nuint)Environment.SystemPageSize);
        var mapping = (nint)HostMemory.Alloc(
            null,
            size,
            HostMemory.MEM_COMMIT | HostMemory.MEM_RESERVE,
            protection);
        Assert.NotEqual((nint)0, mapping);
        return mapping;
    }
}
