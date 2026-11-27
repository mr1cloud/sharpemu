// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Text;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class SpirvOutliningTests
{
    private const uint DontInlineControl = 2u;

    [Fact]
    public void SubdwordBufferStoresShareOneOutlinedHelper()
    {
        var program = ResourceTestProgram.Program(
            ResourceTestProgram.BufferAccess(0, "BufferStoreByte", resourceRegister: 0, offset: 0, vectorData: 4, offsetEnabled: true, vectorAddress: 1),
            ResourceTestProgram.BufferAccess(8, "BufferStoreShort", resourceRegister: 0, offset: 2, vectorData: 5, offsetEnabled: true, vectorAddress: 1),
            ResourceTestProgram.BufferAccess(16, "BufferStoreByte", resourceRegister: 0, offset: 5, vectorData: 6, offsetEnabled: true, vectorAddress: 1),
            ResourceTestProgram.EndProgram(24));
        var request = ResourceTestProgram.Request(program, userDataCount: 4);

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Gen5MipStatisticsTests.ValidateWhenAvailable(shader.Spirv);

        var module = new ParsedModule(shader.Spirv);
        var helper = Assert.Single(module.Functions, fn => module.Names.GetValueOrDefault(fn.Id) == "storeBufferMaskedWord0");
        Assert.Equal(DontInlineControl, helper.Control);
        Assert.Equal(1, module.CountOpcode(SpirvOp.LoopMerge));
        Assert.Equal(1, helper.CountOpcode(SpirvOp.LoopMerge));
        Assert.True(module.CountCallsTo(helper.Id) >= 4, $"expected at least 4 calls to storeBufferMaskedWord0, got {module.CountCallsTo(helper.Id)}");
    }

    [Fact]
    public void SubdwordDeviceStoresShareOneOutlinedHelper()
    {
        var program = ResourceTestProgram.Program(
            ResourceTestProgram.GlobalAccess(0, "GlobalStoreByte", scalarAddress: 0, offset: 0, vectorAddress: 1),
            ResourceTestProgram.GlobalAccess(8, "GlobalStoreShort", scalarAddress: 0, offset: 2, vectorAddress: 1),
            ResourceTestProgram.GlobalAccess(16, "GlobalStoreByte", scalarAddress: 0, offset: 5, vectorAddress: 1),
            ResourceTestProgram.EndProgram(24));
        var request = ResourceTestProgram.Request(program, userDataCount: 2);

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Gen5MipStatisticsTests.ValidateWhenAvailable(shader.Spirv);

        var module = new ParsedModule(shader.Spirv);
        var helper = Assert.Single(module.Functions, fn => module.Names.GetValueOrDefault(fn.Id) == "storeDeviceMaskedWord");
        Assert.Equal(DontInlineControl, helper.Control);
        Assert.Equal(1, module.CountOpcode(SpirvOp.LoopMerge));
        Assert.Equal(1, helper.CountOpcode(SpirvOp.LoopMerge));
        Assert.True(module.CountCallsTo(helper.Id) >= 4, $"expected at least 4 calls to storeDeviceMaskedWord, got {module.CountCallsTo(helper.Id)}");
    }

    [Fact]
    public void FlattenedResourceTableIsNonWritableAndHoistsRepeatedSlotLoads()
    {
        var program = ResourceTestProgram.Program(
            ResourceTestProgram.ScalarLoad(0, baseRegister: 0, destination: 8, count: 1, immediateOffset: 0),
            ResourceTestProgram.Sopc(8, "SCmpEqU32", Gen5Operand.Scalar(2), ResourceTestProgram.Operand(0)),
            ResourceTestProgram.Branch(12, "SCbranchScc0", 2),
            ResourceTestProgram.ScalarLoad(16, baseRegister: 0, destination: 9, count: 1, immediateOffset: 0),
            ResourceTestProgram.Sop2(24, "SAddU32", 10, Gen5Operand.Scalar(8), Gen5Operand.Scalar(9)),
            ResourceTestProgram.MoveVectorFromScalar(28, destination: 4, scalarRegister: 10),
            ResourceTestProgram.BufferStore(32, resourceRegister: 4, offset: 0),
            ResourceTestProgram.EndProgram(40));
        var request = ResourceTestProgram.Request(program, userDataCount: 8);

        Assert.True(request.Memory.TryGetIndex(0, 0, out var firstMemory));
        Assert.True(request.Memory.TryGetIndex(16, 0, out var secondMemory));
        Assert.True(request.Memory[firstMemory].PlanningOnly);
        Assert.True(request.Memory[secondMemory].PlanningOnly);
        Assert.Equal(request.FlattenedSlotByMemoryIndex[firstMemory], request.FlattenedSlotByMemoryIndex[secondMemory]);

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Gen5MipStatisticsTests.ValidateWhenAvailable(shader.Spirv);

        var module = new ParsedModule(shader.Spirv);
        var tableVariable = Assert.Single(module.Names, entry => entry.Value == "flattenedResourceTable").Key;
        var pointerType = module.VariableTypes[tableVariable];
        var structType = module.PointerPointees[pointerType];
        Assert.Contains(SpirvDecoration.NonWritable, module.MemberDecorations[(structType, 0)]);

        var main = Assert.Single(module.Functions, fn => module.EntryPointIds.Contains(fn.Id));
        var tablePointers = new HashSet<uint>();
        var tableLoadIndices = new List<int>();
        var firstTerminatorIndex = -1;
        for (var index = 0; index < main.Instructions.Count; index++)
        {
            var words = main.Instructions[index];
            var opcode = (SpirvOp)(words[0] & 0xFFFF);
            if (firstTerminatorIndex < 0 && opcode is SpirvOp.Branch or SpirvOp.BranchConditional or SpirvOp.Switch or SpirvOp.Return or SpirvOp.ReturnValue or SpirvOp.Kill or SpirvOp.Unreachable)
            {
                firstTerminatorIndex = index;
            }

            if (opcode == SpirvOp.AccessChain && words.Length >= 4 && words[3] == tableVariable)
            {
                tablePointers.Add(words[2]);
            }
            else if (opcode == SpirvOp.Load && words.Length >= 4 && tablePointers.Contains(words[3]))
            {
                tableLoadIndices.Add(index);
            }
        }

        var loadIndex = Assert.Single(tableLoadIndices);
        Assert.True(firstTerminatorIndex < 0 || loadIndex < firstTerminatorIndex,
            $"expected flattenedResourceTable load at instruction {loadIndex} to be hoisted into main's entry block before terminator {firstTerminatorIndex}");
    }

    private sealed class ParsedFunction(uint id, uint control)
    {
        public uint Id { get; } = id;
        public uint Control { get; } = control;
        public List<uint[]> Instructions { get; } = [];

        public int CountOpcode(SpirvOp opcode) =>
            Instructions.Count(words => (SpirvOp)(words[0] & 0xFFFF) == opcode);
    }

    private sealed class ParsedModule
    {
        private const ushort OpMemberDecorate = 72;

        public Dictionary<uint, string> Names { get; } = [];
        public Dictionary<(uint TypeId, uint Member), HashSet<SpirvDecoration>> MemberDecorations { get; } = [];
        public Dictionary<uint, uint> PointerPointees { get; } = [];
        public Dictionary<uint, uint> VariableTypes { get; } = [];
        public HashSet<uint> EntryPointIds { get; } = [];
        public List<ParsedFunction> Functions { get; } = [];
        private readonly List<uint[]> _allInstructions = [];

        public ParsedModule(byte[] spirv)
        {
            var words = new uint[spirv.Length / sizeof(uint)];
            Buffer.BlockCopy(spirv, 0, words, 0, spirv.Length);
            ParsedFunction? current = null;
            for (var offset = 5; offset < words.Length;)
            {
                var count = (int)(words[offset] >> 16);
                var slice = new uint[count];
                Array.Copy(words, offset, slice, 0, count);
                _allInstructions.Add(slice);
                var rawOpcode = (ushort)(slice[0] & 0xFFFF);
                if (rawOpcode == OpMemberDecorate && count >= 4)
                {
                    var key = (slice[1], slice[2]);
                    if (!MemberDecorations.TryGetValue(key, out var set))
                    {
                        set = [];
                        MemberDecorations[key] = set;
                    }

                    set.Add((SpirvDecoration)slice[3]);
                    offset += Math.Max(count, 1);
                    continue;
                }

                var opcode = (SpirvOp)rawOpcode;
                switch (opcode)
                {
                    case SpirvOp.EntryPoint when count >= 3:
                        EntryPointIds.Add(slice[2]);
                        break;
                    case SpirvOp.Name when count >= 3:
                        Names[slice[1]] = DecodeString(slice, 2, count);
                        break;
                    case SpirvOp.TypePointer when count >= 4:
                        PointerPointees[slice[1]] = slice[3];
                        break;
                    case SpirvOp.Variable when count >= 4:
                        VariableTypes[slice[2]] = slice[1];
                        break;
                    case SpirvOp.Function when count >= 5:
                        current = new ParsedFunction(slice[2], slice[3]);
                        Functions.Add(current);
                        break;
                    case SpirvOp.FunctionEnd:
                        current = null;
                        break;
                    default:
                        current?.Instructions.Add(slice);
                        break;
                }

                offset += Math.Max(count, 1);
            }
        }

        public int CountOpcode(SpirvOp opcode) =>
            _allInstructions.Count(words => (SpirvOp)(words[0] & 0xFFFF) == opcode);

        public int CountCallsTo(uint functionId) =>
            _allInstructions.Count(words => (SpirvOp)(words[0] & 0xFFFF) == SpirvOp.FunctionCall && words.Length >= 4 && words[3] == functionId);

        private static string DecodeString(uint[] words, int start, int end)
        {
            var bytes = new List<byte>();
            for (var index = start; index < end; index++)
            {
                foreach (var shift in new[] { 0, 8, 16, 24 })
                {
                    var value = (byte)(words[index] >> shift);
                    if (value == 0)
                    {
                        return Encoding.UTF8.GetString(bytes.ToArray());
                    }

                    bytes.Add(value);
                }
            }

            return Encoding.UTF8.GetString(bytes.ToArray());
        }
    }
}
