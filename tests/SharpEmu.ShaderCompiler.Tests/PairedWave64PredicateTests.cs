// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class PairedWave64PredicateTests
{
    [Fact]
    public void PairedWave64_CompareAndExecMovesAvoid64BitPredicateConversion()
    {
        var program = Program(
            Vopc(0, "VCmpEqU32", Gen5Operand.Vector(0), 1),
            Vop2(4, "VCndmaskB32", 2, Gen5Operand.Vector(0), Gen5Operand.Vector(1)),
            MoveScalarRegister(8, 126, 0),
            MoveScalarRegister(12, 127, 1),
            Vop2(16, "VAddU32", 2, Gen5Operand.Vector(2), Gen5Operand.Vector(0)),
            DataShare(20, "DsWriteB32", gds: false, [Gen5Operand.Vector(0), Gen5Operand.Vector(2)], []),
            EndProgram(28));

        var spirv = CompilePairedWave64(program, localSizeX: 64, userDataCount: 4);
        var instructions = ReadInstructions(spirv);

        Assert.Equal(32u, LocalSizeX(instructions));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.UConvert));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.ShiftLeftLogical));
        Assert.Contains(instructions, static words => Opcode(words) == SpirvOp.IEqual);
        Assert.Contains(instructions, static words => Opcode(words) == SpirvOp.Select);
    }

    [Fact]
    public void PairedWave64_ExecHiMoveUpdatesOnlyExecUpperHalf()
    {
        var program = Program(
            Sopc(0, "SCmpEqU32", Gen5Operand.Scalar(0), Operand(0)),
            Branch(4, "SCbranchScc0", 1),
            MoveScalarRegister(8, 127, 1),
            Vop2(12, "VAddU32", 2, Gen5Operand.Vector(0), Gen5Operand.Vector(1)),
            DataShare(16, "DsWriteB32", gds: false, [Gen5Operand.Vector(0), Gen5Operand.Vector(2)], []),
            EndProgram(24));

        var spirv = CompilePairedWave64(program, localSizeX: 64, userDataCount: 4);
        var inspector = new SpirvModuleInspector(spirv);
        var instructions = ReadInstructions(spirv);

        Assert.Equal(32u, LocalSizeX(instructions));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.UConvert));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.ShiftLeftLogical));

        var execUpperHalf = Assert.Single(inspector.Names, static pair => pair.Value == "execUpperHalf").Key;
        var counts = CountLaneBitPredicateStoresByTarget(instructions);
        // Entry init writes both _lowerExec (1x) and execUpperHalf (1x); SMovB32 127 adds a store only to execUpperHalf.
        Assert.Equal(2, counts.Count);
        Assert.Equal(2, counts[execUpperHalf]);
        Assert.Equal(1, Assert.Single(counts, pair => pair.Key != execUpperHalf).Value);
    }

    [Fact]
    public void PairedWave64_VccHiMoveUpdatesOnlyVccUpperHalf()
    {
        var program = Program(
            Sopc(0, "SCmpEqU32", Gen5Operand.Scalar(0), Operand(0)),
            Branch(4, "SCbranchScc0", 1),
            MoveScalarRegister(8, 107, 1),
            Vop2(12, "VCndmaskB32", 2, Gen5Operand.Vector(0), Gen5Operand.Vector(1)),
            DataShare(16, "DsWriteB32", gds: false, [Gen5Operand.Vector(0), Gen5Operand.Vector(2)], []),
            EndProgram(24));

        var spirv = CompilePairedWave64(program, localSizeX: 64, userDataCount: 4);
        var inspector = new SpirvModuleInspector(spirv);
        var instructions = ReadInstructions(spirv);

        Assert.Equal(32u, LocalSizeX(instructions));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.UConvert));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.ShiftLeftLogical));

        var execUpperHalf = Assert.Single(inspector.Names, static pair => pair.Value == "execUpperHalf").Key;
        var vccUpperHalf = Assert.Single(inspector.Names, static pair => pair.Value == "vccUpperHalf").Key;
        var counts = CountLaneBitPredicateStoresByTarget(instructions);
        // Entry init writes lower _exec + execUpperHalf; SMovB32 107 writes ONLY vccUpperHalf (never lower _vcc).
        Assert.Equal(3, counts.Count);
        Assert.Equal(1, counts[vccUpperHalf]);
        Assert.Equal(1, counts[execUpperHalf]);
        Assert.Equal(1, Assert.Single(counts, pair => pair.Key != vccUpperHalf && pair.Key != execUpperHalf).Value);
    }

    [Theory]
    [InlineData(64u)]
    [InlineData(48u)]
    public void PairedWave64_LocalComparePredicatesAndPartialWavesCompileWithout64BitPredicates(uint localSizeX)
    {
        var program = Program(
            DataShare(0, "DsWriteB32", gds: false, [Gen5Operand.Vector(0), Gen5Operand.Vector(1)], []),
            Vopc(8, "VCmpEqU32", Gen5Operand.Vector(0), 1),
            Vop2(12, "VCndmaskB32", 2, Gen5Operand.Vector(0), Gen5Operand.Vector(1)),
            EndProgram(16));

        var spirv = CompilePairedWave64(program, localSizeX, userDataCount: 0);
        var instructions = ReadInstructions(spirv);

        Assert.Equal(32u, LocalSizeX(instructions));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.UConvert));
        Assert.Equal(0, Count64BitOps(instructions, SpirvOp.ShiftLeftLogical));
        Assert.True(instructions.Count(static words => Opcode(words) == SpirvOp.GroupNonUniformBallot) <= 2);
    }

    private static byte[] CompilePairedWave64(Gen5ShaderProgram program, uint localSizeX, uint userDataCount)
    {
        var (plan, resources, layout) = Prepare(program, userDataCount: userDataCount);
        var request = new ShaderCompileRequest(plan, resources, layout)
        {
            LocalSizeX = localSizeX,
            ThreadCountX = localSizeX,
            WaveSize = 64,
            HostComputeSubgroup32 = true,
            HostWave64Supported = false,
        };

        Assert.True(Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error), error);
        Gen5MipStatisticsTests.ValidateWhenAvailable(shader.Spirv);
        return shader.Spirv;
    }

    private static List<uint[]> ReadInstructions(byte[] spirv)
    {
        var words = new uint[spirv.Length / sizeof(uint)];
        Buffer.BlockCopy(spirv, 0, words, 0, spirv.Length);
        var instructions = new List<uint[]>();
        for (var offset = 5; offset < words.Length;)
        {
            var count = checked((int)(words[offset] >> 16));
            Assert.True(count > 0);
            instructions.Add(words[offset..(offset + count)]);
            offset += count;
        }

        return instructions;
    }

    private static SpirvOp Opcode(uint[] words) => (SpirvOp)(words[0] & 0xFFFF);

    private static uint LocalSizeX(List<uint[]> instructions)
    {
        var localSize = Assert.Single(instructions, static words =>
            Opcode(words) == SpirvOp.ExecutionMode &&
            words.Length >= 6 &&
            words[2] == (uint)SpirvExecutionMode.LocalSize);
        return localSize[3];
    }

    private static int Count64BitOps(List<uint[]> instructions, SpirvOp opcode)
    {
        var int64Types = instructions
            .Where(static words => Opcode(words) == SpirvOp.TypeInt && words.Length >= 4 && words[2] == 64)
            .Select(static words => words[1])
            .ToHashSet();
        return instructions.Count(words => Opcode(words) == opcode && words.Length >= 3 && int64Types.Contains(words[1]));
    }

    private static Dictionary<uint, int> CountLaneBitPredicateStoresByTarget(List<uint[]> instructions)
    {
        var definitions = new Dictionary<uint, uint[]>();
        foreach (var words in instructions)
        {
            if (HasResultId(Opcode(words)) && words.Length >= 3)
            {
                definitions[words[2]] = words;
            }
        }

        var counts = new Dictionary<uint, int>();
        foreach (var words in instructions)
        {
            if (Opcode(words) == SpirvOp.Store && words.Length >= 3 && IsLaneBitPredicateValue(words[2], definitions))
            {
                counts[words[1]] = counts.GetValueOrDefault(words[1]) + 1;
            }
        }

        return counts;
    }

    private static bool HasResultId(SpirvOp opcode) => opcode is
        SpirvOp.Select or
        SpirvOp.INotEqual or
        SpirvOp.BitwiseAnd or
        SpirvOp.ShiftLeftLogical;

    private static bool IsLaneBitPredicateValue(uint valueId, Dictionary<uint, uint[]> definitions)
    {
        if (!definitions.TryGetValue(valueId, out var select) || Opcode(select) != SpirvOp.Select || select.Length < 6)
        {
            return false;
        }

        if (!definitions.TryGetValue(select[3], out var compare) || Opcode(compare) != SpirvOp.INotEqual || compare.Length < 5)
        {
            return false;
        }

        return IsLaneBitMask(compare[3], definitions) || IsLaneBitMask(compare[4], definitions);
    }

    private static bool IsLaneBitMask(uint id, Dictionary<uint, uint[]> definitions)
    {
        if (!definitions.TryGetValue(id, out var bitwiseAnd) || Opcode(bitwiseAnd) != SpirvOp.BitwiseAnd || bitwiseAnd.Length < 5)
        {
            return false;
        }

        return IsShiftLeft(bitwiseAnd[3], definitions) || IsShiftLeft(bitwiseAnd[4], definitions);
    }

    private static bool IsShiftLeft(uint id, Dictionary<uint, uint[]> definitions) =>
        definitions.TryGetValue(id, out var shift) && Opcode(shift) == SpirvOp.ShiftLeftLogical;
}
