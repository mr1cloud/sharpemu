// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using System.Text;
using SharpEmu.ShaderCompiler;
using SharpEmu.ShaderCompiler.Resources;
using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class DynamicVectorArrayPolicyTests
{
    private const uint M0 = 124;

    [Fact]
    public void LargeReadOnlyMovrelsComputeShader_KeepsScalarVgprsAndSkipsNeverWritten()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        // Unbounded M0 loaded from user data s0.
        instructions.Add(ResourceTestProgram.MoveScalarRegister(pc, M0, 0));
        pc += 4;

        // Build a large static VGPR footprint touching v10..v139 and v150 while leaving
        // v3..v9 and v140..v149 never written.
        for (var pass = 0; pass < 12; pass++)
        {
            for (uint register = 10; register <= 139; register++)
            {
                var source = register == 10 ? 0u : register - 1;
                instructions.Add(ResourceTestProgram.Vop2(
                    pc,
                    "VAddU32",
                    register,
                    Gen5Operand.Vector(source),
                    Gen5Operand.Vector(0)));
                pc += 4;
            }
        }

        instructions.Add(ResourceTestProgram.Vop2(
            pc,
            "VAddU32",
            150,
            Gen5Operand.Vector(139),
            Gen5Operand.Vector(0)));
        pc += 4;

        // 12 unbounded read-only VMovrelsB32 over a ~160-VGPR range would exceed the
        // 1024 write-mode budget (12 * 161 = 1932), but stays scalar under the
        // cost-proportional read-only budget.
        for (var read = 0; read < 12; read++)
        {
            instructions.Add(MoveRelative(pc, "VMovrelsB32", destination: 151, source: 0));
            pc += 4;
            instructions.Add(ResourceTestProgram.Vop2(
                pc,
                "VAddU32",
                152,
                Gen5Operand.Vector(152),
                Gen5Operand.Vector(151)));
            pc += 4;
        }

        instructions.Add(ResourceTestProgram.GlobalMemory(
            pc,
            "GlobalStoreDword",
            scalarAddress: 4,
            vectorAddress: 0,
            destination: 0,
            source: 152,
            dwords: 1));
        pc += 8;
        instructions.Add(ResourceTestProgram.EndProgram(pc));

        var spirv = CompileCompute(instructions);
        var info = InspectSpirv(spirv);

        Assert.DoesNotContain("vgpr", info.Names);
        Assert.Empty(info.PrivateArrayVariables);
        Assert.Contains("v10", info.Names);
        Assert.Contains("v150", info.Names);

        for (uint register = 140; register <= 149; register++)
        {
            Assert.DoesNotContain($"v{register}", info.Names);
            Assert.DoesNotContain(register, info.SelectedEqualConstants);
        }
    }

    [Fact]
    public void UnboundedMovreldExceedingBudget_StillUsesPrivateDynamicVectorArray()
    {
        var instructions = new List<Gen5ShaderInstruction>();
        uint pc = 0;

        instructions.Add(ResourceTestProgram.MoveScalarRegister(pc, M0, 0));
        pc += 4;

        // Even with a large static VGPR footprint, unbounded dynamic writes keep the
        // existing 1024-select budget and demote VGPRs to the Private array.
        for (var pass = 0; pass < 12; pass++)
        {
            for (uint register = 10; register <= 120; register++)
            {
                instructions.Add(ResourceTestProgram.Vop2(
                    pc,
                    "VAddU32",
                    register,
                    Gen5Operand.Vector(register - 1),
                    Gen5Operand.Vector(0)));
                pc += 4;
            }
        }

        for (var write = 0; write < 10; write++)
        {
            instructions.Add(MoveRelative(pc, "VMovreldB32", destination: 0, source: 120));
            pc += 4;
        }

        instructions.Add(ResourceTestProgram.EndProgram(pc));

        var spirv = CompileCompute(instructions);
        var info = InspectSpirv(spirv);

        Assert.Contains("vgpr", info.Names);
        Assert.Contains("vgpr", info.PrivateArrayVariables);
    }

    [Fact]
    public void StageInitializedComputeVgprs_AreNeverSkippedByLoadVDynamic()
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            ResourceTestProgram.MoveScalarRegister(0, M0, 0),
            // Never write v0..v2 (stage-initialized thread IDs), never write v3..v4,
            // only write v5.
            ResourceTestProgram.MoveVector(4, 5, 42),
            MoveRelative(8, "VMovrelsB32", destination: 6, source: 0),
            ResourceTestProgram.GlobalMemory(
                12,
                "GlobalStoreDword",
                scalarAddress: 4,
                vectorAddress: 0,
                destination: 0,
                source: 6,
                dwords: 1),
            ResourceTestProgram.EndProgram(20),
        };

        var spirv = CompileCompute(instructions);
        var info = InspectSpirv(spirv);

        Assert.DoesNotContain("vgpr", info.Names);
        Assert.Contains(0u, info.SelectedEqualConstants);
        Assert.Contains(1u, info.SelectedEqualConstants);
        Assert.Contains(2u, info.SelectedEqualConstants);
        Assert.Contains(5u, info.SelectedEqualConstants);

        Assert.DoesNotContain("v3", info.Names);
        Assert.DoesNotContain("v4", info.Names);
        Assert.DoesNotContain(3u, info.SelectedEqualConstants);
        Assert.DoesNotContain(4u, info.SelectedEqualConstants);
    }

    [Fact]
    public void MultiDwordGlobalLoad_MarksAllDestinationVgprsAsPossiblyNonzero()
    {
        var instructions = new List<Gen5ShaderInstruction>
        {
            ResourceTestProgram.MoveScalarRegister(0, M0, 0),
            // global_load_dwordx4 writing v4..v7; v3 and v8..v9 are never written.
            ResourceTestProgram.GlobalMemory(
                4,
                "GlobalLoadDwordx4",
                scalarAddress: 2,
                vectorAddress: 0,
                destination: 4,
                source: 0,
                dwords: 4),
            MoveRelative(12, "VMovrelsB32", destination: 10, source: 4),
            ResourceTestProgram.GlobalMemory(
                16,
                "GlobalStoreDword",
                scalarAddress: 4,
                vectorAddress: 0,
                destination: 0,
                source: 10,
                dwords: 1),
            ResourceTestProgram.EndProgram(24),
        };

        var spirv = CompileCompute(instructions);
        var info = InspectSpirv(spirv);

        Assert.DoesNotContain("vgpr", info.Names);
        Assert.Contains(4u, info.SelectedEqualConstants);
        Assert.Contains(5u, info.SelectedEqualConstants);
        Assert.Contains(6u, info.SelectedEqualConstants);
        Assert.Contains(7u, info.SelectedEqualConstants);

        Assert.DoesNotContain("v3", info.Names);
        Assert.DoesNotContain("v8", info.Names);
        Assert.DoesNotContain("v9", info.Names);
        Assert.DoesNotContain(3u, info.SelectedEqualConstants);
        Assert.DoesNotContain(8u, info.SelectedEqualConstants);
        Assert.DoesNotContain(9u, info.SelectedEqualConstants);
    }

    private static Gen5ShaderInstruction MoveRelative(uint pc, string opcode, uint destination, uint source) =>
        new(
            pc,
            Gen5ShaderEncoding.Vop1,
            opcode,
            [0u],
            [Gen5Operand.Vector(source), Gen5Operand.Scalar(M0)],
            [Gen5Operand.Vector(destination)],
            null);

    private static byte[] CompileCompute(IReadOnlyList<Gen5ShaderInstruction> instructions)
    {
        var program = new Gen5ShaderProgram(0x1000, instructions);
        var request = ResourceTestProgram.Request(program, ShaderStage.Compute);
        Assert.True(
            Gen5SpirvTranslator.TryCompileProgram(request, out var shader, out var error),
            error);
        return shader.Spirv;
    }

    private sealed record SpirvInspection(
        HashSet<string> Names,
        HashSet<string> PrivateArrayVariables,
        HashSet<uint> SelectedEqualConstants);

    private static SpirvInspection InspectSpirv(byte[] spirv)
    {
        Assert.Equal(0, spirv.Length % sizeof(uint));
        Assert.True(spirv.Length >= 5 * sizeof(uint));
        Assert.Equal(0x07230203u, BinaryPrimitives.ReadUInt32LittleEndian(spirv));

        var names = new HashSet<string>(StringComparer.Ordinal);
        var nameById = new Dictionary<uint, string>();
        var arrayTypes = new HashSet<uint>();
        var privateArrayPointerTypes = new HashSet<uint>();
        var privateArrayVariableIds = new List<uint>();
        var uint32Types = new HashSet<uint>();
        var uintConstants = new Dictionary<uint, uint>();
        var equalConstantsByResult = new Dictionary<uint, uint>();
        var selectConditions = new List<uint>();

        for (var offset = 5 * sizeof(uint); offset < spirv.Length;)
        {
            var header = BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset));
            var wordCount = checked((int)(header >> 16));
            var opcode = (SpirvOp)(ushort)header;
            Assert.True(wordCount > 0);

            uint Word(int index) =>
                BinaryPrimitives.ReadUInt32LittleEndian(spirv.AsSpan(offset + index * sizeof(uint)));

            switch (opcode)
            {
                case SpirvOp.Name when wordCount >= 3:
                {
                    var target = Word(1);
                    var literalBytes = spirv.AsSpan(offset + 2 * sizeof(uint), (wordCount - 2) * sizeof(uint));
                    var nul = literalBytes.IndexOf((byte)0);
                    if (nul >= 0)
                    {
                        literalBytes = literalBytes[..nul];
                    }

                    var name = Encoding.UTF8.GetString(literalBytes);
                    names.Add(name);
                    nameById[target] = name;
                    break;
                }
                case SpirvOp.TypeInt when wordCount >= 4 && Word(2) == 32 && Word(3) == 0:
                    uint32Types.Add(Word(1));
                    break;
                case SpirvOp.TypeArray when wordCount >= 4:
                    arrayTypes.Add(Word(1));
                    break;
                case SpirvOp.TypePointer when wordCount >= 4 &&
                    (SpirvStorageClass)Word(2) == SpirvStorageClass.Private &&
                    arrayTypes.Contains(Word(3)):
                    privateArrayPointerTypes.Add(Word(1));
                    break;
                case SpirvOp.Variable when wordCount >= 4 &&
                    (SpirvStorageClass)Word(3) == SpirvStorageClass.Private &&
                    privateArrayPointerTypes.Contains(Word(1)):
                    privateArrayVariableIds.Add(Word(2));
                    break;
                case SpirvOp.Constant when wordCount >= 4 && uint32Types.Contains(Word(1)):
                    uintConstants[Word(2)] = Word(3);
                    break;
                case SpirvOp.ConstantNull when wordCount >= 3 && uint32Types.Contains(Word(1)):
                    uintConstants[Word(2)] = 0;
                    break;
                case SpirvOp.IEqual when wordCount >= 5:
                {
                    var result = Word(2);
                    var left = Word(3);
                    var right = Word(4);
                    if (uintConstants.TryGetValue(right, out var rightConstant))
                    {
                        equalConstantsByResult[result] = rightConstant;
                    }
                    else if (uintConstants.TryGetValue(left, out var leftConstant))
                    {
                        equalConstantsByResult[result] = leftConstant;
                    }

                    break;
                }
                case SpirvOp.Select when wordCount >= 6:
                    selectConditions.Add(Word(3));
                    break;
            }

            offset += wordCount * sizeof(uint);
        }

        var privateArrayVariables = new HashSet<string>(StringComparer.Ordinal);
        foreach (var id in privateArrayVariableIds)
        {
            privateArrayVariables.Add(nameById.TryGetValue(id, out var name) ? name : $"%{id}");
        }

        var selectedEqualConstants = new HashSet<uint>();
        foreach (var condition in selectConditions)
        {
            if (equalConstantsByResult.TryGetValue(condition, out var constant))
            {
                selectedEqualConstants.Add(constant);
            }
        }

        return new SpirvInspection(names, privateArrayVariables, selectedEqualConstants);
    }
}
