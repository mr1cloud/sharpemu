// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Numerics;

namespace SharpEmu.ShaderCompiler.Ir;

// The values M0 can hold at each V_MOVRELS/V_MOVRELD/V_MOVRELSD, when they follow from
// straight-line scalar or vector code (bit-field extracts, masks, shifts, adds, multiplies,
// selects and bit scans, including values moved to SGPRs via V_READFIRSTLANE/V_READLANE).
// A backend that knows the few registers a relative move can reach selects among them with
// constant register numbers; a register file indexed at run time keeps the whole file out of
// hardware registers.
public static class Gen5MoveRelativeOffsets
{
    private const uint M0Register = 124;
    private const uint ExecLow = 126;
    private const uint ExecHigh = 127;
    private const int MaxValues = 64;
    private const int MaxDepth = 10;
    private const int MaxScanSteps = 4096;

    private sealed class AnalysisContext(
        Gen5ShaderProgram program,
        Dictionary<uint, int> targets)
    {
        private IReadOnlySet<uint>? _execFull;

        public IReadOnlyList<Gen5ShaderInstruction> Instructions => program.Instructions;

        public Dictionary<uint, int> Targets => targets;

        public bool IsExecFull(uint pc)
        {
            _execFull ??= Gen5ExecFullAnalysis.Analyze(program, wave32: false);
            return _execFull.Contains(pc);
        }
    }

    // Maps the pc of every resolved relative move to its sorted M0 values; a move that
    // is missing from the map could not be bounded.
    public static IReadOnlyDictionary<uint, uint[]> Analyze(Gen5ShaderProgram program)
    {
        var result = new Dictionary<uint, uint[]>();
        var instructions = program.Instructions;
        var hasRelativeMove = false;
        foreach (var instruction in instructions)
        {
            if (IsBoundedRelativeMove(instruction.Opcode))
            {
                hasRelativeMove = true;
            }

            // A computed jump can land anywhere; no straight-line reasoning holds.
            if (instruction.Opcode is "SSetpcB64" or "SSwappcB64" or "SRfeB64")
            {
                return result;
            }
        }

        if (!hasRelativeMove)
        {
            return result;
        }

        // Branch target pc -> index of the single branch reaching it, or -1 when several
        // branches do.
        var targets = new Dictionary<uint, int>();
        for (var index = 0; index < instructions.Count; index++)
        {
            if (Gen5IrBranchResolver.Instance.TryGetBranchTarget(instructions[index], out var target))
            {
                targets[target] = targets.ContainsKey(target) ? -1 : index;
            }
        }

        var context = new AnalysisContext(program, targets);
        for (var index = 0; index < instructions.Count; index++)
        {
            if (!IsBoundedRelativeMove(instructions[index].Opcode))
            {
                continue;
            }

            var values = new HashSet<uint>();
            if (TryResolveRegister(context, index, M0Register, 0, values) &&
                values.Count > 0 &&
                values.Count <= MaxValues)
            {
                var sorted = values.ToArray();
                Array.Sort(sorted);
                result[instructions[index].Pc] = sorted;
            }
        }

        return result;
    }

    private static bool IsBoundedRelativeMove(string opcode) =>
        opcode is "VMovrelsB32" or "VMovreldB32" or "VMovrelsdB32";

    // The values the scalar register holds when the instruction at index executes.
    private static bool TryResolveRegister(
        AnalysisContext context,
        int index,
        uint register,
        int depth,
        HashSet<uint> values)
    {
        if (depth > MaxDepth)
        {
            return false;
        }

        var instructions = context.Instructions;
        var targets = context.Targets;
        var steps = 0;
        for (var at = index - 1; at >= 0; at--)
        {
            if (++steps > MaxScanSteps)
            {
                return false;
            }

            // A join: follow its one predecessor, or give up when two paths meet.
            if (targets.TryGetValue(instructions[at + 1].Pc, out var branch))
            {
                var fallsThrough = !Gen5IrBranchResolver.IsUnconditionalBranch(instructions[at]) &&
                    !Gen5IrBranchResolver.IsTerminator(instructions[at]);
                if (branch < 0 || fallsThrough)
                {
                    return false;
                }

                at = branch;
                continue;
            }

            var instruction = instructions[at];
            if (!Writes(instruction, register, out var exact))
            {
                continue;
            }

            return exact && TryEvaluate(context, at, instruction, depth, values);
        }

        return false;
    }

    // The values the vector register holds for the queried lane when the instruction at index executes.
    // When requireFullExec is false (V_READFIRSTLANE), any EXEC widening along the path invalidates
    // prior vector writes because an inactive lane can become active with a stale value.
    // When requireFullExec is true (V_READLANE), every vector write in the chain must run under a full EXEC.
    private static bool TryResolveVectorRegister(
        AnalysisContext context,
        int index,
        uint register,
        int depth,
        bool requireFullExec,
        HashSet<uint> values)
    {
        if (depth > MaxDepth)
        {
            return false;
        }

        var instructions = context.Instructions;
        var targets = context.Targets;
        var steps = 0;
        for (var at = index - 1; at >= 0; at--)
        {
            if (++steps > MaxScanSteps)
            {
                return false;
            }

            if (targets.TryGetValue(instructions[at + 1].Pc, out var branch))
            {
                var fallsThrough = !Gen5IrBranchResolver.IsUnconditionalBranch(instructions[at]) &&
                    !Gen5IrBranchResolver.IsTerminator(instructions[at]);
                if (branch < 0 || fallsThrough)
                {
                    return false;
                }

                at = branch;
                continue;
            }

            var instruction = instructions[at];
            if (Gen5IrBranchResolver.IsUnconditionalBranch(instruction) ||
                Gen5IrBranchResolver.IsTerminator(instruction))
            {
                return false;
            }

            if (!requireFullExec && WidensExec(instruction))
            {
                return false;
            }

            if (!WritesVector(instruction, register, out var exact))
            {
                continue;
            }

            if (!exact || (requireFullExec && !context.IsExecFull(instruction.Pc)))
            {
                return false;
            }

            return TryEvaluateVector(context, at, instruction, depth, requireFullExec, values);
        }

        return false;
    }

    // exact: the instruction's only effect on the register is its listed destination.
    private static bool Writes(Gen5ShaderInstruction instruction, uint register, out bool exact)
    {
        exact = false;
        if (instruction.Opcode.Contains("Raw", StringComparison.Ordinal))
        {
            return true;
        }

        var wide = instruction.Opcode.StartsWith('S') &&
            (instruction.Opcode.EndsWith("B64", StringComparison.Ordinal) ||
             instruction.Opcode.EndsWith("U64", StringComparison.Ordinal) ||
             instruction.Opcode.EndsWith("I64", StringComparison.Ordinal));
        foreach (var destination in instruction.Destinations)
        {
            if (destination.Kind != Gen5OperandKind.ScalarRegister)
            {
                continue;
            }

            if (destination.Value == register)
            {
                exact = !wide && instruction.Destinations.Count == 1;
                return true;
            }

            if (wide && destination.Value + 1 == register)
            {
                return true;
            }
        }

        var scalarDestination = instruction.Control switch
        {
            Gen5Vop3Control control => control.ScalarDestination,
            Gen5SdwaControl control => control.ScalarDestination,
            _ => null,
        };
        if (scalarDestination is { } written && (written == register || written + 1 == register))
        {
            return true;
        }

        // Vector compares and carries write VCC or EXEC without listing them; on gfx10 a
        // V_CMPX writes EXEC only.
        if (instruction.Opcode.StartsWith("VCmpx", StringComparison.Ordinal))
        {
            return register is ExecLow or ExecHigh;
        }

        if (register is 106 or 107 &&
            instruction.Opcode.StartsWith('V') &&
            (instruction.Opcode.StartsWith("VCmp", StringComparison.Ordinal) ||
             instruction.Opcode.Contains("Co", StringComparison.Ordinal) ||
             instruction.Opcode.Contains("Div", StringComparison.Ordinal)))
        {
            return true;
        }

        // Relative scalar moves and loads write registers picked at run time.
        return instruction.Opcode.StartsWith("SMovrel", StringComparison.Ordinal);
    }

    // exact: the instruction's only vector effect on the register is its single listed VGPR
    // destination, without DPP/SDWA/VOP3 modifiers that alter lane routing or destination bits.
    private static bool WritesVector(Gen5ShaderInstruction instruction, uint register, out bool exact)
    {
        exact = false;
        var opcode = instruction.Opcode;
        if (opcode.Contains("Raw", StringComparison.Ordinal) ||
            opcode is "VMovreldB32" or "VMovrelsdB32" or "VMovrelsd2B32")
        {
            return true;
        }

        var wide = opcode.Contains("64", StringComparison.Ordinal);
        foreach (var destination in instruction.Destinations)
        {
            if (destination.Kind != Gen5OperandKind.VectorRegister)
            {
                continue;
            }

            if (destination.Value == register)
            {
                exact = !wide &&
                    instruction.Destinations.Count == 1 &&
                    instruction.Encoding is Gen5ShaderEncoding.Vop1 or Gen5ShaderEncoding.Vop2 or Gen5ShaderEncoding.Vop3 &&
                    IsCleanVectorControl(instruction.Control);
                return true;
            }

            if (wide && destination.Value + 1 == register)
            {
                return true;
            }
        }

        return instruction.Control switch
        {
            Gen5ImageControl image when !opcode.StartsWith("ImageStore", StringComparison.Ordinal) =>
                register >= image.VectorData && register < image.VectorData + 5,
            Gen5RayIntersectControl ray =>
                register >= ray.VectorData && register < ray.VectorData + Gen5RayIntersectControl.ResultDwords,
            Gen5BufferMemoryControl buffer when
                opcode.Contains("Load", StringComparison.Ordinal) || opcode.Contains("Atomic", StringComparison.Ordinal) =>
                register >= buffer.VectorData && register < buffer.VectorData + Math.Max(buffer.DwordCount, 1u) + 1,
            Gen5GlobalMemoryControl global when !opcode.Contains("Store", StringComparison.Ordinal) =>
                register >= global.DestinationVectorRegister &&
                register < global.DestinationVectorRegister + Math.Max(global.DwordCount, 1u),
            _ => false,
        };
    }

    private static bool WidensExec(Gen5ShaderInstruction instruction)
    {
        var opcode = instruction.Opcode;
        if (opcode.Contains("Raw", StringComparison.Ordinal))
        {
            return true;
        }

        var scalarDestination = instruction.Control switch
        {
            Gen5Vop3Control control => control.ScalarDestination,
            Gen5SdwaControl control => control.ScalarDestination,
            _ => null,
        };
        if (opcode.StartsWith("VCmpx", StringComparison.Ordinal))
        {
            return false;
        }

        if (opcode.StartsWith("VCmp", StringComparison.Ordinal))
        {
            return scalarDestination is ExecHigh;
        }

        var wide = opcode.StartsWith('S') &&
            (opcode.EndsWith("B64", StringComparison.Ordinal) ||
             opcode.EndsWith("U64", StringComparison.Ordinal) ||
             opcode.EndsWith("I64", StringComparison.Ordinal));
        var writesExec =
            scalarDestination is ExecLow or ExecHigh ||
            opcode.Contains("Saveexec", StringComparison.Ordinal) ||
            opcode.Contains("Wrexec", StringComparison.Ordinal) ||
            opcode.StartsWith("SMovrel", StringComparison.Ordinal) ||
            instruction.Destinations.Any(destination =>
                destination.Kind == Gen5OperandKind.ScalarRegister &&
                (destination.Value is ExecLow or ExecHigh || (wide && destination.Value + 1 == ExecLow)));
        if (!writesExec)
        {
            return false;
        }

        if (opcode is "SAndSaveexecB64" or "SAndSaveexecB32" or "SAndn1SaveexecB64" or "SAndn1SaveexecB32")
        {
            return false;
        }

        var toExec = instruction.Destinations is [{ Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow }];
        var sources = instruction.Sources;
        if (toExec && opcode is "SAndB64" or "SAndB32" && sources.Count == 2 &&
            (IsExec(sources[0]) || IsExec(sources[1])))
        {
            return false;
        }

        if (toExec && opcode is "SAndn2B64" or "SAndn2B32" && sources.Count == 2 && IsExec(sources[0]))
        {
            return false;
        }

        return true;
    }

    private static bool IsExec(Gen5Operand operand) =>
        operand is { Kind: Gen5OperandKind.ScalarRegister, Value: ExecLow };

    private static bool IsCleanVectorControl(Gen5InstructionControl? control) =>
        control switch
        {
            null => true,
            Gen5Vop3Control vop3 => vop3.AbsoluteMask == 0 &&
                vop3.NegateMask == 0 &&
                vop3.OutputModifier == 0 &&
                !vop3.Clamp &&
                vop3.OperandSelect == 0,
            _ => false,
        };

    private static bool TryEvaluate(
        AnalysisContext context,
        int index,
        Gen5ShaderInstruction instruction,
        int depth,
        HashSet<uint> values)
    {
        var sources = instruction.Sources;
        switch (instruction.Opcode)
        {
            case "SMovB32" when sources.Count == 1:
                return TryResolveOperand(context, index, sources[0], depth, values);
            case "VReadfirstlaneB32" when sources.Count >= 1 && IsCleanVectorControl(instruction.Control):
                return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec: false, values);
            case "VReadlaneB32" when sources.Count >= 2 && IsCleanVectorControl(instruction.Control):
                return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec: true, values);
            case "SBfeU32" when sources.Count == 2 && TryConstant(sources[1], out var field):
            {
                var width = (int)((field >> 16) & 0x7F);
                if (width > 6)
                {
                    return false;
                }

                for (var value = 0u; value < 1u << width; value++)
                {
                    values.Add(value);
                }

                return values.Count <= MaxValues;
            }
            case "SAndB32" when sources.Count == 2:
            {
                if (!TryConstant(sources[1], out var mask) && !TryConstant(sources[0], out mask))
                {
                    return false;
                }

                if (mask >= MaxValues)
                {
                    return false;
                }

                for (var value = 0u; value <= mask; value++)
                {
                    if ((value & ~mask) == 0)
                    {
                        values.Add(value);
                    }
                }

                return values.Count <= MaxValues;
            }
            case "SLshlB32" when sources.Count == 2 && TryConstant(sources[1], out var shift):
                return TryCombine(context, index, sources[0], depth, values, value => value << (int)(shift & 31));
            case "SMulI32" or "SAddU32" or "SAddI32" when sources.Count == 2:
            {
                var multiply = instruction.Opcode == "SMulI32";
                if (TryConstant(sources[1], out var right))
                {
                    return TryCombine(context, index, sources[0], depth, values,
                        value => multiply ? value * right : value + right);
                }

                if (TryConstant(sources[0], out var left))
                {
                    return TryCombine(context, index, sources[1], depth, values,
                        value => multiply ? value * left : value + left);
                }

                return false;
            }
            default:
                return false;
        }
    }

    private static bool TryEvaluateVector(
        AnalysisContext context,
        int index,
        Gen5ShaderInstruction instruction,
        int depth,
        bool requireFullExec,
        HashSet<uint> values)
    {
        if (!IsCleanVectorControl(instruction.Control))
        {
            return false;
        }

        var sources = instruction.Sources;
        switch (instruction.Opcode)
        {
            case "VMovB32" when sources.Count == 1:
                return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, values);

            case "VCndmaskB32" when sources.Count >= 2:
            {
                if (sources[0] == sources[1])
                {
                    return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, values);
                }

                var left = new HashSet<uint>();
                var right = new HashSet<uint>();
                if (!TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, left) ||
                    !TryResolveVectorOperand(context, index, sources[1], depth, requireFullExec, right))
                {
                    return false;
                }

                values.UnionWith(left);
                values.UnionWith(right);
                return values.Count > 0 && values.Count <= MaxValues;
            }

            case "VAndB32" when sources.Count == 2:
            {
                if (sources[0] == sources[1])
                {
                    return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, values);
                }

                if (TryConstant(sources[1], out var rightMask))
                {
                    return TryAndWithConstant(context, index, sources[0], rightMask, depth, requireFullExec, values);
                }

                if (TryConstant(sources[0], out var leftMask))
                {
                    return TryAndWithConstant(context, index, sources[1], leftMask, depth, requireFullExec, values);
                }

                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (a, b) => a & b);
            }

            case "VOrB32" when sources.Count == 2:
            {
                if (sources[0] == sources[1])
                {
                    return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, values);
                }

                if ((TryConstant(sources[0], out var c0) && c0 == uint.MaxValue) ||
                    (TryConstant(sources[1], out var c1) && c1 == uint.MaxValue))
                {
                    values.Add(uint.MaxValue);
                    return values.Count <= MaxValues;
                }

                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (a, b) => a | b);
            }

            case "VBfeU32" when sources.Count == 3 && TryConstant(sources[2], out var rawWidth):
            {
                var width = (int)(rawWidth & 31);
                if (width == 0)
                {
                    values.Add(0);
                    return values.Count <= MaxValues;
                }

                var mask = (1u << width) - 1u;
                if (TryConstant(sources[1], out var rawOffset))
                {
                    var offset = (int)(rawOffset & 31);
                    var inputs = new HashSet<uint>();
                    if (TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, inputs))
                    {
                        foreach (var input in inputs)
                        {
                            values.Add((input >> offset) & mask);
                            if (values.Count > MaxValues)
                            {
                                return false;
                            }
                        }

                        return values.Count > 0;
                    }

                    var effectiveWidth = Math.Min(width, 32 - offset);
                    return TryAddBitRange(effectiveWidth, values);
                }

                return TryAddBitRange(width, values);
            }

            case "VLshlrevB32" when sources.Count == 2:
                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (shift, value) => value << (int)(shift & 31));

            case "VLshlB32" when sources.Count == 2:
                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (value, shift) => value << (int)(shift & 31));

            case "VLshrrevB32" when sources.Count == 2:
            {
                var exact = new HashSet<uint>();
                if (TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, exact,
                    static (shift, value) => value >> (int)(shift & 31)))
                {
                    values.UnionWith(exact);
                    return values.Count <= MaxValues;
                }

                return TryConstant(sources[0], out var shift) &&
                    TryAddBitRange(32 - (int)(shift & 31), values);
            }

            case "VLshrB32" when sources.Count == 2:
            {
                var exact = new HashSet<uint>();
                if (TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, exact,
                    static (value, shift) => value >> (int)(shift & 31)))
                {
                    values.UnionWith(exact);
                    return values.Count <= MaxValues;
                }

                return TryConstant(sources[1], out var shift) &&
                    TryAddBitRange(32 - (int)(shift & 31), values);
            }

            case "VAddU32" or "VAddNcU32" or "VAddI32" or "VAddCoU32" when sources.Count == 2:
                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (a, b) => unchecked(a + b));

            case "VMulLoU32" or "VMulLoI32" when sources.Count == 2:
            {
                if ((TryConstant(sources[0], out var m0) && m0 == 0) ||
                    (TryConstant(sources[1], out var m1) && m1 == 0))
                {
                    values.Add(0);
                    return values.Count <= MaxValues;
                }

                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (a, b) => unchecked(a * b));
            }

            case "VMulU32U24" when sources.Count == 2:
            {
                if ((TryConstant(sources[0], out var u0) && (u0 & 0x00FF_FFFFu) == 0) ||
                    (TryConstant(sources[1], out var u1) && (u1 & 0x00FF_FFFFu) == 0))
                {
                    values.Add(0);
                    return values.Count <= MaxValues;
                }

                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (a, b) => unchecked((a & 0x00FF_FFFFu) * (b & 0x00FF_FFFFu)));
            }

            case "VLshlAddU32" when sources.Count == 3:
                return TryCombineVectorTernary(context, index, sources[0], sources[1], sources[2], depth, requireFullExec, values,
                    static (a, shift, c) => unchecked((a << (int)(shift & 31)) + c));

            case "VMadU32U24" when sources.Count == 3:
            {
                if ((TryConstant(sources[0], out var f0) && (f0 & 0x00FF_FFFFu) == 0) ||
                    (TryConstant(sources[1], out var f1) && (f1 & 0x00FF_FFFFu) == 0))
                {
                    return TryResolveVectorOperand(context, index, sources[2], depth, requireFullExec, values);
                }

                return TryCombineVectorTernary(context, index, sources[0], sources[1], sources[2], depth, requireFullExec, values,
                    static (a, b, c) => unchecked(((a & 0x00FF_FFFFu) * (b & 0x00FF_FFFFu)) + c));
            }

            case "VMinU32" when sources.Count == 2:
            {
                if (sources[0] == sources[1])
                {
                    return TryResolveVectorOperand(context, index, sources[0], depth, requireFullExec, values);
                }

                if (TryConstant(sources[1], out var rightCap))
                {
                    return TryMinWithConstant(context, index, sources[0], rightCap, depth, requireFullExec, values);
                }

                if (TryConstant(sources[0], out var leftCap))
                {
                    return TryMinWithConstant(context, index, sources[1], leftCap, depth, requireFullExec, values);
                }

                return TryCombineVectorBinary(context, index, sources[0], sources[1], depth, requireFullExec, values,
                    static (a, b) => Math.Min(a, b));
            }

            case "VFfblB32" when sources.Count == 1:
                return TryBitScan(context, index, sources[0], depth, requireFullExec, values,
                    static value => (uint)BitOperations.TrailingZeroCount(value));

            case "VFfbhU32" when sources.Count == 1:
                return TryBitScan(context, index, sources[0], depth, requireFullExec, values,
                    static value => (uint)BitOperations.LeadingZeroCount(value));

            default:
                return false;
        }
    }

    private static bool TryAndWithConstant(
        AnalysisContext context,
        int index,
        Gen5Operand operand,
        uint mask,
        int depth,
        bool requireFullExec,
        HashSet<uint> values)
    {
        var inputs = new HashSet<uint>();
        if (TryResolveVectorOperand(context, index, operand, depth, requireFullExec, inputs))
        {
            foreach (var input in inputs)
            {
                values.Add(input & mask);
                if (values.Count > MaxValues)
                {
                    return false;
                }
            }

            return values.Count > 0;
        }

        if (BitOperations.PopCount(mask) > 6)
        {
            return false;
        }

        var submask = mask;
        while (true)
        {
            values.Add(submask);
            if (values.Count > MaxValues)
            {
                return false;
            }

            if (submask == 0)
            {
                return true;
            }

            submask = (submask - 1) & mask;
        }
    }

    private static bool TryMinWithConstant(
        AnalysisContext context,
        int index,
        Gen5Operand operand,
        uint cap,
        int depth,
        bool requireFullExec,
        HashSet<uint> values)
    {
        var inputs = new HashSet<uint>();
        if (TryResolveVectorOperand(context, index, operand, depth, requireFullExec, inputs))
        {
            foreach (var input in inputs)
            {
                values.Add(Math.Min(input, cap));
                if (values.Count > MaxValues)
                {
                    return false;
                }
            }

            return values.Count > 0;
        }

        if (cap >= MaxValues)
        {
            return false;
        }

        for (var value = 0u; value <= cap; value++)
        {
            values.Add(value);
            if (values.Count > MaxValues)
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryBitScan(
        AnalysisContext context,
        int index,
        Gen5Operand operand,
        int depth,
        bool requireFullExec,
        HashSet<uint> values,
        Func<uint, uint> scanNonZero)
    {
        var inputs = new HashSet<uint>();
        if (TryResolveVectorOperand(context, index, operand, depth, requireFullExec, inputs))
        {
            foreach (var input in inputs)
            {
                values.Add(input == 0 ? uint.MaxValue : scanNonZero(input));
                if (values.Count > MaxValues)
                {
                    return false;
                }
            }

            return values.Count > 0;
        }

        for (var bit = 0u; bit < 32u; bit++)
        {
            values.Add(bit);
        }

        values.Add(uint.MaxValue);
        return values.Count <= MaxValues;
    }

    private static bool TryAddBitRange(int width, HashSet<uint> values)
    {
        if (width < 0 || width > 6)
        {
            return false;
        }

        var count = 1u << width;
        for (var value = 0u; value < count; value++)
        {
            values.Add(value);
            if (values.Count > MaxValues)
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryCombine(
        AnalysisContext context,
        int index,
        Gen5Operand operand,
        int depth,
        HashSet<uint> values,
        Func<uint, uint> apply)
    {
        var inputs = new HashSet<uint>();
        if (!TryResolveOperand(context, index, operand, depth, inputs))
        {
            return false;
        }

        foreach (var input in inputs)
        {
            values.Add(unchecked(apply(input)));
            if (values.Count > MaxValues)
            {
                return false;
            }
        }

        return values.Count > 0;
    }

    private static bool TryCombineVectorBinary(
        AnalysisContext context,
        int index,
        Gen5Operand leftOperand,
        Gen5Operand rightOperand,
        int depth,
        bool requireFullExec,
        HashSet<uint> values,
        Func<uint, uint, uint> apply)
    {
        if (leftOperand == rightOperand)
        {
            var inputs = new HashSet<uint>();
            if (!TryResolveVectorOperand(context, index, leftOperand, depth, requireFullExec, inputs))
            {
                return false;
            }

            foreach (var input in inputs)
            {
                values.Add(unchecked(apply(input, input)));
                if (values.Count > MaxValues)
                {
                    return false;
                }
            }

            return values.Count > 0;
        }

        var left = new HashSet<uint>();
        var right = new HashSet<uint>();
        if (!TryResolveVectorOperand(context, index, leftOperand, depth, requireFullExec, left) ||
            !TryResolveVectorOperand(context, index, rightOperand, depth, requireFullExec, right))
        {
            return false;
        }

        foreach (var a in left)
        {
            foreach (var b in right)
            {
                values.Add(unchecked(apply(a, b)));
                if (values.Count > MaxValues)
                {
                    return false;
                }
            }
        }

        return values.Count > 0;
    }

    private static bool TryCombineVectorTernary(
        AnalysisContext context,
        int index,
        Gen5Operand op0,
        Gen5Operand op1,
        Gen5Operand op2,
        int depth,
        bool requireFullExec,
        HashSet<uint> values,
        Func<uint, uint, uint, uint> apply)
    {
        if (op0 == op1 && op1 == op2)
        {
            var s0 = new HashSet<uint>();
            if (!TryResolveVectorOperand(context, index, op0, depth, requireFullExec, s0))
            {
                return false;
            }

            foreach (var a in s0)
            {
                values.Add(unchecked(apply(a, a, a)));
                if (values.Count > MaxValues)
                {
                    return false;
                }
            }

            return values.Count > 0;
        }

        if (op0 == op2)
        {
            var s0 = new HashSet<uint>();
            var s1 = new HashSet<uint>();
            if (!TryResolveVectorOperand(context, index, op0, depth, requireFullExec, s0) ||
                !TryResolveVectorOperand(context, index, op1, depth, requireFullExec, s1))
            {
                return false;
            }

            foreach (var a in s0)
            {
                foreach (var b in s1)
                {
                    values.Add(unchecked(apply(a, b, a)));
                    if (values.Count > MaxValues)
                    {
                        return false;
                    }
                }
            }

            return values.Count > 0;
        }

        if (op0 == op1)
        {
            var s0 = new HashSet<uint>();
            var s2 = new HashSet<uint>();
            if (!TryResolveVectorOperand(context, index, op0, depth, requireFullExec, s0) ||
                !TryResolveVectorOperand(context, index, op2, depth, requireFullExec, s2))
            {
                return false;
            }

            foreach (var a in s0)
            {
                foreach (var c in s2)
                {
                    values.Add(unchecked(apply(a, a, c)));
                    if (values.Count > MaxValues)
                    {
                        return false;
                    }
                }
            }

            return values.Count > 0;
        }

        if (op1 == op2)
        {
            var s0 = new HashSet<uint>();
            var s1 = new HashSet<uint>();
            if (!TryResolveVectorOperand(context, index, op0, depth, requireFullExec, s0) ||
                !TryResolveVectorOperand(context, index, op1, depth, requireFullExec, s1))
            {
                return false;
            }

            foreach (var a in s0)
            {
                foreach (var b in s1)
                {
                    values.Add(unchecked(apply(a, b, b)));
                    if (values.Count > MaxValues)
                    {
                        return false;
                    }
                }
            }

            return values.Count > 0;
        }

        var v0 = new HashSet<uint>();
        var v1 = new HashSet<uint>();
        var v2 = new HashSet<uint>();
        if (!TryResolveVectorOperand(context, index, op0, depth, requireFullExec, v0) ||
            !TryResolveVectorOperand(context, index, op1, depth, requireFullExec, v1) ||
            !TryResolveVectorOperand(context, index, op2, depth, requireFullExec, v2))
        {
            return false;
        }

        foreach (var a in v0)
        {
            foreach (var b in v1)
            {
                foreach (var c in v2)
                {
                    values.Add(unchecked(apply(a, b, c)));
                    if (values.Count > MaxValues)
                    {
                        return false;
                    }
                }
            }
        }

        return values.Count > 0;
    }

    private static bool TryResolveOperand(
        AnalysisContext context,
        int index,
        Gen5Operand operand,
        int depth,
        HashSet<uint> values)
    {
        if (TryConstant(operand, out var constant))
        {
            values.Add(constant);
            return values.Count <= MaxValues;
        }

        return operand.Kind == Gen5OperandKind.ScalarRegister &&
            TryResolveRegister(context, index, operand.Value, depth + 1, values) &&
            values.Count <= MaxValues;
    }

    private static bool TryResolveVectorOperand(
        AnalysisContext context,
        int index,
        Gen5Operand operand,
        int depth,
        bool requireFullExec,
        HashSet<uint> values)
    {
        if (TryConstant(operand, out var constant))
        {
            values.Add(constant);
            return values.Count <= MaxValues;
        }

        if (operand.Kind == Gen5OperandKind.ScalarRegister)
        {
            return TryResolveRegister(context, index, operand.Value, depth + 1, values) &&
                values.Count <= MaxValues;
        }

        return operand.Kind == Gen5OperandKind.VectorRegister &&
            TryResolveVectorRegister(context, index, operand.Value, depth + 1, requireFullExec, values) &&
            values.Count <= MaxValues;
    }

    private static bool TryConstant(Gen5Operand operand, out uint value)
    {
        value = 0;
        return operand.Kind switch
        {
            Gen5OperandKind.LiteralConstant => (value = operand.Value) == operand.Value,
            Gen5OperandKind.EncodedConstant => Gen5InlineConstants.TryDecode(operand.Value, out value),
            _ => false,
        };
    }
}
