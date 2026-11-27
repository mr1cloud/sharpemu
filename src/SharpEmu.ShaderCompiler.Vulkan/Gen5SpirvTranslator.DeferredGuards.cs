// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public static partial class Gen5SpirvTranslator
{
    private sealed partial class CompilationContext
    {
        // A vector register write under a partial EXEC keeps the old value in the lanes EXEC leaves out:
        // select(active, new, old). While EXEC and the SPIR-V block stay the same, those lanes of a register are
        // read only by instructions that read other lanes or address memory through VGPR values, so one select
        // per register and region is enough. A write in the region stores the new value in every lane; the first
        // one remembers the value the register had when the region began. The select is emitted when EXEC
        // changes, before the block ends or calls, and before an instruction that reads other lanes.
        // SHARPEMU_DEFERRED_EXEC_GUARDS=0 guards every write again.
        private static readonly bool DeferredExecGuards = !string.Equals(
            Environment.GetEnvironmentVariable("SHARPEMU_DEFERRED_EXEC_GUARDS"), "0", StringComparison.Ordinal);

        private bool _deferGuards;
        private readonly List<(uint Register, uint Old)>[] _deferredGuards = [[], []];
        private readonly HashSet<uint>[] _deferredGuardRegisters = [[], []];

        private int CurrentDeferredGuardHalf =>
            _pairedLanes ? _laneHalf : _pairedMeshLanes ? _logicalHalf : 0;

        private bool HasDeferredGuards =>
            _deferredGuards[0].Count != 0 || _deferredGuards[1].Count != 0;

        // Native waves, half waves, and paired waves (where each logical half tracks its own EXEC and VGPR bank)
        // can coalesce per-write EXEC guards into one select per modified register when EXEC or the block ends.
        private bool CanDeferGuards =>
            DeferredExecGuards && !_hasIndirectControlFlow &&
            _waveLowering is NativeWave32Lowering or NativeWave64Lowering or HalfWave64Lowering or PairedWave64Lowering;

        // A guarded write inside the current region: the new value goes to every lane.
        private bool TryDeferGuard(uint register, uint value)
        {
            if (!_deferGuards)
            {
                return false;
            }

            var half = CurrentDeferredGuardHalf;
            if (_deferredGuardRegisters[half].Add(register))
            {
                _deferredGuards[half].Add((register, LoadV(register)));
            }

            Store(VectorPointer(register), value);
            return true;
        }

        // Ends the region: the lanes EXEC left out get back the values their registers had when it began.
        private void FlushDeferredGuards()
        {
            if (!HasDeferredGuards)
            {
                return;
            }

            FlushDeferredGuardsForHalf(0);
            FlushDeferredGuardsForHalf(1);
        }

        private void FlushDeferredGuardsForHalf(int half)
        {
            var guards = _deferredGuards[half];
            if (guards.Count == 0)
            {
                return;
            }

            var previousHalf = CurrentDeferredGuardHalf;
            var switched = previousHalf != half;
            if (switched)
            {
                SelectDeferredGuardHalf(half);
            }

            try
            {
                var active = Load(_boolType, _exec);
                foreach (var (register, old) in guards)
                {
                    Store(VectorPointer(register), _module.AddInstruction(SpirvOp.Select, _uintType, active, LoadV(register), old));
                }

                guards.Clear();
                _deferredGuardRegisters[half].Clear();
            }
            finally
            {
                if (switched)
                {
                    SelectDeferredGuardHalf(previousHalf);
                }
            }
        }

        private void SelectDeferredGuardHalf(int half)
        {
            if (_pairedLanes)
            {
                SelectLaneHalf(half);
            }
            else if (_pairedMeshLanes)
            {
                SelectMeshHalf(half);
            }
        }

        // Flushes whichever logical half owns an EXEC predicate variable before that predicate is overwritten.
        private void FlushDeferredGuardsBeforeExecStore(uint pointer)
        {
            if (!HasDeferredGuards)
            {
                return;
            }

            if (_pairedLanes)
            {
                if (pointer == _lowerExec)
                {
                    FlushDeferredGuardsForHalf(0);
                }
                else if (pointer == _upperExec)
                {
                    FlushDeferredGuardsForHalf(1);
                }

                return;
            }

            if (_pairedMeshLanes)
            {
                if (pointer == _meshLaneRegisters[0].Execution)
                {
                    FlushDeferredGuardsForHalf(0);
                }
                else if (pointer == _meshLaneRegisters[1].Execution)
                {
                    FlushDeferredGuardsForHalf(1);
                }

                return;
            }

            if (pointer == _exec)
            {
                FlushDeferredGuardsForHalf(0);
            }
        }

        // Instructions that can read the vector registers of other lanes (DPP, lane reads and writes, permutes,
        // image derivatives, exports), or that address memory through VGPR values outside bounds-checked bindings.
        private static bool ReadsOtherLanes(Gen5ShaderInstruction instruction) =>
            instruction.Control is Gen5DppControl or Gen5Dpp8Control or Gen5DataShareControl { Gds: true } ||
            instruction.Encoding is Gen5ShaderEncoding.Mimg or Gen5ShaderEncoding.Flat or Gen5ShaderEncoding.Exp ||
            instruction.Opcode is "VReadlaneB32" or "VReadfirstlaneB32" or "VWritelaneB32" or "VPermlane16B32" or
                "VPermlanex16B32" or "DsSwizzleB32" or "DsBpermuteB32" or "DsPermuteB32" ||
            instruction.Opcode.StartsWith("VMovrel", StringComparison.Ordinal);
    }
}
