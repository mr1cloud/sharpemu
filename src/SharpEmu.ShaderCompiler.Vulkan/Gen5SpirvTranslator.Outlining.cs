// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public static partial class Gen5SpirvTranslator
{
    private sealed partial class CompilationContext
    {
        private uint _markHeapRecordFunction;
        private readonly Dictionary<int, uint> _bufferMaskedStoreFunctions = [];
        private uint _deviceMaskedStoreFunction;

        private void CallMarkHeapRecord(uint wordOffset, uint index, uint active)
        {
            if (_markHeapRecordFunction == 0)
            {
                _markHeapRecordFunction = _module.AllocateId();
            }

            _module.AddInstruction(SpirvOp.FunctionCall, _voidType, _markHeapRecordFunction, wordOffset, index, active);
        }

        private void CallBufferMaskedStore(int bindingIndex, uint dwordAddress, uint value, uint mask)
        {
            if (!_bufferMaskedStoreFunctions.TryGetValue(bindingIndex, out var function))
            {
                function = _module.AllocateId();
                _bufferMaskedStoreFunctions[bindingIndex] = function;
            }

            _module.AddInstruction(SpirvOp.FunctionCall, _voidType, function, dwordAddress, value, mask);
        }

        private void CallDeviceMaskedStore(uint address64, uint value, uint mask, uint allowed)
        {
            if (_deviceMaskedStoreFunction == 0)
            {
                _deviceMaskedStoreFunction = _module.AllocateId();
            }

            _module.AddInstruction(SpirvOp.FunctionCall, _voidType, _deviceMaskedStoreFunction, address64, value, mask, allowed);
        }

        private void EmitStoreDeviceMaskedWordBody(uint address64, uint value, uint mask, uint allowed)
        {
            var touched = LogicalAnd(allowed, _module.AddInstruction(SpirvOp.INotEqual, _boolType, mask, UInt(0)));
            EmitConditional(touched, () =>
            {
                var (pointer, valid) = ResolveDeviceAddress(address64);
                EmitConditional(valid, () =>
                {
                    var wordPointer = DeviceWordPointer(pointer);
                    var full = _module.AddInstruction(SpirvOp.IEqual, _boolType, mask, UInt(uint.MaxValue));
                    EmitConditional(
                        full,
                        () => _module.AddStatement(SpirvOp.Store, wordPointer, value, 2u, 4u),
                        () => EmitAtomicWordUpdate(
                            wordPointer,
                            observed => BitwiseOr(
                                BitwiseAnd(observed, _module.AddInstruction(SpirvOp.Not, _uintType, mask)),
                                value)));
                });
            });
        }

        // After every function that may call them.
        private void EmitOutlinedHelpers()
        {
            EmitMarkHeapRecordFunction();
            EmitBufferMaskedStoreFunctions();
            EmitDeviceMaskedStoreFunction();
        }

        private void EmitMarkHeapRecordFunction()
        {
            if (_markHeapRecordFunction == 0)
            {
                return;
            }

            const uint dontInline = 2;
            var functionType = _module.TypeFunction(_voidType, _uintType, _uintType, _boolType);
            _module.BeginFunction(_voidType, functionType, dontInline, _markHeapRecordFunction);
            _module.AddName(_markHeapRecordFunction, "markHeapRecord");
            var wordOffset = _module.AddFunctionParameter(_uintType);
            var index = _module.AddFunctionParameter(_uintType);
            var active = _module.AddFunctionParameter(_boolType);
            _module.AddLabel();
            var word = IAdd(wordOffset, ShiftRightLogical(index, UInt(5)));
            var bit = ShiftLeftLogical(UInt(1), BitwiseAnd(index, UInt(31)));
            var pointer = BlockWordPointer(HeapFeedback(), word);
            var unmarked = LogicalAnd(
                active,
                _module.AddInstruction(SpirvOp.IEqual, _boolType, BitwiseAnd(Load(_uintType, pointer), bit), UInt(0)));
            EmitConditional(unmarked, () => _module.AddInstruction(SpirvOp.AtomicOr, _uintType, pointer, UInt(1), UInt(0), bit));
            _module.AddStatement(SpirvOp.Return);
            _module.EndFunction();
        }

        private void EmitBufferMaskedStoreFunctions()
        {
            const uint dontInline = 2;
            foreach (var (bindingIndex, function) in _bufferMaskedStoreFunctions.OrderBy(entry => entry.Key))
            {
                var functionType = _module.TypeFunction(_voidType, _uintType, _uintType, _uintType);
                _module.BeginFunction(_voidType, functionType, dontInline, function);
                _module.AddName(function, $"storeBufferMaskedWord{bindingIndex}");
                var dwordAddress = _module.AddFunctionParameter(_uintType);
                var value = _module.AddFunctionParameter(_uintType);
                var mask = _module.AddFunctionParameter(_uintType);
                _module.AddLabel();
                var touched = _module.AddInstruction(
                    SpirvOp.LogicalAnd,
                    _boolType,
                    _module.AddInstruction(SpirvOp.INotEqual, _boolType, mask, UInt(0)),
                    IsBufferWordInRange(bindingIndex, dwordAddress));
                EmitConditional(touched, () =>
                {
                    var pointer = BufferWordPointer(bindingIndex, dwordAddress);
                    var full = _module.AddInstruction(SpirvOp.IEqual, _boolType, mask, UInt(uint.MaxValue));
                    EmitConditional(
                        full,
                        () => Store(pointer, value),
                        () => EmitAtomicWordUpdate(
                            pointer,
                            observed => BitwiseOr(
                                BitwiseAnd(observed, _module.AddInstruction(SpirvOp.Not, _uintType, mask)),
                                value)));
                });
                _module.AddStatement(SpirvOp.Return);
                _module.EndFunction();
            }
        }

        private void EmitDeviceMaskedStoreFunction()
        {
            if (_deviceMaskedStoreFunction == 0)
            {
                return;
            }

            const uint dontInline = 2;
            var functionType = _module.TypeFunction(_voidType, _ulongType, _uintType, _uintType, _boolType);
            _module.BeginFunction(_voidType, functionType, dontInline, _deviceMaskedStoreFunction);
            _module.AddName(_deviceMaskedStoreFunction, "storeDeviceMaskedWord");
            var address64 = _module.AddFunctionParameter(_ulongType);
            var value = _module.AddFunctionParameter(_uintType);
            var mask = _module.AddFunctionParameter(_uintType);
            var allowed = _module.AddFunctionParameter(_boolType);
            _module.AddLabel();
            EmitStoreDeviceMaskedWordBody(address64, value, mask, allowed);
            _module.AddStatement(SpirvOp.Return);
            _module.EndFunction();
        }
    }
}
