// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class SpirvSimplifyHotPathTests
{
    [Fact]
    public void WholeArrayStoreInvalidatesOnlyOwnedElements()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var u32 = module.TypeInt(32, false);
        var arrayType = module.TypeArray(u32, 2);
        var elemPtrType = module.TypePointer(SpirvStorageClass.Private, u32);
        var arrA = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, arrayType), SpirvStorageClass.Private);
        var arrB = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, arrayType), SpirvStorageClass.Private);
        var scalar = module.AddGlobalVariable(elemPtrType, SpirvStorageClass.Private);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [arrA, arrB, scalar, output]);
        module.AddLabel();

        var zero = module.Constant(u32, 0);
        var one = module.Constant(u32, 1);
        var ptrA0 = module.AddInstruction(SpirvOp.AccessChain, elemPtrType, arrA, zero);
        var ptrA1 = module.AddInstruction(SpirvOp.AccessChain, elemPtrType, arrA, one);
        var ptrB0 = module.AddInstruction(SpirvOp.AccessChain, elemPtrType, arrB, zero);

        var v1 = module.Constant(u32, 11);
        var v2 = module.Constant(u32, 22);
        var v3 = module.Constant(u32, 33);
        var v4 = module.Constant(u32, 44);
        module.AddStatement(SpirvOp.Store, ptrA0, v1);
        module.AddStatement(SpirvOp.Store, ptrA1, v2);
        module.AddStatement(SpirvOp.Store, ptrB0, v3);
        module.AddStatement(SpirvOp.Store, scalar, v4);

        var wholeValue = module.ConstantComposite(arrayType, module.Constant(u32, 77), module.Constant(u32, 88));
        module.AddStatement(SpirvOp.Store, arrA, wholeValue);

        // arrA elements must reload from memory; arrB and scalar must keep their forwarded constants.
        var loadA0 = module.AddInstruction(SpirvOp.Load, u32, ptrA0);
        var loadA1 = module.AddInstruction(SpirvOp.Load, u32, ptrA1);
        var loadB0 = module.AddInstruction(SpirvOp.Load, u32, ptrB0);
        var loadScalar = module.AddInstruction(SpirvOp.Load, u32, scalar);

        Assert.NotEqual(v1, loadA0);
        Assert.NotEqual(v2, loadA1);
        Assert.Equal(v3, loadB0);
        Assert.Equal(v4, loadScalar);

        var sum = module.AddInstruction(SpirvOp.IAdd, u32,
            module.AddInstruction(SpirvOp.IAdd, u32, loadA0, loadA1),
            module.AddInstruction(SpirvOp.IAdd, u32, loadB0, loadScalar));
        module.AddStatement(SpirvOp.Store, output, sum);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == loadA0);
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == loadA1);
    }

    [Fact]
    public void DynamicArrayLoadFlushesElementStoresWhileDynamicStoreInvalidatesCachedValues()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var u32 = module.TypeInt(32, false);
        var arrayType = module.TypeArray(u32, 2);
        var elemPtrType = module.TypePointer(SpirvStorageClass.Private, u32);
        var arr = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, arrayType), SpirvStorageClass.Private);
        var indexInput = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
        module.AddDecoration(indexInput, SpirvDecoration.Location, 0);
        module.AddDecoration(indexInput, SpirvDecoration.Flat);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [arr, indexInput, output]);
        module.AddLabel();

        var zero = module.Constant(u32, 0);
        var ptr0 = module.AddInstruction(SpirvOp.AccessChain, elemPtrType, arr, zero);
        var first = module.Constant(u32, 101);
        var second = module.Constant(u32, 202);
        var idx = module.AddInstruction(SpirvOp.Load, u32, indexInput);

        module.AddStatement(SpirvOp.Store, ptr0, first);
        var dynLoaded = module.LoadDynamicArrayElement(u32, elemPtrType, arr, idx);
        // keepValues=true: cached value is still forwarded, and the earlier store must NOT be removed by a later store.
        var stillCached = module.AddInstruction(SpirvOp.Load, u32, ptr0);
        Assert.Equal(first, stillCached);

        module.AddStatement(SpirvOp.Store, ptr0, second);
        module.StoreDynamicArrayElement(elemPtrType, arr, idx, dynLoaded);
        // keepValues=false: cached value must be invalidated.
        var reloaded = module.AddInstruction(SpirvOp.Load, u32, ptr0);
        Assert.NotEqual(second, reloaded);

        var sum = module.AddInstruction(SpirvOp.IAdd, u32,
            module.AddInstruction(SpirvOp.IAdd, u32, dynLoaded, stillCached), reloaded);
        module.AddStatement(SpirvOp.Store, output, sum);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        var ptr0Stores = instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == ptr0).ToArray();
        Assert.Equal(2, ptr0Stores.Length);
        Assert.Equal(first, ptr0Stores[0][2]);
        Assert.Equal(second, ptr0Stores[1][2]);
    }

    [Fact]
    public void TrySplitHighWordHandlesBitwiseAndTreesAndZeroMasks()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        module.AddCapability(SpirvCapability.Int64);
        var u32 = module.TypeInt(32, false);
        var u64 = module.TypeInt(64, false);
        var inputA = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
        var inputB = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
        module.AddDecoration(inputA, SpirvDecoration.Location, 0);
        module.AddDecoration(inputA, SpirvDecoration.Flat);
        module.AddDecoration(inputB, SpirvDecoration.Location, 1);
        module.AddDecoration(inputB, SpirvDecoration.Flat);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [inputA, inputB, output]);
        module.AddLabel();

        var a = module.AddInstruction(SpirvOp.Load, u32, inputA);
        var b = module.AddInstruction(SpirvOp.Load, u32, inputB);
        var shift32 = module.Constant64(u64, 32);

        // Build a depth-6 binary tree of 64-bit BitwiseAnd nodes with non-zero high words.
        var leaves = new uint[16];
        for (var i = 0; i < leaves.Length; i++)
        {
            var word = module.AddInstruction(SpirvOp.IAdd, u32, (i & 1) == 0 ? a : b, module.Constant(u32, (uint)(i + 1)));
            var wide = module.AddInstruction(SpirvOp.UConvert, u64, word);
            leaves[i] = module.AddInstruction(SpirvOp.ShiftLeftLogical, u64, wide, shift32);
        }

        var level = leaves;
        while (level.Length > 1)
        {
            var next = new uint[level.Length / 2];
            for (var i = 0; i < next.Length; i++)
            {
                next[i] = module.AddInstruction(SpirvOp.BitwiseAnd, u64, level[i * 2], level[(i * 2) + 1]);
            }

            level = next;
        }

        // Splitting the high word must succeed and lower to 32-bit BitwiseAnd without 64-bit ShiftRightLogical.
        var highNonZero = module.AddInstruction(SpirvOp.ShiftRightLogical, u64, level[0], shift32);
        var lowOfHigh = module.AddInstruction(SpirvOp.UConvert, u32, highNonZero);

        // Masking with a 32-bit zero-extended value (high word == 0) on either left or right must fold high word to 0.
        var zeroHigh = module.AddInstruction(SpirvOp.UConvert, u64, a);
        var maskedRightZero = module.AddInstruction(SpirvOp.BitwiseAnd, u64, level[0], zeroHigh);
        var highFromRightZero = module.AddInstruction(SpirvOp.ShiftRightLogical, u64, maskedRightZero, shift32);
        var maskedLeftZero = module.AddInstruction(SpirvOp.BitwiseAnd, u64, zeroHigh, level[0]);
        var highFromLeftZero = module.AddInstruction(SpirvOp.ShiftRightLogical, u64, maskedLeftZero, shift32);

        var zero64 = module.Constant64(u64, 0);
        Assert.Equal(zero64, highFromRightZero);
        Assert.Equal(zero64, highFromLeftZero);

        module.AddStatement(SpirvOp.Store, output, lowOfHigh);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        Assert.DoesNotContain(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.ShiftRightLogical);
    }

    [Fact]
    public void DuplicateBranchAndSwitchTargetsDeduplicatePredecessorsAndInheritState()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var u32 = module.TypeInt(32, false);
        var boolean = module.TypeBool();
        var reg = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
        var selectorInput = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
        module.AddDecoration(selectorInput, SpirvDecoration.Location, 0);
        module.AddDecoration(selectorInput, SpirvDecoration.Flat);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [reg, selectorInput, output]);
        module.AddLabel();

        var val = module.Constant(u32, 99);
        var selector = module.AddInstruction(SpirvOp.Load, u32, selectorInput);
        var cond = module.AddInstruction(SpirvOp.INotEqual, boolean, selector, module.Constant(u32, 0));
        module.AddStatement(SpirvOp.Store, reg, val);

        // Conditional branch where true and false targets are identical: must count as 1 predecessor.
        var afterCond = module.AllocateId();
        module.AddStatement(SpirvOp.BranchConditional, cond, afterCond, afterCond);

        module.AddLabel(afterCond);
        var inheritedFromCond = module.AddInstruction(SpirvOp.Load, u32, reg);
        Assert.Equal(val, inheritedFromCond);

        // Switch with duplicate case targets and default target: must deduplicate predecessors.
        var caseTarget = module.AllocateId();
        var otherTarget = module.AllocateId();
        var mergeTarget = module.AllocateId();
        module.AddStatement(SpirvOp.SelectionMerge, mergeTarget, 0);
        module.AddStatement(SpirvOp.Switch, selector, caseTarget, 0u, caseTarget, 1u, otherTarget, 2u, caseTarget);

        module.AddLabel(caseTarget);
        var inheritedCase = module.AddInstruction(SpirvOp.Load, u32, reg);
        Assert.Equal(val, inheritedCase);
        module.AddStatement(SpirvOp.Branch, mergeTarget);

        module.AddLabel(otherTarget);
        var inheritedOther = module.AddInstruction(SpirvOp.Load, u32, reg);
        Assert.Equal(val, inheritedOther);
        module.AddStatement(SpirvOp.Branch, mergeTarget);

        module.AddLabel(mergeTarget);
        var merged = module.AddInstruction(SpirvOp.Load, u32, reg);
        Assert.Equal(val, merged);
        module.AddStatement(SpirvOp.Store, output, merged);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        Assert.True(module.InheritedBlocksHeld);
    }

    private static uint[][] Instructions(byte[] code)
    {
        var words = new uint[code.Length / sizeof(uint)];
        Buffer.BlockCopy(code, 0, words, 0, code.Length);
        var instructions = new List<uint[]>();
        for (var offset = 5; offset < words.Length; offset += (int)(words[offset] >> 16))
        {
            instructions.Add(words.AsSpan(offset, (int)(words[offset] >> 16)).ToArray());
        }

        return instructions.ToArray();
    }
}
