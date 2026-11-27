// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Tests.Resources;
using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;

namespace SharpEmu.ShaderCompiler.Tests;

[CollectionDefinition("SpirvOptimization", DisableParallelization = true)]
public sealed class SpirvOptimizationCollection : ICollectionFixture<SpirvOptimizationDefaults>;

public sealed class SpirvOptimizationDefaults : IDisposable
{
    private readonly bool _previous = SpirvModuleBuilder.KeepsUnreadVariables;

    public SpirvOptimizationDefaults() => SpirvModuleBuilder.KeepsUnreadVariables = false;

    public void Dispose() => SpirvModuleBuilder.KeepsUnreadVariables = _previous;
}

[Collection("SpirvOptimization")]
public sealed class SpirvOptimizationTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ForwardedRegisterValuesKeepTheirOutputWhenUnreadStorageIsRemoved(bool preserve, bool indexed)
    {
        var previous = SpirvModuleBuilder.KeepsUnreadVariables;
        SpirvModuleBuilder.KeepsUnreadVariables = preserve;
        try
        {
            var module = new SpirvModuleBuilder();
            module.AddCapability(SpirvCapability.Shader);
            var integer = module.TypeInt(32, false);
            var storage = indexed ? module.TypeArray(integer, 2) : integer;
            var register = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, storage), SpirvStorageClass.Private);
            module.AddName(register, "register");
            var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
            module.AddDecoration(output, SpirvDecoration.Location, 0);
            var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
            module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [register, output]);
            module.AddLabel();
            var pointer = indexed ? module.AddInstruction(SpirvOp.AccessChain,
                module.TypePointer(SpirvStorageClass.Private, integer), register, module.Constant(integer, 1)) : register;
            var value = module.Constant(integer, 42);
            module.AddStatement(SpirvOp.Store, pointer, value);
            var forwarded = module.AddInstruction(SpirvOp.Load, integer, pointer);
            Assert.Equal(value, forwarded);
            module.AddStatement(SpirvOp.Store, output, forwarded);
            module.AddStatement(SpirvOp.Return);
            module.EndFunction();
            var code = module.Build();
            var inspected = new SpirvModuleInspector(code);
            Assert.Equal(preserve, inspected.VariableStorageClasses.ContainsKey(register));
            Assert.Equal(preserve, inspected.Names.ContainsKey(register));
            var words = new uint[code.Length / sizeof(uint)];
            Buffer.BlockCopy(code, 0, words, 0, code.Length);
            var outputStores = 0;
            for (var offset = 5; offset < words.Length; offset += (int)(words[offset] >> 16))
            {
                if ((SpirvOp)(words[offset] & 0xffff) == SpirvOp.Store && words[offset + 1] == output)
                {
                    Assert.Equal(value, words[offset + 2]);
                    outputStores++;
                }
            }
            Assert.Equal(1, outputStores);
        }
        finally
        {
            SpirvModuleBuilder.KeepsUnreadVariables = previous;
        }
    }

    [Fact]
    public void ExplicitNoneLoadKeepsThePrivateStoreItReadsBeforeALaterStore()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var integer = module.TypeInt(32, false);
        var register = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, integer), SpirvStorageClass.Private);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);
        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [register, output]);
        module.AddLabel();
        var first = module.Constant(integer, 41);
        var second = module.Constant(integer, 73);
        module.AddStatement(SpirvOp.Store, register, first);
        var loaded = module.AddInstruction(SpirvOp.Load, integer, register, 0); // Memory Access: None.
        module.AddStatement(SpirvOp.Store, register, second);
        module.AddStatement(SpirvOp.Store, output, loaded);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        var stores = instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == register).ToArray();
        Assert.Collection(stores,
            words => Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (3u << 16), register, first }, words),
            words => Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (3u << 16), register, second }, words));
        var load = Assert.Single(instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == loaded));
        Assert.Equal(new uint[] { ((uint)SpirvOp.Load) | (5u << 16), integer, loaded, register, 0 }, load);
        Assert.True(Array.IndexOf(instructions, stores[0]) < Array.IndexOf(instructions, load));
        Assert.True(Array.IndexOf(instructions, load) < Array.IndexOf(instructions, stores[1]));
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == output && words[2] == loaded);
    }

    [Fact]
    public void ExplicitNoneElementStoreInvalidatesThePreviouslyForwardedValue()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var integer = module.TypeInt(32, false);
        var array = module.TypeArray(integer, 2);
        var register = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, array), SpirvStorageClass.Private);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);
        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [register, output]);
        module.AddLabel();
        var pointer = module.AddInstruction(SpirvOp.AccessChain,
            module.TypePointer(SpirvStorageClass.Private, integer), register, module.Constant(integer, 1));
        var first = module.Constant(integer, 41);
        var second = module.Constant(integer, 73);
        module.AddStatement(SpirvOp.Store, pointer, first);
        module.AddStatement(SpirvOp.Store, pointer, second, 0); // Memory Access: None.
        var loaded = module.AddInstruction(SpirvOp.Load, integer, pointer);
        Assert.NotEqual(first, loaded);
        module.AddStatement(SpirvOp.Store, output, loaded);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        var explicitStore = Assert.Single(instructions.Where(words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words.Length == 4 && words[1] == pointer));
        Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (4u << 16), pointer, second, 0 }, explicitStore);
        var load = Assert.Single(instructions.Where(words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == loaded));
        Assert.Equal(new uint[] { ((uint)SpirvOp.Load) | (4u << 16), integer, loaded, pointer }, load);
        Assert.True(Array.IndexOf(instructions, explicitStore) < Array.IndexOf(instructions, load));
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == output && words[2] == loaded);
    }

    [Fact]
    public void VolatileLoadKeepsItsStoreWhileAnUntrackedExplicitLoadKeepsPrivateForwarding()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var integer = module.TypeInt(32, false);
        var register = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, integer), SpirvStorageClass.Private);
        var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, integer), SpirvStorageClass.Input);
        module.AddDecoration(input, SpirvDecoration.Location, 0);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);
        var inputOutput = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
        module.AddDecoration(inputOutput, SpirvDecoration.Location, 1);
        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [register, input, output, inputOutput]);
        module.AddLabel();
        var first = module.Constant(integer, 41);
        var second = module.Constant(integer, 73);
        module.AddStatement(SpirvOp.Store, register, first);
        var inputValue = module.AddInstruction(SpirvOp.Load, integer, input, 0); // Untracked Input, explicit None.
        var forwarded = module.AddInstruction(SpirvOp.Load, integer, register);
        Assert.Equal(first, forwarded);
        var loaded = module.AddInstruction(SpirvOp.Load, integer, register, 1); // Memory Access: Volatile.
        module.AddStatement(SpirvOp.Store, register, second);
        module.AddStatement(SpirvOp.Store, output, loaded);
        module.AddStatement(SpirvOp.Store, inputOutput, inputValue);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        var stores = instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == register).ToArray();
        Assert.Collection(stores,
            words => Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (3u << 16), register, first }, words),
            words => Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (3u << 16), register, second }, words));
        var volatileLoad = Assert.Single(instructions.Where(words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == loaded));
        Assert.Equal(new uint[] { ((uint)SpirvOp.Load) | (5u << 16), integer, loaded, register, 1 }, volatileLoad);
        Assert.DoesNotContain(instructions, words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words.Length == 4 && words[3] == register);
        var untrackedLoad = Assert.Single(instructions.Where(words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == inputValue));
        Assert.Equal(new uint[] { ((uint)SpirvOp.Load) | (5u << 16), integer, inputValue, input, 0 }, untrackedLoad);
        Assert.True(Array.IndexOf(instructions, stores[0]) < Array.IndexOf(instructions, volatileLoad));
        Assert.True(Array.IndexOf(instructions, volatileLoad) < Array.IndexOf(instructions, stores[1]));
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == output && words[2] == loaded);
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == inputOutput && words[2] == inputValue);
    }

    [Fact]
    public void ExplicitAccessInAnUndeclaredLoopHeaderKeepsTheInheritedReadGuard()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var integer = module.TypeInt(32, false);
        var register = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, integer), SpirvStorageClass.Private);
        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.GLCompute, function, "main", [register]);
        module.AddExecutionMode(function, SpirvExecutionMode.LocalSize, 1, 1, 1);
        module.AddLabel();
        var header = module.AllocateId();
        var latch = module.AllocateId();
        var done = module.AllocateId();
        module.AddStatement(SpirvOp.Store, register, module.Constant(integer, 1));
        module.AddStatement(SpirvOp.Branch, header);
        module.AddLabel(header); // No DeclareLoopHeader: the header inherits register = 1.
        var inherited = module.AddInstruction(SpirvOp.Load, integer, register);
        module.AddInstruction(SpirvOp.Load, integer, register, 0); // Explicit None: resets the memory state.
        var again = module.AddInstruction(SpirvOp.INotEqual, module.TypeBool(), inherited, module.Constant(integer, 2));
        module.AddStatement(SpirvOp.LoopMerge, done, latch, 0);
        module.AddStatement(SpirvOp.BranchConditional, again, latch, done);
        module.AddLabel(latch);
        module.AddStatement(SpirvOp.Store, register, module.Constant(integer, 2));
        module.AddStatement(SpirvOp.Branch, header);
        module.AddLabel(done);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        // The inherited read happened before the reset, so the header must still request the retry.
        Assert.False(module.InheritedBlocksHeld);
    }

    [Fact]
    public void ElementLoadKeepsTheWholePrivateStoreItReadsBeforeALaterWholeStore()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var integer = module.TypeInt(32, false);
        var array = module.TypeArray(integer, 2);
        var register = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, array), SpirvStorageClass.Private);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
        module.AddDecoration(output, SpirvDecoration.Location, 0);
        var first = module.ConstantComposite(array, module.Constant(integer, 41), module.Constant(integer, 42));
        var second = module.ConstantComposite(array, module.Constant(integer, 73), module.Constant(integer, 74));
        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [register, output]);
        module.AddLabel();
        var pointer = module.AddInstruction(SpirvOp.AccessChain,
            module.TypePointer(SpirvStorageClass.Private, integer), register, module.Constant(integer, 1));
        module.AddStatement(SpirvOp.Store, register, first);
        var loaded = module.AddInstruction(SpirvOp.Load, integer, pointer);
        module.AddStatement(SpirvOp.Store, register, second);
        module.AddStatement(SpirvOp.Store, output, loaded);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var instructions = Instructions(module.Build());
        var stores = instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == register).ToArray();
        Assert.Collection(stores,
            words => Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (3u << 16), register, first }, words),
            words => Assert.Equal(new uint[] { ((uint)SpirvOp.Store) | (3u << 16), register, second }, words));
        var load = Assert.Single(instructions.Where(words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[2] == loaded));
        Assert.Equal(new uint[] { ((uint)SpirvOp.Load) | (4u << 16), integer, loaded, pointer }, load);
        Assert.True(Array.IndexOf(instructions, stores[0]) < Array.IndexOf(instructions, load));
        Assert.True(Array.IndexOf(instructions, load) < Array.IndexOf(instructions, stores[1]));
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == output && words[2] == loaded);
    }

    [Fact]
    public void DynamicArrayElementAccessPreservesConstantIndexTrackingWithoutEscaping()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var integer = module.TypeInt(32, false);
        var arrayType = module.TypeArray(integer, 4);
        var privatePtr = module.TypePointer(SpirvStorageClass.Private, integer);
        var array = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, arrayType), SpirvStorageClass.Private);
        var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, integer), SpirvStorageClass.Input);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, integer), SpirvStorageClass.Output);
        module.AddDecoration(input, SpirvDecoration.Location, 0);
        module.AddDecoration(input, SpirvDecoration.Flat);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [array, input, output]);
        module.AddLabel();

        var c0 = module.Constant(integer, 0);
        var c1 = module.Constant(integer, 1);
        var c42 = module.Constant(integer, 42);
        var c99 = module.Constant(integer, 99);
        var c7 = module.Constant(integer, 7);
        var c8 = module.Constant(integer, 8);
        var dynIndex = module.AddInstruction(SpirvOp.Load, integer, input);

        // Store element 0, dynamically load an element (must keep store to 0), then read element 0 again (must forward c42).
        var ptr0 = module.AddInstruction(SpirvOp.AccessChain, privatePtr, array, c0);
        module.AddStatement(SpirvOp.Store, ptr0, c42);
        var dynLoaded = module.LoadDynamicArrayElement(integer, privatePtr, array, dynIndex);
        var afterDynLoad = module.AddInstruction(SpirvOp.Load, integer, ptr0);

        // Dynamic store invalidates cached element 0, so a read afterwards must reload from memory.
        module.StoreDynamicArrayElement(privatePtr, array, dynIndex, c99);
        var afterDynStore = module.AddInstruction(SpirvOp.Load, integer, ptr0);

        // Subsequent constant-index stores to element 1 still get dead-store elimination and forwarding.
        var ptr1 = module.AddInstruction(SpirvOp.AccessChain, privatePtr, array, c1);
        module.AddStatement(SpirvOp.Store, ptr1, c7);
        module.AddStatement(SpirvOp.Store, ptr1, c8);
        var afterConstStore = module.AddInstruction(SpirvOp.Load, integer, ptr1);

        var sum1 = module.AddInstruction(SpirvOp.IAdd, integer, dynLoaded, afterDynLoad);
        var sum2 = module.AddInstruction(SpirvOp.IAdd, integer, afterDynStore, afterConstStore);
        module.AddStatement(SpirvOp.Store, output, module.AddInstruction(SpirvOp.IAdd, integer, sum1, sum2));
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        var spirv = module.Build();
        Gen5MipStatisticsTests.ValidateWhenAvailable(spirv);
        var instructions = Instructions(spirv);

        // Element 0 read after LoadDynamicArrayElement forwarded c42 without a second Load;
        // element 0 read after StoreDynamicArrayElement emitted a fresh Load;
        // dead store of c7 to ptr1 was eliminated and afterConstStore forwarded c8.
        Assert.Equal(c42, afterDynLoad);
        Assert.NotEqual(c42, afterDynStore);
        Assert.Equal(c8, afterConstStore);
        Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == ptr0 && words[2] == c42);
        Assert.DoesNotContain(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == ptr1 && words[2] == c7);
    }

    [Fact]
    public void LoopHeaderPrivateVariablesArePromotedToSsaPhis()
    {
        var previous = SpirvModuleBuilder.KeepsUnreadVariables;
        SpirvModuleBuilder.KeepsUnreadVariables = false;
        try
        {
            var module = new SpirvModuleBuilder();
            module.AddCapability(SpirvCapability.Shader);
            var u32 = module.TypeInt(32, false);
            var boolean = module.TypeBool();
            var acc = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var counter = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
            var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
            module.AddDecoration(input, SpirvDecoration.Location, 0);
            module.AddDecoration(input, SpirvDecoration.Flat);
            module.AddDecoration(output, SpirvDecoration.Location, 0);

            var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
            module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [acc, counter, input, output]);
            module.AddLabel();

            var zero = module.Constant(u32, 0);
            var one = module.Constant(u32, 1);
            var limit = module.AddInstruction(SpirvOp.Load, u32, input);
            var header = module.AllocateId();
            var body = module.AllocateId();
            var latch = module.AllocateId();
            var done = module.AllocateId();
            module.DeclareLoopHeader(header);

            module.AddStatement(SpirvOp.Store, acc, zero);
            module.AddStatement(SpirvOp.Store, counter, zero);
            module.AddStatement(SpirvOp.Branch, header);

            module.AddLabel(header);
            var curCounter = module.AddInstruction(SpirvOp.Load, u32, counter);
            var cond = module.AddInstruction(SpirvOp.ULessThan, boolean, curCounter, limit);
            module.AddStatement(SpirvOp.LoopMerge, done, latch, 0);
            module.AddStatement(SpirvOp.BranchConditional, cond, body, done);

            module.AddLabel(body);
            var curAcc = module.AddInstruction(SpirvOp.Load, u32, acc);
            var nextAcc = module.AddInstruction(SpirvOp.IAdd, u32, curAcc, curCounter);
            module.AddStatement(SpirvOp.Store, acc, nextAcc);
            module.AddStatement(SpirvOp.Branch, latch);

            module.AddLabel(latch);
            var latchCounter = module.AddInstruction(SpirvOp.Load, u32, counter);
            var nextCounter = module.AddInstruction(SpirvOp.IAdd, u32, latchCounter, one);
            module.AddStatement(SpirvOp.Store, counter, nextCounter);
            module.AddStatement(SpirvOp.Branch, header);

            module.AddLabel(done);
            var finalAcc = module.AddInstruction(SpirvOp.Load, u32, acc);
            module.AddStatement(SpirvOp.Store, output, finalAcc);
            module.AddStatement(SpirvOp.Return);
            module.EndFunction();

            var spirv = module.Build();
            Gen5MipStatisticsTests.ValidateWhenAvailable(spirv);
            var inspected = new SpirvModuleInspector(spirv);
            Assert.False(inspected.VariableStorageClasses.ContainsKey(acc));
            Assert.False(inspected.VariableStorageClasses.ContainsKey(counter));

            var instructions = Instructions(spirv);
            Assert.DoesNotContain(instructions, words =>
                (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && (words[3] == acc || words[3] == counter));
            Assert.DoesNotContain(instructions, words =>
                (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && (words[1] == acc || words[1] == counter));

            var phis = instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Phi).ToArray();
            Assert.Equal(2, phis.Length);
            Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == output);
        }
        finally
        {
            SpirvModuleBuilder.KeepsUnreadVariables = previous;
        }
    }

    [Fact]
    public void CrossBlockDeadStoresToPromotedPrivateVariableDropUpstreamProducers()
    {
        var previous = SpirvModuleBuilder.KeepsUnreadVariables;
        SpirvModuleBuilder.KeepsUnreadVariables = false;
        try
        {
            var module = new SpirvModuleBuilder { InheritBlockState = false };
            module.AddCapability(SpirvCapability.Shader);
            var u32 = module.TypeInt(32, false);
            var boolean = module.TypeBool();
            var scratch = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
            var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
            module.AddDecoration(input, SpirvDecoration.Location, 0);
            module.AddDecoration(input, SpirvDecoration.Flat);
            module.AddDecoration(output, SpirvDecoration.Location, 0);

            var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
            module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [scratch, input, output]);
            module.AddLabel();

            var inVal = module.AddInstruction(SpirvOp.Load, u32, input);
            var deadEntryExpr = module.AddInstruction(SpirvOp.IMul, u32, inVal, module.Constant(u32, 13));
            module.AddStatement(SpirvOp.Store, scratch, deadEntryExpr);

            var cond = module.AddInstruction(SpirvOp.INotEqual, boolean, inVal, module.Constant(u32, 0));
            var left = module.AllocateId();
            var right = module.AllocateId();
            var merge = module.AllocateId();
            module.AddStatement(SpirvOp.SelectionMerge, merge, 0);
            module.AddStatement(SpirvOp.BranchConditional, cond, left, right);

            module.AddLabel(left);
            var liveLeft = module.AddInstruction(SpirvOp.IAdd, u32, inVal, module.Constant(u32, 1));
            module.AddStatement(SpirvOp.Store, scratch, liveLeft);
            module.AddStatement(SpirvOp.Branch, merge);

            module.AddLabel(right);
            var liveRight = module.AddInstruction(SpirvOp.IAdd, u32, inVal, module.Constant(u32, 2));
            module.AddStatement(SpirvOp.Store, scratch, liveRight);
            module.AddStatement(SpirvOp.Branch, merge);

            module.AddLabel(merge);
            var merged = module.AddInstruction(SpirvOp.Load, u32, scratch);
            module.AddStatement(SpirvOp.Store, output, merged);
            var deadFinalExpr = module.AddInstruction(SpirvOp.IMul, u32, inVal, module.Constant(u32, 29));
            module.AddStatement(SpirvOp.Store, scratch, deadFinalExpr);
            module.AddStatement(SpirvOp.Return);
            module.EndFunction();

            var spirv = module.Build();
            Gen5MipStatisticsTests.ValidateWhenAvailable(spirv);
            var inspected = new SpirvModuleInspector(spirv);
            Assert.False(inspected.VariableStorageClasses.ContainsKey(scratch));

            var instructions = Instructions(spirv);
            Assert.DoesNotContain(instructions, words =>
                (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[3] == scratch);
            Assert.DoesNotContain(instructions, words =>
                (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == scratch);
            Assert.DoesNotContain(instructions, words =>
                (SpirvOp)(words[0] & 0xffff) == SpirvOp.IMul && (words[2] == deadEntryExpr || words[2] == deadFinalExpr));
            Assert.Single(instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Phi));
        }
        finally
        {
            SpirvModuleBuilder.KeepsUnreadVariables = previous;
        }
    }

    [Fact]
    public void MarkAndSweepDceCollectsDeadPhiCycle()
    {
        var previous = SpirvModuleBuilder.KeepsUnreadVariables;
        SpirvModuleBuilder.KeepsUnreadVariables = false;
        try
        {
            var module = new SpirvModuleBuilder();
            module.AddCapability(SpirvCapability.Shader);
            var u32 = module.TypeInt(32, false);
            var boolean = module.TypeBool();
            var liveCounter = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var deadAccum = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
            var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
            module.AddDecoration(input, SpirvDecoration.Location, 0);
            module.AddDecoration(input, SpirvDecoration.Flat);
            module.AddDecoration(output, SpirvDecoration.Location, 0);

            var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
            module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [liveCounter, deadAccum, input, output]);
            module.AddLabel();

            var zero = module.Constant(u32, 0);
            var one = module.Constant(u32, 1);
            var seven = module.Constant(u32, 7);
            var limit = module.AddInstruction(SpirvOp.Load, u32, input);
            var header = module.AllocateId();
            var latch = module.AllocateId();
            var done = module.AllocateId();
            module.DeclareLoopHeader(header);

            module.AddStatement(SpirvOp.Store, liveCounter, zero);
            module.AddStatement(SpirvOp.Store, deadAccum, zero);
            module.AddStatement(SpirvOp.Branch, header);

            module.AddLabel(header);
            var curLive = module.AddInstruction(SpirvOp.Load, u32, liveCounter);
            var cond = module.AddInstruction(SpirvOp.ULessThan, boolean, curLive, limit);
            module.AddStatement(SpirvOp.LoopMerge, done, latch, 0);
            module.AddStatement(SpirvOp.BranchConditional, cond, latch, done);

            module.AddLabel(latch);
            var nextLive = module.AddInstruction(SpirvOp.IAdd, u32, curLive, one);
            module.AddStatement(SpirvOp.Store, liveCounter, nextLive);
            // Self-referential dead accumulator: promoted to a Phi <-> IMul cycle that is never observed.
            var curDead = module.AddInstruction(SpirvOp.Load, u32, deadAccum);
            var nextDead = module.AddInstruction(SpirvOp.IMul, u32, curDead, seven);
            module.AddStatement(SpirvOp.Store, deadAccum, nextDead);
            module.AddStatement(SpirvOp.Branch, header);

            module.AddLabel(done);
            var result = module.AddInstruction(SpirvOp.Load, u32, liveCounter);
            module.AddStatement(SpirvOp.Store, output, result);
            module.AddStatement(SpirvOp.Return);
            module.EndFunction();

            var spirv = module.Build();
            Gen5MipStatisticsTests.ValidateWhenAvailable(spirv);
            var inspected = new SpirvModuleInspector(spirv);
            Assert.False(inspected.VariableStorageClasses.ContainsKey(liveCounter));
            Assert.False(inspected.VariableStorageClasses.ContainsKey(deadAccum));

            var instructions = Instructions(spirv);
            // Only the live loop counter Phi survives; the dead accumulator Phi and its IMul cycle are swept.
            Assert.Single(instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Phi));
            Assert.DoesNotContain(instructions, words =>
                (SpirvOp)(words[0] & 0xffff) == SpirvOp.IMul && words[2] == nextDead);
        }
        finally
        {
            SpirvModuleBuilder.KeepsUnreadVariables = previous;
        }
    }

    [Fact]
    public void MultiFunctionShaderPreservesSharedPrivateVariableWhilePromotingSingleFunctionPrivateVariable()
    {
        var previous = SpirvModuleBuilder.KeepsUnreadVariables;
        SpirvModuleBuilder.KeepsUnreadVariables = false;
        try
        {
            var module = new SpirvModuleBuilder();
            module.AddCapability(SpirvCapability.Shader);
            var voidType = module.TypeVoid();
            var fnType = module.TypeFunction(voidType);
            var u32 = module.TypeInt(32, false);
            var boolean = module.TypeBool();
            var sharedVar = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var localOnlyVar = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Private, u32), SpirvStorageClass.Private);
            var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
            var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
            module.AddDecoration(input, SpirvDecoration.Location, 0);
            module.AddDecoration(input, SpirvDecoration.Flat);
            module.AddDecoration(output, SpirvDecoration.Location, 0);

            var helper = module.BeginFunction(voidType, fnType, 2); // DontInline.
            module.AddLabel();
            var helperRead = module.AddInstruction(SpirvOp.Load, u32, sharedVar);
            var helperUpdated = module.AddInstruction(SpirvOp.IAdd, u32, helperRead, module.Constant(u32, 5));
            module.AddStatement(SpirvOp.Store, sharedVar, helperUpdated);
            module.AddStatement(SpirvOp.Return);
            module.EndFunction();

            var main = module.BeginFunction(voidType, fnType);
            module.AddEntryPoint(SpirvExecutionModel.Vertex, main, "main", [sharedVar, localOnlyVar, input, output]);
            module.AddLabel();

            var limit = module.AddInstruction(SpirvOp.Load, u32, input);
            var header = module.AllocateId();
            var latch = module.AllocateId();
            var done = module.AllocateId();
            module.DeclareLoopHeader(header);

            module.AddStatement(SpirvOp.Store, sharedVar, module.Constant(u32, 10));
            module.AddStatement(SpirvOp.Store, localOnlyVar, module.Constant(u32, 0));
            module.AddStatement(SpirvOp.Branch, header);

            module.AddLabel(header);
            var curLocal = module.AddInstruction(SpirvOp.Load, u32, localOnlyVar);
            var cond = module.AddInstruction(SpirvOp.ULessThan, boolean, curLocal, limit);
            module.AddStatement(SpirvOp.LoopMerge, done, latch, 0);
            module.AddStatement(SpirvOp.BranchConditional, cond, latch, done);

            module.AddLabel(latch);
            module.AddInstruction(SpirvOp.FunctionCall, voidType, helper);
            var nextLocal = module.AddInstruction(SpirvOp.IAdd, u32, curLocal, module.Constant(u32, 1));
            module.AddStatement(SpirvOp.Store, localOnlyVar, nextLocal);
            module.AddStatement(SpirvOp.Branch, header);

            module.AddLabel(done);
            var finalShared = module.AddInstruction(SpirvOp.Load, u32, sharedVar);
            var finalLocal = module.AddInstruction(SpirvOp.Load, u32, localOnlyVar);
            module.AddStatement(SpirvOp.Store, output, module.AddInstruction(SpirvOp.IAdd, u32, finalShared, finalLocal));
            module.AddStatement(SpirvOp.Return);
            module.EndFunction();

            var spirv = module.Build();
            Gen5MipStatisticsTests.ValidateWhenAvailable(spirv);
            var inspected = new SpirvModuleInspector(spirv);
            Assert.True(inspected.VariableStorageClasses.ContainsKey(sharedVar));
            Assert.False(inspected.VariableStorageClasses.ContainsKey(localOnlyVar));

            var instructions = Instructions(spirv);
            Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == sharedVar);
            Assert.Contains(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[3] == sharedVar);
            Assert.DoesNotContain(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Store && words[1] == localOnlyVar);
            Assert.DoesNotContain(instructions, words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[3] == localOnlyVar);
            Assert.Single(instructions.Where(words => (SpirvOp)(words[0] & 0xffff) == SpirvOp.Phi));
        }
        finally
        {
            SpirvModuleBuilder.KeepsUnreadVariables = previous;
        }
    }

    [Fact]
    public void MarkImmutableVariableEnablesCseOfRepeatedStorageBufferLoads()
    {
        var module = new SpirvModuleBuilder();
        module.AddCapability(SpirvCapability.Shader);
        var u32 = module.TypeInt(32, false);
        var runtimeArray = module.TypeRuntimeArray(u32);
        module.AddDecoration(runtimeArray, SpirvDecoration.ArrayStride, 4);
        var bufferStruct = module.TypeStruct(runtimeArray);
        module.AddDecoration(bufferStruct, SpirvDecoration.Block);
        module.AddMemberDecoration(bufferStruct, 0, SpirvDecoration.Offset, 0);
        var bufferPtr = module.TypePointer(SpirvStorageClass.StorageBuffer, bufferStruct);
        var elemPtr = module.TypePointer(SpirvStorageClass.StorageBuffer, u32);
        var ssbo = module.AddGlobalVariable(bufferPtr, SpirvStorageClass.StorageBuffer);
        module.AddDecoration(ssbo, SpirvDecoration.DescriptorSet, 0);
        module.AddDecoration(ssbo, SpirvDecoration.Binding, 0);
        module.AddDecoration(ssbo, SpirvDecoration.NonWritable);
        module.MarkImmutableVariable(ssbo);

        var input = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Input, u32), SpirvStorageClass.Input);
        var output = module.AddGlobalVariable(module.TypePointer(SpirvStorageClass.Output, u32), SpirvStorageClass.Output);
        module.AddDecoration(input, SpirvDecoration.Location, 0);
        module.AddDecoration(input, SpirvDecoration.Flat);
        module.AddDecoration(output, SpirvDecoration.Location, 0);

        var function = module.BeginFunction(module.TypeVoid(), module.TypeFunction(module.TypeVoid()));
        module.AddEntryPoint(SpirvExecutionModel.Vertex, function, "main", [ssbo, input, output]);
        module.AddLabel();

        var zero = module.Constant(u32, 0);
        var idx = module.AddInstruction(SpirvOp.Load, u32, input);
        var ptr1 = module.AddInstruction(SpirvOp.AccessChain, elemPtr, ssbo, zero, idx);
        var first = module.AddInstruction(SpirvOp.Load, u32, ptr1);
        var ptr2 = module.AddInstruction(SpirvOp.AccessChain, elemPtr, ssbo, zero, idx);
        var second = module.AddInstruction(SpirvOp.Load, u32, ptr2);

        var nextBlock = module.AllocateId();
        module.AddStatement(SpirvOp.Branch, nextBlock);
        module.AddLabel(nextBlock);
        var ptr3 = module.AddInstruction(SpirvOp.AccessChain, elemPtr, ssbo, zero, idx);
        var third = module.AddInstruction(SpirvOp.Load, u32, ptr3);

        var sum = module.AddInstruction(SpirvOp.IAdd, u32, module.AddInstruction(SpirvOp.IAdd, u32, first, second), third);
        module.AddStatement(SpirvOp.Store, output, sum);
        module.AddStatement(SpirvOp.Return);
        module.EndFunction();

        Assert.Equal(first, second);
        Assert.Equal(first, third);

        var spirv = module.Build();
        Gen5MipStatisticsTests.ValidateWhenAvailable(spirv);
        var instructions = Instructions(spirv);
        var ssboLoads = instructions.Where(words =>
            (SpirvOp)(words[0] & 0xffff) == SpirvOp.Load && words[3] == ptr1).ToArray();
        Assert.Single(ssboLoads);
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
