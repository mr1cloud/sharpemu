// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Ir;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests;

public sealed class VMovrelAnalysisTests
{
    private const uint M0 = 124;
    private const uint Exec = 126;

    private readonly List<Gen5ShaderInstruction> _instructions = [];
    private uint _pc;

    private uint Add(Gen5ShaderInstruction instruction)
    {
        var pc = _pc;
        _instructions.Add(instruction with { Pc = pc });
        _pc += 8;
        return pc;
    }

    private void Narrow() => Add(Vopc(0, "VCmpxLeU32", Operand(40), 2));

    private void SaveExec() => Add(Sop1(0, "SMovB64", 20, Gen5Operand.Scalar(Exec)));

    private void RestoreExec() => Add(Sop1(0, "SMovB64", Exec, Gen5Operand.Scalar(20)));

    private Gen5ShaderProgram BuildProgram(bool appendEnd = true)
    {
        if (appendEnd)
        {
            Add(EndProgram(0));
        }

        return Program([.. _instructions]);
    }

    private Gen5VectorObservationPlan AnalyzeObservation()
    {
        var plan = Gen5VectorObservationAnalysis.Analyze(BuildProgram(), wave32: false);
        Assert.NotNull(plan);
        return plan;
    }

    private IReadOnlyDictionary<uint, uint[]> AnalyzeOffsets() =>
        Gen5MoveRelativeOffsets.Analyze(BuildProgram(appendEnd: false));

    [Fact]
    public void VectorRegisterSetAll_ContainsAll256Registers()
    {
        var all = VectorRegisterSet.All;

        Assert.False(all.IsEmpty);
        Assert.Equal(256, all.Count);
        Assert.True(all.Contains(0));
        Assert.True(all.Contains(127));
        Assert.True(all.Contains(128));
        Assert.True(all.Contains(255));
        Assert.False(all.Contains(256));
    }

    [Fact]
    public void BoundedVMovrelsB32_MarksOnlyReachableReadsObserved()
    {
        SaveExec();
        Narrow();
        var write4 = Add(MoveVector(0, 4, 1));
        var write5 = Add(MoveVector(0, 5, 2));
        var write6 = Add(MoveVector(0, 6, 3));
        var write7 = Add(MoveVector(0, 7, 4));
        var write8 = Add(MoveVector(0, 8, 5));
        RestoreExec();

        // M0 is bounded to {0, 2} -> VMovrelsB32 reads only v4 and v6.
        Add(Sop2(0, "SAndB32", 10, Gen5Operand.Scalar(32), Operand(2)));
        Add(Sop1(0, "SMovB32", M0, Gen5Operand.Scalar(10)));
        Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));
        Add(Vop2(0, "VAddU32", 41, Operand(1), Gen5Operand.Vector(40)));

        var plan = AnalyzeObservation();

        Assert.False(plan.UnobservedWrites(write4).Contains(4));
        Assert.True(plan.UnobservedWrites(write5).Contains(5));
        Assert.False(plan.UnobservedWrites(write6).Contains(6));
        Assert.True(plan.UnobservedWrites(write7).Contains(7));
        Assert.True(plan.UnobservedWrites(write8).Contains(8));
    }

    [Fact]
    public void UnboundedVMovrelsB32_MarksAllPriorReadsObservedWithoutDisablingPlan()
    {
        SaveExec();
        Narrow();
        var writeLow = Add(MoveVector(0, 10, 1));
        var writeHigh = Add(MoveVector(0, 200, 2));
        RestoreExec();

        // Unknown M0 from user SGPR s30 -> conservative VectorRegisterSet.All read.
        Add(Sop1(0, "SMovB32", M0, Gen5Operand.Scalar(30)));
        Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        // A later unrelated narrowed temporary is still eliminated.
        SaveExec();
        Narrow();
        var laterTemp = Add(MoveVector(0, 50, 7));
        Add(Vop2(0, "VAddU32", 51, Operand(1), Gen5Operand.Vector(50)));
        RestoreExec();

        var plan = AnalyzeObservation();

        Assert.False(plan.UnobservedWrites(writeLow).Contains(10));
        Assert.False(plan.UnobservedWrites(writeHigh).Contains(200));
        Assert.True(plan.UnobservedWrites(laterTemp).Contains(50));
    }

    [Fact]
    public void VMovrelsB32_DestinationIsFixedAndKillsUnderFullExec()
    {
        SaveExec();
        Narrow();
        var priorWrite = Add(MoveVector(0, 40, 1));
        var narrowMovrels = Add(Vop1(0, "VMovrelsB32", 41, Gen5Operand.Vector(4)));
        RestoreExec();

        // Bounded M0 = {0}, so VMovrelsB32 reads only v4 and fully overwrites v40 under full EXEC.
        Add(Sop1(0, "SMovB32", M0, Operand(0)));
        Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));
        Add(Vop2(0, "VAddU32", 42, Operand(1), Gen5Operand.Vector(40)));

        var plan = AnalyzeObservation();

        Assert.True(plan.UnobservedWrites(priorWrite).Contains(40));
        Assert.True(plan.UnobservedWrites(narrowMovrels).Contains(41));
    }

    [Fact]
    public void VMovreldB32_DoesNotKillTargetAndMarksSourceRead()
    {
        SaveExec();
        Narrow();
        var writeTarget = Add(MoveVector(0, 10, 1));
        var writeSource = Add(MoveVector(0, 20, 2));
        var writeUnused = Add(MoveVector(0, 30, 3));
        RestoreExec();

        // Even under full EXEC with M0 = {2}, VMovreldB32 conservatively kills nothing and has no unobserved writes.
        Add(Sop1(0, "SMovB32", M0, Operand(2)));
        var movreld = Add(Vop1(0, "VMovreldB32", 8, Gen5Operand.Vector(20)));
        Add(Vop2(0, "VAddU32", 40, Operand(1), Gen5Operand.Vector(10)));

        var plan = AnalyzeObservation();

        Assert.True(plan.UnobservedWrites(movreld).IsEmpty);
        Assert.False(plan.UnobservedWrites(writeTarget).Contains(10));
        Assert.False(plan.UnobservedWrites(writeSource).Contains(20));
        Assert.True(plan.UnobservedWrites(writeUnused).Contains(30));
    }

    [Fact]
    public void VMovrelsdB32_BoundedReadsObservedAndKillsNothing()
    {
        SaveExec();
        Narrow();
        var writeReadCandidate = Add(MoveVector(0, 6, 1));
        var writeTargetCandidate = Add(MoveVector(0, 12, 2));
        var writeUnrelated = Add(MoveVector(0, 30, 3));
        RestoreExec();

        // M0 in {0, 2}: reads v4/v6, writes v10/v12 without killing v12.
        Add(Sop2(0, "SAndB32", 10, Gen5Operand.Scalar(32), Operand(2)));
        Add(Sop1(0, "SMovB32", M0, Gen5Operand.Scalar(10)));
        var movrelsd = Add(Vop1(0, "VMovrelsdB32", 10, Gen5Operand.Vector(4)));
        Add(Vop2(0, "VAddU32", 40, Operand(1), Gen5Operand.Vector(12)));

        var plan = AnalyzeObservation();

        Assert.True(plan.UnobservedWrites(movrelsd).IsEmpty);
        Assert.False(plan.UnobservedWrites(writeReadCandidate).Contains(6));
        Assert.False(plan.UnobservedWrites(writeTargetCandidate).Contains(12));
        Assert.True(plan.UnobservedWrites(writeUnrelated).Contains(30));
    }

    [Fact]
    public void VMovrelsd2B32_ConservativelyReadsAllAndKillsNothing()
    {
        SaveExec();
        Narrow();
        var writeHigh = Add(MoveVector(0, 150, 1));
        RestoreExec();

        Add(Sop1(0, "SMovB32", M0, Operand(0)));
        var movrelsd2 = Add(Vop1(0, "VMovrelsd2B32", 10, Gen5Operand.Vector(4)));

        var plan = AnalyzeObservation();

        Assert.True(plan.UnobservedWrites(movrelsd2).IsEmpty);
        Assert.False(plan.UnobservedWrites(writeHigh).Contains(150));
    }

    [Fact]
    public void ReadFirstLane_CndmaskOrFfblLshlAdd_ProducesExactPointwiseOffsets()
    {
        // mask = (c0 ? 1 : 0) | (c1 ? 2 : 0) | (c2 ? 4 : 0) | 8 -> bits 0..3, always non-zero
        // idx = ffbl(mask) -> {0, 1, 2, 3}
        // offset = (idx << 1) + idx -> {0, 3, 6, 9} (pointwise, not Cartesian)
        Add(Vop3(0, "VCndmaskB32", 1, Operand(0), Operand(1), Gen5Operand.Scalar(10)));
        Add(Vop3(0, "VCndmaskB32", 2, Operand(0), Operand(2), Gen5Operand.Scalar(12)));
        Add(Vop2(0, "VOrB32", 1, Gen5Operand.Vector(1), Gen5Operand.Vector(2)));
        Add(Vop3(0, "VCndmaskB32", 2, Operand(0), Operand(4), Gen5Operand.Scalar(14)));
        Add(Vop2(0, "VOrB32", 1, Gen5Operand.Vector(1), Gen5Operand.Vector(2)));
        Add(Vop2(0, "VOrB32", 1, Gen5Operand.Vector(1), Operand(8)));
        Add(Vop1(0, "VFfblB32", 0, Gen5Operand.Vector(1)));
        Add(Vop3(0, "VLshlAddU32", 0, Gen5Operand.Vector(0), Operand(1), Gen5Operand.Vector(0)));
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        var offsets = AnalyzeOffsets();

        Assert.True(offsets.ContainsKey(move));
        Assert.Equal(new uint[] { 0, 3, 6, 9 }, offsets[move]);
    }

    [Fact]
    public void ReadFirstLane_FfblZeroIncluded_PreservesAllOnesSentinel()
    {
        Add(Vop3(0, "VCndmaskB32", 1, Operand(0), Operand(1), Gen5Operand.Scalar(10)));
        Add(Vop3(0, "VCndmaskB32", 2, Operand(0), Operand(4), Gen5Operand.Scalar(12)));
        Add(Vop2(0, "VOrB32", 1, Gen5Operand.Vector(1), Gen5Operand.Vector(2)));
        Add(Vop1(0, "VFfblB32", 0, Gen5Operand.Vector(1)));
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        var offsets = AnalyzeOffsets();

        Assert.Equal(new uint[] { 0, 2, uint.MaxValue }, offsets[move]);
    }

    [Fact]
    public void ReadFirstLane_UnboundedFfblClampedByVMinU32_BoundsToClampRange()
    {
        Add(Vop1(0, "VFfblB32", 0, Gen5Operand.Vector(10)));
        Add(Vop2(0, "VMinU32", 0, Gen5Operand.Vector(0), Operand(3)));
        Add(Vop3(0, "VLshlAddU32", 0, Gen5Operand.Vector(0), Operand(1), Gen5Operand.Vector(0)));
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        var offsets = AnalyzeOffsets();

        Assert.Equal(new uint[] { 0, 3, 6, 9 }, offsets[move]);
    }

    [Fact]
    public void ReadLane_UnderFullExec_TracesAndShiftAndBfeAndMad()
    {
        // v0 = (v10 & 3) << 1 -> {0, 2, 4, 6}
        Add(Vop2(0, "VAndB32", 0, Gen5Operand.Vector(10), Operand(3)));
        Add(Vop2(0, "VLshlrevB32", 0, Operand(1), Gen5Operand.Vector(0)));
        Add(ReadLane(0, M0, 0, 0));
        var move1 = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        // v1 = bfe(v11, 4, 2) * 3 + 1 -> {1, 4, 7, 10}
        Add(Vop3(0, "VBfeU32", 1, Gen5Operand.Vector(11), Operand(4), Operand(2)));
        Add(Vop3(0, "VMadU32U24", 1, Gen5Operand.Vector(1), Operand(3), Operand(1)));
        Add(ReadLane(0, M0, 1, 0));
        var move2 = Add(Vop1(0, "VMovrelsB32", 41, Gen5Operand.Vector(4)));

        // v2 = ffbh(cndmask(0x80000000, 0x00010000)) -> {0, 15}; v2 = v2 + v2 (pointwise) -> {0, 30}
        Add(Vop3(0, "VCndmaskB32", 2, Operand(0x8000_0000u), Operand(0x0001_0000u), Gen5Operand.Scalar(10)));
        Add(Vop1(0, "VFfbhU32", 2, Gen5Operand.Vector(2)));
        Add(Vop2(0, "VAddNcU32", 2, Gen5Operand.Vector(2), Gen5Operand.Vector(2)));
        Add(ReadFirstLane(0, M0, 2));
        var move3 = Add(Vop1(0, "VMovrelsB32", 42, Gen5Operand.Vector(4)));

        var offsets = AnalyzeOffsets();

        Assert.Equal(new uint[] { 0, 2, 4, 6 }, offsets[move1]);
        Assert.Equal(new uint[] { 1, 4, 7, 10 }, offsets[move2]);
        Assert.Equal(new uint[] { 0, 30 }, offsets[move3]);
    }

    [Fact]
    public void ReadFirstLane_ClobberedVectorRegister_StaysUnbounded()
    {
        Add(Vop2(0, "VAndB32", 0, Gen5Operand.Vector(10), Operand(3)));
        // Intervening unknown write to v0 (e.g. buffer load overwriting v0).
        Add(BufferAccess(0, "BufferLoadDword", resourceRegister: 0, vectorData: 0));
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        Assert.False(AnalyzeOffsets().ContainsKey(move));
    }

    [Fact]
    public void ReadFirstLane_DynamicVMovreldClobber_StaysUnbounded()
    {
        Add(Vop2(0, "VAndB32", 0, Gen5Operand.Vector(10), Operand(3)));
        Add(Vop1(0, "VMovreldB32", 0, Gen5Operand.Vector(12)));
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        Assert.False(AnalyzeOffsets().ContainsKey(move));
    }

    [Fact]
    public void ReadFirstLane_ExecWideningBetweenVectorWriteAndRead_StaysUnbounded()
    {
        SaveExec();
        Narrow();
        Add(MoveVector(0, 0, 3));
        RestoreExec();
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        Assert.False(AnalyzeOffsets().ContainsKey(move));
    }

    [Fact]
    public void ReadLane_AfterNarrowedExec_StaysUnbounded()
    {
        SaveExec();
        Narrow();
        Add(MoveVector(0, 0, 3));
        Add(ReadLane(0, M0, 0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        Assert.False(AnalyzeOffsets().ContainsKey(move));
    }

    [Fact]
    public void ReadFirstLane_DppModifiedVectorWrite_StaysUnbounded()
    {
        Add(new Gen5ShaderInstruction(0, Gen5ShaderEncoding.Vop1, "VMovB32", [0u, 0u],
            [Operand(3)], [Gen5Operand.Vector(0)],
            new Gen5DppControl(0x111, false, false, 0, 0, 0xF, 0xF)));
        Add(ReadFirstLane(0, M0, 0));
        var move = Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));

        Assert.False(AnalyzeOffsets().ContainsKey(move));
    }

    [Fact]
    public void VectorObservation_UsesVectorDerivedOffsetsEndToEnd()
    {
        SaveExec();
        Narrow();
        var write4 = Add(MoveVector(0, 4, 1));
        var write5 = Add(MoveVector(0, 5, 2));
        var write6 = Add(MoveVector(0, 6, 3));
        var write7 = Add(MoveVector(0, 7, 4));
        var write10 = Add(MoveVector(0, 10, 5));
        RestoreExec();

        // idx in {0, 1, 2} -> offset in {0, 3, 6} -> reads only v4, v7, v10.
        Add(Vop3(0, "VCndmaskB32", 1, Operand(0), Operand(1), Gen5Operand.Scalar(10)));
        Add(Vop3(0, "VCndmaskB32", 2, Operand(0), Operand(2), Gen5Operand.Scalar(12)));
        Add(Vop2(0, "VOrB32", 1, Gen5Operand.Vector(1), Gen5Operand.Vector(2)));
        Add(Vop2(0, "VOrB32", 1, Gen5Operand.Vector(1), Operand(4)));
        Add(Vop1(0, "VFfblB32", 0, Gen5Operand.Vector(1)));
        Add(Vop3(0, "VLshlAddU32", 0, Gen5Operand.Vector(0), Operand(1), Gen5Operand.Vector(0)));
        Add(ReadFirstLane(0, M0, 0));
        Add(Vop1(0, "VMovrelsB32", 40, Gen5Operand.Vector(4)));
        Add(Vop2(0, "VAddU32", 41, Operand(1), Gen5Operand.Vector(40)));

        var plan = AnalyzeObservation();

        Assert.False(plan.UnobservedWrites(write4).Contains(4));
        Assert.True(plan.UnobservedWrites(write5).Contains(5));
        Assert.True(plan.UnobservedWrites(write6).Contains(6));
        Assert.False(plan.UnobservedWrites(write7).Contains(7));
        Assert.False(plan.UnobservedWrites(write10).Contains(10));
    }
}
