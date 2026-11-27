// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Globalization;

namespace SharpEmu.ShaderCompiler.Resources;

// Chooses the branch-free device-address lowering for a compute program that reads many dwords through device
// addresses. Gen5SpirvTranslator's compact memory access was first used only by programs with shader calls (#1120);
// the default lowering gives every device-address dword its own page-table branch, unmapped-page branch, conditional
// load and fault atomic. A 26 KB compute program with ~2,000 such dwords became tens of thousands of SPIR-V blocks,
// and one native driver compilation of it held gigabytes of host memory for minutes. Compact access resolves a page
// with a select, reads an unmade access from the host's fallback address and drops it, shares one page lookup between
// the dwords of a single access, and reports the invocation's missing pages when it ends. The program then binds
// DescriptorBindingKind.DeviceAddressFallback (ShaderResourceInfo.UsesDeviceAddressFallback).
//
// The count is of device-address dwords, not of memory instructions: a program whose buffers are mostly bound keeps the
// default lowering (and its existing driver-cache entries), since compact access makes such a module larger without
// removing a significant number of blocks.
public static class CompactMemoryAccessPolicy
{
    // Device-address dwords of a compute program at or above which compact access is chosen. The count models the
    // page-table lookups the default lowering emits, including paired wave64 lane doubling, formatted device loads/stores,
    // subdword device accesses, and BVH ray intersections.
    public const int DefaultThreshold = 128;

    public const string EnvironmentVariable = "SHARPEMU_COMPACT_MEMORY_ACCESS_THRESHOLD";

    // Read once: one process never mixes two lowerings of the same program.
    public static int Threshold { get; } = ParseThreshold(Environment.GetEnvironmentVariable(EnvironmentVariable));

    // Unset or empty keeps the default; "off" or 0 disables the policy; a positive count replaces the threshold.
    public static int ParseThreshold(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return DefaultThreshold;
        }

        var text = value.Trim();
        if (string.Equals(text, "off", StringComparison.OrdinalIgnoreCase))
        {
            return int.MaxValue;
        }

        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var threshold))
        {
            throw new FormatException($"{EnvironmentVariable} must be a non-negative dword count or 'off', not '{value}'.");
        }

        return threshold == 0 ? int.MaxValue : threshold;
    }

    // Data dwords of the tracked accesses that go through the device-address page table: buffers whose descriptors are
    // read from device memory, BVH ray intersections, and global, flat and scalar-address accesses. Scratch is the
    // invocation's own. When pairedWave64 is true, vector memory accesses are emitted for both 32-lane halves.
    public static int CountDeviceAddressDwords(ShaderResourcePlan plan, bool pairedWave64 = false)
    {
        ArgumentNullException.ThrowIfNull(plan);
        var count = 0;
        foreach (var memory in plan.Memory.Entries)
        {
            if (memory.PlanningOnly)
            {
                continue;
            }

            var dwords = (int)Math.Max(1u, memory.DataDwords);
            int weight;
            if (memory.DeviceDescriptor)
            {
                if (memory.Formatted || memory.Typed)
                {
                    // Non-compact formatted device loads issue two page-table lookups per component plus inline
                    // format decoding (~80 blocks per load); formatted device stores expand across format shapes
                    // with masked atomic CAS loops.
                    weight = memory.Access == MemoryAccess.Read ? Math.Max(16, dwords * 4) : Math.Max(32, dwords * 8);
                }
                else if (memory.DataBits is > 0 and < 32)
                {
                    // Subdword device loads/stores emit two word accesses.
                    weight = Math.Max(2, dwords);
                }
                else
                {
                    weight = dwords;
                }
            }
            else if (memory.Opcode is "ImageBvhIntersectRay" or "ImageBvh64IntersectRay")
            {
                // Three ResolveDeviceAddress lookups per ray intersection.
                weight = 3;
            }
            else if (memory.Kind is MemoryResourceKind.Global or MemoryResourceKind.Flat or MemoryResourceKind.ScalarAddress ||
                (memory.Kind == MemoryResourceKind.ScalarBuffer && memory.Resource == MemoryAccessInfo.NoResource) ||
                (memory.Kind == MemoryResourceKind.Buffer &&
                 memory.BufferDescriptor?.ChooseStrategy(memory.Typed, memory.Formatted, memory.Access == MemoryAccess.Atomic) == BufferLoweringStrategy.PhysicalStorageBuffer))
            {
                weight = memory.Kind is MemoryResourceKind.Global or MemoryResourceKind.Flat or MemoryResourceKind.Buffer &&
                    memory.DataBits is > 0 and < 32
                    ? Math.Max(2, dwords)
                    : dwords;
            }
            else
            {
                continue;
            }

            var laneMultiplier = pairedWave64 && IsVectorMemory(memory) ? 2 : 1;
            count += weight * laneMultiplier;
        }

        return count;
    }

    private static bool IsVectorMemory(MemoryAccessInfo memory) =>
        memory.Kind is MemoryResourceKind.Buffer or MemoryResourceKind.Global or MemoryResourceKind.Flat ||
        memory.Opcode is "ImageBvhIntersectRay" or "ImageBvh64IntersectRay";

    public static bool Prefers(ShaderResourcePlan plan, ShaderStage stage) =>
        Prefers(plan, stage, Threshold, pairedWave64: false);

    public static bool Prefers(ShaderResourcePlan plan, ShaderStage stage, bool pairedWave64) =>
        Prefers(plan, stage, Threshold, pairedWave64);

    public static bool Prefers(ShaderResourcePlan plan, ShaderStage stage, int threshold, bool pairedWave64 = false) =>
        stage == ShaderStage.Compute && threshold != int.MaxValue && CountDeviceAddressDwords(plan, pairedWave64) >= threshold;
}
