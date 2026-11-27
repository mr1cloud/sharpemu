// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.ShaderCompiler.Vulkan;

public sealed partial class SpirvModuleBuilder
{
    private sealed class BasicBlock(uint label)
    {
        public uint Label { get; } = label;

        public List<int> InstOffsets { get; } = [];

        public List<int> Succs { get; } = [];

        public uint MergeTarget { get; set; }

        public uint ContinueTarget { get; set; }
    }

    private sealed class PlacedPhi(int varIndex, uint type, uint id)
    {
        public int VarIndex { get; } = varIndex;

        public uint Type { get; } = type;

        public uint Id { get; } = id;

        public List<(uint Value, uint PredLabel)> Incoming { get; } = [];

        public bool Removed { get; set; }
    }

    private List<uint> MaterializeFunctions()
    {
        var words = _functions;
        var materialized = new List<uint>(words.Count);
        List<uint>? prologue = null;
        for (var offset = 0; offset < words.Count;)
        {
            var wordCount = (int)(words[offset] >> 16);
            if (_removedInstructions.Contains(offset))
            {
                offset += wordCount;
                continue;
            }

            var opcode = (SpirvOp)(words[offset] & 0xFFFF);
            if (prologue is not null && opcode != SpirvOp.Variable)
            {
                materialized.AddRange(prologue);
                prologue = null;
            }

            for (var word = 0; word < wordCount; word++)
            {
                materialized.Add(words[offset + word]);
            }

            if (_entryPrologues.TryGetValue(offset, out var entry))
            {
                prologue = entry;
            }

            offset += wordCount;
        }

        if (prologue is not null)
        {
            materialized.AddRange(prologue);
        }

        return materialized;
    }

    private List<uint> PromoteLocalVariables(
        List<uint> functions,
        HashSet<uint> removedIds,
        HashSet<uint> removedVariables)
    {
        if (functions.Count == 0 || _localVariables.Count == 0)
        {
            return functions;
        }

        var funcRanges = CollectFunctionRanges(functions);
        if (funcRanges.Count == 0)
        {
            return functions;
        }

        var eligibleByFunction = CollectEligibleVariables(functions, funcRanges);
        if (eligibleByFunction.Count == 0)
        {
            return functions;
        }

        var promoted = new List<uint>(functions.Count);
        foreach (var (funcId, start, end) in funcRanges)
        {
            if (!eligibleByFunction.TryGetValue(funcId, out var eligibleVars) || eligibleVars.Count == 0)
            {
                for (var word = start; word < end; word++)
                {
                    promoted.Add(functions[word]);
                }

                continue;
            }

            PromoteFunction(functions, start, end, eligibleVars, promoted, removedIds, removedVariables);
        }

        return promoted;
    }

    private static List<(uint Id, int Start, int End)> CollectFunctionRanges(List<uint> functions)
    {
        var ranges = new List<(uint Id, int Start, int End)>();
        uint currentId = 0;
        var currentStart = -1;
        for (var offset = 0; offset < functions.Count;)
        {
            var wordCount = (int)(functions[offset] >> 16);
            var opcode = (SpirvOp)(functions[offset] & 0xFFFF);
            if (opcode == SpirvOp.Function)
            {
                currentId = functions[offset + 2];
                currentStart = offset;
            }
            else if (opcode == SpirvOp.FunctionEnd && currentStart >= 0)
            {
                ranges.Add((currentId, currentStart, offset + wordCount));
                currentStart = -1;
            }

            offset += wordCount;
        }

        return ranges;
    }

    private Dictionary<uint, List<uint>> CollectEligibleVariables(
        List<uint> functions,
        List<(uint Id, int Start, int End)> funcRanges)
    {
        var isLocal = new bool[_nextId];
        var ineligible = new bool[_nextId];
        var ownerFunc = new uint[_nextId];
        foreach (var variable in _localVariables)
        {
            isLocal[variable] = true;
            if (_escapedVariables.Contains(variable) ||
                _immutableVariables.Contains(variable) ||
                (KeepsUnreadVariables && _privateVariables.Contains(variable)) ||
                !_variableTypes.TryGetValue(variable, out var elementType) ||
                elementType == 0)
            {
                ineligible[variable] = true;
            }
        }

        void ClaimOwner(uint variable, uint funcId)
        {
            if (ownerFunc[variable] == 0)
            {
                ownerFunc[variable] = funcId;
            }
            else if (ownerFunc[variable] != funcId)
            {
                ineligible[variable] = true;
            }
        }

        foreach (var (funcId, start, end) in funcRanges)
        {
            for (var offset = start; offset < end;)
            {
                var wordCount = (int)(functions[offset] >> 16);
                var opcode = (SpirvOp)(functions[offset] & 0xFFFF);
                switch (opcode)
                {
                    case SpirvOp.Variable:
                    {
                        var varId = functions[offset + 2];
                        if (varId < (uint)isLocal.Length && isLocal[varId])
                        {
                            ClaimOwner(varId, funcId);
                        }

                        if (wordCount > 4)
                        {
                            var initId = functions[offset + 4];
                            if (initId < (uint)isLocal.Length && isLocal[initId])
                            {
                                ineligible[initId] = true;
                            }
                        }

                        break;
                    }
                    case SpirvOp.Load:
                    {
                        var ptr = functions[offset + 3];
                        if (ptr < (uint)isLocal.Length && isLocal[ptr])
                        {
                            ClaimOwner(ptr, funcId);
                            if (wordCount != 4 || !_variableTypes.TryGetValue(ptr, out var elemType) ||
                                functions[offset + 1] != elemType)
                            {
                                ineligible[ptr] = true;
                            }
                        }

                        for (var word = 4; word < wordCount; word++)
                        {
                            if (IsLiteralOperand(opcode, functions, offset, word))
                            {
                                continue;
                            }

                            var id = functions[offset + word];
                            if (id < (uint)isLocal.Length && isLocal[id])
                            {
                                ineligible[id] = true;
                            }
                        }

                        break;
                    }
                    case SpirvOp.Store:
                    {
                        var ptr = functions[offset + 1];
                        if (ptr < (uint)isLocal.Length && isLocal[ptr])
                        {
                            ClaimOwner(ptr, funcId);
                            if (wordCount != 3)
                            {
                                ineligible[ptr] = true;
                            }
                        }

                        var valueId = functions[offset + 2];
                        if (valueId < (uint)isLocal.Length && isLocal[valueId])
                        {
                            ineligible[valueId] = true;
                        }

                        for (var word = 3; word < wordCount; word++)
                        {
                            if (IsLiteralOperand(opcode, functions, offset, word))
                            {
                                continue;
                            }

                            var id = functions[offset + word];
                            if (id < (uint)isLocal.Length && isLocal[id])
                            {
                                ineligible[id] = true;
                            }
                        }

                        break;
                    }
                    default:
                    {
                        var first = FirstOperandWordIndex(opcode);
                        for (var word = first; word < wordCount; word++)
                        {
                            if (IsLiteralOperand(opcode, functions, offset, word))
                            {
                                continue;
                            }

                            var id = functions[offset + word];
                            if (id < (uint)isLocal.Length && isLocal[id])
                            {
                                ineligible[id] = true;
                            }
                        }

                        break;
                    }
                }

                offset += wordCount;
            }
        }

        var byFunction = new Dictionary<uint, List<uint>>();
        foreach (var variable in _localVariables)
        {
            if (ineligible[variable])
            {
                continue;
            }

            var owner = ownerFunc[variable];
            if (owner == 0)
            {
                continue;
            }

            if (!byFunction.TryGetValue(owner, out var list))
            {
                list = [];
                byFunction.Add(owner, list);
            }

            list.Add(variable);
        }

        return byFunction;
    }

    private void PromoteFunction(
        List<uint> functions,
        int funcStart,
        int funcEnd,
        List<uint> eligibleVars,
        List<uint> output,
        HashSet<uint> removedIds,
        HashSet<uint> removedVariables)
    {
        // Emit OpFunction and any OpFunctionParameter instructions up to the first OpLabel.
        var cursor = funcStart;
        while (cursor < funcEnd)
        {
            var wordCount = (int)(functions[cursor] >> 16);
            var opcode = (SpirvOp)(functions[cursor] & 0xFFFF);
            if (opcode is SpirvOp.Label or SpirvOp.FunctionEnd)
            {
                break;
            }

            for (var word = 0; word < wordCount; word++)
            {
                output.Add(functions[cursor + word]);
            }

            cursor += wordCount;
        }

        // Split function body into basic blocks.
        var blocks = new List<BasicBlock>();
        var labelToBlock = new Dictionary<uint, int>();
        BasicBlock? current = null;
        while (cursor < funcEnd)
        {
            var wordCount = (int)(functions[cursor] >> 16);
            var opcode = (SpirvOp)(functions[cursor] & 0xFFFF);
            if (opcode == SpirvOp.FunctionEnd)
            {
                break;
            }

            if (opcode == SpirvOp.Label)
            {
                var labelId = functions[cursor + 1];
                current = new BasicBlock(labelId);
                labelToBlock[labelId] = blocks.Count;
                blocks.Add(current);
            }
            else if (current is not null)
            {
                current.InstOffsets.Add(cursor);
            }

            cursor += wordCount;
        }

        if (blocks.Count == 0)
        {
            Emit(output, SpirvOp.FunctionEnd);
            return;
        }

        // Build CFG edges and record structured merge/continue targets.
        foreach (var block in blocks)
        {
            foreach (var instOffset in block.InstOffsets)
            {
                var wordCount = (int)(functions[instOffset] >> 16);
                var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                switch (opcode)
                {
                    case SpirvOp.SelectionMerge:
                        block.MergeTarget = functions[instOffset + 1];
                        break;
                    case SpirvOp.LoopMerge:
                        block.MergeTarget = functions[instOffset + 1];
                        block.ContinueTarget = functions[instOffset + 2];
                        break;
                    case SpirvOp.Branch:
                        AddUniqueSuccessor(block.Succs, labelToBlock, functions[instOffset + 1]);
                        break;
                    case SpirvOp.BranchConditional:
                        AddUniqueSuccessor(block.Succs, labelToBlock, functions[instOffset + 2]);
                        AddUniqueSuccessor(block.Succs, labelToBlock, functions[instOffset + 3]);
                        break;
                    case SpirvOp.Switch:
                    {
                        AddUniqueSuccessor(block.Succs, labelToBlock, functions[instOffset + 2]);
                        var literalWords = SwitchLiteralWords(functions[instOffset + 1]);
                        var step = literalWords + 1;
                        for (var word = 3 + literalWords; word < wordCount; word += step)
                        {
                            AddUniqueSuccessor(block.Succs, labelToBlock, functions[instOffset + word]);
                        }

                        break;
                    }
                }
            }
        }

        // Compute RPO over CFG-reachable blocks from entry block 0 (iterative DFS).
        var blockCount = blocks.Count;
        var visited = new bool[blockCount];
        var postOrder = new List<int>(blockCount);
        var rpoStack = new Stack<(int Block, int NextSucc)>();
        visited[0] = true;
        rpoStack.Push((0, 0));
        while (rpoStack.Count > 0)
        {
            var (blockIdx, nextSucc) = rpoStack.Pop();
            var succs = blocks[blockIdx].Succs;
            if (nextSucc < succs.Count)
            {
                rpoStack.Push((blockIdx, nextSucc + 1));
                var succ = succs[nextSucc];
                if (!visited[succ])
                {
                    visited[succ] = true;
                    rpoStack.Push((succ, 0));
                }
            }
            else
            {
                postOrder.Add(blockIdx);
            }
        }

        postOrder.Reverse();
        var rpo = postOrder.ToArray();
        var rpoIndex = new int[blockCount];
        Array.Fill(rpoIndex, -1);
        for (var i = 0; i < rpo.Length; i++)
        {
            rpoIndex[rpo[i]] = i;
        }

        // Preserve any unreachable block referenced by a kept block's merge/continue target or branch
        // so structured control-flow targets remain valid SPIR-V.
        var kept = new bool[blockCount];
        var keepQueue = new Queue<int>();
        foreach (var b in rpo)
        {
            kept[b] = true;
            keepQueue.Enqueue(b);
        }

        while (keepQueue.Count > 0)
        {
            var b = keepQueue.Dequeue();
            var block = blocks[b];
            if (block.MergeTarget != 0 && labelToBlock.TryGetValue(block.MergeTarget, out var mergeIdx) && !kept[mergeIdx])
            {
                kept[mergeIdx] = true;
                keepQueue.Enqueue(mergeIdx);
            }

            if (block.ContinueTarget != 0 && labelToBlock.TryGetValue(block.ContinueTarget, out var contIdx) && !kept[contIdx])
            {
                kept[contIdx] = true;
                keepQueue.Enqueue(contIdx);
            }

            foreach (var succ in block.Succs)
            {
                if (!kept[succ])
                {
                    kept[succ] = true;
                    keepQueue.Enqueue(succ);
                }
            }
        }

        var anyBlockDropped = false;
        for (var b = 0; b < blockCount; b++)
        {
            if (kept[b])
            {
                continue;
            }

            anyBlockDropped = true;
            removedIds.Add(blocks[b].Label);
            foreach (var instOffset in blocks[b].InstOffsets)
            {
                var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                if (FirstOperandWordIndex(opcode) == 3)
                {
                    removedIds.Add(functions[instOffset + 2]);
                }
            }
        }

        // Reachable predecessor lists for dominator and dominance-frontier computation.
        var reachablePreds = new List<int>[blockCount];
        for (var b = 0; b < blockCount; b++)
        {
            reachablePreds[b] = [];
        }

        foreach (var b in rpo)
        {
            foreach (var s in blocks[b].Succs)
            {
                if (rpoIndex[s] >= 0)
                {
                    reachablePreds[s].Add(b);
                }
            }
        }

        // Cooper-Harvey-Kennedy iterative immediate dominators.
        var idom = new int[blockCount];
        Array.Fill(idom, -1);
        idom[0] = 0;

        int Intersect(int b1, int b2)
        {
            while (b1 != b2)
            {
                while (rpoIndex[b1] > rpoIndex[b2])
                {
                    b1 = idom[b1];
                }

                while (rpoIndex[b2] > rpoIndex[b1])
                {
                    b2 = idom[b2];
                }
            }

            return b1;
        }

        var domChanged = true;
        while (domChanged)
        {
            domChanged = false;
            for (var i = 1; i < rpo.Length; i++)
            {
                var b = rpo[i];
                var newIdom = -1;
                foreach (var p in reachablePreds[b])
                {
                    if (idom[p] == -1)
                    {
                        continue;
                    }

                    newIdom = newIdom == -1 ? p : Intersect(p, newIdom);
                }

                if (newIdom != -1 && idom[b] != newIdom)
                {
                    idom[b] = newIdom;
                    domChanged = true;
                }
            }
        }

        // Dominator-tree children and Dominance Frontiers.
        var domChildren = new List<int>[blockCount];
        var df = new HashSet<int>[blockCount];
        for (var b = 0; b < blockCount; b++)
        {
            domChildren[b] = [];
            df[b] = [];
        }

        for (var i = 1; i < rpo.Length; i++)
        {
            var b = rpo[i];
            if (idom[b] >= 0)
            {
                domChildren[idom[b]].Add(b);
            }
        }

        foreach (var b in rpo)
        {
            if (b == 0 || reachablePreds[b].Count < 2)
            {
                continue;
            }

            foreach (var p in reachablePreds[b])
            {
                var runner = p;
                while (runner != idom[b] && runner != -1)
                {
                    df[runner].Add(b);
                    runner = idom[runner];
                }
            }
        }

        // Backward bitset liveness for eligible variables.
        var varCount = eligibleVars.Count;
        var varIndex = new Dictionary<uint, int>(varCount);
        var varElementTypes = new uint[varCount];
        var defBlocks = new List<int>[varCount];
        for (var vi = 0; vi < varCount; vi++)
        {
            var v = eligibleVars[vi];
            varIndex[v] = vi;
            varElementTypes[vi] = _variableTypes[v];
            defBlocks[vi] = [];
        }

        var bitWords = (varCount + 63) >> 6;
        var useBits = new ulong[blockCount * bitWords];
        var defBits = new ulong[blockCount * bitWords];
        var liveIn = new ulong[blockCount * bitWords];
        var liveOut = new ulong[blockCount * bitWords];

        foreach (var b in rpo)
        {
            var bBase = b * bitWords;
            foreach (var instOffset in blocks[b].InstOffsets)
            {
                var wordCount = (int)(functions[instOffset] >> 16);
                var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                if (opcode == SpirvOp.Load && wordCount == 4 &&
                    varIndex.TryGetValue(functions[instOffset + 3], out var loadVi))
                {
                    var w = loadVi >> 6;
                    var mask = 1UL << (loadVi & 63);
                    if ((defBits[bBase + w] & mask) == 0)
                    {
                        useBits[bBase + w] |= mask;
                    }
                }
                else if (opcode == SpirvOp.Store && wordCount == 3 &&
                    varIndex.TryGetValue(functions[instOffset + 1], out var storeVi))
                {
                    var w = storeVi >> 6;
                    var mask = 1UL << (storeVi & 63);
                    if ((defBits[bBase + w] & mask) == 0)
                    {
                        defBits[bBase + w] |= mask;
                        defBlocks[storeVi].Add(b);
                    }
                }
            }
        }

        var liveChanged = true;
        while (liveChanged)
        {
            liveChanged = false;
            for (var i = rpo.Length - 1; i >= 0; i--)
            {
                var b = rpo[i];
                var bBase = b * bitWords;
                var succs = blocks[b].Succs;
                for (var w = 0; w < bitWords; w++)
                {
                    ulong outVal = 0;
                    foreach (var s in succs)
                    {
                        if (rpoIndex[s] >= 0)
                        {
                            outVal |= liveIn[(s * bitWords) + w];
                        }
                    }

                    liveOut[bBase + w] = outVal;
                    var inVal = useBits[bBase + w] | (outVal & ~defBits[bBase + w]);
                    if (inVal != liveIn[bBase + w])
                    {
                        liveIn[bBase + w] = inVal;
                        liveChanged = true;
                    }
                }
            }
        }

        // Place pruned OpPhi nodes at IDF(defs(v)) intersected with liveIn[b].
        var blockPhis = new List<PlacedPhi>[blockCount];
        for (var b = 0; b < blockCount; b++)
        {
            blockPhis[b] = [];
        }

        var hasPhiEpoch = new int[blockCount];
        var workEpoch = new int[blockCount];
        var idfWorklist = new Queue<int>();
        for (var vi = 0; vi < varCount; vi++)
        {
            var epoch = vi + 1;
            foreach (var b in defBlocks[vi])
            {
                workEpoch[b] = epoch;
                idfWorklist.Enqueue(b);
            }

            var w = vi >> 6;
            var mask = 1UL << (vi & 63);
            var elemType = varElementTypes[vi];
            while (idfWorklist.Count > 0)
            {
                var x = idfWorklist.Dequeue();
                foreach (var y in df[x])
                {
                    if (hasPhiEpoch[y] == epoch || (liveIn[(y * bitWords) + w] & mask) == 0)
                    {
                        continue;
                    }

                    hasPhiEpoch[y] = epoch;
                    var phiId = AllocateId();
                    NoteResultType(phiId, elemType);
                    blockPhis[y].Add(new PlacedPhi(vi, elemType, phiId));
                    if (workEpoch[y] != epoch)
                    {
                        workEpoch[y] = epoch;
                        idfWorklist.Enqueue(y);
                    }
                }
            }
        }

        // Dominator-tree preorder DFS renaming (iterative).
        var stacks = new Stack<uint>[varCount];
        for (var vi = 0; vi < varCount; vi++)
        {
            stacks[vi] = new Stack<uint>();
        }

        var initialValues = new uint[varCount];
        var replacements = new Dictionary<uint, uint>();
        var droppedInsts = new HashSet<int>();
        var pushTrail = new List<int>();

        uint GetInitialValue(int vi)
        {
            if (initialValues[vi] == 0)
            {
                initialValues[vi] = _variableInitializers.TryGetValue(eligibleVars[vi], out var init) && init != 0
                    ? init
                    : ConstantNull(varElementTypes[vi]);
            }

            return initialValues[vi];
        }

        uint PeekValue(int vi) => stacks[vi].Count > 0 ? stacks[vi].Peek() : GetInitialValue(vi);

        uint Resolve(uint id)
        {
            if (replacements.Count == 0 || !replacements.TryGetValue(id, out var next))
            {
                return id;
            }

            var root = next;
            while (replacements.TryGetValue(root, out var further) && further != root)
            {
                root = further;
            }

            var cur = id;
            while (cur != root && replacements.TryGetValue(cur, out var nxt))
            {
                replacements[cur] = root;
                cur = nxt;
            }

            return root;
        }

        var renameStack = new Stack<(int Block, int SavedTrail, bool IsExit)>();
        renameStack.Push((0, 0, false));
        while (renameStack.Count > 0)
        {
            var (b, savedTrail, isExit) = renameStack.Pop();
            if (isExit)
            {
                while (pushTrail.Count > savedTrail)
                {
                    var vi = pushTrail[^1];
                    pushTrail.RemoveAt(pushTrail.Count - 1);
                    stacks[vi].Pop();
                }

                continue;
            }

            var trailStart = pushTrail.Count;
            renameStack.Push((b, trailStart, true));

            foreach (var phi in blockPhis[b])
            {
                stacks[phi.VarIndex].Push(phi.Id);
                pushTrail.Add(phi.VarIndex);
            }

            foreach (var instOffset in blocks[b].InstOffsets)
            {
                var wordCount = (int)(functions[instOffset] >> 16);
                var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                if (opcode == SpirvOp.Variable && varIndex.ContainsKey(functions[instOffset + 2]))
                {
                    droppedInsts.Add(instOffset);
                }
                else if (opcode == SpirvOp.Load && wordCount == 4 &&
                    varIndex.TryGetValue(functions[instOffset + 3], out var loadVi))
                {
                    var resultId = functions[instOffset + 2];
                    replacements[resultId] = PeekValue(loadVi);
                    droppedInsts.Add(instOffset);
                    removedIds.Add(resultId);
                }
                else if (opcode == SpirvOp.Store && wordCount == 3 &&
                    varIndex.TryGetValue(functions[instOffset + 1], out var storeVi))
                {
                    var valueId = Resolve(functions[instOffset + 2]);
                    stacks[storeVi].Push(valueId);
                    pushTrail.Add(storeVi);
                    droppedInsts.Add(instOffset);
                }
            }

            var blockLabel = blocks[b].Label;
            foreach (var succ in blocks[b].Succs)
            {
                if (!kept[succ])
                {
                    continue;
                }

                foreach (var phi in blockPhis[succ])
                {
                    phi.Incoming.Add((PeekValue(phi.VarIndex), blockLabel));
                }
            }

            var children = domChildren[b];
            for (var i = children.Count - 1; i >= 0; i--)
            {
                renameStack.Push((children[i], 0, false));
            }
        }

        // Handle unreachable-but-kept structured merge/continue blocks so they have no leftover uses
        // of promoted variables and supply a dominating constant to any successor OpPhi.
        for (var b = 0; b < blockCount; b++)
        {
            if (!kept[b] || rpoIndex[b] >= 0)
            {
                continue;
            }

            foreach (var instOffset in blocks[b].InstOffsets)
            {
                var wordCount = (int)(functions[instOffset] >> 16);
                var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                if (opcode == SpirvOp.Variable && varIndex.ContainsKey(functions[instOffset + 2]))
                {
                    droppedInsts.Add(instOffset);
                }
                else if (opcode == SpirvOp.Load && wordCount == 4 &&
                    varIndex.TryGetValue(functions[instOffset + 3], out var loadVi))
                {
                    var resultId = functions[instOffset + 2];
                    replacements[resultId] = GetInitialValue(loadVi);
                    droppedInsts.Add(instOffset);
                    removedIds.Add(resultId);
                }
                else if (opcode == SpirvOp.Store && wordCount == 3 &&
                    varIndex.ContainsKey(functions[instOffset + 1]))
                {
                    droppedInsts.Add(instOffset);
                }
            }

            var blockLabel = blocks[b].Label;
            foreach (var succ in blocks[b].Succs)
            {
                if (!kept[succ])
                {
                    continue;
                }

                foreach (var phi in blockPhis[succ])
                {
                    phi.Incoming.Add((GetInitialValue(phi.VarIndex), blockLabel));
                }
            }
        }

        // If any unreachable block was dropped and an existing OpPhi loses all incoming edges, replace it.
        if (anyBlockDropped)
        {
            for (var b = 0; b < blockCount; b++)
            {
                if (!kept[b])
                {
                    continue;
                }

                foreach (var instOffset in blocks[b].InstOffsets)
                {
                    var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                    if (opcode != SpirvOp.Phi)
                    {
                        break;
                    }

                    var wordCount = (int)(functions[instOffset] >> 16);
                    var hasKeptIncoming = false;
                    for (var word = 4; word < wordCount; word += 2)
                    {
                        if (labelToBlock.TryGetValue(functions[instOffset + word], out var predIdx) && kept[predIdx])
                        {
                            hasKeptIncoming = true;
                            break;
                        }
                    }

                    if (!hasKeptIncoming)
                    {
                        var phiId = functions[instOffset + 2];
                        replacements[phiId] = ConstantNull(functions[instOffset + 1]);
                        droppedInsts.Add(instOffset);
                        removedIds.Add(phiId);
                    }
                }
            }
        }

        // Mark promoted variables removed from module globals, entry points, debug and annotations.
        foreach (var variable in eligibleVars)
        {
            removedVariables.Add(variable);
            removedIds.Add(variable);
        }

        // Iteratively simplify trivial placed OpPhis.
        var phiChanged = true;
        while (phiChanged)
        {
            phiChanged = false;
            for (var b = 0; b < blockCount; b++)
            {
                if (!kept[b])
                {
                    continue;
                }

                foreach (var phi in blockPhis[b])
                {
                    if (phi.Removed)
                    {
                        continue;
                    }

                    uint unique = 0;
                    var trivial = true;
                    foreach (var (rawValue, _) in phi.Incoming)
                    {
                        var val = Resolve(rawValue);
                        if (val == phi.Id)
                        {
                            continue;
                        }

                        if (unique == 0)
                        {
                            unique = val;
                        }
                        else if (unique != val)
                        {
                            trivial = false;
                            break;
                        }
                    }

                    if (trivial)
                    {
                        replacements[phi.Id] = unique != 0 ? unique : ConstantNull(phi.Type);
                        phi.Removed = true;
                        removedIds.Add(phi.Id);
                        phiChanged = true;
                    }
                }
            }
        }

        // Emit surviving blocks in original order, placing Function OpVariables before other entry
        // instructions and placed OpPhis immediately after each non-entry OpLabel.
        var filteredPhiWords = new List<uint>();
        for (var b = 0; b < blockCount; b++)
        {
            if (!kept[b])
            {
                continue;
            }

            var block = blocks[b];
            Emit(output, SpirvOp.Label, block.Label);

            // Emit surviving OpVariables first (entry block only).
            if (b == 0)
            {
                foreach (var instOffset in block.InstOffsets)
                {
                    if (droppedInsts.Contains(instOffset))
                    {
                        continue;
                    }

                    var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                    if (opcode != SpirvOp.Variable)
                    {
                        continue;
                    }

                    var wordCount = (int)(functions[instOffset] >> 16);
                    for (var word = 0; word < wordCount; word++)
                    {
                        output.Add(functions[instOffset + word]);
                    }
                }
            }

            // Emit placed OpPhis immediately after OpLabel.
            foreach (var phi in blockPhis[b])
            {
                if (phi.Removed)
                {
                    continue;
                }

                var phiWordCount = 3 + (phi.Incoming.Count * 2);
                output.Add(((uint)phiWordCount << 16) | (ushort)SpirvOp.Phi);
                output.Add(phi.Type);
                output.Add(phi.Id);
                foreach (var (rawValue, predLabel) in phi.Incoming)
                {
                    output.Add(Resolve(rawValue));
                    output.Add(predLabel);
                }
            }

            // Emit remaining surviving instructions with rewritten SSA operands.
            foreach (var instOffset in block.InstOffsets)
            {
                if (droppedInsts.Contains(instOffset))
                {
                    continue;
                }

                var wordCount = (int)(functions[instOffset] >> 16);
                var opcode = (SpirvOp)(functions[instOffset] & 0xFFFF);
                if (b == 0 && opcode == SpirvOp.Variable)
                {
                    continue;
                }

                if (opcode == SpirvOp.Phi)
                {
                    filteredPhiWords.Clear();
                    filteredPhiWords.Add(functions[instOffset + 1]);
                    filteredPhiWords.Add(functions[instOffset + 2]);
                    for (var word = 3; word < wordCount; word += 2)
                    {
                        var predLabel = functions[instOffset + word + 1];
                        if (anyBlockDropped && (!labelToBlock.TryGetValue(predLabel, out var predIdx) || !kept[predIdx]))
                        {
                            continue;
                        }

                        filteredPhiWords.Add(Resolve(functions[instOffset + word]));
                        filteredPhiWords.Add(predLabel);
                    }

                    output.Add(((uint)(filteredPhiWords.Count + 1) << 16) | (ushort)SpirvOp.Phi);
                    output.AddRange(filteredPhiWords);
                    continue;
                }

                var firstOperand = FirstOperandWordIndex(opcode);
                var headerWords = Math.Min(firstOperand, wordCount);
                for (var word = 0; word < headerWords; word++)
                {
                    output.Add(functions[instOffset + word]);
                }

                for (var word = firstOperand; word < wordCount; word++)
                {
                    var raw = functions[instOffset + word];
                    output.Add(IsLiteralOperand(opcode, functions, instOffset, word) ? raw : Resolve(raw));
                }
            }
        }

        Emit(output, SpirvOp.FunctionEnd);
    }

    private List<uint> EliminateDeadCodeAndUnreadVariables(
        List<uint> functions,
        HashSet<uint> removedIds,
        HashSet<uint> removedVariables)
    {
        functions = MarkAndSweepDce(functions, removedIds);
        if (KeepsUnreadVariables || _localVariables.Count == 0)
        {
            return functions;
        }

        while (true)
        {
            var unread = FindUnreadVariables(functions, removedVariables, out var unreadPointers);
            if (unread.Count == 0)
            {
                break;
            }

            removedVariables.UnionWith(unread);
            removedIds.UnionWith(unread);
            functions = StripUnreadVariableInstructions(functions, unreadPointers);
            functions = MarkAndSweepDce(functions, removedIds);
        }

        return functions;
    }

    private List<uint> MarkAndSweepDce(List<uint> functions, HashSet<uint> removedIds)
    {
        if (functions.Count == 0)
        {
            return functions;
        }

        var starts = new List<int>();
        for (var offset = 0; offset < functions.Count; offset += (int)(functions[offset] >> 16))
        {
            starts.Add(offset);
        }

        var defInst = new int[_nextId];
        var live = new bool[starts.Count];
        var worklist = new Stack<int>();
        for (var index = 0; index < starts.Count; index++)
        {
            var offset = starts[index];
            var wordCount = (int)(functions[offset] >> 16);
            if (IsRemovableDefinition(functions, offset, wordCount))
            {
                var resultId = functions[offset + 2];
                if (resultId < (uint)defInst.Length)
                {
                    defInst[resultId] = index + 1;
                }
            }
            else
            {
                live[index] = true;
                worklist.Push(index);
            }
        }

        while (worklist.Count > 0)
        {
            var index = worklist.Pop();
            var offset = starts[index];
            var wordCount = (int)(functions[offset] >> 16);
            var opcode = (SpirvOp)(functions[offset] & 0xFFFF);
            var first = FirstOperandWordIndex(opcode);
            for (var word = first; word < wordCount; word++)
            {
                if (IsLiteralOperand(opcode, functions, offset, word))
                {
                    continue;
                }

                var id = functions[offset + word];
                if (id < (uint)defInst.Length)
                {
                    var def = defInst[id];
                    if (def != 0 && !live[def - 1])
                    {
                        live[def - 1] = true;
                        worklist.Push(def - 1);
                    }
                }
            }
        }

        var anyDead = false;
        for (var index = 0; index < live.Length; index++)
        {
            if (!live[index])
            {
                anyDead = true;
                break;
            }
        }

        if (!anyDead)
        {
            return functions;
        }

        var compacted = new List<uint>(functions.Count);
        for (var index = 0; index < starts.Count; index++)
        {
            var offset = starts[index];
            var wordCount = (int)(functions[offset] >> 16);
            if (!live[index])
            {
                removedIds.Add(functions[offset + 2]);
                continue;
            }

            for (var word = 0; word < wordCount; word++)
            {
                compacted.Add(functions[offset + word]);
            }
        }

        return compacted;
    }

    private HashSet<uint> FindUnreadVariables(
        List<uint> words,
        HashSet<uint> alreadyRemoved,
        out HashSet<uint> unreadPointers)
    {
        var local = new bool[_nextId];
        foreach (var variable in _localVariables)
        {
            if (!alreadyRemoved.Contains(variable))
            {
                local[variable] = true;
            }
        }

        var chainBases = new uint[_nextId];
        var chains = new List<uint>();
        var read = new bool[_nextId];
        for (var offset = 0; offset < words.Count; offset += (int)(words[offset] >> 16))
        {
            var wordCount = (int)(words[offset] >> 16);
            var opcode = (SpirvOp)(words[offset] & 0xFFFF);
            var first = FirstOperandWordIndex(opcode);
            switch (opcode)
            {
                case SpirvOp.Variable:
                    continue;
                case SpirvOp.Store when wordCount == 3:
                    first = 2;
                    break;
                case SpirvOp.AccessChain when wordCount >= 4 && words[offset + 3] < (uint)local.Length:
                {
                    var baseId = words[offset + 3];
                    var root = local[baseId] ? baseId : chainBases[baseId];
                    if (root != 0)
                    {
                        var chainId = words[offset + 2];
                        if (chainId < (uint)chainBases.Length)
                        {
                            chains.Add(chainId);
                            chainBases[chainId] = root;
                        }

                        first = 4;
                    }

                    break;
                }
            }

            for (var word = first; word < wordCount; word++)
            {
                if (IsLiteralOperand(opcode, words, offset, word))
                {
                    continue;
                }

                var id = words[offset + word];
                if (id >= (uint)local.Length)
                {
                    continue;
                }

                if (local[id])
                {
                    read[id] = true;
                }
                else if (chainBases[id] != 0)
                {
                    read[chainBases[id]] = true;
                }
            }
        }

        var unread = new HashSet<uint>();
        foreach (var variable in _localVariables)
        {
            if (!alreadyRemoved.Contains(variable) && !read[variable])
            {
                unread.Add(variable);
            }
        }

        unreadPointers = new HashSet<uint>(unread);
        foreach (var chain in chains)
        {
            if (unread.Contains(chainBases[chain]))
            {
                unreadPointers.Add(chain);
            }
        }

        return unread;
    }

    private static List<uint> StripUnreadVariableInstructions(List<uint> functions, HashSet<uint> unreadPointers)
    {
        var filtered = new List<uint>(functions.Count);
        for (var offset = 0; offset < functions.Count;)
        {
            var wordCount = (int)(functions[offset] >> 16);
            var opcode = (SpirvOp)(functions[offset] & 0xFFFF);
            var dropped = (opcode == SpirvOp.Store && wordCount == 3 && unreadPointers.Contains(functions[offset + 1])) ||
                (opcode == SpirvOp.Variable && unreadPointers.Contains(functions[offset + 2]));
            if (!dropped)
            {
                for (var word = 0; word < wordCount; word++)
                {
                    filtered.Add(functions[offset + word]);
                }
            }

            offset += wordCount;
        }

        return filtered;
    }

    private static void AddUniqueSuccessor(List<int> succs, Dictionary<uint, int> labelToBlock, uint targetLabel)
    {
        if (labelToBlock.TryGetValue(targetLabel, out var targetBlock) && !succs.Contains(targetBlock))
        {
            succs.Add(targetBlock);
        }
    }

    private static int FirstOperandWordIndex(SpirvOp opcode) => opcode switch
    {
        SpirvOp.Nop or SpirvOp.FunctionEnd or SpirvOp.Kill or SpirvOp.Return or SpirvOp.Unreachable => int.MaxValue,
        SpirvOp.Label => 2,
        SpirvOp.Store or SpirvOp.ImageWrite or SpirvOp.ControlBarrier or SpirvOp.MemoryBarrier or
            SpirvOp.LoopMerge or SpirvOp.SelectionMerge or SpirvOp.Branch or SpirvOp.BranchConditional or
            SpirvOp.Switch or SpirvOp.ReturnValue or SpirvOp.EmitMeshTasksExt or SpirvOp.SetMeshOutputsExt => 1,
        _ => 3,
    };

    private bool IsLiteralOperand(SpirvOp opcode, List<uint> words, int offset, int wordIndex) => opcode switch
    {
        SpirvOp.ExtInst => wordIndex == 4,
        SpirvOp.Function => wordIndex == 3,
        SpirvOp.Variable => wordIndex == 3,
        SpirvOp.Load => wordIndex == 4 || (wordIndex == 5 && (words[offset + 4] & 0x2u) != 0),
        SpirvOp.Store => wordIndex == 3 || (wordIndex == 4 && (words[offset + 3] & 0x2u) != 0),
        SpirvOp.ArrayLength => wordIndex == 4,
        SpirvOp.VectorShuffle => wordIndex >= 5,
        SpirvOp.CompositeExtract => wordIndex >= 4,
        SpirvOp.CompositeInsert => wordIndex >= 5,
        SpirvOp.ImageSampleImplicitLod or SpirvOp.ImageSampleExplicitLod or
            SpirvOp.ImageFetch or SpirvOp.ImageRead => wordIndex == 5,
        SpirvOp.ImageSampleDrefImplicitLod or SpirvOp.ImageSampleDrefExplicitLod or
            SpirvOp.ImageGather or SpirvOp.ImageDrefGather => wordIndex == 6,
        SpirvOp.ImageWrite => wordIndex == 4,
        SpirvOp.LoopMerge => wordIndex >= 3,
        SpirvOp.SelectionMerge => wordIndex == 2,
        SpirvOp.BranchConditional => wordIndex >= 4,
        SpirvOp.Switch => IsSwitchLiteralWord(words[offset + 1], wordIndex),
        SpirvOp.GroupNonUniformBallotBitCount or SpirvOp.GroupNonUniformIAdd or
            SpirvOp.GroupNonUniformSMin or SpirvOp.GroupNonUniformUMin or
            SpirvOp.GroupNonUniformSMax or SpirvOp.GroupNonUniformUMax or
            SpirvOp.GroupNonUniformBitwiseAnd or SpirvOp.GroupNonUniformBitwiseOr or
            SpirvOp.GroupNonUniformBitwiseXor => wordIndex == 4,
        _ => false,
    };

    private bool IsSwitchLiteralWord(uint selectorId, int wordIndex)
    {
        if (wordIndex < 3)
        {
            return false;
        }

        var literalWords = SwitchLiteralWords(selectorId);
        return ((wordIndex - 3) % (literalWords + 1)) < literalWords;
    }

    private int SwitchLiteralWords(uint selectorId) => IntegerWidth(TypeOf(selectorId)) == 64 ? 2 : 1;
}
