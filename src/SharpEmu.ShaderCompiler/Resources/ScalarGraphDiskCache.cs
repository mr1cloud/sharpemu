// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers;
using System.Buffers.Binary;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace SharpEmu.ShaderCompiler.Resources;

// The running game's folder for the scalar graph disk cache: the presenter names it beside the game's pipeline cache.
public static class ShaderAnalysisCache
{
    public static void UseGameFolder(string folder) => ScalarGraphDiskCache.UseGameFolder(folder);
}

internal static class ScalarGraphDiskCache
{
    // 6: direct binary control encoding in cache keys.
    private const int Version = 6;
    private static readonly byte[] ModuleVersionIdBytes = typeof(ScalarValueGraph).Module.ModuleVersionId.ToByteArray();
    private const int MaxEntryBytes = 32 * 1024 * 1024;
    // Each game keeps its own cache, up to 8 GiB of compressed entries, trimmed by last use. One small folder shared by
    // every game and trimmed by write time cannot hold one game's graphs: a game's opening and half an hour of play can
    // build hundreds of megabytes of them, and play keeps adding graphs. A warm graph takes about a third off planning.
    private const long MaxCacheBytes = 8L * 1024 * 1024 * 1024;
    // A trim that passes the cap stops at this size, so the writes after it do not each list and trim the folder again.
    private const long TrimTargetBytes = MaxCacheBytes / 10 * 9;
    // SHARPEMU_SHADER_ANALYSIS_CACHE_PATH names the folder ("0" turns the cache off). Otherwise it is the running game's
    // folder (UseGameFolder), kept like its pipeline binaries and prewarm list, so no game trims another's graphs; until a
    // game names its folder the cache is off.
    private static readonly string? CachePath =
        Environment.GetEnvironmentVariable("SHARPEMU_SHADER_ANALYSIS_CACHE_PATH") is { Length: > 0 } configured ? configured : null;
    private static volatile string? _gameFolder;

    internal static void UseGameFolder(string folder) => _gameFolder = folder;
    private static readonly bool Trace = Environment.GetEnvironmentVariable("SHARPEMU_PROFILE_SHADER_PLANNING") == "1";
    private static readonly object WriteLock = new();
    // Each folder's size as last listed plus what this process wrote since (under WriteLock).
    private static readonly Dictionary<string, long> CountedBytes = new(StringComparer.OrdinalIgnoreCase);

    internal static ScalarValueGraph Build(Gen5ShaderProgram program, uint userDataBase, uint userDataCount,
        IReadOnlySet<uint>? fixedFunctionVertexLoads, uint waveSize, string? cachePath = null, ulong excludedUserDataRegisters = 0, int dynamicUserData = -1)
    {
        var directory = cachePath ?? CachePath ?? _gameFolder ?? "0";
        if (directory == "0") return ScalarValueGraph.Build(program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
        var key = Key(program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
        var path = Path.Combine(directory, key + ".graph");
        try
        {
            using var file = File.OpenRead(path);
            if (file.Length is > 32 and <= MaxEntryBytes)
            {
                var digest = new byte[32];
                file.ReadExactly(digest);
                var bytes = new byte[checked((int)file.Length - 32)];
                file.ReadExactly(bytes);
                if (CryptographicOperations.FixedTimeEquals(digest, SHA256.HashData(bytes)))
                {
                    using var reader = new BinaryReader(Inflate(bytes));
                    if (reader.ReadString() != key) throw new InvalidDataException("Graph cache key mismatch.");
                    var cached = ScalarValueGraph.ReadSnapshot(reader, program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
                    MarkUsed(path);
                    if (Trace) Console.Error.WriteLine($"[SHADER_ANALYSIS_CACHE] hit key={key} bytes={bytes.Length}");
                    return cached;
                }
            }
        }
        catch (Exception error) when (IsCacheFailure(error)) { }

        var graph = ScalarValueGraph.Build(program, userDataBase, userDataCount, fixedFunctionVertexLoads, waveSize, excludedUserDataRegisters, dynamicUserData);
        try
        {
            using var stream = new MemoryStream();
            using (var writer = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                writer.Write(key);
                graph.WriteSnapshot(writer);
            }
            if (stream.Length <= MaxEntryBytes - 32)
            {
                using var compressed = new MemoryStream();
                using (var zlib = new ZLibStream(compressed, CompressionLevel.Fastest, leaveOpen: true))
                {
                    zlib.Write(stream.GetBuffer(), 0, (int)stream.Length);
                }

                var bytes = compressed.GetBuffer().AsSpan(0, (int)compressed.Length);
                lock (WriteLock)
                {
                    Directory.CreateDirectory(directory);
                    // Keep a bounded cache across compiler versions too. No live
                    // analysis objects or guest memory are retained by the cache.
                    var entryBytes = compressed.Length + 32;
                    if (!CountedBytes.TryGetValue(directory, out var counted) || counted + entryBytes > MaxCacheBytes)
                    {
                        counted = Trim(directory, entryBytes, MaxCacheBytes, TrimTargetBytes);
                    }

                    CountedBytes[directory] = counted + entryBytes;
                    var temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
                    try
                    {
                        using (var file = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                        {
                            file.Write(SHA256.HashData(bytes));
                            file.Write(bytes);
                        }
                        File.Move(temporary, path, true);
                    }
                    finally { File.Delete(temporary); }
                }
                if (Trace) Console.Error.WriteLine($"[SHADER_ANALYSIS_CACHE] miss key={key} bytes={bytes.Length}");
            }
        }
        catch (Exception error) when (IsCacheFailure(error)) { }
        return graph;
    }

    internal static string Key(Gen5ShaderProgram program, uint userDataBase, uint userDataCount,
        IReadOnlySet<uint>? fixedFunctionVertexLoads, uint waveSize, ulong excludedUserDataRegisters = 0, int dynamicUserData = -1)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> buffer = stackalloc byte[4096];
        var writer = new Sha256KeyWriter(hash, buffer);

        writer.WriteInt32(Version);
        writer.WriteBytes(ModuleVersionIdBytes);
        writer.WriteUInt32(userDataBase);
        writer.WriteUInt32(userDataCount);
        writer.WriteUInt32(waveSize);
        writer.WriteUInt64(excludedUserDataRegisters);
        writer.WriteInt32(dynamicUserData);

        if (fixedFunctionVertexLoads is null || fixedFunctionVertexLoads.Count == 0)
        {
            writer.WriteInt32(0);
        }
        else
        {
            var loads = new uint[fixedFunctionVertexLoads.Count];
            var loadIndex = 0;
            foreach (var load in fixedFunctionVertexLoads)
            {
                loads[loadIndex++] = load;
            }

            Array.Sort(loads);
            writer.WriteInt32(loads.Length);
            for (var index = 0; index < loads.Length; index++)
            {
                writer.WriteUInt32(loads[index]);
            }
        }

        var instructions = program.Instructions;
        var instructionCount = instructions.Count;
        writer.WriteInt32(instructionCount);
        for (var index = 0; index < instructionCount; index++)
        {
            var instruction = instructions[index];
            writer.WriteUInt32(instruction.Pc);
            writer.WriteUInt64(instruction.ProgramOffset);
            writer.WriteInt32((int)instruction.Encoding);
            writer.WriteString(instruction.Opcode);

            var words = instruction.Words;
            var wordCount = words.Count;
            writer.WriteInt32(wordCount);
            for (var wordIndex = 0; wordIndex < wordCount; wordIndex++)
            {
                writer.WriteUInt32(words[wordIndex]);
            }

            var sources = instruction.Sources;
            var sourceCount = sources.Count;
            writer.WriteInt32(sourceCount);
            for (var sourceIndex = 0; sourceIndex < sourceCount; sourceIndex++)
            {
                var operand = sources[sourceIndex];
                writer.WriteInt32((int)operand.Kind);
                writer.WriteUInt32(operand.Value);
            }

            var destinations = instruction.Destinations;
            var destinationCount = destinations.Count;
            writer.WriteInt32(destinationCount);
            for (var destinationIndex = 0; destinationIndex < destinationCount; destinationIndex++)
            {
                var operand = destinations[destinationIndex];
                writer.WriteInt32((int)operand.Kind);
                writer.WriteUInt32(operand.Value);
            }

            if (instruction.Control is { } control)
            {
                WriteControl(ref writer, control);
            }
            else
            {
                writer.WriteByte(0);
            }
        }

        return writer.FinishHex();
    }

    private static void WriteControl(ref Sha256KeyWriter writer, Gen5InstructionControl control)
    {
        // Direct binary encoding of every concrete Gen5InstructionControl subtype in this assembly,
        // keyed by a unique type tag, with a JSON fallback for external or future subtypes.
        switch (control)
        {
            case Gen5Vop3Control vop3:
                writer.WriteByte(1);
                writer.WriteUInt32(vop3.AbsoluteMask);
                writer.WriteUInt32(vop3.NegateMask);
                writer.WriteUInt32(vop3.OutputModifier);
                writer.WriteBool(vop3.Clamp);
                writer.WriteUInt32(vop3.OperandSelect);
                writer.WriteNullableUInt32(vop3.ScalarDestination);
                return;
            case Gen5ScalarMemoryControl scalarMemory:
                writer.WriteByte(2);
                writer.WriteUInt32(scalarMemory.DestinationCount);
                writer.WriteInt32(scalarMemory.ImmediateOffsetBytes);
                writer.WriteNullableUInt32(scalarMemory.DynamicOffsetRegister);
                return;
            case Gen5GlobalMemoryControl globalMemory:
                writer.WriteByte(3);
                writer.WriteUInt32(globalMemory.DwordCount);
                writer.WriteUInt32(globalMemory.VectorAddress);
                writer.WriteUInt32(globalMemory.SourceVectorRegister);
                writer.WriteUInt32(globalMemory.DestinationVectorRegister);
                writer.WriteUInt32(globalMemory.ScalarAddress);
                writer.WriteInt32(globalMemory.OffsetBytes);
                writer.WriteBool(globalMemory.Glc);
                writer.WriteBool(globalMemory.Slc);
                writer.WriteBool(globalMemory.UsesFlatAddress);
                writer.WriteNullableUInt32(globalMemory.DynamicOffsetRegister);
                writer.WriteBool(globalMemory.SourceIsScalar);
                return;
            case Gen5BufferMemoryControl bufferMemory:
                writer.WriteByte(4);
                writer.WriteUInt32(bufferMemory.DwordCount);
                writer.WriteUInt32(bufferMemory.VectorAddress);
                writer.WriteUInt32(bufferMemory.VectorData);
                writer.WriteUInt32(bufferMemory.ScalarResource);
                writer.WriteInt32(bufferMemory.OffsetBytes);
                writer.WriteBool(bufferMemory.IndexEnabled);
                writer.WriteBool(bufferMemory.OffsetEnabled);
                writer.WriteBool(bufferMemory.Glc);
                writer.WriteBool(bufferMemory.Slc);
                writer.WriteBool(bufferMemory.Typed);
                writer.WriteUInt32(bufferMemory.TypedFormat);
                writer.WriteBool(bufferMemory.SplitDescriptorFormat);
                return;
            case Gen5ImageControl image:
                writer.WriteByte(5);
                writer.WriteUInt32(image.Dmask);
                writer.WriteUInt32(image.VectorAddress);
                writer.WriteUInt32List(image.AddressRegisters);
                writer.WriteUInt32(image.VectorData);
                writer.WriteUInt32(image.ScalarResource);
                writer.WriteUInt32(image.ScalarSampler);
                writer.WriteUInt32(image.Dimension);
                writer.WriteBool(image.IsArray);
                writer.WriteBool(image.Glc);
                writer.WriteBool(image.Slc);
                writer.WriteBool(image.A16);
                writer.WriteBool(image.D16);
                return;
            case Gen5SdwaControl sdwa:
                writer.WriteByte(6);
                writer.WriteUInt32(sdwa.DestinationSelect);
                writer.WriteUInt32(sdwa.DestinationUnused);
                writer.WriteUInt32(sdwa.Source0Select);
                writer.WriteUInt32(sdwa.Source1Select);
                writer.WriteBool(sdwa.Source0SignExtend);
                writer.WriteBool(sdwa.Source1SignExtend);
                writer.WriteUInt32(sdwa.AbsoluteMask);
                writer.WriteUInt32(sdwa.NegateMask);
                writer.WriteUInt32(sdwa.OutputModifier);
                writer.WriteBool(sdwa.Clamp);
                writer.WriteNullableUInt32(sdwa.ScalarDestination);
                return;
            case Gen5DataShareControl dataShare:
                writer.WriteByte(7);
                writer.WriteUInt32(dataShare.Offset0);
                writer.WriteUInt32(dataShare.Offset1);
                writer.WriteBool(dataShare.Gds);
                return;
            case Gen5Vop3pControl vop3p:
                writer.WriteByte(8);
                writer.WriteUInt32(vop3p.OpSelMask);
                writer.WriteUInt32(vop3p.OpSelHiMask);
                writer.WriteUInt32(vop3p.NegLoMask);
                writer.WriteUInt32(vop3p.NegHiMask);
                writer.WriteBool(vop3p.Clamp);
                return;
            case Gen5DppControl dpp:
                writer.WriteByte(9);
                writer.WriteUInt32(dpp.Control);
                writer.WriteBool(dpp.FetchInactive);
                writer.WriteBool(dpp.BoundControl);
                writer.WriteUInt32(dpp.AbsoluteMask);
                writer.WriteUInt32(dpp.NegateMask);
                writer.WriteUInt32(dpp.BankMask);
                writer.WriteUInt32(dpp.RowMask);
                return;
            case Gen5Dpp8Control dpp8:
                writer.WriteByte(10);
                writer.WriteUInt32(dpp8.LaneSelectors);
                writer.WriteBool(dpp8.FetchInactive);
                return;
            case Gen5ExportControl export:
                writer.WriteByte(11);
                writer.WriteUInt32(export.Target);
                writer.WriteUInt32(export.EnableMask);
                writer.WriteBool(export.Compressed);
                writer.WriteBool(export.Done);
                writer.WriteBool(export.ValidMask);
                return;
            case Gen5InterpolationControl interpolation:
                writer.WriteByte(12);
                writer.WriteUInt32(interpolation.Attribute);
                writer.WriteUInt32(interpolation.Channel);
                writer.WriteBool(interpolation.HighAttribute);
                writer.WriteBool(interpolation.HighDestination);
                return;
            case Gen5RayIntersectControl rayIntersect:
                writer.WriteByte(13);
                writer.WriteUInt32(rayIntersect.VectorAddress);
                writer.WriteUInt32List(rayIntersect.AddressRegisters);
                writer.WriteUInt32(rayIntersect.VectorData);
                writer.WriteUInt32(rayIntersect.ScalarResource);
                writer.WriteBool(rayIntersect.A16);
                return;
            case ShaderCallMatchControl callMatch:
                writer.WriteByte(14);
                writer.WriteUInt32(callMatch.TargetPc);
                writer.WriteUInt32(callMatch.AddressRegister);
                writer.WriteUInt32(callMatch.ArgumentRegister);
                writer.WriteUInt64(callMatch.Address);
                writer.WriteUInt64(callMatch.Argument);
                writer.WriteBool(callMatch.RuntimeArgument);
                return;
            case ShaderCallEntryControl callEntry:
                writer.WriteByte(15);
                writer.WriteUInt32(callEntry.ReturnRegister);
                writer.WriteUInt64(callEntry.ReturnAddress);
                writer.WriteUInt32(callEntry.ArgumentRegister);
                writer.WriteUInt64(callEntry.Argument);
                writer.WriteBool(callEntry.RuntimeArgument);
                writer.WriteUInt64(callEntry.Callee);
                return;
            case ShaderCallFaultControl callFault:
                writer.WriteByte(16);
                writer.WriteUInt32(callFault.CallPc);
                writer.WriteUInt32(callFault.AddressRegister);
                writer.WriteNullableUInt32(callFault.ReturnPc);
                return;
            case ShaderFunctionPcControl functionPc:
                writer.WriteByte(17);
                writer.WriteUInt64(functionPc.FunctionAddress);
                writer.WriteUInt64(functionPc.FunctionOffset);
                return;
            case ShaderLinkedBranchControl linkedBranch:
                writer.WriteByte(18);
                writer.WriteUInt32(linkedBranch.TargetPc);
                return;
            case ShaderCallGroupControl callGroup:
                writer.WriteByte(19);
                writer.WriteUInt32(callGroup.TargetPc);
                writer.WriteUInt32(callGroup.Group);
                return;
            default:
                writer.WriteByte(byte.MaxValue);
                var type = control.GetType();
                writer.WriteString(type.FullName ?? "");
                writer.WriteString(JsonSerializer.Serialize(control, type));
                return;
        }
    }

    private ref struct Sha256KeyWriter(IncrementalHash hash, Span<byte> buffer)
    {
        private readonly IncrementalHash _hash = hash;
        private readonly Span<byte> _buffer = buffer;
        private int _pos;

        public void WriteByte(byte value)
        {
            if (_pos == _buffer.Length)
            {
                Flush();
            }

            _buffer[_pos++] = value;
        }

        public void WriteBool(bool value) => WriteByte(value ? (byte)1 : (byte)0);

        public void WriteInt32(int value)
        {
            if (_buffer.Length - _pos < sizeof(int))
            {
                Flush();
            }

            BinaryPrimitives.WriteInt32LittleEndian(_buffer[_pos..], value);
            _pos += sizeof(int);
        }

        public void WriteUInt32(uint value)
        {
            if (_buffer.Length - _pos < sizeof(uint))
            {
                Flush();
            }

            BinaryPrimitives.WriteUInt32LittleEndian(_buffer[_pos..], value);
            _pos += sizeof(uint);
        }

        public void WriteUInt64(ulong value)
        {
            if (_buffer.Length - _pos < sizeof(ulong))
            {
                Flush();
            }

            BinaryPrimitives.WriteUInt64LittleEndian(_buffer[_pos..], value);
            _pos += sizeof(ulong);
        }

        public void WriteNullableUInt32(uint? value)
        {
            if (value.HasValue)
            {
                WriteByte(1);
                WriteUInt32(value.GetValueOrDefault());
            }
            else
            {
                WriteByte(0);
            }
        }

        public void WriteUInt32List(IReadOnlyList<uint>? values)
        {
            if (values is null)
            {
                WriteInt32(-1);
                return;
            }

            var count = values.Count;
            WriteInt32(count);
            for (var index = 0; index < count; index++)
            {
                WriteUInt32(values[index]);
            }
        }

        public void WriteBytes(ReadOnlySpan<byte> data)
        {
            if (data.Length <= _buffer.Length - _pos)
            {
                data.CopyTo(_buffer[_pos..]);
                _pos += data.Length;
                return;
            }

            Flush();
            if (data.Length <= _buffer.Length)
            {
                data.CopyTo(_buffer);
                _pos = data.Length;
                return;
            }

            _hash.AppendData(data);
        }

        public void WriteString(string value)
        {
            if (value.Length == 0)
            {
                WriteInt32(0);
                return;
            }

            var maxBytes = Encoding.UTF8.GetMaxByteCount(value.Length);
            if (maxBytes <= _buffer.Length - sizeof(int))
            {
                if (_buffer.Length - _pos < sizeof(int) + maxBytes)
                {
                    Flush();
                }

                var written = Encoding.UTF8.GetBytes(value.AsSpan(), _buffer[(_pos + sizeof(int))..]);
                BinaryPrimitives.WriteInt32LittleEndian(_buffer[_pos..], written);
                _pos += sizeof(int) + written;
                return;
            }

            var byteCount = Encoding.UTF8.GetByteCount(value);
            WriteInt32(byteCount);
            Flush();
            var rented = ArrayPool<byte>.Shared.Rent(byteCount);
            try
            {
                var written = Encoding.UTF8.GetBytes(value.AsSpan(), rented.AsSpan());
                _hash.AppendData(rented.AsSpan(0, written));
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }

        public string FinishHex()
        {
            Flush();
            Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
            _hash.GetHashAndReset(digest);
            return Convert.ToHexString(digest);
        }

        private void Flush()
        {
            if (_pos != 0)
            {
                _hash.AppendData(_buffer[.._pos]);
                _pos = 0;
            }
        }
    }

    // An entry's snapshot, inflated (at most MaxEntryBytes; the reader needs a seekable stream).
    private static MemoryStream Inflate(byte[] compressed)
    {
        var inflated = new MemoryStream();
        using (var zlib = new ZLibStream(new MemoryStream(compressed, false), CompressionMode.Decompress))
        {
            var buffer = new byte[81920];
            int read;
            while ((read = zlib.Read(buffer, 0, buffer.Length)) > 0)
            {
                if (inflated.Length + read > MaxEntryBytes) throw new InvalidDataException("Graph cache entry too large.");
                inflated.Write(buffer, 0, read);
            }
        }

        inflated.Position = 0;
        return inflated;
    }

    // A hit marks its entry used: Trim drops the entries used longest ago, not the ones written longest ago.
    private static void MarkUsed(string path)
    {
        try
        {
            File.SetLastAccessTimeUtc(path, DateTime.UtcNow);
        }
        catch (Exception error) when (IsCacheFailure(error)) { }
    }

    // Lists the folder; when an entry of entryBytes would take it past capBytes, deletes the entries used longest ago until
    // it fits within targetBytes. Returns the bytes that remain.
    internal static long Trim(string directory, long entryBytes, long capBytes, long targetBytes)
    {
        var entries = new DirectoryInfo(directory).GetFiles("*.graph");
        var total = entries.Sum(static entry => entry.Length);
        if (total + entryBytes <= capBytes) return total;
        foreach (var entry in entries.OrderBy(LastUse))
        {
            if (total + entryBytes <= targetBytes) break;
            total -= entry.Length;
            entry.Delete();
        }

        return total;
    }

    private static DateTime LastUse(FileInfo entry) =>
        entry.LastAccessTimeUtc > entry.LastWriteTimeUtc ? entry.LastAccessTimeUtc : entry.LastWriteTimeUtc;

    private static bool IsCacheFailure(Exception error) => error is IOException or UnauthorizedAccessException or
        InvalidDataException or ArgumentException or OverflowException;
}
