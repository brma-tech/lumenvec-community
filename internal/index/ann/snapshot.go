package ann

import (
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
)

const annCompactSnapshotVersion = 1

const annSnapshotVersion = 1

var annBinaryMagicV2 = [8]byte{'L', 'V', 'A', 'N', 'N', '0', '0', '2'}
var annBinaryMagicV3 = [8]byte{'L', 'V', 'A', 'N', 'N', '0', '0', '3'}
var annBinaryMagic = [8]byte{'L', 'V', 'A', 'N', 'N', '0', '0', '4'}
var segmentedBinaryMagic = [8]byte{'L', 'V', 'S', 'E', 'G', '0', '0', '2'}

type indexSnapshot struct {
	Version        int            `json:"version"`
	Nodes          []nodeSnapshot `json:"nodes"`
	Deleted        []bool         `json:"deleted"`
	DeletedCount   int            `json:"deleted_count"`
	Entrypoint     int            `json:"entrypoint"`
	HasEntrypoint  bool           `json:"has_entrypoint"`
	Dimension      int            `json:"dimension"`
	M              int            `json:"m"`
	EfConstruction int            `json:"ef_construction"`
	EfSearch       int            `json:"ef_search"`
	Seed           int64          `json:"seed"`
}

type nodeSnapshot struct {
	Slot      int       `json:"slot"`
	ID        int       `json:"id"`
	Vector    []float32 `json:"vector"`
	Neighbors []int     `json:"neighbors"`
}

func (a *AnnIndex) MarshalBinary() ([]byte, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	count := a.nodeCountLocked()
	deleted := make([]bool, count)
	for i := range deleted {
		deleted[i] = a.isDeletedLocked(i)
	}
	snapshot := indexSnapshot{
		Version:        annSnapshotVersion,
		Deleted:        deleted,
		DeletedCount:   a.deletedCount,
		Entrypoint:     a.entrypoint,
		HasEntrypoint:  a.hasEntrypoint,
		Dimension:      a.dim,
		M:              a.m,
		EfConstruction: a.efConstruction,
		EfSearch:       a.efSearch,
		Seed:           a.seed,
		Nodes:          make([]nodeSnapshot, count),
	}
	for i := 0; i < count; i++ {
		rawNeighbors := a.neighborsForSlot(i)
		neighbors := make([]int, len(rawNeighbors))
		for j, slot := range rawNeighbors {
			neighbors[j] = int(slot)
		}
		snapshot.Nodes[i] = nodeSnapshot{
			Slot:      i,
			ID:        a.nodeIDLocked(i),
			Vector:    append([]float32(nil), a.nodeVectorLocked(i)...),
			Neighbors: neighbors,
		}
	}
	return json.Marshal(snapshot)
}

// MarshalCompactBinary encodes an ANN snapshot in a binary representation;
// MarshalBinary remains the JSON-compatible format used by existing callers.
func (a *AnnIndex) MarshalCompactBinary() ([]byte, error) {
	var out bytes.Buffer
	if _, err := a.WriteCompactBinary(&out); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// CompactBinarySize returns the exact number of bytes produced by
// WriteCompactBinary without allocating a serialized representation.
func (a *AnnIndex) CompactBinarySize() uint64 {
	a.mu.RLock()
	defer a.mu.RUnlock()
	size := uint64(len(annBinaryMagic) + 8*8 + 2)
	count := a.nodeCountLocked()
	vectorWidth := uint64(4)
	if a.quantized {
		vectorWidth = 1
		size += uint64(a.dim) * 8
	}
	size += uint64(count)*9 + uint64(count)*uint64(a.dim)*vectorWidth
	size += uint64(count+1) * 8
	for i := 0; i < count; i++ {
		size += uint64(a.neighborCountLocked(i)) * 4
	}
	return size
}

// WriteCompactBinary streams the compact v2 representation directly to w.
func (a *AnnIndex) WriteCompactBinary(w io.Writer) (uint64, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	count := a.nodeCountLocked()
	var written uint64
	write := func(data []byte) error {
		n, err := w.Write(data)
		written += uint64(n)
		if err == nil && n != len(data) {
			return io.ErrShortWrite
		}
		return err
	}
	var scalar [8]byte
	writeI64 := func(value int64) error {
		binary.LittleEndian.PutUint64(scalar[:], uint64(value))
		return write(scalar[:8])
	}
	writeU32 := func(value uint32) error {
		binary.LittleEndian.PutUint32(scalar[:4], value)
		return write(scalar[:4])
	}
	writeByte := func(value byte) error {
		scalar[0] = value
		return write(scalar[:1])
	}
	if err := write(annBinaryMagic[:]); err != nil {
		return written, err
	}
	for _, value := range []int64{
		int64(a.dim), int64(a.m), int64(a.efConstruction),
		int64(a.efSearch), a.seed, int64(a.entrypoint),
		int64(a.deletedCount), int64(count),
	} {
		if err := writeI64(value); err != nil {
			return written, err
		}
	}
	if a.hasEntrypoint {
		if err := writeByte(1); err != nil {
			return written, err
		}
	} else {
		if err := writeByte(0); err != nil {
			return written, err
		}
	}
	if a.quantized {
		if err := writeByte(1); err != nil {
			return written, err
		}
		for _, values := range [][]float32{a.quantMin, a.quantMax} {
			for _, value := range values {
				if err := writeU32(math.Float32bits(value)); err != nil {
					return written, err
				}
			}
		}
	} else if err := writeByte(0); err != nil {
		return written, err
	}
	for i := 0; i < count; i++ {
		if err := writeI64(int64(a.nodeIDLocked(i))); err != nil {
			return written, err
		}
	}
	for i := 0; i < count; i++ {
		if a.isDeletedLocked(i) {
			if err := writeByte(1); err != nil {
				return written, err
			}
		} else {
			if err := writeByte(0); err != nil {
				return written, err
			}
		}
	}
	for i := 0; i < count; i++ {
		if a.quantized {
			if err := write(a.qvectorLocked(i)); err != nil {
				return written, err
			}
		} else {
			for _, value := range a.nodes[i].vector {
				if err := writeU32(math.Float32bits(value)); err != nil {
					return written, err
				}
			}
		}
	}
	var edgeOffset uint64
	if err := writeI64(0); err != nil {
		return written, err
	}
	for i := 0; i < count; i++ {
		edgeOffset += uint64(a.neighborCountLocked(i))
		if edgeOffset > math.MaxInt64 {
			return written, errors.New("compact ANN adjacency overflow")
		}
		if err := writeI64(int64(edgeOffset)); err != nil {
			return written, err
		}
	}
	for i := 0; i < count; i++ {
		for position, count := 0, a.neighborCountLocked(i); position < count; position++ {
			if err := writeU32(uint32(a.neighborAtLocked(i, position))); err != nil {
				return written, err
			}
		}
	}
	return written, nil
}

func UnmarshalCompactBinary(data []byte) (*AnnIndex, error) {
	if len(data) >= len(annBinaryMagic) && bytes.Equal(data[:len(annBinaryMagic)], annBinaryMagic[:]) {
		return unmarshalANNCompactV4(data)
	}
	if len(data) >= len(annBinaryMagicV3) && bytes.Equal(data[:len(annBinaryMagicV3)], annBinaryMagicV3[:]) {
		return unmarshalANNCompactV3(data)
	}
	if len(data) >= len(annBinaryMagicV2) && bytes.Equal(data[:len(annBinaryMagicV2)], annBinaryMagicV2[:]) {
		return unmarshalANNCompactV2(data)
	}
	// Backward compatibility for the short-lived gob compact format.
	var value struct {
		Version  int
		Snapshot indexSnapshot
	}
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&value); err != nil {
		return nil, err
	}
	if value.Version != annCompactSnapshotVersion {
		return nil, fmt.Errorf("unsupported compact ANN snapshot version %d", value.Version)
	}
	encoded, err := json.Marshal(value.Snapshot)
	if err != nil {
		return nil, err
	}
	return UnmarshalBinary(encoded)
}

// OpenCompactBinaryFile validates a snapshot through the regular decoder and
// retains v4 columnar sections as read-only file mappings. Older formats are
// copied by their compatibility decoders and release the mapping immediately.
func OpenCompactBinaryFile(path string) (*AnnIndex, error) {
	mapping, err := mapSnapshotFile(path)
	if err != nil {
		return nil, err
	}
	idx, err := UnmarshalCompactBinary(mapping.data)
	if err != nil {
		_ = mapping.Close()
		return nil, err
	}
	if len(mapping.data) >= len(annBinaryMagic) &&
		bytes.Equal(mapping.data[:len(annBinaryMagic)], annBinaryMagic[:]) &&
		idx.quantized {
		idx.mappedOwner = mapping
		return idx, nil
	}
	if err := mapping.Close(); err != nil {
		_ = idx.Close()
		return nil, err
	}
	return idx, nil
}

func unmarshalANNCompactV4(data []byte) (*AnnIndex, error) {
	offset := len(annBinaryMagic)
	readI64 := func() (int64, error) {
		if offset > len(data)-8 {
			return 0, io.ErrUnexpectedEOF
		}
		value := int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
		offset += 8
		return value, nil
	}
	readByte := func() (byte, error) {
		if offset >= len(data) {
			return 0, io.ErrUnexpectedEOF
		}
		value := data[offset]
		offset++
		return value, nil
	}
	var header [8]int64
	for i := range header {
		value, err := readI64()
		if err != nil {
			return nil, err
		}
		header[i] = value
	}
	dim, m, efc, efs := header[0], header[1], header[2], header[3]
	seed, entrypoint, deletedCount, count := header[4], header[5], header[6], header[7]
	if dim < 0 || m <= 0 || efc <= 0 || efs <= 0 || count < 0 ||
		count > int64(len(data)) || deletedCount < 0 || deletedCount > count {
		return nil, errors.New("invalid compact ANN v4 header")
	}
	hasEntrypoint, err := readByte()
	if err != nil || hasEntrypoint > 1 {
		return nil, errors.New("invalid compact ANN v4 entrypoint")
	}
	quantizedFlag, err := readByte()
	if err != nil || quantizedFlag > 1 {
		return nil, errors.New("invalid compact ANN v4 quantization flag")
	}
	quantized := quantizedFlag == 1
	if count > 0 && dim > int64((^uint(0)>>1)/uint(count)) {
		return nil, errors.New("compact ANN v4 vector arena overflow")
	}
	vectorElements := uint64(count) * uint64(dim)
	vectorWidth := uint64(4)
	if quantized {
		vectorWidth = 1
	}
	if vectorElements > uint64(^uint(0)>>1)/vectorWidth {
		return nil, errors.New("compact ANN v4 vector bytes overflow")
	}

	var quantMin, quantMax []float32
	if quantized {
		rangeBytes := uint64(dim) * 8
		if rangeBytes > uint64(len(data)-offset) {
			return nil, errors.New("invalid compact ANN v4 quantization ranges")
		}
		quantMin, quantMax = make([]float32, int(dim)), make([]float32, int(dim))
		for _, values := range [][]float32{quantMin, quantMax} {
			for d := range values {
				values[d] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset:]))
				offset += 4
			}
		}
	}

	idsBytes := uint64(count) * 8
	tombstoneBytes := uint64(count)
	vectorsBytes := vectorElements * vectorWidth
	offsetsBytes := uint64(count+1) * 8
	fixedBytes := idsBytes + tombstoneBytes + vectorsBytes + offsetsBytes
	if fixedBytes > uint64(len(data)-offset) {
		return nil, io.ErrUnexpectedEOF
	}
	idsData := data[offset : offset+int(idsBytes)]
	offset += int(idsBytes)
	tombstones := data[offset : offset+int(tombstoneBytes)]
	offset += int(tombstoneBytes)
	vectorsData := data[offset : offset+int(vectorsBytes)]
	offset += int(vectorsBytes)
	adjacencyOffsets := data[offset : offset+int(offsetsBytes)]
	offset += int(offsetsBytes)

	if binary.LittleEndian.Uint64(adjacencyOffsets[:8]) != 0 {
		return nil, errors.New("invalid compact ANN v4 first adjacency offset")
	}
	var previous uint64
	for i := 1; i <= int(count); i++ {
		current := binary.LittleEndian.Uint64(adjacencyOffsets[i*8:])
		if current < previous {
			return nil, errors.New("invalid compact ANN v4 adjacency offsets")
		}
		previous = current
	}
	if previous > uint64((len(data)-offset)/4) || previous*4 != uint64(len(data)-offset) {
		return nil, errors.New("invalid compact ANN v4 adjacency payload")
	}
	adjacencyData := data[offset:]
	for edge := uint64(0); edge < previous; edge++ {
		neighbor := int32(binary.LittleEndian.Uint32(adjacencyData[edge*4:]))
		if neighbor < 0 || int64(neighbor) >= count {
			return nil, errors.New("invalid compact ANN v4 neighbor")
		}
	}

	idx := NewAnnIndexWithOptions(Options{M: int(m), EfConstruction: int(efc), EfSearch: int(efs), Seed: seed})
	idx.dim, idx.entrypoint = int(dim), int(entrypoint)
	idx.hasEntrypoint, idx.deletedCount = hasEntrypoint == 1, int(deletedCount)
	idx.quantized, idx.quantMin, idx.quantMax = quantized, quantMin, quantMax
	idx.idsSorted = quantized
	var vectorArena []float32
	if quantized {
		// The caller-owned byte slice is retained by these views. File-backed
		// callers attach a mapping owner after validation.
		idx.qvectors = vectorsData
		idx.mappedIDs = idsData
		idx.immutableNodeCount = int(count)
		idx.deletedBits = make([]uint64, (int(count)+63)/64)
		idx.idToSlot = nil
	} else {
		idx.nodes = make([]node, int(count))
		idx.deleted = make([]bool, int(count))
		idx.idToSlot = make(map[int]int, int(count))
		vectorArena = make([]float32, int(vectorElements))
	}
	for i := 0; i < int(count); i++ {
		id64 := int64(binary.LittleEndian.Uint64(idsData[i*8:]))
		id := int(id64)
		if int64(id) != id64 {
			return nil, errors.New("compact ANN v4 ID overflow")
		}
		if tombstones[i] > 1 {
			return nil, errors.New("invalid compact ANN v4 tombstone")
		}
		var vector []float32
		if !quantized {
			vector = vectorArena[i*int(dim) : (i+1)*int(dim)]
			start := i * int(dim) * 4
			for d := range vector {
				vector[d] = math.Float32frombits(binary.LittleEndian.Uint32(vectorsData[start+d*4:]))
			}
		}
		if quantized {
			if tombstones[i] == 1 {
				idx.deletedBits[i>>6] |= uint64(1) << uint(i&63)
			}
			if i > 0 {
				previous := int(int64(binary.LittleEndian.Uint64(idsData[(i-1)*8:])))
				if previous >= id {
					if idx.idToSlot == nil {
						idx.idToSlot = make(map[int]int, int(count))
						for prior := 0; prior < i; prior++ {
							priorID := int(int64(binary.LittleEndian.Uint64(idsData[prior*8:])))
							idx.idToSlot[priorID] = prior
						}
					}
					idx.idsSorted = false
				}
			}
			if idx.idToSlot != nil {
				if _, exists := idx.idToSlot[id]; exists {
					return nil, errors.New("duplicate compact ANN v4 ID")
				}
				idx.idToSlot[id] = i
			}
		} else {
			if _, exists := idx.idToSlot[id]; exists {
				return nil, errors.New("duplicate compact ANN v4 ID")
			}
			idx.nodes[i] = node{id: id, vector: vector}
			idx.deleted[i] = tombstones[i] == 1
			idx.idToSlot[id] = i
		}
	}
	if quantized {
		idx.mappedOffsets = adjacencyOffsets
		idx.mappedAdjacency = adjacencyData
		if idx.idsSorted {
			idx.idToSlot = nil
		}
	} else {
		for slot := range idx.nodes {
			start := binary.LittleEndian.Uint64(adjacencyOffsets[slot*8:])
			end := binary.LittleEndian.Uint64(adjacencyOffsets[(slot+1)*8:])
			neighbors := make([]int32, int(end-start))
			for position := range neighbors {
				neighbors[position] = int32(binary.LittleEndian.Uint32(adjacencyData[(start+uint64(position))*4:]))
			}
			idx.nodes[slot].neighbors = neighbors
		}
	}
	if idx.hasEntrypoint && (idx.entrypoint < 0 || idx.entrypoint >= idx.nodeCountLocked()) {
		return nil, errors.New("invalid compact ANN v4 entrypoint")
	}
	idx.rebuildRouteSignaturesLocked()
	idx.rnd = rand.New(rand.NewSource(seed)) // #nosec G404 -- snapshot reproducibility
	return idx, nil
}

func unmarshalANNCompactV3(data []byte) (*AnnIndex, error) {
	offset := len(annBinaryMagicV3)
	readI64 := func() (int64, error) {
		if offset > len(data)-8 {
			return 0, io.ErrUnexpectedEOF
		}
		value := int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
		offset += 8
		return value, nil
	}
	readByte := func() (byte, error) {
		if offset >= len(data) {
			return 0, io.ErrUnexpectedEOF
		}
		value := data[offset]
		offset++
		return value, nil
	}
	readFloat32 := func() (float32, error) {
		if offset > len(data)-4 {
			return 0, io.ErrUnexpectedEOF
		}
		value := math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4
		return value, nil
	}
	var values [8]int64
	for i := range values {
		var err error
		if values[i], err = readI64(); err != nil {
			return nil, err
		}
	}
	dim, m, efc, efs, seed, entrypoint, deletedCount, count := values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]
	if dim < 0 || m <= 0 || efc <= 0 || efs <= 0 || count < 0 || count > int64(len(data)) || deletedCount < 0 || deletedCount > count {
		return nil, errors.New("invalid compact ANN header")
	}
	hasEntrypoint, err := readByte()
	if err != nil || hasEntrypoint > 1 {
		return nil, errors.New("invalid compact ANN entrypoint")
	}
	quantizedByte, err := readByte()
	if err != nil || quantizedByte > 1 {
		return nil, errors.New("invalid compact ANN quantization flag")
	}
	quantized := quantizedByte == 1
	var quantMin, quantMax []float32
	if quantized {
		if dim > int64((len(data)-offset)/8) {
			return nil, errors.New("invalid compact ANN quantization ranges")
		}
		quantMin, quantMax = make([]float32, int(dim)), make([]float32, int(dim))
		for _, values := range [][]float32{quantMin, quantMax} {
			for d := range values {
				if values[d], err = readFloat32(); err != nil {
					return nil, err
				}
			}
		}
	}

	vectorWidth := 4
	if quantized {
		vectorWidth = 1
	}
	recordsOffset := offset
	totalNeighbors := 0
	for i := 0; i < int(count); i++ {
		if _, err = readI64(); err != nil {
			return nil, err
		}
		deleted, readErr := readByte()
		if readErr != nil || deleted > 1 {
			return nil, errors.New("invalid compact ANN tombstone")
		}
		vectorCount, readErr := readI64()
		if readErr != nil || vectorCount != dim || vectorCount < 0 || vectorCount > int64((len(data)-offset)/vectorWidth) {
			return nil, errors.New("invalid compact ANN vector")
		}
		offset += int(vectorCount) * vectorWidth
		neighborCount, readErr := readI64()
		if readErr != nil || neighborCount < 0 || neighborCount > int64((len(data)-offset)/4) {
			return nil, errors.New("invalid compact ANN neighbors")
		}
		if neighborCount > int64(^uint(0)>>1)-int64(totalNeighbors) {
			return nil, errors.New("compact ANN neighbors overflow")
		}
		for j := int64(0); j < neighborCount; j++ {
			neighbor := int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
			if neighbor < 0 || int64(neighbor) >= count {
				return nil, errors.New("invalid compact ANN neighbor")
			}
			offset += 4
		}
		totalNeighbors += int(neighborCount)
	}
	if offset != len(data) {
		return nil, errors.New("invalid compact ANN trailer")
	}

	idx := NewAnnIndexWithOptions(Options{M: int(m), EfConstruction: int(efc), EfSearch: int(efs), Seed: seed})
	idx.dim, idx.entrypoint, idx.hasEntrypoint, idx.deletedCount = int(dim), int(entrypoint), hasEntrypoint == 1, int(deletedCount)
	idx.quantized, idx.quantMin, idx.quantMax = quantized, quantMin, quantMax
	idx.idsSorted = quantized
	idx.nodes = make([]node, int(count))
	idx.deleted = make([]bool, int(count))
	idx.idToSlot = make(map[int]int, int(count))
	if count > 0 && dim > int64((^uint(0)>>1)/uint(count)) {
		return nil, errors.New("compact ANN vector arena overflow")
	}
	var vectorArena []float32
	var quantizedArena []byte
	if quantized {
		quantizedArena = make([]byte, int(count*dim))
		idx.qvectors = quantizedArena
	} else {
		vectorArena = make([]float32, int(count*dim))
	}
	neighborArena := make([]int32, totalNeighbors)
	offset = recordsOffset
	vectorOffset, neighborOffset := 0, 0
	for i := 0; i < int(count); i++ {
		id64, readErr := readI64()
		if readErr != nil {
			return nil, readErr
		}
		deleted, readErr := readByte()
		if readErr != nil || deleted > 1 {
			return nil, errors.New("invalid compact ANN tombstone")
		}
		vectorCount, readErr := readI64()
		if readErr != nil || vectorCount != dim {
			return nil, errors.New("invalid compact ANN vector")
		}
		var vector []float32
		if quantized {
			copy(quantizedArena[vectorOffset:vectorOffset+int(vectorCount)], data[offset:offset+int(vectorCount)])
			offset += int(vectorCount)
		} else {
			vector = vectorArena[vectorOffset : vectorOffset+int(vectorCount)]
			for d := range vector {
				vector[d] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
				offset += 4
			}
		}
		vectorOffset += int(vectorCount)
		neighborCount, readErr := readI64()
		if readErr != nil || neighborCount < 0 {
			return nil, errors.New("invalid compact ANN neighbors")
		}
		neighbors := neighborArena[neighborOffset : neighborOffset+int(neighborCount)]
		for n := range neighbors {
			neighbors[n] = int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4
		}
		neighborOffset += int(neighborCount)
		id := int(id64)
		if int64(id) != id64 {
			return nil, errors.New("compact ANN ID overflow")
		}
		if _, exists := idx.idToSlot[id]; exists {
			return nil, errors.New("duplicate compact ANN ID")
		}
		if quantized && i > 0 && idx.nodes[i-1].id >= id {
			idx.idsSorted = false
		}
		idx.nodes[i] = node{id: id, vector: vector, neighbors: neighbors}
		idx.deleted[i] = deleted == 1
		idx.idToSlot[id] = i
	}
	if idx.quantized && idx.idsSorted {
		idx.idToSlot = nil
	}
	if idx.quantized {
		idx.compactAdjacencyLocked()
		idx.immutableNodeCount = len(idx.nodes)
		idx.immutableIDs = make([]int64, len(idx.nodes))
		idx.deletedBits = make([]uint64, (len(idx.nodes)+63)/64)
		for slot := range idx.nodes {
			idx.immutableIDs[slot] = int64(idx.nodes[slot].id)
			if idx.deleted[slot] {
				idx.deletedBits[slot>>6] |= uint64(1) << uint(slot&63)
			}
		}
		idx.nodes = nil
		idx.deleted = nil
	}
	if offset != len(data) || (idx.hasEntrypoint && (idx.entrypoint < 0 || idx.entrypoint >= idx.nodeCountLocked())) {
		return nil, errors.New("invalid compact ANN trailer")
	}
	idx.rebuildRouteSignaturesLocked()
	idx.rnd = rand.New(rand.NewSource(seed)) // #nosec G404 -- snapshot reproducibility
	return idx, nil
}

func unmarshalANNCompactV2(data []byte) (*AnnIndex, error) {
	offset := len(annBinaryMagicV2)
	readI64 := func() (int64, error) {
		if offset > len(data)-8 {
			return 0, io.ErrUnexpectedEOF
		}
		value := int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
		offset += 8
		return value, nil
	}
	readByte := func() (byte, error) {
		if offset >= len(data) {
			return 0, io.ErrUnexpectedEOF
		}
		value := data[offset]
		offset++
		return value, nil
	}
	var values [8]int64
	for i := range values {
		var err error
		if values[i], err = readI64(); err != nil {
			return nil, err
		}
	}
	dim, m, efc, efs, seed, entrypoint, deletedCount, count := values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]
	if dim < 0 || m <= 0 || efc <= 0 || efs <= 0 || count < 0 || count > int64(len(data)) || deletedCount < 0 || deletedCount > count {
		return nil, errors.New("invalid compact ANN header")
	}
	hasEntrypoint, err := readByte()
	if err != nil || hasEntrypoint > 1 {
		return nil, errors.New("invalid compact ANN entrypoint")
	}
	// Validate the variable-width records and determine one contiguous
	// neighbor arena before allocating the index. Snapshot segments contain
	// thousands of nodes; allocating vectors and neighbors per node creates
	// millions of objects at million-vector scale and makes GC dominate
	// recovery.
	recordsOffset := offset
	totalNeighbors := 0
	for i := 0; i < int(count); i++ {
		if _, err = readI64(); err != nil {
			return nil, err
		}
		deleted, readErr := readByte()
		if readErr != nil || deleted > 1 {
			return nil, errors.New("invalid compact ANN tombstone")
		}
		vectorCount, readErr := readI64()
		if readErr != nil || vectorCount != dim || vectorCount < 0 || vectorCount > int64((len(data)-offset)/4) {
			return nil, errors.New("invalid compact ANN vector")
		}
		offset += int(vectorCount) * 4
		neighborCount, readErr := readI64()
		if readErr != nil || neighborCount < 0 || neighborCount > int64((len(data)-offset)/4) {
			return nil, errors.New("invalid compact ANN neighbors")
		}
		if neighborCount > int64(^uint(0)>>1)-int64(totalNeighbors) {
			return nil, errors.New("compact ANN neighbors overflow")
		}
		for j := int64(0); j < neighborCount; j++ {
			neighbor := int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
			if neighbor < 0 || int64(neighbor) >= count {
				return nil, errors.New("invalid compact ANN neighbor")
			}
			offset += 4
		}
		totalNeighbors += int(neighborCount)
	}
	if offset != len(data) {
		return nil, errors.New("invalid compact ANN trailer")
	}

	idx := NewAnnIndexWithOptions(Options{M: int(m), EfConstruction: int(efc), EfSearch: int(efs), Seed: seed})
	idx.dim, idx.entrypoint, idx.hasEntrypoint, idx.deletedCount = int(dim), int(entrypoint), hasEntrypoint == 1, int(deletedCount)
	idx.nodes = make([]node, int(count))
	idx.deleted = make([]bool, int(count))
	idx.idToSlot = make(map[int]int, int(count))
	if count > 0 && dim > int64((^uint(0)>>1)/uint(count)) {
		return nil, errors.New("compact ANN vector arena overflow")
	}
	vectorArena := make([]float32, int(count*dim))
	neighborArena := make([]int32, totalNeighbors)
	offset = recordsOffset
	vectorOffset, neighborOffset := 0, 0
	for i := 0; i < int(count); i++ {
		id64, err := readI64()
		if err != nil {
			return nil, err
		}
		deleted, err := readByte()
		if err != nil || deleted > 1 {
			return nil, errors.New("invalid compact ANN tombstone")
		}
		vectorCount, err := readI64()
		if err != nil || vectorCount != dim {
			return nil, errors.New("invalid compact ANN vector")
		}
		vector := vectorArena[vectorOffset : vectorOffset+int(vectorCount)]
		for j := range vector {
			vector[j] = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4
		}
		vectorOffset += int(vectorCount)
		neighborCount, err := readI64()
		if err != nil || neighborCount < 0 {
			return nil, errors.New("invalid compact ANN neighbors")
		}
		neighbors := neighborArena[neighborOffset : neighborOffset+int(neighborCount)]
		for j := range neighbors {
			neighbors[j] = int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4
		}
		neighborOffset += int(neighborCount)
		id := int(id64)
		if int64(id) != id64 {
			return nil, errors.New("compact ANN ID overflow")
		}
		if _, exists := idx.idToSlot[id]; exists {
			return nil, errors.New("duplicate compact ANN ID")
		}
		idx.nodes[i] = node{id: id, vector: vector, neighbors: neighbors}
		idx.deleted[i] = deleted == 1
		idx.idToSlot[id] = i
	}
	if offset != len(data) || (idx.hasEntrypoint && (idx.entrypoint < 0 || idx.entrypoint >= len(idx.nodes))) {
		return nil, errors.New("invalid compact ANN trailer")
	}
	idx.rebuildRouteSignaturesLocked()
	idx.rnd = rand.New(rand.NewSource(seed)) // #nosec G404 -- snapshot reproducibility
	return idx, nil
}

func UnmarshalBinary(data []byte) (*AnnIndex, error) {
	var snapshot indexSnapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return nil, err
	}
	if snapshot.Version != annSnapshotVersion {
		return nil, fmt.Errorf("unsupported ANN snapshot version %d", snapshot.Version)
	}
	if len(snapshot.Deleted) != len(snapshot.Nodes) || snapshot.DeletedCount < 0 || snapshot.DeletedCount > len(snapshot.Nodes) {
		return nil, errors.New("invalid ANN snapshot tombstones")
	}
	idx := NewAnnIndexWithOptions(Options{
		M:              snapshot.M,
		EfConstruction: snapshot.EfConstruction,
		EfSearch:       snapshot.EfSearch,
		Seed:           snapshot.Seed,
	})
	idx.nodes = make([]node, len(snapshot.Nodes))
	idx.deleted = append([]bool(nil), snapshot.Deleted...)
	idx.deletedCount = snapshot.DeletedCount
	idx.entrypoint = snapshot.Entrypoint
	idx.hasEntrypoint = snapshot.HasEntrypoint
	idx.dim = snapshot.Dimension
	idx.idToSlot = make(map[int]int, len(snapshot.Nodes))
	for i, n := range snapshot.Nodes {
		if n.Slot != i || len(n.Vector) != snapshot.Dimension {
			return nil, errors.New("invalid ANN snapshot node")
		}
		for _, neighbor := range n.Neighbors {
			if neighbor < 0 || neighbor >= len(snapshot.Nodes) {
				return nil, errors.New("invalid ANN snapshot neighbor")
			}
		}
		if _, exists := idx.idToSlot[n.ID]; exists {
			return nil, errors.New("duplicate ANN snapshot ID")
		}
		neighbors := make([]int32, len(n.Neighbors))
		for j, slot := range n.Neighbors {
			neighbors[j] = int32(slot)
		}
		idx.nodes[i] = node{id: n.ID, vector: append([]float32(nil), n.Vector...), neighbors: neighbors}
		idx.idToSlot[n.ID] = i
	}
	if idx.hasEntrypoint && (idx.entrypoint < 0 || idx.entrypoint >= len(idx.nodes)) {
		return nil, errors.New("invalid ANN snapshot entrypoint")
	}
	idx.rebuildRouteSignaturesLocked()
	idx.rnd = rand.New(rand.NewSource(snapshot.Seed)) // #nosec G404 -- snapshot reproducibility, not security randomness
	return idx, nil
}

type segmentedSnapshot struct {
	Version         int               `json:"version"`
	Options         Options           `json:"options"`
	SegmentMaxNodes int               `json:"segment_max_nodes"`
	DeltaNodes      int               `json:"delta_nodes"`
	Segments        []json.RawMessage `json:"segments"`
	Delta           json.RawMessage   `json:"delta"`
}

func (s *SegmentedIndex) MarshalBinary() ([]byte, error) {
	s.mu.RLock()
	segments := append([]*AnnIndex(nil), s.segments...)
	delta := s.delta
	deltaNodes := s.deltaNodes
	maxNodes := s.segmentMaxNodes
	options := s.options
	s.mu.RUnlock()

	snapshot := segmentedSnapshot{Version: annSnapshotVersion, Options: options, SegmentMaxNodes: maxNodes, DeltaNodes: deltaNodes}
	for _, segment := range segments {
		data, err := segment.MarshalBinary()
		if err != nil {
			return nil, err
		}
		snapshot.Segments = append(snapshot.Segments, data)
	}
	data, err := delta.MarshalBinary()
	if err != nil {
		return nil, err
	}
	snapshot.Delta = data
	return json.Marshal(snapshot)
}

func UnmarshalSegmentedBinary(data []byte) (*SegmentedIndex, error) {
	var snapshot segmentedSnapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return nil, err
	}
	if snapshot.Version != annSnapshotVersion || snapshot.SegmentMaxNodes <= 0 || snapshot.DeltaNodes < 0 {
		return nil, errors.New("invalid segmented ANN snapshot")
	}
	idx := NewSegmentedIndex(snapshot.Options, snapshot.SegmentMaxNodes)
	idx.segments = make([]*AnnIndex, 0, len(snapshot.Segments))
	for _, raw := range snapshot.Segments {
		segment, err := UnmarshalBinary(raw)
		if err != nil {
			return nil, err
		}
		idx.segments = append(idx.segments, segment)
	}
	idx.segmentEpoch++
	delta, err := UnmarshalBinary(snapshot.Delta)
	if err != nil {
		return nil, err
	}
	if snapshot.DeltaNodes != delta.Stats().Nodes {
		return nil, errors.New("invalid segmented ANN delta count")
	}
	idx.delta = delta
	idx.deltaNodes = snapshot.DeltaNodes
	idx.mu.Lock()
	idx.requestCompactionLocked()
	idx.mu.Unlock()
	return idx, nil
}

// MarshalSegmentedCompactBinary avoids JSON/base64 and writes each immutable
// graph as a length-delimited binary payload.
func (s *SegmentedIndex) MarshalSegmentedCompactBinary() ([]byte, error) {
	var out bytes.Buffer
	if _, err := s.WriteSegmentedCompactBinary(&out); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// WriteSegmentedCompactBinary streams a length-delimited segmented snapshot.
func (s *SegmentedIndex) WriteSegmentedCompactBinary(w io.Writer) (uint64, error) {
	s.mu.RLock()
	segments := append([]*AnnIndex(nil), s.segments...)
	delta := s.delta
	deltaNodes := s.deltaNodes
	maxNodes := s.segmentMaxNodes
	options := s.options
	s.mu.RUnlock()
	var written uint64
	write := func(data []byte) error {
		n, err := w.Write(data)
		written += uint64(n)
		if err == nil && n != len(data) {
			return io.ErrShortWrite
		}
		return err
	}
	writeU64 := func(value uint64) error {
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], value)
		return write(buf[:])
	}
	if err := write(segmentedBinaryMagic[:]); err != nil {
		return written, err
	}
	for _, v := range []int64{int64(options.M), int64(options.EfConstruction), int64(options.EfSearch), options.Seed, int64(maxNodes), int64(deltaNodes), int64(len(segments))} {
		if err := writeU64(uint64(v)); err != nil {
			return written, err
		}
	}
	writeIndex := func(idx *AnnIndex) error {
		size := idx.CompactBinarySize()
		if err := writeU64(size); err != nil {
			return err
		}
		n, err := idx.WriteCompactBinary(w)
		written += n
		if err == nil && n != size {
			return fmt.Errorf("compact ANN size changed while writing: got %d want %d", n, size)
		}
		return err
	}
	for _, segment := range segments {
		if err := writeIndex(segment); err != nil {
			return written, err
		}
	}
	if err := writeIndex(delta); err != nil {
		return written, err
	}
	return written, nil
}

func UnmarshalSegmentedCompactBinary(data []byte) (*SegmentedIndex, error) {
	if len(data) < len(segmentedBinaryMagic) || !bytes.Equal(data[:len(segmentedBinaryMagic)], segmentedBinaryMagic[:]) {
		return nil, errors.New("invalid compact segmented ANN magic")
	}
	r := bytes.NewReader(data[len(segmentedBinaryMagic):])
	readI64 := func() (int64, error) { var v int64; err := binary.Read(r, binary.LittleEndian, &v); return v, err }
	v := make([]int64, 7)
	for i := range v {
		var err error
		if v[i], err = readI64(); err != nil {
			return nil, err
		}
	}
	if v[0] <= 0 || v[1] <= 0 || v[2] <= 0 || v[4] <= 0 || v[5] < 0 || v[6] < 0 || v[6] > int64(len(data)) {
		return nil, errors.New("invalid compact segmented ANN header")
	}
	idx := NewSegmentedIndex(Options{M: int(v[0]), EfConstruction: int(v[1]), EfSearch: int(v[2]), Seed: v[3]}, int(v[4]))
	idx.segments = nil
	readIndex := func() (*AnnIndex, error) {
		var size uint64
		if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
			return nil, err
		}
		if size > uint64(r.Len()) {
			return nil, io.ErrUnexpectedEOF
		}
		start := len(data) - r.Len()
		payload := data[start : start+int(size)]
		if _, err := r.Seek(int64(size), io.SeekCurrent); err != nil {
			return nil, err
		}
		return UnmarshalCompactBinary(payload)
	}
	for i := int64(0); i < v[6]; i++ {
		segment, err := readIndex()
		if err != nil {
			idx.Close()
			return nil, err
		}
		idx.segments = append(idx.segments, segment)
	}
	delta, err := readIndex()
	if err != nil {
		idx.Close()
		return nil, err
	}
	if r.Len() != 0 || delta.Stats().Nodes != int(v[5]) {
		idx.Close()
		return nil, errors.New("invalid compact segmented ANN delta")
	}
	idx.delta = delta
	idx.deltaNodes = int(v[5])
	idx.mutationEpoch++
	idx.segmentEpoch++
	return idx, nil
}

// OpenSegmentedCompactBinaryFileRegion maps a length-delimited segmented
// snapshot embedded in a larger durable artifact. The mapping is owned by the
// returned index and remains valid until Close.
func OpenSegmentedCompactBinaryFileRegion(path string, offset, size int64) (*SegmentedIndex, error) {
	mapping, err := mapSnapshotFile(path)
	if err != nil {
		return nil, err
	}
	if offset < 0 || size < 0 || offset > int64(len(mapping.data)) ||
		size > int64(len(mapping.data))-offset {
		_ = mapping.Close()
		return nil, io.ErrUnexpectedEOF
	}
	idx, err := UnmarshalSegmentedCompactBinary(mapping.data[offset : offset+size])
	if err != nil {
		_ = mapping.Close()
		return nil, err
	}
	idx.mappedOwner = mapping
	return idx, nil
}
