package core

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

const annCheckpointVersion = 1 // legacy JSON version

var annCheckpointBinaryMagic = [8]byte{'L', 'V', 'C', 'H', 'K', '0', '0', '2'}

type annCheckpoint struct {
	Version     int             `json:"version"`
	Fingerprint string          `json:"fingerprint"`
	IDs         map[string]int  `json:"ids"`
	Index       json.RawMessage `json:"index"`
}

func (s *Service) annCheckpointPath() string {
	if s.vectorPath == "" {
		return ""
	}
	return filepath.Join(s.vectorPath, "ann-checkpoint.bin")
}

func (s *Service) legacyANNCheckpointPath() string {
	return filepath.Join(s.vectorPath, "ann-checkpoint.json")
}

func (s *Service) saveANNCheckpoint(vectors []index.Vector) error {
	return s.saveANNCheckpointWithFingerprint(vectorSetFingerprint(vectors))
}

// saveANNCheckpointFromStore fingerprints the canonical store without
// materializing every persisted float32 payload as a float64 index.Vector.
// This keeps graceful shutdown memory bounded for large segmented stores.
func (s *Service) saveANNCheckpointFromStore() error {
	if vectors, ok := s.vectorStore.(rangeVectorByID32Reader); ok {
		h := sha256.New()
		var buf [8]byte
		count := 0
		vectors.RangeVectorsByID32(func(id string, values []float32) bool {
			writeVectorFingerprint(h, buf[:], id, values)
			count++
			return true
		})
		stats := s.currentANNIndex().Stats()
		if count == stats.Nodes-stats.Deleted {
			return s.finishANNCheckpoint(s.saveANNCheckpointWithFingerprint(hex.EncodeToString(h.Sum(nil))))
		}
	}
	if ids, ok := s.vectorStore.(rangeVectorIDReader); ok {
		if values, ok := s.vectorStore.(readOnlyVector32Reader); ok {
			h := sha256.New()
			var buf [8]byte
			valid := true
			ids.RangeVectorIDs(func(id string) bool {
				vectorValues, err := values.GetVectorReadOnly32(id)
				if err != nil {
					valid = false
					return false
				}
				writeVectorFingerprint(h, buf[:], id, vectorValues)
				return true
			})
			if valid {
				return s.finishANNCheckpoint(s.saveANNCheckpointWithFingerprint(hex.EncodeToString(h.Sum(nil))))
			}
		}
	}
	return s.finishANNCheckpoint(s.saveANNCheckpoint(s.vectorStore.ListVectors()))
}

func (s *Service) finishANNCheckpoint(checkpointErr error) error {
	if checkpointErr != nil {
		return checkpointErr
	}
	if writer, ok := s.vectorStore.(interface{ SaveRecoveryIndex() error }); ok {
		return writer.SaveRecoveryIndex()
	}
	return nil
}

func (s *Service) saveANNCheckpointWithFingerprint(fingerprint string) error {
	if !s.annSegmented {
		return nil
	}
	path := s.annCheckpointPath()
	if path == "" {
		return nil
	}
	segmented, ok := s.currentANNIndex().(*ann.SegmentedIndex)
	if !ok {
		return nil
	}
	resolver, ok := s.idResolver.(checkpointIDResolver)
	if !ok {
		return nil
	}
	var entries []idResolverEntry
	if errorAware, ok := resolver.(errorAwareCheckpointIDResolver); ok {
		var err error
		entries, err = errorAware.snapshotSortedE()
		if err != nil {
			return err
		}
	} else {
		entries = resolver.snapshotSorted()
	}
	if err := os.MkdirAll(filepath.Dir(path), s.storageSecurity.DirMode); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_RDWR|os.O_TRUNC, s.storageSecurity.FileMode)
	if err != nil {
		return err
	}
	cleanup := func(writeErr error) error {
		closeErr := f.Close()
		if writeErr == nil {
			writeErr = closeErr
		}
		if writeErr != nil {
			_ = os.Remove(tmp)
		}
		return writeErr
	}
	var scalar [8]byte
	writeU32 := func(value uint32) error {
		binary.LittleEndian.PutUint32(scalar[:4], value)
		_, err := f.Write(scalar[:4])
		return err
	}
	writeU64 := func(value uint64) error {
		binary.LittleEndian.PutUint64(scalar[:], value)
		_, err := f.Write(scalar[:])
		return err
	}
	writeString := func(value string) error {
		if err := writeU32(uint32(len(value))); err != nil {
			return err
		}
		_, err := io.WriteString(f, value)
		return err
	}
	if _, err = f.Write(annCheckpointBinaryMagic[:]); err != nil {
		return cleanup(err)
	}
	if err = writeString(fingerprint); err != nil {
		return cleanup(err)
	}
	if err = writeU64(uint64(len(entries))); err != nil {
		return cleanup(err)
	}
	for _, entry := range entries {
		if err = writeString(entry.id); err != nil {
			return cleanup(err)
		}
		if err = writeU64(uint64(entry.internalID)); err != nil {
			return cleanup(err)
		}
	}
	indexSizeOffset, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return cleanup(err)
	}
	if err = writeU64(0); err != nil {
		return cleanup(err)
	}
	indexStart, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return cleanup(err)
	}
	if _, err = segmented.WriteSegmentedCompactBinary(f); err != nil {
		return cleanup(err)
	}
	bodyEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return cleanup(err)
	}
	if _, err = f.Seek(indexSizeOffset, io.SeekStart); err != nil {
		return cleanup(err)
	}
	if err = writeU64(uint64(bodyEnd - indexStart)); err != nil {
		return cleanup(err)
	}
	if _, err = f.Seek(0, io.SeekStart); err != nil {
		return cleanup(err)
	}
	h := sha256.New()
	if _, err = io.CopyN(h, f, bodyEnd); err != nil {
		return cleanup(err)
	}
	if _, err = f.Seek(bodyEnd, io.SeekStart); err != nil {
		return cleanup(err)
	}
	if _, err = f.Write(h.Sum(nil)); err == nil {
		err = f.Sync()
	}
	if err = cleanup(err); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (s *Service) loadANNCheckpoint(vectors []index.Vector) bool {
	if !s.annSegmented {
		return false
	}
	if fingerprint, entries, loaded, err := parseANNCheckpointBinaryFile(s.annCheckpointPath()); err == nil {
		return s.installANNCheckpointEntries(vectors, fingerprint, entries, loaded)
	}
	data, err := os.ReadFile(s.legacyANNCheckpointPath())
	if err != nil {
		return false
	}
	var checkpoint annCheckpoint
	if json.Unmarshal(data, &checkpoint) != nil || checkpoint.Version != annCheckpointVersion {
		return false
	}
	loaded, err := ann.UnmarshalSegmentedBinary(checkpoint.Index)
	if err != nil {
		return false
	}
	return s.installANNCheckpoint(vectors, checkpoint.Fingerprint, checkpoint.IDs, loaded)
}

// loadANNCheckpointFromStore validates the v2 checkpoint against the native
// float32 store without materializing a second full vector collection.
func (s *Service) loadANNCheckpointFromStore() bool {
	if !s.annSegmented {
		return false
	}
	vectorsReader, streamingOK := s.vectorStore.(rangeVectorByID32Reader)
	idsReader, idsOK := s.vectorStore.(rangeVectorIDReader)
	valuesReader, valuesOK := s.vectorStore.(readOnlyVector32Reader)
	if !streamingOK && (!idsOK || !valuesOK) {
		return false
	}
	started := time.Now()
	fingerprint, entries, loaded, err := parseANNCheckpointBinaryFile(s.annCheckpointPath())
	startupTrace("ann-checkpoint-map-parse", started)
	if err != nil {
		return false
	}
	h := sha256.New()
	var buf [8]byte
	count, valid := 0, true
	visit := func(id string, values []float32) bool {
		writeVectorFingerprint(h, buf[:], id, values)
		count++
		return true
	}
	if streamingOK {
		started := time.Now()
		vectorsReader.RangeVectorsByID32(visit)
		startupTrace("ann-checkpoint-fingerprint", started)
	} else {
		idsReader.RangeVectorIDs(func(id string) bool {
			values, readErr := valuesReader.GetVectorReadOnly32(id)
			if readErr != nil {
				valid = false
				return false
			}
			return visit(id, values)
		})
	}
	if !valid || count != len(entries) || hex.EncodeToString(h.Sum(nil)) != fingerprint {
		_ = loaded.Close()
		return false
	}
	return s.activateANNCheckpointEntries(entries, loaded, count)
}

func parseANNCheckpointBinaryFile(path string) (string, []idResolverEntry, *ann.SegmentedIndex, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", nil, nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || info.Size() < int64(len(annCheckpointBinaryMagic)+sha256.Size) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	bodySize := info.Size() - sha256.Size
	magic := make([]byte, len(annCheckpointBinaryMagic))
	if _, err = io.ReadFull(file, magic); err != nil || !bytes.Equal(magic, annCheckpointBinaryMagic[:]) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	readString := func() (string, error) {
		var size uint32
		if err := binary.Read(file, binary.LittleEndian, &size); err != nil {
			return "", err
		}
		position, seekErr := file.Seek(0, io.SeekCurrent)
		if seekErr != nil || int64(size) > bodySize-position {
			return "", io.ErrUnexpectedEOF
		}
		value := make([]byte, int(size))
		_, readErr := io.ReadFull(file, value)
		return string(value), readErr
	}
	fingerprint, err := readString()
	if err != nil {
		return "", nil, nil, err
	}
	var count uint64
	if err = binary.Read(file, binary.LittleEndian, &count); err != nil || count > uint64(bodySize/12) || count > uint64(^uint(0)>>1) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	entries := make([]idResolverEntry, 0, int(count))
	previousID := ""
	for i := uint64(0); i < count; i++ {
		id, readErr := readString()
		if readErr != nil {
			return "", nil, nil, readErr
		}
		var value int64
		if readErr = binary.Read(file, binary.LittleEndian, &value); readErr != nil {
			return "", nil, nil, readErr
		}
		converted := int(value)
		if int64(converted) != value {
			return "", nil, nil, io.ErrUnexpectedEOF
		}
		if id == "" || (i > 0 && id <= previousID) {
			return "", nil, nil, io.ErrUnexpectedEOF
		}
		entries = append(entries, idResolverEntry{id: id, internalID: converted})
		previousID = id
	}
	var indexSize uint64
	if err = binary.Read(file, binary.LittleEndian, &indexSize); err != nil {
		return "", nil, nil, err
	}
	indexOffset, err := file.Seek(0, io.SeekCurrent)
	if err != nil || indexSize > uint64(bodySize-indexOffset) || indexOffset+int64(indexSize) != bodySize {
		return "", nil, nil, io.ErrUnexpectedEOF
	}

	if _, err = file.Seek(0, io.SeekStart); err != nil {
		return "", nil, nil, err
	}
	h := sha256.New()
	if _, err = io.CopyN(h, file, bodySize); err != nil {
		return "", nil, nil, err
	}
	expected := make([]byte, sha256.Size)
	if _, err = io.ReadFull(file, expected); err != nil || !bytes.Equal(expected, h.Sum(nil)) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	loaded, err := ann.OpenSegmentedCompactBinaryFileRegion(path, indexOffset, int64(indexSize))
	if err != nil {
		return "", nil, nil, err
	}
	return fingerprint, entries, loaded, nil
}

func parseANNCheckpointBinary(data []byte) (string, map[string]int, *ann.SegmentedIndex, error) {
	if len(data) < len(annCheckpointBinaryMagic)+sha256.Size || !bytes.Equal(data[:len(annCheckpointBinaryMagic)], annCheckpointBinaryMagic[:]) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	body, checksum := data[:len(data)-sha256.Size], data[len(data)-sha256.Size:]
	calculated := sha256.Sum256(body)
	if !bytes.Equal(checksum, calculated[:]) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	r := bytes.NewReader(body[len(annCheckpointBinaryMagic):])
	readString := func() (string, error) {
		var size uint32
		if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
			return "", err
		}
		if uint64(size) > uint64(r.Len()) {
			return "", io.ErrUnexpectedEOF
		}
		value := make([]byte, int(size))
		_, err := io.ReadFull(r, value)
		return string(value), err
	}
	fingerprint, err := readString()
	if err != nil {
		return "", nil, nil, err
	}
	var count uint64
	if err = binary.Read(r, binary.LittleEndian, &count); err != nil || count > uint64(r.Len()/12) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	ids := make(map[string]int, int(count))
	for i := uint64(0); i < count; i++ {
		id, e := readString()
		if e != nil {
			return "", nil, nil, e
		}
		var value int64
		if e = binary.Read(r, binary.LittleEndian, &value); e != nil {
			return "", nil, nil, e
		}
		converted := int(value)
		if int64(converted) != value {
			return "", nil, nil, io.ErrUnexpectedEOF
		}
		if _, exists := ids[id]; exists {
			return "", nil, nil, io.ErrUnexpectedEOF
		}
		ids[id] = converted
	}
	var indexSize uint64
	if err = binary.Read(r, binary.LittleEndian, &indexSize); err != nil || indexSize > uint64(r.Len()) {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	start := len(body) - r.Len()
	payload := body[start : start+int(indexSize)]
	if _, err = r.Seek(int64(indexSize), io.SeekCurrent); err != nil || r.Len() != 0 {
		return "", nil, nil, io.ErrUnexpectedEOF
	}
	loaded, err := ann.UnmarshalSegmentedCompactBinary(payload)
	return fingerprint, ids, loaded, err
}

func (s *Service) installANNCheckpoint(vectors []index.Vector, fingerprint string, ids map[string]int, loaded *ann.SegmentedIndex) bool {
	if fingerprint != vectorSetFingerprint(vectors) || len(ids) != len(vectors) {
		_ = loaded.Close()
		return false
	}
	for _, vec := range vectors {
		if _, ok := ids[vec.ID]; !ok {
			_ = loaded.Close()
			return false
		}
	}
	return s.activateANNCheckpoint(ids, loaded, len(vectors))
}

func (s *Service) installANNCheckpointEntries(vectors []index.Vector, fingerprint string, entries []idResolverEntry, loaded *ann.SegmentedIndex) bool {
	if fingerprint != vectorSetFingerprint(vectors) || len(entries) != len(vectors) {
		_ = loaded.Close()
		return false
	}
	return s.activateANNCheckpointEntries(entries, loaded, len(vectors))
}

func (s *Service) activateANNCheckpoint(ids map[string]int, loaded *ann.SegmentedIndex, vectorCount int) bool {
	loaded.SetSegmentRouting(s.annOptions.SegmentRouting)
	loaded.SetDiversifiedPruning(s.annOptions.DiversifiedPruning)
	if s.annOptions.QuantizeSegments {
		if err := loaded.EnableSegmentQuantization(); err != nil {
			_ = loaded.Close()
			return false
		}
	}
	stats := loaded.Stats()
	if stats.Nodes-stats.Deleted != vectorCount {
		_ = loaded.Close()
		return false
	}
	resolver, ok := s.idResolver.(checkpointIDResolver)
	if !ok || resolver.Restore(ids) != nil {
		_ = loaded.Close()
		return false
	}
	s.annMu.Lock()
	previous := s.annIndex
	s.annIndex = loaded
	s.annMu.Unlock()
	if closer, ok := previous.(interface{ Close() error }); ok {
		_ = closer.Close()
	}
	s.annCheckpointLoaded.Store(true)
	return true
}

func (s *Service) activateANNCheckpointEntries(entries []idResolverEntry, loaded *ann.SegmentedIndex, vectorCount int) bool {
	loaded.SetSegmentRouting(s.annOptions.SegmentRouting)
	loaded.SetDiversifiedPruning(s.annOptions.DiversifiedPruning)
	if s.annOptions.QuantizeSegments {
		if err := loaded.EnableSegmentQuantization(); err != nil {
			_ = loaded.Close()
			return false
		}
	}
	stats := loaded.Stats()
	if stats.Nodes-stats.Deleted != vectorCount {
		_ = loaded.Close()
		return false
	}
	resolver, ok := s.idResolver.(checkpointIDResolver)
	if !ok || resolver.restoreEntries(entries) != nil {
		_ = loaded.Close()
		return false
	}
	s.annMu.Lock()
	previous := s.annIndex
	s.annIndex = loaded
	s.annMu.Unlock()
	if closer, ok := previous.(interface{ Close() error }); ok {
		_ = closer.Close()
	}
	s.annCheckpointLoaded.Store(true)
	return true
}

func vectorSetFingerprint(vectors []index.Vector) string {
	ordered := append([]index.Vector(nil), vectors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ID < ordered[j].ID })
	h := sha256.New()
	var buf [8]byte
	for _, vec := range ordered {
		values := make([]float32, len(vec.Values))
		for i, value := range vec.Values {
			values[i] = float32(value)
		}
		writeVectorFingerprint(h, buf[:], vec.ID, values)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeVectorFingerprint(h interface{ Write([]byte) (int, error) }, buf []byte, id string, values []float32) {
	binary.LittleEndian.PutUint64(buf, uint64(len(id)))
	_, _ = h.Write(buf)
	_, _ = h.Write([]byte(id))
	binary.LittleEndian.PutUint64(buf, uint64(len(values)))
	_, _ = h.Write(buf)
	for _, value := range values {
		binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(value))
		_, _ = h.Write(buf[:4])
	}
}
