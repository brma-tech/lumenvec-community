package core

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"lumenvec/internal/index"
	"lumenvec/internal/vector"
)

const segmentManifestVersion = 1
const defaultSegmentCompactThreshold = 32
const defaultSegmentCompactionBudgetBytes = 64 << 20

type segmentManifest struct {
	Version        int      `json:"version"`
	NextGeneration uint64   `json:"next_generation"`
	BytesWritten   uint64   `json:"bytes_written,omitempty"`
	Segments       []string `json:"segments"`
}

type segmentLocation struct {
	segment string
	meta    fileVectorRecordMeta
}

// segmentVectorStore persists every committed batch as an immutable segment.
// The manifest is the commit point: orphan segment files are ignored on reopen.
type segmentVectorStore struct {
	basePath           string
	manifestPath       string
	recoveryPath       string
	locationMarkerPath string
	security           StorageSecurityOptions
	deltaWAL           *DeltaWAL

	mu               sync.RWMutex
	manifest         segmentManifest
	manifestHash     [sha256.Size]byte
	locations        map[string]segmentLocation
	diskLocations    *diskLocationIndex
	locationCapacity uint64
	ids              *orderedIDIndex
	recoveryLoaded   bool
	closed           bool
	writes           uint64
	compacts         atomic.Uint64
	bytesWritten     atomic.Uint64
	compacting       atomic.Bool

	compactThreshold   int
	compactBudgetBytes uint64
	compactCh          chan struct{}
	stopCh             chan struct{}
	stopOnce           sync.Once
	workerWG           sync.WaitGroup
}

func newSegmentVectorStore(basePath string, security StorageSecurityOptions) *segmentVectorStore {
	return newSegmentVectorStoreWithLocationCapacity(basePath, security, 0)
}

func newSegmentVectorStoreWithLocationCapacity(basePath string, security StorageSecurityOptions, locationCapacity uint64) *segmentVectorStore {
	store := &segmentVectorStore{
		basePath:           basePath,
		manifestPath:       filepath.Join(basePath, "manifest.json"),
		recoveryPath:       filepath.Join(basePath, "locations.idx"),
		locationMarkerPath: filepath.Join(basePath, "locations.hash.manifest"),
		security:           normalizeStorageSecurityOptions(security),
		locationCapacity:   locationCapacity,
		// The locations map is sufficient for point reads and ANN recovery.
		// Build the ordered B-tree only when an API actually requests ordered
		// IDs; eagerly inserting one million IDs dominated restart time.
		ids:                nil,
		compactThreshold:   defaultSegmentCompactThreshold,
		compactBudgetBytes: defaultSegmentCompactionBudgetBytes,
		compactCh:          make(chan struct{}, 1),
		stopCh:             make(chan struct{}),
	}
	if locationCapacity == 0 {
		store.locations = make(map[string]segmentLocation)
	}
	if err := store.open(); err != nil {
		panic(err)
	}
	wal, err := OpenDeltaWAL(filepath.Join(basePath, "delta.wal"))
	if err != nil {
		panic(err)
	}
	store.deltaWAL = wal
	store.workerWG.Add(1)
	go store.compactionWorker()
	return store
}

func parseSegmentLocationName(name string) (uint64, bool, error) {
	compact := false
	prefix := "segment-"
	if strings.HasPrefix(name, "compact-") {
		compact = true
		prefix = "compact-"
	}
	if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, ".dat") {
		return 0, false, fmt.Errorf("invalid segment filename %q", name)
	}
	generation, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(name, prefix), ".dat"), 10, 64)
	if err != nil || generation == 0 {
		return 0, false, fmt.Errorf("invalid segment generation in %q", name)
	}
	return generation, compact, nil
}

func diskLocationToSegmentLocation(value diskLocationValue) segmentLocation {
	prefix := "segment"
	if value.Compact {
		prefix = "compact"
	}
	return segmentLocation{
		segment: fmt.Sprintf("%s-%020d.dat", prefix, value.SegmentGeneration),
		meta: fileVectorRecordMeta{
			recordOffset: value.RecordOffset,
			recordLength: value.RecordLength,
		},
	}
}

func segmentLocationToDiskLocation(location segmentLocation) (diskLocationValue, error) {
	generation, compact, err := parseSegmentLocationName(location.segment)
	if err != nil {
		return diskLocationValue{}, err
	}
	return diskLocationValue{
		SegmentGeneration: generation,
		Compact:           compact,
		RecordOffset:      location.meta.recordOffset,
		RecordLength:      location.meta.recordLength,
	}, nil
}

func (s *segmentVectorStore) locationGetLocked(id string) (segmentLocation, bool, error) {
	if s.diskLocations == nil {
		location, ok := s.locations[id]
		return location, ok, nil
	}
	value, ok, err := s.diskLocations.Get(id)
	if err != nil || !ok {
		return segmentLocation{}, false, err
	}
	return diskLocationToSegmentLocation(value), true, nil
}

func (s *segmentVectorStore) locationPutLocked(id string, location segmentLocation) error {
	if s.diskLocations == nil {
		s.locations[id] = location
		return nil
	}
	value, err := segmentLocationToDiskLocation(location)
	if err != nil {
		return err
	}
	return s.diskLocations.Put(id, value)
}

func (s *segmentVectorStore) locationDeleteLocked(id string) error {
	if s.diskLocations == nil {
		delete(s.locations, id)
		return nil
	}
	_, err := s.diskLocations.Delete(id)
	return err
}

func (s *segmentVectorStore) locationCountLocked() int {
	if s.diskLocations == nil {
		return len(s.locations)
	}
	return int(s.diskLocations.Count())
}

func (s *segmentVectorStore) VectorCount() (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return 0, errors.New("segment vector store is closed")
	}
	return s.locationCountLocked(), nil
}

func (s *segmentVectorStore) locationSnapshotLocked() (map[string]segmentLocation, error) {
	if s.diskLocations == nil {
		locations := make(map[string]segmentLocation, len(s.locations))
		for id, location := range s.locations {
			locations[id] = location
		}
		return locations, nil
	}
	locations := make(map[string]segmentLocation, s.locationCountLocked())
	err := s.diskLocations.Range(func(id string, value diskLocationValue) bool {
		locations[id] = diskLocationToSegmentLocation(value)
		return true
	})
	return locations, err
}

func (s *segmentVectorStore) locationReplaceLocked(locations map[string]segmentLocation) error {
	if s.diskLocations == nil {
		s.locations = locations
		return nil
	}
	if uint64(len(locations)) > s.locationCapacity {
		return ErrLocationIndexFull
	}
	if err := s.diskLocations.Reset(); err != nil {
		return err
	}
	entries := make([]diskLocationEntry, 0, len(locations))
	for id, location := range locations {
		value, err := segmentLocationToDiskLocation(location)
		if err != nil {
			return err
		}
		entries = append(entries, diskLocationEntry{Key: id, Value: value})
	}
	return s.diskLocations.PutBatch(entries)
}

func (s *segmentVectorStore) initializeDiskLocationsLocked() (bool, error) {
	if s.locationCapacity == 0 {
		return false, nil
	}
	tablePath := filepath.Join(s.basePath, "locations.hash")
	keysPath := filepath.Join(s.basePath, "locations.keys")
	marker, markerErr := os.ReadFile(s.locationMarkerPath)
	markerMatches := markerErr == nil && bytes.Equal(marker, s.manifestHash[:])
	if !markerMatches {
		_ = os.Remove(tablePath)
		_ = os.Remove(keysPath)
		_ = os.Remove(s.locationMarkerPath)
	}
	index, err := openDiskLocationIndex(tablePath, keysPath, s.locationCapacity)
	if err != nil && markerMatches {
		_ = os.Remove(tablePath)
		_ = os.Remove(keysPath)
		_ = os.Remove(s.locationMarkerPath)
		index, err = openDiskLocationIndex(tablePath, keysPath, s.locationCapacity)
		markerMatches = false
	}
	if err != nil {
		return false, err
	}
	s.diskLocations = index
	if markerMatches {
		s.recoveryLoaded = true
	}
	return markerMatches, nil
}

func (s *segmentVectorStore) persistDiskLocationMarkerLocked() error {
	if s.diskLocations == nil {
		return nil
	}
	if err := s.diskLocations.Sync(); err != nil {
		return err
	}
	tmp := s.locationMarkerPath + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, s.security.FileMode)
	if err != nil {
		return err
	}
	if _, err = f.Write(s.manifestHash[:]); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, s.locationMarkerPath)
}

func (s *segmentVectorStore) open() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.MkdirAll(s.basePath, s.security.DirMode); err != nil {
		return err
	}
	s.manifest = segmentManifest{Version: segmentManifestVersion, NextGeneration: 1}
	data, err := os.ReadFile(s.manifestPath)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	if err == nil {
		if err := json.Unmarshal(data, &s.manifest); err != nil {
			return fmt.Errorf("decode segment manifest: %w", err)
		}
		if s.manifest.Version != segmentManifestVersion {
			return fmt.Errorf("unsupported segment manifest version %d", s.manifest.Version)
		}
		s.manifestHash = sha256.Sum256(data)
	}
	if s.manifest.NextGeneration == 0 {
		s.manifest.NextGeneration = uint64(len(s.manifest.Segments)) + 1
	}
	s.bytesWritten.Store(s.manifest.BytesWritten)
	diskLoaded, err := s.initializeDiskLocationsLocked()
	if err != nil {
		return err
	}
	recoveryLoaded := diskLoaded
	if !recoveryLoaded && s.diskLocations == nil {
		recoveryLoaded = s.loadRecoveryIndexLocked()
	}
	if !recoveryLoaded {
		for _, name := range s.manifest.Segments {
			if err := s.applySegmentLocked(name); err != nil {
				return err
			}
		}
		if err := s.persistDiskLocationMarkerLocked(); err != nil {
			return err
		}
	}
	s.cleanupOrphansLocked()
	s.closed = false
	return nil
}

func (s *segmentVectorStore) UpsertVector(vec index.Vector) error {
	return s.UpsertVectors([]index.Vector{vec})
}

func (s *segmentVectorStore) UpsertVectors(vectors []index.Vector) error {
	if len(vectors) == 0 {
		return nil
	}
	records := make([][]byte, 0, len(vectors))
	for _, vec := range vectors {
		record, err := encodeFileVectorRecord(fileVectorStoreOpPut, vec.ID, vec.Values)
		if err != nil {
			return err
		}
		records = append(records, record)
	}
	if err := s.commitRecords(records); err != nil {
		return err
	}
	return s.appendDeltaVectors(vectors)
}

// UpsertVectors32 writes native float32 payloads without widening them. The
// record format is the same as the existing segment format, so older readers
// remain compatible.
func (s *segmentVectorStore) UpsertVectors32(vectors []vectorBatch32) error {
	if len(vectors) == 0 {
		return nil
	}
	records := make([][]byte, 0, len(vectors))
	for _, vec := range vectors {
		record, err := encodeFileVectorRecord32(fileVectorStoreOpPut, vec.ID, vec.Values)
		if err != nil {
			return err
		}
		records = append(records, record)
	}
	if err := s.commitRecords(records); err != nil {
		return err
	}
	deltaRecords := make([]DeltaRecord, len(vectors))
	for i, vec := range vectors {
		// AppendBatch marshals synchronously while holding the WAL lock and
		// never retains the slice. Reuse the caller-owned payload here to
		// avoid one float32 allocation per vector during native bulk ingest.
		deltaRecords[i] = DeltaRecord{VectorID: vec.ID, Values: vec.Values}
	}
	_, err := s.deltaWAL.AppendBatch(deltaRecords)
	return err
}

func (s *segmentVectorStore) appendDeltaVectors(vectors []index.Vector) error {
	records := make([]DeltaRecord, len(vectors))
	for i, vec := range vectors {
		values := make([]float32, len(vec.Values))
		for i, value := range vec.Values {
			values[i] = float32(value)
		}
		records[i] = DeltaRecord{VectorID: vec.ID, Values: values}
	}
	_, err := s.deltaWAL.AppendBatch(records)
	return err
}

func (s *segmentVectorStore) DeleteVector(id string) error {
	s.mu.RLock()
	_, ok, lookupErr := s.locationGetLocked(id)
	s.mu.RUnlock()
	if lookupErr != nil {
		return lookupErr
	}
	if !ok {
		return index.ErrVectorNotFound
	}
	record, err := encodeFileVectorRecord(fileVectorStoreOpDelete, id, nil)
	if err != nil {
		return err
	}
	if err := s.commitRecords([][]byte{record}); err != nil {
		return err
	}
	_, err = s.deltaWAL.Append(DeltaRecord{VectorID: id, Deleted: true})
	return err
}

func (s *segmentVectorStore) commitRecords(records [][]byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return os.ErrClosed
	}
	if err := s.ensureLocationCapacityLocked(records); err != nil {
		return err
	}
	name := fmt.Sprintf("segment-%020d.dat", s.manifest.NextGeneration)
	tmpPath := filepath.Join(s.basePath, name+".tmp")
	finalPath := filepath.Join(s.basePath, name)
	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, s.security.FileMode)
	if err != nil {
		return err
	}
	// Coalesce the batch into one write. A large ingest otherwise performs one
	// syscall per vector while holding the store lock, extending the critical
	// section and competing with readers. The encoded records are already
	// immutable, so this does not alter the on-disk format or commit semantics.
	var payload bytes.Buffer
	total := 0
	for _, record := range records {
		total += len(record)
	}
	payload.Grow(total)
	for _, record := range records {
		_, _ = payload.Write(record)
	}
	if _, err = f.Write(payload.Bytes()); err != nil {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err = f.Sync(); err == nil {
		err = f.Close()
	} else {
		_ = f.Close()
	}
	if err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}

	next := s.manifest
	next.NextGeneration++
	for _, record := range records {
		next.BytesWritten += uint64(len(record))
	}
	next.Segments = append(append([]string(nil), s.manifest.Segments...), name)
	if err := s.saveManifestLocked(next); err != nil {
		_ = os.Remove(finalPath)
		return err
	}
	s.manifest = next
	if err := s.applySegmentLocked(name); err != nil {
		return err
	}
	if err := s.persistDiskLocationMarkerLocked(); err != nil {
		return err
	}
	s.writes += uint64(len(records))
	s.bytesWritten.Store(next.BytesWritten)
	s.requestCompactionLocked()
	return nil
}

func (s *segmentVectorStore) ensureLocationCapacityLocked(records [][]byte) error {
	if s.diskLocations == nil {
		return nil
	}
	count := s.diskLocations.Count()
	states := make(map[string]bool, len(records))
	known := make(map[string]bool, len(records))
	for _, record := range records {
		if len(record) < 13 {
			return io.ErrUnexpectedEOF
		}
		idLength := binary.LittleEndian.Uint32(record[5:9])
		if uint64(idLength) > uint64(len(record)-13) {
			return io.ErrUnexpectedEOF
		}
		id := string(record[9 : 9+idLength])
		live := states[id]
		if !known[id] {
			var err error
			_, live, err = s.locationGetLocked(id)
			if err != nil {
				return err
			}
			known[id] = true
		}
		nextLive := record[4] != fileVectorStoreOpDelete
		if !live && nextLive {
			count++
		} else if live && !nextLive {
			count--
		}
		states[id] = nextLive
		if count > s.locationCapacity {
			return ErrLocationIndexFull
		}
	}
	return nil
}

func (s *segmentVectorStore) requestCompactionLocked() {
	if s.compactThreshold <= 0 || len(s.manifest.Segments) < s.compactThreshold {
		return
	}
	select {
	case s.compactCh <- struct{}{}:
	default:
	}
}

func (s *segmentVectorStore) compactionWorker() {
	defer s.workerWG.Done()
	for {
		select {
		case <-s.compactCh:
			_ = s.compactOnce()
		case <-s.stopCh:
			return
		}
	}
}

func (s *segmentVectorStore) compactOnce() error {
	if !s.compacting.CompareAndSwap(false, true) {
		return nil
	}
	defer s.compacting.Store(false)
	s.mu.RLock()
	if s.closed || len(s.manifest.Segments) < s.compactThreshold {
		s.mu.RUnlock()
		return nil
	}
	baseManifest := cloneSegmentManifest(s.manifest)
	locations, err := s.locationSnapshotLocked()
	if err != nil {
		s.mu.RUnlock()
		return err
	}
	s.mu.RUnlock()
	// Compaction is deliberately budgeted: if the rewrite would exceed the
	// configured I/O budget, leave immutable segments in place and let a later
	// maintenance cycle retry. Ingestion and reads never wait on this path.
	if s.compactBudgetBytes > 0 {
		var estimated uint64
		for _, location := range locations {
			estimated += uint64(location.meta.recordLength)
		}
		if estimated > s.compactBudgetBytes {
			return nil
		}
	}

	ids := make([]string, 0, len(locations))
	for id := range locations {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	name := fmt.Sprintf("compact-%020d.dat", baseManifest.NextGeneration)
	tmpPath := filepath.Join(s.basePath, name+".tmp")
	finalPath := filepath.Join(s.basePath, name)
	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, s.security.FileMode)
	if err != nil {
		return err
	}
	newLocations := make(map[string]segmentLocation, len(ids))
	var offset int64
	for _, id := range ids {
		values, err := s.readLocation32(locations[id])
		if err != nil {
			_ = f.Close()
			_ = os.Remove(tmpPath)
			return err
		}
		record, err := encodeFileVectorRecord(fileVectorStoreOpPut, id, vector.ToFloat64(values))
		if err != nil {
			_ = f.Close()
			_ = os.Remove(tmpPath)
			return err
		}
		if _, err := f.Write(record); err != nil {
			_ = f.Close()
			_ = os.Remove(tmpPath)
			return err
		}
		newLocations[id] = segmentLocation{segment: name, meta: fileVectorRecordMeta{recordOffset: offset, recordLength: boundedRecordLength(len(record))}}
		offset += int64(len(record))
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || !sameSegmentManifest(baseManifest, s.manifest) {
		_ = os.Remove(finalPath)
		if !s.closed {
			s.requestCompactionLocked()
		}
		return nil
	}
	next := segmentManifest{
		Version:        segmentManifestVersion,
		NextGeneration: s.manifest.NextGeneration,
		BytesWritten:   s.manifest.BytesWritten + uint64(offset),
		Segments:       []string{name},
	}
	if err := s.saveManifestLocked(next); err != nil {
		_ = os.Remove(finalPath)
		return err
	}
	s.manifest = next
	if err := s.locationReplaceLocked(newLocations); err != nil {
		return err
	}
	if err := s.persistDiskLocationMarkerLocked(); err != nil {
		return err
	}
	if s.ids != nil {
		s.ids = newOrderedIDIndex()
		for _, id := range ids {
			s.ids.Upsert(id)
		}
	}
	s.compacts.Add(1)
	s.bytesWritten.Store(next.BytesWritten)
	// Superseded files remain until the next reopen. A reader may already have
	// captured a location from the previous manifest without holding store.mu.
	return nil
}

func cloneSegmentManifest(manifest segmentManifest) segmentManifest {
	manifest.Segments = append([]string(nil), manifest.Segments...)
	return manifest
}

func sameSegmentManifest(a, b segmentManifest) bool {
	if a.Version != b.Version || a.NextGeneration != b.NextGeneration || len(a.Segments) != len(b.Segments) {
		return false
	}
	for i := range a.Segments {
		if a.Segments[i] != b.Segments[i] {
			return false
		}
	}
	return true
}

func (s *segmentVectorStore) cleanupOrphansLocked() {
	active := make(map[string]struct{}, len(s.manifest.Segments))
	for _, name := range s.manifest.Segments {
		active[name] = struct{}{}
	}
	entries, err := os.ReadDir(s.basePath)
	if err != nil {
		return
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || (filepath.Ext(name) != ".dat" && filepath.Ext(name) != ".tmp") {
			continue
		}
		if _, ok := active[name]; !ok {
			_ = os.Remove(filepath.Join(s.basePath, name))
		}
	}
}

func (s *segmentVectorStore) saveManifestLocked(manifest segmentManifest) error {
	data, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	tmp := s.manifestPath + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, s.security.FileMode)
	if err != nil {
		return err
	}
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, s.manifestPath); err != nil {
		return err
	}
	s.manifestHash = sha256.Sum256(data)
	return nil
}

func (s *segmentVectorStore) applySegmentLocked(name string) error {
	path := filepath.Join(s.basePath, name)
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open segment %q: %w", name, err)
	}
	defer func() { _ = f.Close() }()
	reader := bufio.NewReader(f)
	var diskMutations map[string]*segmentLocation
	if s.diskLocations != nil {
		diskMutations = make(map[string]*segmentLocation)
	}
	flushDiskMutations := func() error {
		if diskMutations == nil {
			return nil
		}
		puts := make([]diskLocationEntry, 0, len(diskMutations))
		for id, location := range diskMutations {
			if location == nil {
				if _, err := s.diskLocations.Delete(id); err != nil {
					return err
				}
				continue
			}
			value, err := segmentLocationToDiskLocation(*location)
			if err != nil {
				return err
			}
			puts = append(puts, diskLocationEntry{Key: id, Value: value})
		}
		return s.diskLocations.PutBatch(puts)
	}
	var offset int64
	var lengthHeader [4]byte
	var recordHeader [5]byte
	var valuesHeader [4]byte
	var idBuffer []byte
	for {
		_, err := io.ReadFull(reader, lengthHeader[:])
		if errors.Is(err, io.EOF) {
			return flushDiskMutations()
		}
		if err != nil {
			return fmt.Errorf("read segment %q: %w", name, err)
		}
		length := binary.LittleEndian.Uint32(lengthHeader[:])
		if length < 9 {
			return fmt.Errorf("decode segment %q: %w", name, io.ErrUnexpectedEOF)
		}
		if _, err := io.ReadFull(reader, recordHeader[:]); err != nil {
			return fmt.Errorf("read segment %q record header: %w", name, err)
		}
		op := recordHeader[0]
		valueBytes, err := fileVectorRecordValueBytes(op)
		if err != nil {
			return fmt.Errorf("decode segment %q: %w", name, err)
		}
		idLength := binary.LittleEndian.Uint32(recordHeader[1:])
		if uint64(idLength)+9 > uint64(length) {
			return fmt.Errorf("decode segment %q id: %w", name, io.ErrUnexpectedEOF)
		}
		if cap(idBuffer) < int(idLength) {
			idBuffer = make([]byte, int(idLength))
		} else {
			idBuffer = idBuffer[:int(idLength)]
		}
		if _, err := io.ReadFull(reader, idBuffer); err != nil {
			return fmt.Errorf("read segment %q id: %w", name, err)
		}
		if _, err := io.ReadFull(reader, valuesHeader[:]); err != nil {
			return fmt.Errorf("read segment %q values header: %w", name, err)
		}
		valuesLength := binary.LittleEndian.Uint32(valuesHeader[:])
		payloadBytes := uint64(valuesLength) * uint64(valueBytes)
		if uint64(9)+uint64(idLength)+payloadBytes != uint64(length) {
			return fmt.Errorf("decode segment %q values: %w", name, io.ErrUnexpectedEOF)
		}
		if payloadBytes > uint64(^uint(0)>>1) {
			return fmt.Errorf("decode segment %q values: length overflows int", name)
		}
		if discarded, err := reader.Discard(int(payloadBytes)); err != nil || uint64(discarded) != payloadBytes {
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return fmt.Errorf("read segment %q values: %w", name, err)
		}
		id := string(idBuffer)
		meta := fileVectorRecordMeta{recordOffset: offset, recordLength: 4 + length}
		if op == fileVectorStoreOpDelete {
			if diskMutations != nil {
				diskMutations[id] = nil
			} else {
				if err := s.locationDeleteLocked(id); err != nil {
					return err
				}
			}
			if s.ids != nil {
				s.ids.Delete(id)
			}
		} else {
			location := segmentLocation{segment: name, meta: meta}
			if diskMutations != nil {
				diskMutations[id] = &location
			} else {
				if err := s.locationPutLocked(id, location); err != nil {
					return err
				}
			}
			if s.ids != nil {
				s.ids.Upsert(id)
			}
		}
		offset += int64(4 + length)
	}
}

func (s *segmentVectorStore) GetVector(id string) (index.Vector, error) {
	values, err := s.GetVectorReadOnly32(id)
	if err != nil {
		return index.Vector{}, err
	}
	return index.Vector{ID: id, Values: vector.ToFloat64(values)}, nil
}

func (s *segmentVectorStore) GetVectorReadOnly(id string) (index.Vector, error) {
	return s.GetVector(id)
}

func (s *segmentVectorStore) GetVectorReadOnly32(id string) ([]float32, error) {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return nil, os.ErrClosed
	}
	location, ok, lookupErr := s.locationGetLocked(id)
	s.mu.RUnlock()
	if lookupErr != nil {
		return nil, lookupErr
	}
	if !ok {
		return nil, index.ErrVectorNotFound
	}
	return s.readLocation32(location)
}

func (s *segmentVectorStore) readLocation32(location segmentLocation) ([]float32, error) {
	f, err := os.Open(filepath.Join(s.basePath, location.segment))
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	record, err := readFileVectorRecord32At(f, location.meta)
	if err != nil {
		return nil, err
	}
	return record.values, nil
}

func (s *segmentVectorStore) ListVectors() []index.Vector {
	s.mu.RLock()
	locations, err := s.locationSnapshotLocked()
	s.mu.RUnlock()
	if err != nil {
		return nil
	}
	ids := make([]string, 0, len(locations))
	for id := range locations {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	out := make([]index.Vector, 0, len(ids))
	for _, id := range ids {
		if vec, err := s.GetVector(id); err == nil {
			out = append(out, vec)
		}
	}
	return out
}

// RangeVectors32 streams persisted vectors in their native representation.
// Values are valid only for the duration of the callback and must not be
// retained or mutated by the caller.
func (s *segmentVectorStore) RangeVectors32(fn func(id string, values []float32) bool) {
	s.mu.RLock()
	locations, snapshotErr := s.locationSnapshotLocked()
	if snapshotErr != nil {
		s.mu.RUnlock()
		return
	}
	type entry struct {
		segment  string
		id       string
		location segmentLocation
	}
	entries := make([]entry, 0, len(locations))
	for id, location := range locations {
		entries = append(entries, entry{segment: location.segment, id: id, location: location})
	}
	s.mu.RUnlock()
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].segment == entries[j].segment {
			return entries[i].id < entries[j].id
		}
		return entries[i].segment < entries[j].segment
	})
	var currentSegment string
	var f *os.File
	closeFile := func() {
		if f != nil {
			_ = f.Close()
			f = nil
		}
	}
	defer closeFile()
	var recordBuffer []byte
	var valueBuffer []float32
	for _, item := range entries {
		if item.segment != currentSegment {
			closeFile()
			f, _ = os.Open(filepath.Join(s.basePath, item.segment))
			if f == nil {
				return
			}
			currentSegment = item.segment
		}
		if len(recordBuffer) < int(item.location.meta.recordLength) {
			recordBuffer = make([]byte, item.location.meta.recordLength)
		}
		record, readErr := readFileVectorRecord32AtBufferScratch(f, item.location.meta, recordBuffer, valueBuffer)
		if readErr != nil || !fn(item.id, record.values) {
			return
		}
		valueBuffer = record.values
	}
}

// RangeVectorsByID32 streams vectors in canonical ID order while reusing its
// record/value buffers. Segment files stay open for the duration of the scan,
// avoiding one allocation and open syscall per fingerprinted vector.
func (s *segmentVectorStore) RangeVectorsByID32(fn func(id string, values []float32) bool) {
	s.mu.RLock()
	locations, snapshotErr := s.locationSnapshotLocked()
	if snapshotErr != nil {
		s.mu.RUnlock()
		return
	}
	type entry struct {
		id       string
		location segmentLocation
	}
	entries := make([]entry, 0, len(locations))
	for id, location := range locations {
		entries = append(entries, entry{id: id, location: location})
	}
	segmentCount := len(s.manifest.Segments)
	s.mu.RUnlock()
	sort.Slice(entries, func(i, j int) bool { return entries[i].id < entries[j].id })

	files := make(map[string]*os.File, segmentCount)
	defer func() {
		for _, f := range files {
			_ = f.Close()
		}
	}()
	var recordBuffer []byte
	var valueBuffer []float32
	for _, item := range entries {
		f := files[item.location.segment]
		if f == nil {
			var err error
			f, err = os.Open(filepath.Join(s.basePath, item.location.segment))
			if err != nil {
				return
			}
			files[item.location.segment] = f
		}
		if len(recordBuffer) < int(item.location.meta.recordLength) {
			recordBuffer = make([]byte, item.location.meta.recordLength)
		}
		record, err := readFileVectorRecord32AtBufferScratch(f, item.location.meta, recordBuffer, valueBuffer)
		if err != nil || !fn(item.id, record.values) {
			return
		}
		valueBuffer = record.values
	}
}

// RangeSelectedVectors32 reads an indexed candidate set in physical
// segment/offset order. Filtered search therefore opens each immutable file
// once instead of issuing one open+ReadAt pair per candidate ID.
func (s *segmentVectorStore) RangeSelectedVectors32(candidates map[string]struct{}, fn func(id string, values []float32) bool) {
	if len(candidates) == 0 {
		return
	}
	s.mu.RLock()
	type entry struct {
		id       string
		location segmentLocation
	}
	entries := make([]entry, 0, len(candidates))
	for id := range candidates {
		if location, ok, err := s.locationGetLocked(id); err == nil && ok {
			entries = append(entries, entry{id: id, location: location})
		}
	}
	s.mu.RUnlock()
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].location.segment == entries[j].location.segment {
			return entries[i].location.meta.recordOffset < entries[j].location.meta.recordOffset
		}
		return entries[i].location.segment < entries[j].location.segment
	})

	var currentSegment string
	var f *os.File
	closeFile := func() {
		if f != nil {
			_ = f.Close()
			f = nil
		}
	}
	defer closeFile()
	var recordBuffer []byte
	var valueBuffer []float32
	for _, item := range entries {
		if item.location.segment != currentSegment {
			closeFile()
			var err error
			f, err = os.Open(filepath.Join(s.basePath, item.location.segment))
			if err != nil {
				return
			}
			currentSegment = item.location.segment
		}
		if len(recordBuffer) < int(item.location.meta.recordLength) {
			recordBuffer = make([]byte, item.location.meta.recordLength)
		}
		record, err := readFileVectorRecord32AtBufferScratch(f, item.location.meta, recordBuffer, valueBuffer)
		if err != nil || !fn(item.id, record.values) {
			return
		}
		valueBuffer = record.values
	}
}

func (s *segmentVectorStore) RangeSelectedVectorIDs32(candidates []string, fn func(id string, values []float32) bool) {
	if len(candidates) == 0 {
		return
	}
	s.mu.RLock()
	type entry struct {
		id       string
		location segmentLocation
	}
	entries := make([]entry, 0, len(candidates))
	for _, id := range candidates {
		if location, ok, err := s.locationGetLocked(id); err == nil && ok {
			entries = append(entries, entry{id: id, location: location})
		}
	}
	s.mu.RUnlock()
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].location.segment == entries[j].location.segment {
			return entries[i].location.meta.recordOffset < entries[j].location.meta.recordOffset
		}
		return entries[i].location.segment < entries[j].location.segment
	})

	var currentSegment string
	var f *os.File
	closeFile := func() {
		if f != nil {
			_ = f.Close()
			f = nil
		}
	}
	defer closeFile()
	var recordBuffer []byte
	var valueBuffer []float32
	for _, item := range entries {
		if item.location.segment != currentSegment {
			closeFile()
			var err error
			f, err = os.Open(filepath.Join(s.basePath, item.location.segment))
			if err != nil {
				return
			}
			currentSegment = item.location.segment
		}
		if len(recordBuffer) < int(item.location.meta.recordLength) {
			recordBuffer = make([]byte, item.location.meta.recordLength)
		}
		record, err := readFileVectorRecord32AtBufferScratch(f, item.location.meta, recordBuffer, valueBuffer)
		if err != nil || !fn(item.id, record.values) {
			return
		}
		valueBuffer = record.values
	}
}

// RangeVectors32From resumes the deterministic segment/id enumeration after
// offset records. It preserves the same callback buffer contract as
// RangeVectors32 and is used by restartable native resharding.
func (s *segmentVectorStore) RangeVectors32From(offset uint64, fn func(id string, values []float32) bool) {
	var seen uint64
	s.RangeVectors32(func(id string, values []float32) bool {
		if seen < offset {
			seen++
			return true
		}
		return fn(id, values)
	})
}

func (s *segmentVectorStore) RangeVectorIDs(fn func(string) bool) {
	s.mu.Lock()
	s.ensureOrderedIDsLocked()
	s.mu.Unlock()
	s.mu.RLock()
	defer s.mu.RUnlock()
	s.ids.Range(fn)
}

func (s *segmentVectorStore) PageVectorIDs(afterID string, limit int) []string {
	s.mu.Lock()
	s.ensureOrderedIDsLocked()
	s.mu.Unlock()
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.ids.PageAfter(afterID, limit)
}

func (s *segmentVectorStore) ensureOrderedIDsLocked() {
	if s.ids != nil {
		return
	}
	locations, err := s.locationSnapshotLocked()
	if err != nil {
		s.ids = newOrderedIDIndex()
		return
	}
	s.ids = newOrderedIDIndexFromMap(locations)
}

func (s *segmentVectorStore) IsPersistent() bool { return true }

func (s *segmentVectorStore) DiskStats() DiskStoreStats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	stats := DiskStoreStats{Records: uint64(s.locationCountLocked()), Segments: uint64(len(s.manifest.Segments)), CompactionBudgetBytes: s.compactBudgetBytes}
	stats.Compactions = s.compacts.Load()
	stats.BytesWritten = s.manifest.BytesWritten
	stats.CompactionActive = s.compacting.Load()
	for _, name := range s.manifest.Segments {
		if info, err := os.Stat(filepath.Join(s.basePath, name)); err == nil {
			stats.FileBytes += nonNegativeUint64(info.Size())
		}
	}
	if stats.FileBytes > 0 {
		stats.WriteAmplification = float64(stats.BytesWritten) / float64(stats.FileBytes)
	}
	return stats
}

func (s *segmentVectorStore) Close() error {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.stopOnce.Do(func() { close(s.stopCh) })
	s.workerWG.Wait()
	var locationErr error
	if s.diskLocations != nil {
		locationErr = s.diskLocations.Close()
	}
	if s.deltaWAL != nil {
		if err := s.deltaWAL.Close(); err != nil {
			return err
		}
	}
	return locationErr
}
