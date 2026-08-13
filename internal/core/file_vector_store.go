package core

import (
	"bufio"
	"container/list"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"

	"lumenvec/internal/index"
	"lumenvec/internal/vector"
)

const (
	fileVectorStoreDataFile               = "vectors.dat"
	fileVectorStoreOpPutFloat64           = byte(1)
	fileVectorStoreOpDelete               = byte(2)
	fileVectorStoreOpPut                  = byte(3)
	fileVectorStoreCompactMinStaleRecords = 1
	fileVectorStoreCompactStaleDivisor    = 2
	fileVectorStoreHotCacheMaxItems       = 1024
)

type fileVectorStore struct {
	basePath  string
	dataPath  string
	security  StorageSecurityOptions
	syncEvery int

	mu       sync.RWMutex
	file     *os.File
	offsets  map[string]fileVectorRecordMeta
	ids      *orderedIDIndex
	writes   int64
	stale    int64
	compacts int64
	pending  int64

	hotMu       sync.Mutex
	hotCache    map[string]*list.Element
	hotOrder    *list.List
	hotMaxItems int
}

type fileVectorRecordMeta struct {
	recordOffset int64
	recordLength uint32
	deleted      bool
}

func newFileVectorStore(basePath string) *fileVectorStore {
	return newFileVectorStoreWithSecurity(basePath, DefaultStorageSecurityOptions())
}

func newFileVectorStoreWithSecurity(basePath string, security StorageSecurityOptions) *fileVectorStore {
	return newFileVectorStoreWithOptions(basePath, security, 1)
}

func newFileVectorStoreWithOptions(basePath string, security StorageSecurityOptions, syncEvery int) *fileVectorStore {
	store := &fileVectorStore{
		basePath:    basePath,
		dataPath:    filepath.Join(basePath, fileVectorStoreDataFile),
		security:    normalizeStorageSecurityOptions(security),
		syncEvery:   normalizeSyncEvery(syncEvery),
		offsets:     make(map[string]fileVectorRecordMeta),
		ids:         newOrderedIDIndex(),
		hotCache:    make(map[string]*list.Element),
		hotOrder:    list.New(),
		hotMaxItems: fileVectorStoreHotCacheMaxItems,
	}
	if err := store.open(); err != nil {
		panic(err)
	}
	return store
}

func (s *fileVectorStore) UpsertVector(vec index.Vector) error {
	return s.UpsertVectors([]index.Vector{vec})
}

func (s *fileVectorStore) UpsertVectors(vectors []index.Vector) error {
	if len(vectors) == 0 {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.ensureOpenLocked(); err != nil {
		return err
	}

	type pendingRecord struct {
		vec    index.Vector
		record []byte
		offset int64
	}
	records := make([]pendingRecord, 0, len(vectors))
	totalBytes := int64(0)
	for _, vec := range vectors {
		record, err := encodeFileVectorRecord(fileVectorStoreOpPut, vec.ID, vec.Values)
		if err != nil {
			return err
		}
		records = append(records, pendingRecord{
			vec:    index.Vector{ID: vec.ID, Values: cloneVectorValues(vec.Values)},
			record: record,
		})
		totalBytes += int64(len(record))
	}

	offset, err := s.file.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}
	buf := make([]byte, 0, totalBytes)
	for i := range records {
		records[i].offset = offset + int64(len(buf))
		buf = append(buf, records[i].record...)
	}
	if _, err := s.file.Write(buf); err != nil {
		return err
	}
	for _, pending := range records {
		if _, exists := s.offsets[pending.vec.ID]; exists {
			s.stale++
		} else {
			s.orderedIDsLocked().Upsert(pending.vec.ID)
		}
		s.offsets[pending.vec.ID] = fileVectorRecordMeta{
			recordOffset: pending.offset,
			recordLength: boundedRecordLength(len(pending.record)),
		}
		s.writes++
		s.putHotVector(pending.vec.ID, vector.ToFloat32(pending.vec.Values))
	}
	s.pending += int64(len(records))
	if err := s.syncPendingLocked(false); err != nil {
		return err
	}
	return s.maybeCompactLocked()
}

func (s *fileVectorStore) GetVector(id string) (index.Vector, error) {
	vec, err := s.GetVectorReadOnly(id)
	if err != nil {
		return index.Vector{}, err
	}
	return index.Vector{ID: vec.ID, Values: cloneVectorValues(vec.Values)}, nil
}

func (s *fileVectorStore) GetVectorReadOnly(id string) (index.Vector, error) {
	record, err := s.readVectorRecord(id)
	if err != nil {
		return index.Vector{}, err
	}
	return index.Vector{ID: record.id, Values: record.values}, nil
}

func (s *fileVectorStore) GetVectorReadOnly32(id string) ([]float32, error) {
	record, err := s.readVectorRecord32(id)
	if err != nil {
		return nil, err
	}
	return record.values, nil
}

func (s *fileVectorStore) readVectorRecord(id string) (fileVectorRecord, error) {
	s.mu.RLock()
	meta, ok := s.offsets[id]
	file := s.file
	s.mu.RUnlock()

	if !ok || meta.deleted {
		return fileVectorRecord{}, index.ErrVectorNotFound
	}
	if file == nil {
		return fileVectorRecord{}, os.ErrClosed
	}

	record, err := readFileVectorRecordAt(file, meta)
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return fileVectorRecord{}, index.ErrVectorNotFound
		}
		return fileVectorRecord{}, err
	}
	if record.op == fileVectorStoreOpDelete {
		return fileVectorRecord{}, index.ErrVectorNotFound
	}
	return record, nil
}

func (s *fileVectorStore) readVectorRecord32(id string) (fileVectorRecord32, error) {
	s.mu.RLock()
	meta, ok := s.offsets[id]
	file := s.file
	s.mu.RUnlock()

	if !ok || meta.deleted {
		return fileVectorRecord32{}, index.ErrVectorNotFound
	}
	if file == nil {
		return fileVectorRecord32{}, os.ErrClosed
	}

	if values, ok := s.getHotVector(id); ok {
		return fileVectorRecord32{op: fileVectorStoreOpPut, id: id, values: values}, nil
	}

	record, err := readFileVectorRecord32At(file, meta)
	if err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return fileVectorRecord32{}, index.ErrVectorNotFound
		}
		return fileVectorRecord32{}, err
	}
	if record.op == fileVectorStoreOpDelete {
		return fileVectorRecord32{}, index.ErrVectorNotFound
	}
	s.putHotVector(record.id, record.values)
	return record, nil
}

func (s *fileVectorStore) DeleteVector(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	meta, ok := s.offsets[id]
	if !ok || meta.deleted {
		return index.ErrVectorNotFound
	}
	if err := s.ensureOpenLocked(); err != nil {
		return err
	}
	record, err := encodeFileVectorRecord(fileVectorStoreOpDelete, id, nil)
	if err != nil {
		return err
	}
	if _, err := s.file.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	if _, err := s.file.Write(record); err != nil {
		return err
	}
	delete(s.offsets, id)
	s.orderedIDsLocked().Delete(id)
	s.deleteHotVector(id)
	s.writes++
	s.stale++
	s.pending++
	if err := s.syncPendingLocked(false); err != nil {
		return err
	}
	return s.maybeCompactLocked()
}

func (s *fileVectorStore) ListVectors() []index.Vector {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.file == nil {
		return nil
	}

	out := make([]index.Vector, 0, len(s.offsets))
	for id, meta := range s.offsets {
		if meta.deleted {
			continue
		}
		record, err := readFileVectorRecordAt(s.file, meta)
		if err != nil || record.op == fileVectorStoreOpDelete {
			continue
		}
		out = append(out, index.Vector{ID: id, Values: cloneVectorValues(record.values)})
	}
	return out
}

func (s *fileVectorStore) RangeVectorIDs(fn func(string) bool) {
	s.ensureOrderedIDs()
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.file == nil {
		return
	}
	s.ids.Range(fn)
}

func (s *fileVectorStore) PageVectorIDs(afterID string, limit int) []string {
	s.ensureOrderedIDs()
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.file == nil {
		return nil
	}
	return s.ids.PageAfter(afterID, limit)
}

func (s *fileVectorStore) IsPersistent() bool {
	return true
}

func (s *fileVectorStore) DiskStats() DiskStoreStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	stats := DiskStoreStats{
		Records:      uint64(len(s.offsets)),
		StaleRecords: nonNegativeUint64(s.stale),
		Compactions:  nonNegativeUint64(s.compacts),
	}
	if info, err := os.Stat(s.dataPath); err == nil {
		stats.FileBytes = nonNegativeUint64(info.Size())
	}
	return stats
}

func nonNegativeUint64(value int64) uint64 {
	if value <= 0 {
		return 0
	}
	return uint64(value) // #nosec G115 -- value is proven positive and int64 fits uint64
}

func (s *fileVectorStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.file == nil {
		return nil
	}
	if err := s.syncPendingLocked(true); err != nil {
		return err
	}
	err := s.file.Close()
	s.file = nil
	s.clearHotVectors()
	return err
}

func (s *fileVectorStore) open() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.openLocked()
}

func (s *fileVectorStore) openLocked() error {
	if err := os.MkdirAll(s.basePath, s.security.DirMode); err != nil {
		return err
	}
	file, err := os.OpenFile(s.dataPath, os.O_CREATE|os.O_RDWR, s.security.FileMode)
	if err != nil {
		return err
	}
	offsets, indexStats, err := rebuildFileVectorStoreIndex(file)
	if err != nil {
		_ = file.Close()
		return err
	}
	s.file = file
	s.offsets = offsets
	s.ids = newOrderedIDIndexFromFileOffsets(offsets)
	s.writes = indexStats.totalRecords
	s.stale = indexStats.staleRecords
	s.clearHotVectors()
	return s.maybeCompactLocked()
}

func (s *fileVectorStore) ensureOpenLocked() error {
	if s.file != nil {
		return nil
	}
	return s.openLocked()
}

func (s *fileVectorStore) maybeCompactLocked() error {
	if s.stale < fileVectorStoreCompactMinStaleRecords {
		return nil
	}
	if s.stale*fileVectorStoreCompactStaleDivisor < s.writes {
		return nil
	}
	return s.compactLocked()
}

func (s *fileVectorStore) compactLocked() error {
	if s.file == nil {
		return nil
	}
	if err := s.syncPendingLocked(true); err != nil {
		return err
	}

	live := make([]index.Vector, 0, len(s.offsets))
	for id, meta := range s.offsets {
		record, err := readFileVectorRecordAt(s.file, meta)
		if err != nil || record.op == fileVectorStoreOpDelete {
			continue
		}
		live = append(live, index.Vector{ID: id, Values: cloneVectorValues(record.values)})
	}

	tmpPath := s.dataPath + ".compact"
	tmpFile, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_RDWR|os.O_TRUNC, s.security.FileMode)
	if err != nil {
		return err
	}

	newOffsets := make(map[string]fileVectorRecordMeta, len(live))
	newIDs := newOrderedIDIndex()
	var offset int64
	for _, vec := range live {
		record, err := encodeFileVectorRecord(fileVectorStoreOpPut, vec.ID, vec.Values)
		if err != nil {
			_ = tmpFile.Close()
			_ = os.Remove(tmpPath)
			return err
		}
		if _, err := tmpFile.Write(record); err != nil {
			_ = tmpFile.Close()
			_ = os.Remove(tmpPath)
			return err
		}
		newOffsets[vec.ID] = fileVectorRecordMeta{
			recordOffset: offset,
			recordLength: boundedRecordLength(len(record)),
		}
		newIDs.Upsert(vec.ID)
		offset += int64(len(record))
	}
	if err := tmpFile.Sync(); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := s.file.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, s.dataPath); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}

	file, err := os.OpenFile(s.dataPath, os.O_CREATE|os.O_RDWR, s.security.FileMode)
	if err != nil {
		return err
	}
	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		_ = file.Close()
		return err
	}

	s.file = file
	s.offsets = newOffsets
	s.ids = newIDs
	s.writes = int64(len(newOffsets))
	s.stale = 0
	s.pending = 0
	s.compacts++
	s.clearHotVectors()
	return nil
}

func (s *fileVectorStore) syncPendingLocked(force bool) error {
	if s.file == nil || s.pending == 0 {
		return nil
	}
	if !force && s.pending < int64(s.syncEvery) {
		return nil
	}
	if err := s.file.Sync(); err != nil {
		return err
	}
	s.pending = 0
	return nil
}

func (s *fileVectorStore) orderedIDsLocked() *orderedIDIndex {
	if s.ids == nil {
		s.ids = newOrderedIDIndexFromFileOffsets(s.offsets)
	}
	return s.ids
}

func (s *fileVectorStore) ensureOrderedIDs() {
	s.mu.RLock()
	ready := s.ids != nil
	s.mu.RUnlock()
	if ready {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = s.orderedIDsLocked()
}

type fileVectorHotEntry struct {
	id     string
	values []float32
}

func (s *fileVectorStore) getHotVector(id string) ([]float32, bool) {
	s.hotMu.Lock()
	defer s.hotMu.Unlock()

	elem, ok := s.hotCache[id]
	if !ok {
		return nil, false
	}
	s.hotOrder.MoveToFront(elem)
	entry := elem.Value.(fileVectorHotEntry)
	return entry.values, true
}

func (s *fileVectorStore) putHotVector(id string, values []float32) {
	s.hotMu.Lock()
	defer s.hotMu.Unlock()

	s.ensureHotCacheLocked()
	if s.hotMaxItems <= 0 {
		return
	}
	if elem, ok := s.hotCache[id]; ok {
		elem.Value = fileVectorHotEntry{id: id, values: values}
		s.hotOrder.MoveToFront(elem)
		return
	}
	elem := s.hotOrder.PushFront(fileVectorHotEntry{id: id, values: values})
	s.hotCache[id] = elem
	for len(s.hotCache) > s.hotMaxItems {
		s.evictOldestHotVectorLocked()
	}
}

func (s *fileVectorStore) deleteHotVector(id string) {
	s.hotMu.Lock()
	defer s.hotMu.Unlock()

	if s.hotCache == nil {
		return
	}
	if elem, ok := s.hotCache[id]; ok {
		s.hotOrder.Remove(elem)
		delete(s.hotCache, id)
	}
}

func (s *fileVectorStore) clearHotVectors() {
	s.hotMu.Lock()
	defer s.hotMu.Unlock()

	if s.hotCache == nil {
		return
	}
	if s.hotOrder == nil {
		s.hotOrder = list.New()
	}
	clear(s.hotCache)
	s.hotOrder.Init()
}

func (s *fileVectorStore) ensureHotCacheLocked() {
	if s.hotCache == nil {
		s.hotCache = make(map[string]*list.Element)
	}
	if s.hotOrder == nil {
		s.hotOrder = list.New()
	}
	if s.hotMaxItems == 0 {
		s.hotMaxItems = fileVectorStoreHotCacheMaxItems
	}
}

func (s *fileVectorStore) evictOldestHotVectorLocked() {
	elem := s.hotOrder.Back()
	if elem == nil {
		return
	}
	entry := elem.Value.(fileVectorHotEntry)
	delete(s.hotCache, entry.id)
	s.hotOrder.Remove(elem)
}

func newOrderedIDIndexFromFileOffsets(offsets map[string]fileVectorRecordMeta) *orderedIDIndex {
	idx := newOrderedIDIndex()
	for id, meta := range offsets {
		if meta.deleted {
			continue
		}
		idx.Upsert(id)
	}
	return idx
}

type fileVectorRecord struct {
	op     byte
	id     string
	values []float64
}

type fileVectorRecord32 struct {
	op     byte
	id     string
	values []float32
}

func encodeFileVectorRecord(op byte, id string, values []float64) ([]byte, error) {
	idBytes := []byte(id)
	valueBytes, err := fileVectorRecordValueBytes(op)
	if err != nil {
		return nil, err
	}
	const maxUint32 = uint64(^uint32(0))
	if boundedLengthU64(len(idBytes)) > maxUint32 || boundedLengthU64(len(values)) > maxUint32 {
		return nil, errors.New("vector record field exceeds uint32 wire limit")
	}
	baseLength := uint64(9) + boundedLengthU64(len(idBytes))
	if baseLength > maxUint32 {
		return nil, errors.New("vector record exceeds uint32 wire limit")
	}
	if boundedLengthU64(len(values)) > (maxUint32-baseLength)/boundedLengthU64(valueBytes) {
		return nil, errors.New("vector record exceeds uint32 wire limit")
	}
	recordLength := 1 + 4 + len(idBytes) + 4 + len(values)*valueBytes
	if boundedLengthU64(recordLength) > maxUint32 {
		return nil, errors.New("vector record exceeds uint32 wire limit")
	}
	buf := make([]byte, 4+recordLength)
	binary.LittleEndian.PutUint32(buf[:4], boundedRecordLength(recordLength))
	pos := 4
	buf[pos] = op
	pos++
	binary.LittleEndian.PutUint32(buf[pos:pos+4], boundedRecordLength(len(idBytes)))
	pos += 4
	copy(buf[pos:pos+len(idBytes)], idBytes)
	pos += len(idBytes)
	binary.LittleEndian.PutUint32(buf[pos:pos+4], boundedRecordLength(len(values)))
	pos += 4
	for _, value := range values {
		if op == fileVectorStoreOpPutFloat64 {
			binary.LittleEndian.PutUint64(buf[pos:pos+8], math.Float64bits(value))
			pos += 8
			continue
		}
		binary.LittleEndian.PutUint32(buf[pos:pos+4], math.Float32bits(float32(value)))
		pos += 4
	}
	return buf, nil
}

func encodeFileVectorRecord32(op byte, id string, values []float32) ([]byte, error) {
	if op != fileVectorStoreOpPut {
		return encodeFileVectorRecord(op, id, nil)
	}
	idBytes := []byte(id)
	const maxUint32 = uint64(^uint32(0))
	if uint64(len(idBytes)) > maxUint32 || uint64(len(values)) > maxUint32 {
		return nil, errors.New("vector record field exceeds uint32 wire limit")
	}
	recordLength := 1 + 4 + len(idBytes) + 4 + len(values)*4
	if uint64(recordLength) > maxUint32 {
		return nil, errors.New("vector record exceeds uint32 wire limit")
	}
	buf := make([]byte, 4+recordLength)
	binary.LittleEndian.PutUint32(buf[:4], uint32(recordLength))
	pos := 4
	buf[pos] = op
	pos++
	binary.LittleEndian.PutUint32(buf[pos:pos+4], uint32(len(idBytes)))
	pos += 4
	copy(buf[pos:], idBytes)
	pos += len(idBytes)
	binary.LittleEndian.PutUint32(buf[pos:pos+4], uint32(len(values)))
	pos += 4
	for _, value := range values {
		binary.LittleEndian.PutUint32(buf[pos:pos+4], math.Float32bits(value))
		pos += 4
	}
	return buf, nil
}

func boundedRecordLength(length int) uint32 {
	// encodeFileVectorRecord rejects lengths above the uint32 wire limit.
	return uint32(length) // #nosec G115 -- validated by the encoder before storage
}

func boundedLengthU64(length int) uint64 {
	return uint64(length) // #nosec G115 -- lengths are non-negative slice sizes
}

func fileVectorRecordValueBytes(op byte) (int, error) {
	switch op {
	case fileVectorStoreOpPut:
		return 4, nil
	case fileVectorStoreOpPutFloat64, fileVectorStoreOpDelete:
		return 8, nil
	default:
		return 0, errors.New("unknown vector record op")
	}
}

type fileVectorStoreIndexStats struct {
	totalRecords int64
	staleRecords int64
}

func rebuildFileVectorStoreIndex(file *os.File) (map[string]fileVectorRecordMeta, fileVectorStoreIndexStats, error) {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, fileVectorStoreIndexStats{}, err
	}
	reader := bufio.NewReader(file)
	offsets := make(map[string]fileVectorRecordMeta)
	stats := fileVectorStoreIndexStats{}
	var offset int64

	for {
		recordOffset := offset
		lengthBuf := make([]byte, 4)
		_, err := io.ReadFull(reader, lengthBuf)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fileVectorStoreIndexStats{}, err
		}
		recordLength := binary.LittleEndian.Uint32(lengthBuf)
		offset += 4

		payload := make([]byte, recordLength)
		if _, err := io.ReadFull(reader, payload); err != nil {
			return nil, fileVectorStoreIndexStats{}, err
		}
		offset += int64(recordLength)
		stats.totalRecords++

		record, err := decodeFileVectorPayload(payload)
		if err != nil {
			return nil, fileVectorStoreIndexStats{}, err
		}
		if record.op == fileVectorStoreOpDelete {
			stats.staleRecords++
			delete(offsets, record.id)
			continue
		}
		if _, exists := offsets[record.id]; exists {
			stats.staleRecords++
		}
		offsets[record.id] = fileVectorRecordMeta{
			recordOffset: recordOffset,
			recordLength: 4 + recordLength,
		}
	}

	if _, err := file.Seek(0, io.SeekEnd); err != nil {
		return nil, fileVectorStoreIndexStats{}, err
	}
	return offsets, stats, nil
}

func readFileVectorRecordAt(file *os.File, meta fileVectorRecordMeta) (fileVectorRecord, error) {
	buf := make([]byte, meta.recordLength)
	if _, err := file.ReadAt(buf, meta.recordOffset); err != nil {
		return fileVectorRecord{}, err
	}
	recordLength := binary.LittleEndian.Uint32(buf[:4])
	if uint64(len(buf)) < 4+uint64(recordLength) {
		return fileVectorRecord{}, io.ErrUnexpectedEOF
	}
	return decodeFileVectorPayload(buf[4 : 4+recordLength])
}

func readFileVectorRecord32At(file *os.File, meta fileVectorRecordMeta) (fileVectorRecord32, error) {
	buf := make([]byte, meta.recordLength)
	return readFileVectorRecord32AtBuffer(file, meta, buf)
}

func readFileVectorRecord32AtBuffer(file *os.File, meta fileVectorRecordMeta, buf []byte) (fileVectorRecord32, error) {
	if len(buf) < int(meta.recordLength) {
		return fileVectorRecord32{}, io.ErrShortBuffer
	}
	buf = buf[:meta.recordLength]
	if _, err := file.ReadAt(buf, meta.recordOffset); err != nil {
		return fileVectorRecord32{}, err
	}
	recordLength := binary.LittleEndian.Uint32(buf[:4])
	if uint64(len(buf)) < 4+uint64(recordLength) {
		return fileVectorRecord32{}, io.ErrUnexpectedEOF
	}
	return decodeFileVectorPayload32(buf[4 : 4+recordLength])
}

func readFileVectorRecord32AtBufferScratch(file *os.File, meta fileVectorRecordMeta, buf []byte, scratch []float32) (fileVectorRecord32, error) {
	if len(buf) < int(meta.recordLength) {
		return fileVectorRecord32{}, io.ErrShortBuffer
	}
	buf = buf[:meta.recordLength]
	if _, err := file.ReadAt(buf, meta.recordOffset); err != nil {
		return fileVectorRecord32{}, err
	}
	recordLength := binary.LittleEndian.Uint32(buf[:4])
	if uint64(len(buf)) < 4+uint64(recordLength) {
		return fileVectorRecord32{}, io.ErrUnexpectedEOF
	}
	return decodeFileVectorValues32Into(buf[4:4+recordLength], scratch)
}

func decodeFileVectorPayload(payload []byte) (fileVectorRecord, error) {
	record, err := decodeFileVectorPayload32(payload)
	if err != nil {
		return fileVectorRecord{}, err
	}
	return fileVectorRecord{
		op:     record.op,
		id:     record.id,
		values: vector.ToFloat64(record.values),
	}, nil
}

func decodeFileVectorPayload32(payload []byte) (fileVectorRecord32, error) {
	return decodeFileVectorPayload32Into(payload, nil)
}

func decodeFileVectorPayload32Into(payload []byte, scratch []float32) (fileVectorRecord32, error) {
	return decodeFileVectorPayload32Options(payload, scratch, true)
}

func decodeFileVectorValues32Into(payload []byte, scratch []float32) (fileVectorRecord32, error) {
	return decodeFileVectorPayload32Options(payload, scratch, false)
}

func decodeFileVectorPayload32Options(payload []byte, scratch []float32, includeID bool) (fileVectorRecord32, error) {
	if len(payload) < 9 {
		return fileVectorRecord32{}, io.ErrUnexpectedEOF
	}
	pos := 0
	record := fileVectorRecord32{op: payload[pos]}
	pos++

	idLen := int(binary.LittleEndian.Uint32(payload[pos : pos+4]))
	pos += 4
	if len(payload[pos:]) < idLen+4 {
		return fileVectorRecord32{}, io.ErrUnexpectedEOF
	}
	if includeID {
		record.id = string(payload[pos : pos+idLen])
	}
	pos += idLen

	valuesLen := int(binary.LittleEndian.Uint32(payload[pos : pos+4]))
	pos += 4
	valueBytes, err := fileVectorRecordValueBytes(record.op)
	if err != nil {
		return fileVectorRecord32{}, err
	}
	if len(payload[pos:]) < valuesLen*valueBytes {
		return fileVectorRecord32{}, io.ErrUnexpectedEOF
	}
	if cap(scratch) < valuesLen {
		scratch = make([]float32, valuesLen)
	} else {
		scratch = scratch[:valuesLen]
	}
	record.values = scratch
	for i := 0; i < valuesLen; i++ {
		if record.op == fileVectorStoreOpPutFloat64 {
			record.values[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(payload[pos : pos+8])))
			pos += 8
			continue
		}
		record.values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[pos : pos+4]))
		pos += 4
	}
	return record, nil
}
