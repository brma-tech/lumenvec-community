package core

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// DeltaRecord is a replayable mutation captured during an online migration.
// Offset is strictly monotonic within one WAL file.
type DeltaRecord struct {
	Offset   uint64    `json:"offset"`
	VectorID string    `json:"vector_id"`
	Deleted  bool      `json:"deleted,omitempty"`
	Values   []float32 `json:"values,omitempty"`
}

func (r DeltaRecord) validate() error {
	if r.Offset == 0 || r.VectorID == "" {
		return errors.New("invalid delta record identity")
	}
	if r.Deleted && len(r.Values) != 0 {
		return errors.New("deleted delta cannot contain values")
	}
	if !r.Deleted && len(r.Values) == 0 {
		return errors.New("upsert delta requires values")
	}
	return nil
}

type DeltaWAL struct {
	mu               sync.Mutex
	path             string
	checkpointPath   string
	next             uint64
	closed           bool
	checkpointLoaded bool
	format           deltaWALFormat
	file             *os.File
	// appendPositions is a sparse in-memory index: one entry per appended
	// batch, not per record. Online migration normally captures an offset and
	// replays in the same process, so this avoids O(total WAL) scans while
	// keeping memory proportional to the number of fsync batches.
	appendPositions []deltaWALPosition
}

type deltaWALPosition struct {
	firstOffset uint64
	fileOffset  int64
}

const deltaWALTailHashBytes = 64 << 10
const deltaWALMaxRecordBytes = 64 << 20

var deltaWALBinaryMagic = [8]byte{'L', 'V', 'D', 'W', 'A', 'L', '0', '2'}
var deltaWALCRC32Table = crc32.MakeTable(crc32.Castagnoli)

type deltaWALFormat uint8

const (
	deltaWALLegacyJSON deltaWALFormat = iota + 1
	deltaWALBinaryV2
)

type deltaWALCheckpoint struct {
	Version    int    `json:"version"`
	Size       int64  `json:"size"`
	Next       uint64 `json:"next"`
	TailSHA256 string `json:"tail_sha256"`
	Format     string `json:"format,omitempty"`
}

func (w *DeltaWAL) CurrentOffset() uint64 {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.next
}

func OpenDeltaWAL(path string) (*DeltaWAL, error) {
	if path == "" {
		return nil, errors.New("WAL path is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o640)
	if err != nil {
		return nil, err
	}
	wal := &DeltaWAL{path: path, checkpointPath: path + ".offset", file: f}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	if info.Size() == 0 {
		if _, err := f.Write(deltaWALBinaryMagic[:]); err != nil {
			_ = f.Close()
			return nil, err
		}
		if err := f.Sync(); err != nil {
			_ = f.Close()
			return nil, err
		}
		wal.format = deltaWALBinaryV2
	} else {
		var magic [len(deltaWALBinaryMagic)]byte
		if _, err := f.ReadAt(magic[:], 0); err == nil && magic == deltaWALBinaryMagic {
			wal.format = deltaWALBinaryV2
		} else {
			wal.format = deltaWALLegacyJSON
		}
	}
	if wal.loadCheckpointLocked() {
		wal.checkpointLoaded = true
	} else {
		if err := wal.scanLocked(); err != nil {
			_ = f.Close()
			return nil, err
		}
		_ = wal.saveCheckpointLocked()
	}
	return wal, nil
}

func (w *DeltaWAL) scanLocked() error {
	if w.format == deltaWALBinaryV2 {
		return w.scanBinaryLocked(nil, 0, int64(len(deltaWALBinaryMagic)))
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	scanner := bufio.NewScanner(w.file)
	for scanner.Scan() {
		var record DeltaRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return fmt.Errorf("decode delta WAL: %w", err)
		}
		if err := record.validate(); err != nil {
			return err
		}
		if record.Offset > w.next {
			w.next = record.Offset
		}
	}
	return scanner.Err()
}

func (w *DeltaWAL) Append(record DeltaRecord) (uint64, error) {
	offsets, err := w.AppendBatch([]DeltaRecord{record})
	if err != nil {
		return 0, err
	}
	return offsets[0], nil
}

// AppendBatch appends and durably syncs a group of mutations with one fsync.
// Offsets are assigned while holding the WAL lock, so replay ordering remains
// identical to repeated Append calls while avoiding one disk barrier per
// vector during bulk ingest and resharding.
func (w *DeltaWAL) AppendBatch(records []DeltaRecord) ([]uint64, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil, errors.New("delta WAL is closed")
	}
	if len(records) == 0 {
		return []uint64{}, nil
	}
	var payload []byte
	offsets := make([]uint64, len(records))
	for i, record := range records {
		if record.VectorID == "" {
			return nil, errors.New("vector ID is required")
		}
		if record.Deleted && len(record.Values) != 0 {
			return nil, errors.New("deleted delta cannot contain values")
		}
		if !record.Deleted && len(record.Values) == 0 {
			return nil, errors.New("upsert delta requires values")
		}
		record.Offset = w.next + uint64(i) + 1
		if w.format == deltaWALBinaryV2 {
			data, err := marshalDeltaBinaryRecord(record)
			if err != nil {
				return nil, err
			}
			payload = append(payload, data...)
		} else {
			data, err := json.Marshal(record)
			if err != nil {
				return nil, err
			}
			payload = append(payload, data...)
			payload = append(payload, '\n')
		}
		offsets[i] = record.Offset
	}
	info, err := w.file.Stat()
	if err != nil {
		return nil, err
	}
	fileOffset := info.Size()
	if _, err := w.file.Write(payload); err != nil {
		return nil, err
	}
	if err := w.file.Sync(); err != nil {
		return nil, err
	}
	w.next = offsets[len(offsets)-1]
	w.appendPositions = append(w.appendPositions, deltaWALPosition{
		firstOffset: offsets[0],
		fileOffset:  fileOffset,
	})
	_ = w.saveCheckpointLocked()
	return offsets, nil
}

// ReplaySince replays records after offset. Duplicate offsets or vector IDs
// are delivered once, making retries from an earlier checkpoint idempotent.
func (w *DeltaWAL) ReplaySince(offset uint64, fn func(DeltaRecord) error) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return errors.New("delta WAL is closed")
	}
	if offset >= w.next {
		return nil
	}
	if w.format == deltaWALBinaryV2 {
		return w.scanBinaryLocked(fn, offset, w.replayStartLocked(offset))
	}
	if _, err := w.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	seen := make(map[string]struct{})
	scanner := bufio.NewScanner(w.file)
	for scanner.Scan() {
		var record DeltaRecord
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return err
		}
		if err := record.validate(); err != nil {
			return err
		}
		if record.Offset <= offset {
			continue
		}
		key := fmt.Sprintf("%d:%s", record.Offset, record.VectorID)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		if err := fn(record); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func (w *DeltaWAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	return w.file.Close()
}

func (w *DeltaWAL) loadCheckpointLocked() bool {
	data, err := os.ReadFile(w.checkpointPath)
	if err != nil {
		return false
	}
	var checkpoint deltaWALCheckpoint
	if json.Unmarshal(data, &checkpoint) != nil || checkpoint.Version != 1 || checkpoint.Size < 0 {
		return false
	}
	if checkpoint.Format != "" && checkpoint.Format != w.formatName() {
		return false
	}
	info, err := w.file.Stat()
	if err != nil || info.Size() != checkpoint.Size {
		return false
	}
	tail, err := w.tailHashLocked(info.Size())
	if err != nil || fmt.Sprintf("%x", tail[:]) != checkpoint.TailSHA256 {
		return false
	}
	w.next = checkpoint.Next
	return true
}

func (w *DeltaWAL) saveCheckpointLocked() error {
	info, err := w.file.Stat()
	if err != nil {
		return err
	}
	tail, err := w.tailHashLocked(info.Size())
	if err != nil {
		return err
	}
	data, err := json.Marshal(deltaWALCheckpoint{
		Version:    1,
		Size:       info.Size(),
		Next:       w.next,
		TailSHA256: fmt.Sprintf("%x", tail[:]),
		Format:     w.formatName(),
	})
	if err != nil {
		return err
	}
	tmp := w.checkpointPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o640); err != nil {
		return err
	}
	if err := os.Rename(tmp, w.checkpointPath); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func (w *DeltaWAL) formatName() string {
	if w.format == deltaWALBinaryV2 {
		return "binary-v2"
	}
	return "json-v1"
}

func marshalDeltaBinaryRecord(record DeltaRecord) ([]byte, error) {
	if err := record.validate(); err != nil {
		return nil, err
	}
	if uint64(len(record.VectorID)) > math.MaxUint32 || uint64(len(record.Values)) > math.MaxUint32 {
		return nil, errors.New("delta WAL record is too large")
	}
	bodyLength := 8 + 1 + 4 + 4 + len(record.VectorID) + len(record.Values)*4
	if bodyLength > deltaWALMaxRecordBytes {
		return nil, errors.New("delta WAL record exceeds size limit")
	}
	out := make([]byte, 4+bodyLength+4)
	binary.LittleEndian.PutUint32(out[0:4], uint32(bodyLength))
	body := out[4 : 4+bodyLength]
	binary.LittleEndian.PutUint64(body[0:8], record.Offset)
	if record.Deleted {
		body[8] = 1
	}
	binary.LittleEndian.PutUint32(body[9:13], uint32(len(record.VectorID)))
	binary.LittleEndian.PutUint32(body[13:17], uint32(len(record.Values)))
	cursor := 17
	copy(body[cursor:], record.VectorID)
	cursor += len(record.VectorID)
	for _, value := range record.Values {
		binary.LittleEndian.PutUint32(body[cursor:cursor+4], math.Float32bits(value))
		cursor += 4
	}
	binary.LittleEndian.PutUint32(out[4+bodyLength:], crc32.Checksum(body, deltaWALCRC32Table))
	return out, nil
}

func unmarshalDeltaBinaryRecord(body []byte) (DeltaRecord, error) {
	if len(body) < 17 {
		return DeltaRecord{}, errors.New("binary delta WAL record is truncated")
	}
	idLength := int(binary.LittleEndian.Uint32(body[9:13]))
	vectorLength := int(binary.LittleEndian.Uint32(body[13:17]))
	expected := 17 + idLength + vectorLength*4
	if idLength < 0 || vectorLength < 0 || expected != len(body) {
		return DeltaRecord{}, errors.New("binary delta WAL record has invalid lengths")
	}
	record := DeltaRecord{
		Offset:   binary.LittleEndian.Uint64(body[0:8]),
		Deleted:  body[8]&1 != 0,
		VectorID: string(body[17 : 17+idLength]),
	}
	if body[8]&^byte(1) != 0 {
		return DeltaRecord{}, errors.New("binary delta WAL record has invalid flags")
	}
	if vectorLength > 0 {
		record.Values = make([]float32, vectorLength)
		cursor := 17 + idLength
		for i := range record.Values {
			record.Values[i] = math.Float32frombits(binary.LittleEndian.Uint32(body[cursor : cursor+4]))
			cursor += 4
		}
	}
	if err := record.validate(); err != nil {
		return DeltaRecord{}, err
	}
	return record, nil
}

func (w *DeltaWAL) replayStartLocked(after uint64) int64 {
	start := int64(len(deltaWALBinaryMagic))
	// Search backwards because migration offsets are normally near the tail
	// and the number of entries equals append batches rather than records.
	for i := len(w.appendPositions) - 1; i >= 0; i-- {
		if w.appendPositions[i].firstOffset <= after+1 {
			return w.appendPositions[i].fileOffset
		}
	}
	return start
}

func (w *DeltaWAL) scanBinaryLocked(fn func(DeltaRecord) error, after uint64, offset int64) error {
	info, err := w.file.Stat()
	if err != nil {
		return err
	}
	size := info.Size()
	if size < offset {
		return errors.New("binary delta WAL header is truncated")
	}
	seen := make(map[string]struct{})
	var lengthBytes [4]byte
	for offset < size {
		if size-offset < 4 {
			return errors.New("binary delta WAL length is truncated")
		}
		if _, err := w.file.ReadAt(lengthBytes[:], offset); err != nil {
			return err
		}
		bodyLength := int64(binary.LittleEndian.Uint32(lengthBytes[:]))
		if bodyLength < 17 || bodyLength > deltaWALMaxRecordBytes || size-offset < 4+bodyLength+4 {
			return errors.New("binary delta WAL record length is invalid")
		}
		payload := make([]byte, bodyLength+4)
		if _, err := w.file.ReadAt(payload, offset+4); err != nil {
			return err
		}
		body := payload[:bodyLength]
		expectedCRC := binary.LittleEndian.Uint32(payload[bodyLength:])
		if crc32.Checksum(body, deltaWALCRC32Table) != expectedCRC {
			return errors.New("binary delta WAL checksum mismatch")
		}
		record, err := unmarshalDeltaBinaryRecord(body)
		if err != nil {
			return err
		}
		if record.Offset > w.next {
			w.next = record.Offset
		}
		if fn != nil && record.Offset > after {
			key := fmt.Sprintf("%d:%s", record.Offset, record.VectorID)
			if _, ok := seen[key]; !ok {
				seen[key] = struct{}{}
				if err := fn(record); err != nil {
					return err
				}
			}
		}
		offset += 4 + bodyLength + 4
	}
	return nil
}

func (w *DeltaWAL) tailHashLocked(size int64) ([sha256.Size]byte, error) {
	start := size - deltaWALTailHashBytes
	if start < 0 {
		start = 0
	}
	tail := make([]byte, int(size-start))
	if len(tail) > 0 {
		if _, err := w.file.ReadAt(tail, start); err != nil {
			return [sha256.Size]byte{}, err
		}
	}
	return sha256.Sum256(tail), nil
}
