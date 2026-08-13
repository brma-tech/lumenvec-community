package core

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"os"
	"sync"

	"github.com/cespare/xxhash/v2"
)

const (
	diskLocationHeaderSize = 64
	diskLocationSlotSize   = 56
	diskLocationVersion    = 1
	diskLocationEmpty      = byte(0)
	diskLocationLive       = byte(1)
	diskLocationTombstone  = byte(2)
)

var (
	diskLocationMagic    = [8]byte{'L', 'V', 'L', 'O', 'C', '0', '0', '1'}
	diskLocationCRC      = crc32.MakeTable(crc32.Castagnoli)
	ErrLocationIndexFull = errors.New("disk location index is full")
)

type diskLocationValue struct {
	SegmentGeneration uint64
	Compact           bool
	RecordOffset      int64
	RecordLength      uint32
}

type diskLocationSlot struct {
	state     byte
	hash      uint64
	keyOffset uint64
	keyLength uint32
	value     diskLocationValue
}

type diskLocationEntry struct {
	Key   string
	Value diskLocationValue
}

type diskKeyReference struct {
	Offset uint64
	Length uint32
}

// diskLocationIndex is a derived, fixed-capacity hash index. Slot metadata and
// complete keys live in files rather than the Go heap. Full key comparison
// makes collisions exact; a damaged/torn slot fails closed through CRC32C.
type diskLocationIndex struct {
	mu             sync.RWMutex
	table          *os.File
	tableMap       *mappedFile
	keys           *os.File
	keysMap        *mappedFile
	capacity       uint64
	count          uint64
	tombs          uint64
	keysSize       uint64
	closed         bool
	hashID         func(string) uint64
	pendingKeyBase uint64
	pendingKeys    []byte
}

func openDiskLocationIndex(tablePath, keysPath string, requestedCapacity uint64) (*diskLocationIndex, error) {
	if requestedCapacity < 8 {
		return nil, errors.New("disk location index capacity must be at least 8")
	}
	table, err := os.OpenFile(tablePath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	keys, err := os.OpenFile(keysPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		_ = table.Close()
		return nil, err
	}
	idx := &diskLocationIndex{table: table, keys: keys, hashID: xxhash.Sum64String}
	info, err := table.Stat()
	if err != nil {
		_ = idx.closeFiles()
		return nil, err
	}
	if info.Size() == 0 {
		idx.capacity = requestedCapacity
		tableSize := int64(diskLocationHeaderSize) + int64(requestedCapacity)*diskLocationSlotSize
		if tableSize <= 0 || uint64(tableSize) > uint64(^uint(0)>>1) {
			_ = idx.closeFiles()
			return nil, errors.New("disk location index exceeds address space")
		}
		if err := table.Truncate(tableSize); err != nil {
			_ = idx.closeFiles()
			return nil, err
		}
		idx.tableMap, err = mapReadWriteFile(table, int(tableSize))
		if err != nil {
			_ = idx.closeFiles()
			return nil, err
		}
		if err := idx.writeHeaderLocked(); err != nil {
			_ = idx.closeFiles()
			return nil, err
		}
		if err := idx.ensureKeysMappedLocked(1); err != nil {
			_ = idx.closeFiles()
			return nil, err
		}
		return idx, nil
	}
	if info.Size() <= 0 || uint64(info.Size()) > uint64(^uint(0)>>1) {
		_ = idx.closeFiles()
		return nil, errors.New("disk location index exceeds address space")
	}
	idx.tableMap, err = mapReadWriteFile(table, int(info.Size()))
	if err != nil {
		_ = idx.closeFiles()
		return nil, err
	}
	if err := idx.readHeaderLocked(); err != nil {
		_ = idx.closeFiles()
		return nil, err
	}
	if idx.capacity != requestedCapacity {
		_ = idx.closeFiles()
		return nil, fmt.Errorf("disk location index capacity mismatch: file=%d requested=%d", idx.capacity, requestedCapacity)
	}
	if info.Size() != diskLocationHeaderSize+int64(idx.capacity)*diskLocationSlotSize {
		_ = idx.closeFiles()
		return nil, errors.New("invalid disk location index file size")
	}
	keysInfo, err := keys.Stat()
	if err != nil || uint64(keysInfo.Size()) < idx.keysSize {
		_ = idx.closeFiles()
		return nil, errors.New("truncated disk location key log")
	}
	if err := idx.ensureKeysMappedLocked(max(idx.keysSize, 1)); err != nil {
		_ = idx.closeFiles()
		return nil, err
	}
	return idx, nil
}

func (d *diskLocationIndex) Put(key string, value diskLocationValue) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := d.putLocked(key, value); err != nil {
		return err
	}
	return d.writeHeaderLocked()
}

// PutBatch amortizes header persistence across a committed segment batch.
// Individual slots remain checksummed, so partial derived-index writes are
// detected and can be rebuilt from immutable segments.
func (d *diskLocationIndex) PutBatch(entries []diskLocationEntry) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(entries) == 0 {
		return nil
	}
	d.pendingKeyBase = d.keysSize
	d.pendingKeys = make([]byte, 0, len(entries)*16)
	for _, entry := range entries {
		if err := d.putLocked(entry.Key, entry.Value); err != nil {
			flushErr := d.flushPendingKeysLocked()
			_ = d.writeHeaderLocked()
			if flushErr != nil {
				return flushErr
			}
			return err
		}
	}
	if err := d.flushPendingKeysLocked(); err != nil {
		return err
	}
	return d.writeHeaderLocked()
}

func (d *diskLocationIndex) putLocked(key string, value diskLocationValue) error {
	if key == "" || uint64(len(key)) > uint64(^uint32(0)) || value.RecordOffset < 0 || value.RecordLength == 0 {
		return errors.New("invalid disk location entry")
	}
	if d.closed {
		return os.ErrClosed
	}
	hash := d.hashID(key)
	slotIndex, existing, err := d.findSlotLocked(key, hash, true)
	if err != nil {
		return err
	}
	if existing {
		slot, err := d.readSlotLocked(slotIndex)
		if err != nil {
			return err
		}
		slot.value = value
		return d.writeSlotLocked(slotIndex, slot)
	}
	keyOffset := d.keysSize
	if d.pendingKeys != nil {
		d.pendingKeys = append(d.pendingKeys, key...)
	} else {
		if err := d.ensureKeysMappedLocked(keyOffset + uint64(len(key))); err != nil {
			return err
		}
		copy(d.keysMap.data[keyOffset:keyOffset+uint64(len(key))], key)
	}
	d.keysSize += uint64(len(key))
	previous, err := d.readSlotLocked(slotIndex)
	if err != nil {
		return err
	}
	if previous.state == diskLocationTombstone {
		d.tombs--
	}
	if err := d.writeSlotLocked(slotIndex, diskLocationSlot{
		state: diskLocationLive, hash: hash, keyOffset: keyOffset,
		keyLength: uint32(len(key)), value: value,
	}); err != nil {
		return err
	}
	d.count++
	return nil
}

func (d *diskLocationIndex) Get(key string) (diskLocationValue, bool, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return diskLocationValue{}, false, os.ErrClosed
	}
	slotIndex, found, err := d.findSlotLocked(key, d.hashID(key), false)
	if err != nil || !found {
		return diskLocationValue{}, false, err
	}
	slot, err := d.readSlotLocked(slotIndex)
	return slot.value, err == nil, err
}

func (d *diskLocationIndex) KeyReference(key string) (diskKeyReference, bool, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return diskKeyReference{}, false, os.ErrClosed
	}
	slotIndex, found, err := d.findSlotLocked(key, d.hashID(key), false)
	if err != nil || !found {
		return diskKeyReference{}, false, err
	}
	slot, err := d.readSlotLocked(slotIndex)
	if err != nil {
		return diskKeyReference{}, false, err
	}
	return diskKeyReference{Offset: slot.keyOffset, Length: slot.keyLength}, true, nil
}

func (d *diskLocationIndex) ReadKeyReference(reference diskKeyReference) (string, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return "", os.ErrClosed
	}
	end := reference.Offset + uint64(reference.Length)
	if end < reference.Offset || end > d.keysSize || end > uint64(len(d.keysMap.data)) {
		return "", errors.New("disk location key reference is out of bounds")
	}
	return string(d.keysMap.data[reference.Offset:end]), nil
}

func (d *diskLocationIndex) Delete(key string) (bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return false, os.ErrClosed
	}
	slotIndex, found, err := d.findSlotLocked(key, d.hashID(key), false)
	if err != nil || !found {
		return false, err
	}
	slot, err := d.readSlotLocked(slotIndex)
	if err != nil {
		return false, err
	}
	slot.state = diskLocationTombstone
	if err := d.writeSlotLocked(slotIndex, slot); err != nil {
		return false, err
	}
	d.count--
	d.tombs++
	return true, d.writeHeaderLocked()
}

func (d *diskLocationIndex) Range(fn func(string, diskLocationValue) bool) error {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if d.closed {
		return os.ErrClosed
	}
	for index := uint64(0); index < d.capacity; index++ {
		slot, err := d.readSlotLocked(index)
		if err != nil {
			return err
		}
		if slot.state != diskLocationLive {
			continue
		}
		key, err := d.readKeyLocked(slot)
		if err != nil {
			return err
		}
		if !fn(key, slot.value) {
			return nil
		}
	}
	return nil
}

func (d *diskLocationIndex) Count() uint64 {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.count
}

func (d *diskLocationIndex) Reset() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return os.ErrClosed
	}
	clear(d.tableMap.data)
	if d.keysMap != nil {
		if err := d.keysMap.Close(); err != nil {
			return err
		}
		d.keysMap = nil
	}
	if err := d.keys.Truncate(0); err != nil {
		return err
	}
	if err := d.ensureKeysMappedLocked(1); err != nil {
		return err
	}
	d.count, d.tombs, d.keysSize = 0, 0, 0
	d.pendingKeys, d.pendingKeyBase = nil, 0
	return d.writeHeaderLocked()
}

func (d *diskLocationIndex) Sync() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return os.ErrClosed
	}
	if err := d.writeHeaderLocked(); err != nil {
		return err
	}
	if err := d.keysMap.Flush(); err != nil {
		return err
	}
	if err := d.keys.Sync(); err != nil {
		return err
	}
	if err := d.tableMap.Flush(); err != nil {
		return err
	}
	return d.table.Sync()
}

func (d *diskLocationIndex) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	if err := d.writeHeaderLocked(); err != nil {
		d.closed = true
		_ = d.closeFiles()
		return err
	}
	if err := d.keysMap.Flush(); err != nil {
		d.closed = true
		_ = d.closeFiles()
		return err
	}
	if err := d.keys.Sync(); err != nil {
		d.closed = true
		_ = d.closeFiles()
		return err
	}
	if err := d.tableMap.Flush(); err != nil {
		d.closed = true
		_ = d.closeFiles()
		return err
	}
	if err := d.table.Sync(); err != nil {
		d.closed = true
		_ = d.closeFiles()
		return err
	}
	d.closed = true
	return d.closeFiles()
}

func (d *diskLocationIndex) closeFiles() error {
	var mapErr error
	if d.keysMap != nil {
		mapErr = d.keysMap.Close()
		d.keysMap = nil
	}
	if d.tableMap != nil {
		if err := d.tableMap.Close(); mapErr == nil {
			mapErr = err
		}
		d.tableMap = nil
	}
	keysErr := d.keys.Close()
	tableErr := d.table.Close()
	if mapErr != nil {
		return mapErr
	}
	if keysErr != nil {
		return keysErr
	}
	return tableErr
}

func (d *diskLocationIndex) findSlotLocked(key string, hash uint64, forInsert bool) (uint64, bool, error) {
	firstTombstone := d.capacity
	for probe := uint64(0); probe < d.capacity; probe++ {
		index := (hash + probe) % d.capacity
		slot, err := d.readSlotLocked(index)
		if err != nil {
			return 0, false, err
		}
		switch slot.state {
		case diskLocationEmpty:
			if forInsert {
				if firstTombstone != d.capacity {
					return firstTombstone, false, nil
				}
				return index, false, nil
			}
			return 0, false, nil
		case diskLocationTombstone:
			if forInsert && firstTombstone == d.capacity {
				firstTombstone = index
			}
		case diskLocationLive:
			if slot.hash != hash || slot.keyLength != uint32(len(key)) {
				continue
			}
			equal, err := d.keyEqualsLocked(slot, key)
			if err != nil {
				return 0, false, err
			}
			if equal {
				return index, true, nil
			}
		default:
			return 0, false, errors.New("invalid disk location slot state")
		}
	}
	if forInsert && firstTombstone != d.capacity {
		return firstTombstone, false, nil
	}
	return 0, false, ErrLocationIndexFull
}

func (d *diskLocationIndex) readKeyLocked(slot diskLocationSlot) (string, error) {
	if uint64(slot.keyLength) > d.keysSize || slot.keyOffset > d.keysSize-uint64(slot.keyLength) {
		return "", errors.New("invalid disk location key bounds")
	}
	if d.pendingKeys != nil && slot.keyOffset >= d.pendingKeyBase {
		start := slot.keyOffset - d.pendingKeyBase
		if start > uint64(len(d.pendingKeys)) || uint64(slot.keyLength) > uint64(len(d.pendingKeys))-start {
			return "", errors.New("invalid pending disk location key bounds")
		}
		return string(d.pendingKeys[start : start+uint64(slot.keyLength)]), nil
	}
	if d.keysMap == nil || slot.keyOffset+uint64(slot.keyLength) > uint64(len(d.keysMap.data)) {
		return "", errors.New("disk location key exceeds mapping")
	}
	return string(d.keysMap.data[slot.keyOffset : slot.keyOffset+uint64(slot.keyLength)]), nil
}

func (d *diskLocationIndex) keyEqualsLocked(slot diskLocationSlot, key string) (bool, error) {
	if uint64(slot.keyLength) > d.keysSize || slot.keyOffset > d.keysSize-uint64(slot.keyLength) {
		return false, errors.New("invalid disk location key bounds")
	}
	if d.pendingKeys != nil && slot.keyOffset >= d.pendingKeyBase {
		start := slot.keyOffset - d.pendingKeyBase
		if start > uint64(len(d.pendingKeys)) || uint64(slot.keyLength) > uint64(len(d.pendingKeys))-start {
			return false, errors.New("invalid pending disk location key bounds")
		}
		return bytesEqualString(d.pendingKeys[start:start+uint64(slot.keyLength)], key), nil
	}
	if d.keysMap == nil || slot.keyOffset+uint64(slot.keyLength) > uint64(len(d.keysMap.data)) {
		return false, errors.New("disk location key exceeds mapping")
	}
	return bytesEqualString(d.keysMap.data[slot.keyOffset:slot.keyOffset+uint64(slot.keyLength)], key), nil
}

func bytesEqualString(data []byte, value string) bool {
	if len(data) != len(value) {
		return false
	}
	for i, b := range data {
		if b != value[i] {
			return false
		}
	}
	return true
}

func (d *diskLocationIndex) flushPendingKeysLocked() error {
	if d.pendingKeys == nil {
		return nil
	}
	pending := d.pendingKeys
	base := d.pendingKeyBase
	d.pendingKeys = nil
	d.pendingKeyBase = 0
	if len(pending) == 0 {
		return nil
	}
	if err := d.ensureKeysMappedLocked(base + uint64(len(pending))); err != nil {
		return err
	}
	copy(d.keysMap.data[base:base+uint64(len(pending))], pending)
	return nil
}

func (d *diskLocationIndex) ensureKeysMappedLocked(required uint64) error {
	if required == 0 {
		required = 1
	}
	if d.keysMap != nil && required <= uint64(len(d.keysMap.data)) {
		return nil
	}
	size := uint64(1 << 20)
	for size < required {
		if size > uint64(^uint(0)>>1)/2 {
			return errors.New("disk location key log exceeds address space")
		}
		size *= 2
	}
	if size > uint64(^uint(0)>>1) {
		return errors.New("disk location key log exceeds address space")
	}
	if d.keysMap != nil {
		if err := d.keysMap.Flush(); err != nil {
			return err
		}
		if err := d.keysMap.Close(); err != nil {
			return err
		}
		d.keysMap = nil
	}
	if err := d.keys.Truncate(int64(size)); err != nil {
		return err
	}
	mapped, err := mapReadWriteFile(d.keys, int(size))
	if err != nil {
		return err
	}
	d.keysMap = mapped
	return nil
}

func (d *diskLocationIndex) readSlotLocked(index uint64) (diskLocationSlot, error) {
	offset := diskLocationHeaderSize + int(index)*diskLocationSlotSize
	raw := d.tableMap.data[offset : offset+diskLocationSlotSize]
	if raw[0] == diskLocationEmpty {
		for _, value := range raw {
			if value != 0 {
				return diskLocationSlot{}, errors.New("invalid non-zero empty disk location slot")
			}
		}
		return diskLocationSlot{}, nil
	}
	if got, want := binary.LittleEndian.Uint32(raw[52:56]), crc32.Checksum(raw[:52], diskLocationCRC); got != want {
		return diskLocationSlot{}, errors.New("disk location slot checksum mismatch")
	}
	if raw[1] > 1 {
		return diskLocationSlot{}, errors.New("invalid disk location segment kind")
	}
	return diskLocationSlot{
		state: raw[0], hash: binary.LittleEndian.Uint64(raw[8:16]),
		keyOffset: binary.LittleEndian.Uint64(raw[16:24]),
		keyLength: binary.LittleEndian.Uint32(raw[24:28]),
		value: diskLocationValue{
			SegmentGeneration: binary.LittleEndian.Uint64(raw[32:40]),
			Compact:           raw[1] == 1,
			RecordOffset:      int64(binary.LittleEndian.Uint64(raw[40:48])),
			RecordLength:      binary.LittleEndian.Uint32(raw[48:52]),
		},
	}, nil
}

func (d *diskLocationIndex) writeSlotLocked(index uint64, slot diskLocationSlot) error {
	var raw [diskLocationSlotSize]byte
	raw[0] = slot.state
	if slot.value.Compact {
		raw[1] = 1
	}
	binary.LittleEndian.PutUint64(raw[8:16], slot.hash)
	binary.LittleEndian.PutUint64(raw[16:24], slot.keyOffset)
	binary.LittleEndian.PutUint32(raw[24:28], slot.keyLength)
	binary.LittleEndian.PutUint64(raw[32:40], slot.value.SegmentGeneration)
	binary.LittleEndian.PutUint64(raw[40:48], uint64(slot.value.RecordOffset))
	binary.LittleEndian.PutUint32(raw[48:52], slot.value.RecordLength)
	binary.LittleEndian.PutUint32(raw[52:56], crc32.Checksum(raw[:52], diskLocationCRC))
	offset := diskLocationHeaderSize + int(index)*diskLocationSlotSize
	copy(d.tableMap.data[offset:offset+diskLocationSlotSize], raw[:])
	return nil
}

func (d *diskLocationIndex) readHeaderLocked() error {
	var raw [diskLocationHeaderSize]byte
	copy(raw[:], d.tableMap.data[:diskLocationHeaderSize])
	if !bytes.Equal(raw[:8], diskLocationMagic[:]) ||
		binary.LittleEndian.Uint32(raw[8:12]) != diskLocationVersion ||
		binary.LittleEndian.Uint32(raw[12:16]) != diskLocationSlotSize {
		return errors.New("invalid disk location index header")
	}
	if got, want := binary.LittleEndian.Uint32(raw[48:52]), crc32.Checksum(raw[:48], diskLocationCRC); got != want {
		return errors.New("disk location header checksum mismatch")
	}
	d.capacity = binary.LittleEndian.Uint64(raw[16:24])
	d.count = binary.LittleEndian.Uint64(raw[24:32])
	d.tombs = binary.LittleEndian.Uint64(raw[32:40])
	d.keysSize = binary.LittleEndian.Uint64(raw[40:48])
	if d.capacity < 8 || d.count+d.tombs > d.capacity {
		return errors.New("invalid disk location index counters")
	}
	return nil
}

func (d *diskLocationIndex) writeHeaderLocked() error {
	var raw [diskLocationHeaderSize]byte
	copy(raw[:8], diskLocationMagic[:])
	binary.LittleEndian.PutUint32(raw[8:12], diskLocationVersion)
	binary.LittleEndian.PutUint32(raw[12:16], diskLocationSlotSize)
	binary.LittleEndian.PutUint64(raw[16:24], d.capacity)
	binary.LittleEndian.PutUint64(raw[24:32], d.count)
	binary.LittleEndian.PutUint64(raw[32:40], d.tombs)
	binary.LittleEndian.PutUint64(raw[40:48], d.keysSize)
	binary.LittleEndian.PutUint32(raw[48:52], crc32.Checksum(raw[:48], diskLocationCRC))
	copy(d.tableMap.data[:diskLocationHeaderSize], raw[:])
	return nil
}

var _ io.Closer = (*diskLocationIndex)(nil)
