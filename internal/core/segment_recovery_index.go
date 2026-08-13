package core

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"time"
)

var segmentRecoveryIndexMagic = [8]byte{'L', 'V', 'L', 'O', 'C', '0', '0', '1'}

// SaveRecoveryIndex persists the current ID-to-segment offset map. It is a
// derived artifact: the segment manifest remains the commit point and a
// missing, stale, or corrupt index is rebuilt by scanning immutable segments.
func (s *segmentVectorStore) SaveRecoveryIndex() error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	segmentHashes := make([][sha256.Size]byte, len(s.manifest.Segments))
	segmentOrdinals := make(map[string]uint32, len(s.manifest.Segments))
	for i, name := range s.manifest.Segments {
		sum, err := hashFileSHA256(filepath.Join(s.basePath, name))
		if err != nil {
			return err
		}
		segmentHashes[i] = sum
		segmentOrdinals[name] = uint32(i) // #nosec G115 -- segment count is bounded by memory
	}

	tmp := s.recoveryPath + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, s.security.FileMode)
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
	h := sha256.New()
	buffered := bufio.NewWriterSize(io.MultiWriter(f, h), 1<<20)
	var scalar [8]byte
	write := func(data []byte) error {
		n, writeErr := buffered.Write(data)
		if writeErr == nil && n != len(data) {
			return io.ErrShortWrite
		}
		return writeErr
	}
	writeU32 := func(value uint32) error {
		binary.LittleEndian.PutUint32(scalar[:4], value)
		return write(scalar[:4])
	}
	writeU64 := func(value uint64) error {
		binary.LittleEndian.PutUint64(scalar[:], value)
		return write(scalar[:])
	}
	if err = write(segmentRecoveryIndexMagic[:]); err == nil {
		err = write(s.manifestHash[:])
	}
	if err == nil {
		err = writeU32(uint32(len(segmentHashes))) // #nosec G115 -- bounded by memory
	}
	for i := range segmentHashes {
		if err == nil {
			err = write(segmentHashes[i][:])
		}
	}
	if err == nil {
		err = writeU64(uint64(s.locationCountLocked()))
	}
	writeLocation := func(id string, location segmentLocation) bool {
		ordinal, ok := segmentOrdinals[location.segment]
		if !ok {
			err = io.ErrUnexpectedEOF
			return false
		}
		if err = writeU32(uint32(len(id))); err == nil { // #nosec G115 -- record IDs already use uint32
			err = write([]byte(id))
		}
		if err == nil {
			err = writeU32(ordinal)
		}
		if err == nil {
			err = writeU64(uint64(location.meta.recordOffset))
		}
		if err == nil {
			err = writeU32(location.meta.recordLength)
		}
		return err == nil
	}
	if err == nil {
		if s.diskLocations == nil {
			for id, location := range s.locations {
				if !writeLocation(id, location) {
					break
				}
			}
		} else {
			rangeErr := s.diskLocations.Range(func(id string, value diskLocationValue) bool {
				return writeLocation(id, diskLocationToSegmentLocation(value))
			})
			if err == nil {
				err = rangeErr
			}
		}
	}
	if err == nil {
		err = buffered.Flush()
	}
	if err == nil {
		_, err = f.Write(h.Sum(nil))
	}
	if err == nil {
		err = f.Sync()
	}
	if err = cleanup(err); err != nil {
		return err
	}
	return os.Rename(tmp, s.recoveryPath)
}

func (s *segmentVectorStore) loadRecoveryIndexLocked() bool {
	if len(s.manifest.Segments) == 0 {
		return false
	}
	started := time.Now()
	data, err := os.ReadFile(s.recoveryPath)
	if err != nil || len(data) < len(segmentRecoveryIndexMagic)+sha256.Size*2 {
		return false
	}
	startupTrace("segment-recovery-read", started)
	body := data[:len(data)-sha256.Size]
	checksum := data[len(data)-sha256.Size:]
	calculated := sha256.Sum256(body)
	if !bytes.Equal(checksum, calculated[:]) {
		return false
	}
	r := bytes.NewReader(body)
	var magic [8]byte
	var manifestHash [sha256.Size]byte
	if _, err = io.ReadFull(r, magic[:]); err != nil || magic != segmentRecoveryIndexMagic {
		return false
	}
	if _, err = io.ReadFull(r, manifestHash[:]); err != nil || manifestHash != s.manifestHash {
		return false
	}
	var segmentCount uint32
	if binary.Read(r, binary.LittleEndian, &segmentCount) != nil || int(segmentCount) != len(s.manifest.Segments) {
		return false
	}
	segmentSizes := make([]int64, segmentCount)
	for i, name := range s.manifest.Segments {
		var expected [sha256.Size]byte
		if _, err = io.ReadFull(r, expected[:]); err != nil {
			return false
		}
		path := filepath.Join(s.basePath, name)
		info, statErr := os.Stat(path)
		if statErr != nil {
			return false
		}
		segmentSizes[i] = info.Size()
	}
	var count uint64
	if binary.Read(r, binary.LittleEndian, &count) != nil || count > uint64(r.Len()/20) {
		return false
	}
	started = time.Now()
	locations := make(map[string]segmentLocation, int(count))
	for i := uint64(0); i < count; i++ {
		var idLength uint32
		if binary.Read(r, binary.LittleEndian, &idLength) != nil || uint64(idLength) > uint64(r.Len()) {
			return false
		}
		idBytes := make([]byte, int(idLength))
		if _, err = io.ReadFull(r, idBytes); err != nil {
			return false
		}
		var ordinal uint32
		var offset uint64
		var length uint32
		if binary.Read(r, binary.LittleEndian, &ordinal) != nil ||
			binary.Read(r, binary.LittleEndian, &offset) != nil ||
			binary.Read(r, binary.LittleEndian, &length) != nil ||
			int(ordinal) >= len(s.manifest.Segments) ||
			offset > uint64(^uint64(0))-uint64(length) ||
			offset+uint64(length) > uint64(segmentSizes[ordinal]) {
			return false
		}
		id := string(idBytes)
		if id == "" {
			return false
		}
		if _, exists := locations[id]; exists {
			return false
		}
		locations[id] = segmentLocation{
			segment: s.manifest.Segments[ordinal],
			meta: fileVectorRecordMeta{
				recordOffset: int64(offset), // #nosec G115 -- bounded by non-negative file size above
				recordLength: length,
			},
		}
	}
	if r.Len() != 0 {
		return false
	}
	if s.locationReplaceLocked(locations) != nil {
		return false
	}
	s.ids = nil
	s.recoveryLoaded = true
	startupTrace("segment-recovery-map", started)
	return true
}

func (s *segmentVectorStore) RecoveryIndexLoaded() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.recoveryLoaded
}

// RebuildRecoveryIndex discards the derived offset map and reconstructs it
// from canonical immutable segments. It is the fail-closed fallback when ANN
// fingerprint validation detects a stale or damaged recovery sidecar.
func (s *segmentVectorStore) RebuildRecoveryIndex() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.locationReplaceLocked(make(map[string]segmentLocation)); err != nil {
		return err
	}
	s.ids = nil
	s.recoveryLoaded = false
	for _, name := range s.manifest.Segments {
		if err := s.applySegmentLocked(name); err != nil {
			return err
		}
	}
	return s.persistDiskLocationMarkerLocked()
}

func hashFileSHA256(path string) ([sha256.Size]byte, error) {
	var result [sha256.Size]byte
	f, err := os.Open(path)
	if err != nil {
		return result, err
	}
	h := sha256.New()
	_, copyErr := io.Copy(h, f)
	closeErr := f.Close()
	if copyErr != nil {
		return result, copyErr
	}
	if closeErr != nil {
		return result, closeErr
	}
	copy(result[:], h.Sum(nil))
	return result, nil
}
