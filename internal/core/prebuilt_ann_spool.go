package core

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"

	"lumenvec/internal/index/ann"
)

const maxPrebuiltRecordIDBytes = 16 << 20

// PrebuiltANNRecordSpool is the durable, rewindable record side of an ANN
// transfer. Its final path is deterministic so an applying journal can resume
// after process failure without asking the source to resend vectors.
type PrebuiltANNRecordSpool struct {
	path     string
	tmpPath  string
	file     *os.File
	manifest ann.SegmentManifest
	count    int
	sealed   bool
}

func NewPrebuiltANNRecordSpool(dir string, manifest ann.SegmentManifest) (*PrebuiltANNRecordSpool, error) {
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	file, err := os.CreateTemp(dir, ".prebuilt-records-*.tmp")
	if err != nil {
		return nil, err
	}
	return &PrebuiltANNRecordSpool{
		path: filepath.Join(dir, manifest.SegmentID+".records"), tmpPath: file.Name(),
		file: file, manifest: manifest,
	}, nil
}

func (s *PrebuiltANNRecordSpool) Append(record PrebuiltANNRecord) error {
	values := make([]float32, len(record.Values))
	for i, value := range record.Values {
		values[i] = float32(value)
	}
	return s.Append32(record.ID, record.InternalID, values)
}

func (s *PrebuiltANNRecordSpool) Append32(id string, internalID int, values []float32) error {
	if s == nil || s.file == nil || s.sealed {
		return errors.New("prebuilt record spool is closed")
	}
	if id == "" || len(id) > maxPrebuiltRecordIDBytes || internalID <= 0 || internalID > math.MaxInt32 || len(values) != s.manifest.Dimension {
		return errors.New("invalid prebuilt vector record")
	}
	var scalar [8]byte
	binary.LittleEndian.PutUint32(scalar[:4], uint32(len(id)))
	if _, err := s.file.Write(scalar[:4]); err != nil {
		return err
	}
	binary.LittleEndian.PutUint64(scalar[:8], uint64(internalID))
	if _, err := s.file.Write(scalar[:8]); err != nil {
		return err
	}
	if _, err := s.file.WriteString(id); err != nil {
		return err
	}
	for _, value := range values {
		binary.LittleEndian.PutUint32(scalar[:4], math.Float32bits(value))
		if _, err := s.file.Write(scalar[:4]); err != nil {
			return err
		}
	}
	s.count++
	return nil
}

func (s *PrebuiltANNRecordSpool) Seal() error {
	if s == nil || s.file == nil || s.sealed {
		return errors.New("prebuilt record spool is closed")
	}
	if err := s.file.Sync(); err != nil {
		s.Abort()
		return err
	}
	if err := s.file.Close(); err != nil {
		s.file = nil
		s.sealed = true
		_ = os.Remove(s.tmpPath)
		return err
	}
	s.file = nil
	s.sealed = true
	if _, err := os.Stat(s.path); err == nil {
		equal, compareErr := equalPrebuiltSpoolFiles(s.path, s.tmpPath)
		if compareErr != nil {
			_ = os.Remove(s.tmpPath)
			return compareErr
		}
		if !equal {
			_ = os.Remove(s.tmpPath)
			return errors.New("prebuilt record spool already exists with different content")
		}
		_ = os.Remove(s.tmpPath)
		return nil
	} else if !os.IsNotExist(err) {
		_ = os.Remove(s.tmpPath)
		return err
	}
	return os.Rename(s.tmpPath, s.path)
}

func equalPrebuiltSpoolFiles(left, right string) (bool, error) {
	digest := func(path string) ([sha256.Size]byte, error) {
		var out [sha256.Size]byte
		file, err := os.Open(path)
		if err != nil {
			return out, err
		}
		defer file.Close()
		hash := sha256.New()
		if _, err := io.Copy(hash, file); err != nil {
			return out, err
		}
		copy(out[:], hash.Sum(nil))
		return out, nil
	}
	leftHash, err := digest(left)
	if err != nil {
		return false, err
	}
	rightHash, err := digest(right)
	if err != nil {
		return false, err
	}
	return leftHash == rightHash, nil
}

func OpenPrebuiltANNRecordReplay(path string, manifest ann.SegmentManifest) (PrebuiltANNReplay, error) {
	if path == "" {
		return nil, errors.New("prebuilt record spool path is required")
	}
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	return func(visit func(PrebuiltANNRecord) error) error {
		file, err := os.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		reader := bufio.NewReaderSize(file, 256<<10)
		var scalar [8]byte
		for {
			if _, err := io.ReadFull(reader, scalar[:4]); errors.Is(err, io.EOF) {
				return nil
			} else if err != nil {
				return err
			}
			idBytes := binary.LittleEndian.Uint32(scalar[:4])
			if idBytes == 0 || idBytes > maxPrebuiltRecordIDBytes {
				return errors.New("invalid staged prebuilt record ID length")
			}
			if _, err := io.ReadFull(reader, scalar[:8]); err != nil {
				return err
			}
			internalID := binary.LittleEndian.Uint64(scalar[:8])
			if internalID == 0 || internalID > math.MaxInt32 {
				return errors.New("invalid staged prebuilt internal ID")
			}
			id := make([]byte, int(idBytes))
			if _, err := io.ReadFull(reader, id); err != nil {
				return err
			}
			values := make([]float64, manifest.Dimension)
			for i := range values {
				if _, err := io.ReadFull(reader, scalar[:4]); err != nil {
					return err
				}
				values[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(scalar[:4])))
			}
			if err := visit(PrebuiltANNRecord{ID: string(id), InternalID: int(internalID), Values: values}); err != nil {
				return err
			}
		}
	}, nil
}

func (s *PrebuiltANNRecordSpool) Replay(visit func(PrebuiltANNRecord) error) error {
	if s == nil || !s.sealed {
		return errors.New("prebuilt record spool is not sealed")
	}
	replay, err := OpenPrebuiltANNRecordReplay(s.path, s.manifest)
	if err != nil {
		return err
	}
	return replay(visit)
}

func (s *PrebuiltANNRecordSpool) Path() string { return s.path }
func (s *PrebuiltANNRecordSpool) Count() int   { return s.count }

func (s *PrebuiltANNRecordSpool) Abort() {
	if s == nil || s.sealed {
		return
	}
	if s.file != nil {
		_ = s.file.Close()
		s.file = nil
	}
	s.sealed = true
	_ = os.Remove(s.tmpPath)
}

func (s *PrebuiltANNRecordSpool) Discard() {
	if s == nil {
		return
	}
	s.Abort()
	_ = os.Remove(s.path)
}
