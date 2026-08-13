package ann

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"hash"
	"io"
	"math"
	"os"
	"path/filepath"
)

var segmentPayloadMagic = [4]byte{'L', 'V', 'S', '1'}

const segmentPayloadVersion uint16 = 1

type SegmentPayloadWriter struct {
	w      *bufio.Writer
	hash   hash.Hash
	count  int
	dim    int
	closed bool
	bytes  int64
}

type SegmentPayloadHeader struct {
	Count     uint64
	Dimension uint32
}

func NewSegmentPayloadWriter(w io.Writer, dimension int) (*SegmentPayloadWriter, error) {
	if w == nil || dimension <= 0 {
		return nil, errors.New("payload writer requires destination and dimension")
	}
	p := &SegmentPayloadWriter{w: bufio.NewWriterSize(w, 256*1024), hash: sha256.New(), dim: dimension}
	if _, err := p.write(segmentPayloadMagic[:]); err != nil {
		return nil, err
	}
	if err := p.writeBinary(segmentPayloadVersion); err != nil {
		return nil, err
	}
	if err := p.writeBinary(uint32(dimension)); err != nil {
		return nil, err
	}
	return p, nil
}

func (p *SegmentPayloadWriter) WriteVector(id int, values []float32) error {
	if p == nil || p.closed || len(values) != p.dim {
		return errors.New("invalid or closed segment payload writer")
	}
	if err := p.writeBinary(int64(id)); err != nil {
		return err
	}
	for _, value := range values {
		if err := p.writeBinary(value); err != nil {
			return err
		}
	}
	p.count++
	return nil
}

func (p *SegmentPayloadWriter) Close() (SegmentPayloadHeader, error) {
	if p == nil || p.closed {
		return SegmentPayloadHeader{}, errors.New("segment payload writer already closed")
	}
	p.closed = true
	if err := p.writeBinary(uint64(p.count)); err != nil {
		return SegmentPayloadHeader{}, err
	}
	if err := p.writeBinary(uint32(p.dim)); err != nil {
		return SegmentPayloadHeader{}, err
	}
	if err := p.w.Flush(); err != nil {
		return SegmentPayloadHeader{}, err
	}
	return SegmentPayloadHeader{Count: uint64(p.count), Dimension: uint32(p.dim)}, nil
}

func (p *SegmentPayloadWriter) write(data []byte) (int, error) {
	if _, err := p.w.Write(data); err != nil {
		return 0, err
	}
	return p.hash.Write(data)
}

func (p *SegmentPayloadWriter) writeBinary(value any) error {
	var buf bytes.Buffer
	if err := binary.Write(&buf, binary.LittleEndian, value); err != nil {
		return err
	}
	_, err := p.write(buf.Bytes())
	return err
}

func (p *SegmentPayloadWriter) SHA256() string {
	if p == nil || p.hash == nil {
		return ""
	}
	return hex.EncodeToString(p.hash.Sum(nil))
}

// ReadSegmentPayload streams records from a sealed payload without retaining
// the complete vector set in memory.
func ReadSegmentPayload(r io.ReadSeeker, visit func(int, []float32) error) (SegmentPayloadHeader, error) {
	if r == nil || visit == nil {
		return SegmentPayloadHeader{}, errors.New("payload reader requires source and callback")
	}
	var magic [4]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil || magic != segmentPayloadMagic {
		return SegmentPayloadHeader{}, errors.New("invalid segment payload magic")
	}
	var version uint16
	var dimension uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil || version != segmentPayloadVersion {
		return SegmentPayloadHeader{}, errors.New("unsupported segment payload version")
	}
	if err := binary.Read(r, binary.LittleEndian, &dimension); err != nil || dimension == 0 {
		return SegmentPayloadHeader{}, errors.New("invalid segment payload dimension")
	}
	if _, err := r.Seek(-12, io.SeekEnd); err != nil {
		return SegmentPayloadHeader{}, err
	}
	footer := make([]byte, 12)
	if _, err := io.ReadFull(r, footer); err != nil {
		return SegmentPayloadHeader{}, errors.New("truncated segment payload")
	}
	count := binary.LittleEndian.Uint64(footer[:8])
	footerDim := binary.LittleEndian.Uint32(footer[8:])
	if footerDim != dimension {
		return SegmentPayloadHeader{}, errors.New("segment payload dimension mismatch")
	}
	if _, err := r.Seek(10, io.SeekStart); err != nil {
		return SegmentPayloadHeader{}, err
	}
	values := make([]float32, dimension)
	valueBytes := make([]byte, int(dimension)*4)
	idBytes := make([]byte, 8)
	for n := uint64(0); n < count; n++ {
		if _, err := io.ReadFull(r, idBytes); err != nil {
			return SegmentPayloadHeader{}, err
		}
		id := int(int64(binary.LittleEndian.Uint64(idBytes)))
		if _, err := io.ReadFull(r, valueBytes); err != nil {
			return SegmentPayloadHeader{}, err
		}
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(valueBytes[i*4:]))
		}
		if err := visit(id, values); err != nil {
			return SegmentPayloadHeader{}, err
		}
	}
	return SegmentPayloadHeader{Count: count, Dimension: dimension}, nil
}

// WriteSegmentPayloadFile streams a segment to a temporary file and atomically
// seals it. The callback is the only owner of the writer during construction.
func WriteSegmentPayloadFile(path string, dimension int, write func(*SegmentPayloadWriter) error) (SegmentPayloadHeader, string, error) {
	if write == nil {
		return SegmentPayloadHeader{}, "", errors.New("segment payload callback is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return SegmentPayloadHeader{}, "", err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".segment-payload-*.tmp")
	if err != nil {
		return SegmentPayloadHeader{}, "", err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	writer, err := NewSegmentPayloadWriter(tmp, dimension)
	if err == nil {
		err = write(writer)
	}
	var header SegmentPayloadHeader
	if err == nil {
		header, err = writer.Close()
	}
	if err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return SegmentPayloadHeader{}, "", err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return SegmentPayloadHeader{}, "", err
	}
	return header, writer.SHA256(), nil
}
