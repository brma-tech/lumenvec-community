package ann

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

const segmentManifestVersion = 1

type SegmentState string

const (
	SegmentBuilding  SegmentState = "building"
	SegmentSealed    SegmentState = "sealed"
	SegmentPublished SegmentState = "published"
)

// SegmentManifest is the durable catalog entry for an immutable ANN segment.
// The manifest is written only after the segment payload has been fsynced.
type SegmentManifest struct {
	Version        int          `json:"version"`
	SegmentID      string       `json:"segment_id"`
	State          SegmentState `json:"state"`
	FirstID        int          `json:"first_id"`
	LastID         int          `json:"last_id"`
	Nodes          int          `json:"nodes"`
	Dimension      int          `json:"dimension"`
	Metric         string       `json:"metric"`
	M              int          `json:"m"`
	EfConstruction int          `json:"ef_construction"`
	PayloadBytes   int64        `json:"payload_bytes"`
	PayloadSHA256  string       `json:"payload_sha256"`
	UpdatedAt      time.Time    `json:"updated_at"`
}

func (m SegmentManifest) Validate() error {
	if m.Version != segmentManifestVersion {
		return fmt.Errorf("unsupported segment manifest version %d", m.Version)
	}
	if m.SegmentID == "" || m.State == "" || m.Nodes <= 0 || m.Dimension <= 0 {
		return errors.New("invalid segment manifest identity or dimensions")
	}
	if m.FirstID > m.LastID || m.PayloadBytes < 0 || len(m.PayloadSHA256) != sha256.Size*2 {
		return errors.New("invalid segment manifest bounds or checksum")
	}
	if _, err := hex.DecodeString(m.PayloadSHA256); err != nil {
		return fmt.Errorf("invalid segment checksum: %w", err)
	}
	return nil
}

func NewSegmentManifest(segmentID string, state SegmentState, payload []byte) SegmentManifest {
	sum := sha256.Sum256(payload)
	return SegmentManifest{
		Version:       segmentManifestVersion,
		SegmentID:     segmentID,
		State:         state,
		PayloadSHA256: hex.EncodeToString(sum[:]),
		PayloadBytes:  int64(len(payload)),
		UpdatedAt:     time.Now().UTC(),
	}
}

func (m SegmentManifest) VerifyPayload(payload []byte) error {
	if err := m.Validate(); err != nil {
		return err
	}
	sum := sha256.Sum256(payload)
	if hex.EncodeToString(sum[:]) != m.PayloadSHA256 || int64(len(payload)) != m.PayloadBytes {
		return errors.New("segment payload checksum or size mismatch")
	}
	return nil
}

// VerifyPayloadFile validates a durable payload without loading it into memory.
func (m SegmentManifest) VerifyPayloadFile(path string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return err
	}
	if stat.Size() != m.PayloadBytes {
		return fmt.Errorf("segment payload size mismatch: got %d want %d", stat.Size(), m.PayloadBytes)
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if hex.EncodeToString(h.Sum(nil)) != m.PayloadSHA256 {
		return errors.New("segment payload checksum mismatch")
	}
	return nil
}

// VerifyArtifactFile validates either a vector payload or a pre-built ANN
// snapshot using the checksum recorded in the manifest.
func (m SegmentManifest) VerifyArtifactFile(path string) error {
	return m.VerifyPayloadFile(path)
}

// WriteAtomically persists a manifest via a same-directory rename.
func (m SegmentManifest) WriteAtomically(path string) error {
	if err := m.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".segment-manifest-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

func ReadSegmentManifest(path string) (SegmentManifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return SegmentManifest{}, err
	}
	var manifest SegmentManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return SegmentManifest{}, err
	}
	if err := manifest.Validate(); err != nil {
		return SegmentManifest{}, err
	}
	return manifest, nil
}
