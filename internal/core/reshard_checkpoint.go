package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const reshardCheckpointVersion = 1

// ReshardCheckpoint is the durable progress record for one migration window.
// Cursor and WAL offsets are opaque to the checkpoint layer and can therefore
// be resumed by either the paged or native float32 migration path.
type ReshardCheckpoint struct {
	Version          int       `json:"version"`
	MigrationID      string    `json:"migration_id"`
	SourceGeneration uint64    `json:"source_generation"`
	TargetShard      int       `json:"target_shard"`
	Cursor           string    `json:"cursor,omitempty"`
	AppliedWAL       uint64    `json:"applied_wal"`
	VectorsCopied    uint64    `json:"vectors_copied"`
	State            string    `json:"state"`
	UpdatedAt        time.Time `json:"updated_at"`
}

func (c ReshardCheckpoint) validate() error {
	if c.Version != reshardCheckpointVersion || c.MigrationID == "" || c.TargetShard < 0 {
		return errors.New("invalid reshard checkpoint identity")
	}
	if c.State != "copying" && c.State != "draining" && c.State != "complete" {
		return fmt.Errorf("invalid reshard checkpoint state %q", c.State)
	}
	return nil
}

// SaveReshardCheckpoint atomically persists a validated checkpoint and fsyncs
// it before rename, so a crash leaves either the previous or new record.
func SaveReshardCheckpoint(path string, checkpoint ReshardCheckpoint) error {
	if checkpoint.Version == 0 {
		checkpoint.Version = reshardCheckpointVersion
	}
	if checkpoint.UpdatedAt.IsZero() {
		checkpoint.UpdatedAt = time.Now().UTC()
	}
	if err := checkpoint.validate(); err != nil {
		return err
	}
	// #nosec G703 -- path is constructed internally below the configured,
	// service-owned checkpoint directory; this helper never consumes HTTP input.
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".reshard-checkpoint-*.tmp")
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
	return os.Rename(tmpName, path) // #nosec G703 -- same service-owned destination validated by the caller
}

func LoadReshardCheckpoint(path string) (ReshardCheckpoint, error) {
	data, err := os.ReadFile(path) // #nosec G703 -- internal service-owned checkpoint path, not request input
	if err != nil {
		return ReshardCheckpoint{}, err
	}
	var checkpoint ReshardCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return ReshardCheckpoint{}, err
	}
	if err := checkpoint.validate(); err != nil {
		return ReshardCheckpoint{}, err
	}
	return checkpoint, nil
}
