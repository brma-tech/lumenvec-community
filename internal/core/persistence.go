package core

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"lumenvec/internal/index"
)

// WALReplicator is the provider-neutral durability boundary between storage
// and consensus. The payload is newline-delimited WAL JSON.
type WALReplicator interface{ ReplicateWAL([]byte) error }

type replicatedPersistence struct {
	backend    PersistenceBackend
	replicator WALReplicator
}

func (p *replicatedPersistence) SaveSnapshot(v []index.Vector) error {
	return p.backend.SaveSnapshot(v)
}
func (p *replicatedPersistence) LoadSnapshot() (map[string][]float64, error) {
	return p.backend.LoadSnapshot()
}
func (p *replicatedPersistence) AppendWAL(op walOp) error { return p.AppendWALBatch([]walOp{op}) }
func (p *replicatedPersistence) AppendWALBatch(ops []walOp) error {
	if len(ops) == 0 {
		return nil
	}
	var payload bytes.Buffer
	enc := json.NewEncoder(&payload)
	for _, op := range ops {
		if err := enc.Encode(op); err != nil {
			return err
		}
	}
	if err := p.replicator.ReplicateWAL(payload.Bytes()); err != nil {
		return err
	}
	if batcher, ok := p.backend.(interface{ AppendWALBatch([]walOp) error }); ok {
		return batcher.AppendWALBatch(ops)
	}
	for _, op := range ops {
		if err := p.backend.AppendWAL(op); err != nil {
			return err
		}
	}
	return nil
}
func (p *replicatedPersistence) ReplayWAL(fn func(walOp) error) error { return p.backend.ReplayWAL(fn) }
func (p *replicatedPersistence) TruncateWAL() error                   { return p.backend.TruncateWAL() }

type PersistenceBackend interface {
	SaveSnapshot(vectors []index.Vector) error
	LoadSnapshot() (map[string][]float64, error)
	AppendWAL(op walOp) error
	ReplayWAL(func(walOp) error) error
	TruncateWAL() error
}

type snapshotWALBackend struct {
	snapshotPath string
	walPath      string
	security     StorageSecurityOptions
	syncEvery    int
	pendingOps   int
	mu           sync.Mutex
}

func newSnapshotWALBackend(snapshotPath, walPath string) *snapshotWALBackend {
	return newSnapshotWALBackendWithSecurity(snapshotPath, walPath, DefaultStorageSecurityOptions())
}

func newSnapshotWALBackendWithSecurity(snapshotPath, walPath string, security StorageSecurityOptions) *snapshotWALBackend {
	return &snapshotWALBackend{
		snapshotPath: snapshotPath,
		walPath:      walPath,
		security:     normalizeStorageSecurityOptions(security),
		syncEvery:    1,
	}
}

func newSnapshotWALBackendWithOptions(snapshotPath, walPath string, security StorageSecurityOptions, syncEvery int) *snapshotWALBackend {
	backend := newSnapshotWALBackendWithSecurity(snapshotPath, walPath, security)
	backend.syncEvery = normalizeSyncEvery(syncEvery)
	return backend
}

func (b *snapshotWALBackend) SaveSnapshot(vectors []index.Vector) error {
	payload := make(map[string][]float64, len(vectors))
	for _, vec := range vectors {
		payload[vec.ID] = append([]float64(nil), vec.Values...)
	}

	if err := os.MkdirAll(filepath.Dir(b.snapshotPath), b.security.DirMode); err != nil {
		return err
	}
	tmp := b.snapshotPath + ".tmp"
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, b.security.FileMode)
	if err != nil {
		return err
	}
	if _, err = f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return publishDurableSnapshot(tmp, b.snapshotPath)
}

func (b *snapshotWALBackend) LoadSnapshot() (map[string][]float64, error) {
	data, err := os.ReadFile(b.snapshotPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var payload map[string][]float64
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	return payload, nil
}

func (b *snapshotWALBackend) AppendWAL(op walOp) error {
	return b.AppendWALBatch([]walOp{op})
}

func (b *snapshotWALBackend) AppendWALBatch(ops []walOp) error {
	if len(ops) == 0 {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(b.walPath), b.security.DirMode); err != nil {
		return err
	}
	f, err := os.OpenFile(b.walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, b.security.FileMode)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	writer := bufio.NewWriter(f)
	for _, op := range ops {
		data, err := json.Marshal(op)
		if err != nil {
			return err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return err
		}
	}
	if err := writer.Flush(); err != nil {
		return err
	}
	b.pendingOps += len(ops)
	if b.pendingOps < b.syncEvery {
		return nil
	}
	if err := f.Sync(); err != nil {
		return err
	}
	b.pendingOps = 0
	return nil
}

func (b *snapshotWALBackend) ReplayWAL(apply func(walOp) error) error {
	f, err := os.Open(b.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer func() { _ = f.Close() }()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var op walOp
		if err := json.Unmarshal([]byte(line), &op); err != nil {
			return err
		}
		if err := apply(op); err != nil && !errors.Is(err, errSkipWALOp) {
			return err
		}
	}
	return scanner.Err()
}

func (b *snapshotWALBackend) TruncateWAL() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(b.walPath), b.security.DirMode); err != nil {
		return err
	}
	f, err := os.OpenFile(b.walPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, b.security.FileMode)
	if err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	b.pendingOps = 0
	return nil
}

func (b *snapshotWALBackend) Sync() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.pendingOps == 0 {
		return nil
	}
	f, err := os.OpenFile(b.walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, b.security.FileMode)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	b.pendingOps = 0
	return nil
}
