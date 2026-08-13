package core

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"lumenvec/internal/index"
)

type recordingWALReplicator struct {
	payloads [][]byte
	err      error
}

func (r *recordingWALReplicator) ReplicateWAL(payload []byte) error {
	if r.err != nil {
		return r.err
	}
	r.payloads = append(r.payloads, append([]byte(nil), payload...))
	return nil
}

func TestReplicatedPersistenceReplicatesBeforeLocalAppend(t *testing.T) {
	base := t.TempDir()
	backend := newSnapshotWALBackend(filepath.Join(base, "snapshot.json"), filepath.Join(base, "wal.log"))
	repl := &recordingWALReplicator{}
	p := &replicatedPersistence{backend: backend, replicator: repl}
	if err := p.AppendWALBatch([]walOp{{Op: "upsert", ID: "a", Values: []float64{1}}}); err != nil {
		t.Fatal(err)
	}
	if len(repl.payloads) != 1 || !bytes.Contains(repl.payloads[0], []byte(`"id":"a"`)) {
		t.Fatalf("payloads=%q", repl.payloads)
	}
	if err := backend.ReplayWAL(func(op walOp) error {
		if op.ID != "a" {
			t.Fatalf("op=%+v", op)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestReplicatedPersistenceDoesNotAppendWhenReplicationFails(t *testing.T) {
	base := t.TempDir()
	backend := newSnapshotWALBackend(filepath.Join(base, "snapshot.json"), filepath.Join(base, "wal.log"))
	repl := &recordingWALReplicator{err: errors.New("quorum unavailable")}
	p := &replicatedPersistence{backend: backend, replicator: repl}
	if err := p.AppendWAL(walOp{Op: "upsert", ID: "a"}); err == nil {
		t.Fatal("expected quorum error")
	}
	if err := backend.ReplayWAL(func(walOp) error { t.Fatal("local WAL changed after replication failure"); return nil }); err != nil {
		t.Fatal(err)
	}
}

func TestSnapshotWALBackendRoundTrip(t *testing.T) {
	base := t.TempDir()
	backend := newSnapshotWALBackend(filepath.Join(base, "snapshot.json"), filepath.Join(base, "wal.log"))

	if err := backend.SaveSnapshot([]index.Vector{{ID: "a", Values: []float64{1, 2, 3}}}); err != nil {
		t.Fatal(err)
	}
	payload, err := backend.LoadSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(payload) != 1 || len(payload["a"]) != 3 {
		t.Fatal("expected snapshot payload")
	}

	if err := backend.AppendWAL(walOp{Op: "upsert", ID: "a", Values: []float64{1, 2, 3}}); err != nil {
		t.Fatal(err)
	}
	if err := backend.AppendWALBatch([]walOp{{Op: "delete", ID: "a"}}); err != nil {
		t.Fatal(err)
	}
	if err := backend.AppendWALBatch(nil); err != nil {
		t.Fatal(err)
	}

	seen := 0
	if err := backend.ReplayWAL(func(op walOp) error {
		seen++
		if op.ID == "" {
			t.Fatal("expected wal op id")
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if seen != 2 {
		t.Fatalf("expected 2 wal ops, got %d", seen)
	}

	if err := backend.TruncateWAL(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(filepath.Join(base, "wal.log"))
	if err != nil || info.Size() != 0 {
		t.Fatal("expected truncated wal")
	}
}

func TestSnapshotWALBackendSyncEveryDefersAndFlushes(t *testing.T) {
	base := t.TempDir()
	backend := newSnapshotWALBackendWithOptions(
		filepath.Join(base, "snapshot.json"),
		filepath.Join(base, "wal.log"),
		DefaultStorageSecurityOptions(),
		3,
	)

	if err := backend.AppendWAL(walOp{Op: "upsert", ID: "a", Values: []float64{1}}); err != nil {
		t.Fatal(err)
	}
	if backend.pendingOps != 1 {
		t.Fatalf("pendingOps = %d, want 1", backend.pendingOps)
	}
	if err := backend.AppendWALBatch([]walOp{
		{Op: "upsert", ID: "b", Values: []float64{2}},
		{Op: "upsert", ID: "c", Values: []float64{3}},
	}); err != nil {
		t.Fatal(err)
	}
	if backend.pendingOps != 0 {
		t.Fatalf("pendingOps after threshold = %d, want 0", backend.pendingOps)
	}

	if err := backend.AppendWAL(walOp{Op: "delete", ID: "a"}); err != nil {
		t.Fatal(err)
	}
	if err := backend.Sync(); err != nil {
		t.Fatal(err)
	}
	if backend.pendingOps != 0 {
		t.Fatalf("pendingOps after Sync = %d, want 0", backend.pendingOps)
	}
}

func TestSnapshotWALBackendReplaySkipAndMissingFiles(t *testing.T) {
	base := t.TempDir()
	backend := newSnapshotWALBackend(filepath.Join(base, "snapshot.json"), filepath.Join(base, "wal.log"))

	payload, err := backend.LoadSnapshot()
	if err != nil || payload != nil {
		t.Fatal("expected nil snapshot for missing file")
	}

	if err := backend.ReplayWAL(func(walOp) error { return errSkipWALOp }); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(base, "wal.log"), []byte("{\"op\":\"upsert\",\"id\":\"a\",\"values\":[1]}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := backend.ReplayWAL(func(walOp) error { return errSkipWALOp }); err != nil {
		t.Fatal(err)
	}
}

func TestSnapshotWALBackendReplayApplyError(t *testing.T) {
	base := t.TempDir()
	backend := newSnapshotWALBackend(filepath.Join(base, "snapshot.json"), filepath.Join(base, "wal.log"))
	if err := os.WriteFile(filepath.Join(base, "wal.log"), []byte("{\"op\":\"upsert\",\"id\":\"a\",\"values\":[1]}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	want := errors.New("apply failed")
	if err := backend.ReplayWAL(func(walOp) error { return want }); !errors.Is(err, want) {
		t.Fatalf("expected apply failure, got %v", err)
	}
}

func TestSnapshotWALBackendErrorBranches(t *testing.T) {
	base := t.TempDir()
	blocker := filepath.Join(base, "not-a-dir")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	blocked := newSnapshotWALBackend(filepath.Join(blocker, "snapshot.json"), filepath.Join(blocker, "wal.log"))
	if err := blocked.SaveSnapshot([]index.Vector{{ID: "a", Values: []float64{1}}}); err == nil {
		t.Fatal("expected SaveSnapshot mkdir failure")
	}
	if err := blocked.AppendWAL(walOp{Op: "upsert", ID: "a", Values: []float64{1}}); err == nil {
		t.Fatal("expected AppendWAL mkdir failure")
	}
	if err := blocked.TruncateWAL(); err == nil {
		t.Fatal("expected TruncateWAL mkdir failure")
	}

	dirSnapshot := newSnapshotWALBackend(base, filepath.Join(base, "wal.log"))
	if _, err := dirSnapshot.LoadSnapshot(); err == nil {
		t.Fatal("expected LoadSnapshot to fail when snapshot path is a directory")
	}

	invalidSnapshot := newSnapshotWALBackend(filepath.Join(base, "invalid.json"), filepath.Join(base, "wal-invalid.log"))
	if err := os.WriteFile(invalidSnapshot.snapshotPath, []byte("{bad-json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := invalidSnapshot.LoadSnapshot(); err == nil {
		t.Fatal("expected invalid snapshot JSON to fail")
	}

	if err := os.WriteFile(invalidSnapshot.walPath, []byte("\n{bad-json}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := invalidSnapshot.ReplayWAL(func(walOp) error { return nil }); err == nil {
		t.Fatal("expected invalid WAL JSON to fail")
	}
}
