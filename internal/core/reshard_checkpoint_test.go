package core

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReshardCheckpointRoundTripAndReplace(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state", "checkpoint.json")
	first := ReshardCheckpoint{MigrationID: "m1", SourceGeneration: 7, TargetShard: 2, Cursor: "v-10", AppliedWAL: 11, VectorsCopied: 10, State: "copying"}
	if err := SaveReshardCheckpoint(path, first); err != nil {
		t.Fatal(err)
	}
	second := first
	second.Cursor, second.AppliedWAL, second.VectorsCopied, second.State = "v-20", 21, 20, "draining"
	if err := SaveReshardCheckpoint(path, second); err != nil {
		t.Fatal(err)
	}
	got, err := LoadReshardCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Cursor != "v-20" || got.AppliedWAL != 21 || got.State != "draining" || got.Version != reshardCheckpointVersion {
		t.Fatalf("checkpoint = %+v", got)
	}
}

func TestReshardCheckpointRejectsInvalidOrCorrupt(t *testing.T) {
	path := filepath.Join(t.TempDir(), "checkpoint.json")
	if err := SaveReshardCheckpoint(path, ReshardCheckpoint{MigrationID: "m", TargetShard: 0, State: "bad"}); err == nil {
		t.Fatal("expected invalid state rejection")
	}
	if err := os.WriteFile(path, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadReshardCheckpoint(path); err == nil {
		t.Fatal("expected corrupt checkpoint rejection")
	}
}
