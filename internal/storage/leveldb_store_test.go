package storage

import (
	"math"
	"testing"
)

func TestLevelDBStoreCRUD(t *testing.T) {
	store, err := NewLevelDBStore("ignored")
	if err != nil {
		t.Fatal(err)
	}

	type payload struct {
		Name string `json:"name"`
	}

	if err := store.Put("item-1", payload{Name: "alpha"}); err != nil {
		t.Fatal(err)
	}

	var got payload
	if err := store.Get("item-1", &got); err != nil {
		t.Fatal(err)
	}
	if got.Name != "alpha" {
		t.Fatal("expected stored payload")
	}

	if err := store.Delete("item-1"); err != nil {
		t.Fatal(err)
	}
	got = payload{}
	if err := store.Get("item-1", &got); err != nil {
		t.Fatal(err)
	}
	if got.Name != "" {
		t.Fatal("expected deleted key to behave like missing key")
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLevelDBStoreErrorPaths(t *testing.T) {
	store, err := NewLevelDBStore("ignored")
	if err != nil {
		t.Fatal(err)
	}

	if err := store.Put("bad", math.NaN()); err == nil {
		t.Fatal("expected marshal error")
	}

	store.store["bad-json"] = []byte("{bad")
	var got map[string]any
	if err := store.Get("bad-json", &got); err == nil {
		t.Fatal("expected unmarshal error")
	}
}

func TestLevelDBStoreIterateBranches(t *testing.T) {
	store, err := NewLevelDBStore("ignored")
	if err != nil {
		t.Fatal(err)
	}

	if err := store.Put("keep-1", map[string]any{"value": 1}); err != nil {
		t.Fatal(err)
	}
	if err := store.Put("keep-2", map[string]any{"value": 2}); err != nil {
		t.Fatal(err)
	}
	store.store["keep-bad"] = []byte("{bad-json")
	if err := store.Put("skip", map[string]any{"value": 3}); err != nil {
		t.Fatal(err)
	}

	seen := 0
	if err := store.Iterate("keep", func(key string, value interface{}) bool {
		seen++
		return key != "keep-1"
	}); err != nil {
		t.Fatal(err)
	}
	if seen == 0 {
		t.Fatal("expected iterate to visit matching keys")
	}

	seen = 0
	if err := store.Iterate("", func(key string, value interface{}) bool {
		seen++
		return true
	}); err != nil {
		t.Fatal(err)
	}
	if seen < 3 {
		t.Fatalf("expected empty prefix to visit all valid keys, got %d", seen)
	}
}
