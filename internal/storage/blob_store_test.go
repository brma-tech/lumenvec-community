package storage

import (
	"errors"
	"io"
	"strings"
	"testing"
)

type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errors.New("read failed") }

func TestLocalBlobStoreAtomicCRUDAndKeyValidation(t *testing.T) {
	store, err := NewLocalBlobStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewLocalBlobStore(""); err == nil {
		t.Fatal("expected root validation")
	}
	if err := store.Put("failed", failingReader{}); err == nil {
		t.Fatal("expected reader failure")
	}
	if err := store.Put("segments/0001", strings.NewReader("segment-data")); err != nil {
		t.Fatal(err)
	}
	r, err := store.Get("segments/0001")
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(r)
	_ = r.Close()
	if err != nil || string(data) != "segment-data" {
		t.Fatalf("data=%q err=%v", data, err)
	}
	if err := store.Put("segments/0001", strings.NewReader("replacement")); err != nil {
		t.Fatal(err)
	}
	r, err = store.Get("segments/0001")
	if err != nil {
		t.Fatal(err)
	}
	data, _ = io.ReadAll(r)
	_ = r.Close()
	if string(data) != "replacement" {
		t.Fatalf("replacement=%q", data)
	}
	if err := store.Delete("segments/0001"); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete("segments/0001"); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"../escape", "/absolute", "."} {
		if err := store.Put(key, strings.NewReader("x")); err == nil {
			t.Fatalf("expected invalid key %q", key)
		}
	}
	if _, err := store.Get("../escape"); err == nil {
		t.Fatal("expected invalid get key")
	}
	if err := store.Delete("../escape"); err == nil {
		t.Fatal("expected invalid delete key")
	}
	if err := store.Put("segments/keep", strings.NewReader("x")); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete("segments"); err == nil {
		t.Fatal("expected directory delete error")
	}
	if _, err := store.Get("missing"); err == nil {
		t.Fatal("expected missing blob error")
	}
}
