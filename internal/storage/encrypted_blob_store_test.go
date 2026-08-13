package storage

import (
	"io"
	"testing"
)

func TestEncryptedBlobStoreRoundTrip(t *testing.T) {
	inner, err := NewLocalBlobStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	key := []byte("01234567890123456789012345678901")
	store, err := NewEncryptedBlobStore(inner, key)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Put("segments/a", stringsReader("secret payload")); err != nil {
		t.Fatal(err)
	}
	r, err := store.Get("segments/a")
	if err != nil {
		t.Fatal(err)
	}
	got, err := io.ReadAll(r)
	_ = r.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "secret payload" {
		t.Fatalf("got %q", got)
	}
	raw, err := inner.Get("segments/a")
	if err != nil {
		t.Fatal(err)
	}
	ciphertext, _ := io.ReadAll(raw)
	_ = raw.Close()
	if string(ciphertext) == "secret payload" {
		t.Fatal("blob was stored in plaintext")
	}
}

func TestEncryptedBlobStoreRejectsWrongKeyAndTampering(t *testing.T) {
	inner, err := NewLocalBlobStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	key := []byte("01234567890123456789012345678901")
	store, err := NewEncryptedBlobStore(inner, key)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Put("a", stringsReader("x")); err != nil {
		t.Fatal(err)
	}
	wrong, err := NewEncryptedBlobStore(inner, []byte("abcdefghijklmnopqrstuvwxyz123456"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wrong.Get("a"); err == nil {
		t.Fatal("wrong key decrypted blob")
	}
	if err := inner.Put("a", stringsReader("tampered")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get("a"); err == nil {
		t.Fatal("tampered blob was accepted")
	}
}

func TestNewEncryptedBlobStoreRequiresKey(t *testing.T) {
	inner, err := NewLocalBlobStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewEncryptedBlobStore(inner, []byte("short")); err == nil {
		t.Fatal("short key accepted")
	}
}

type stringReader struct{ data []byte }

func (r *stringReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	return n, nil
}
func stringsReader(s string) io.Reader { return &stringReader{data: []byte(s)} }
