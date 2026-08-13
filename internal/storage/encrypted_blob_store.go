package storage

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"io"
)

// EncryptedBlobStore encrypts every blob with AES-256-GCM before delegating to
// another BlobStore. The object key is authenticated as associated data, so a
// ciphertext cannot be moved to a different key without failing Open.
type EncryptedBlobStore struct {
	inner BlobStore
	aead  cipher.AEAD
}

func NewEncryptedBlobStore(inner BlobStore, key []byte) (*EncryptedBlobStore, error) {
	if inner == nil {
		return nil, fmt.Errorf("blob store is required")
	}
	if len(key) != 32 {
		return nil, fmt.Errorf("encryption key must be 32 bytes")
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return &EncryptedBlobStore{inner: inner, aead: aead}, nil
}

func (s *EncryptedBlobStore) Put(key string, r io.Reader) error {
	if s == nil || s.inner == nil || s.aead == nil {
		return fmt.Errorf("encrypted blob store is not initialized")
	}
	plain, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	nonce := make([]byte, s.aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return err
	}
	sealed := s.aead.Seal(nonce, nonce, plain, []byte(key))
	return s.inner.Put(key, bytesReader(sealed))
}

func (s *EncryptedBlobStore) Get(key string) (io.ReadCloser, error) {
	if s == nil || s.inner == nil || s.aead == nil {
		return nil, fmt.Errorf("encrypted blob store is not initialized")
	}
	r, err := s.inner.Get(key)
	if err != nil {
		return nil, err
	}
	sealed, readErr := io.ReadAll(r)
	_ = r.Close()
	if readErr != nil {
		return nil, readErr
	}
	if len(sealed) < s.aead.NonceSize() {
		return nil, fmt.Errorf("encrypted blob is truncated")
	}
	nonce, ciphertext := sealed[:s.aead.NonceSize()], sealed[s.aead.NonceSize():]
	plain, err := s.aead.Open(nil, nonce, ciphertext, []byte(key))
	if err != nil {
		return nil, fmt.Errorf("decrypt blob %q: %w", key, err)
	}
	return io.NopCloser(bytesReader(plain)), nil
}

func (s *EncryptedBlobStore) Delete(key string) error { return s.inner.Delete(key) }

// bytesReader avoids exposing a mutable buffer to the delegated store.
func bytesReader(b []byte) io.Reader { return &byteReader{b: b} }

type byteReader struct{ b []byte }

func (r *byteReader) Read(p []byte) (int, error) {
	if len(r.b) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.b)
	r.b = r.b[n:]
	return n, nil
}
