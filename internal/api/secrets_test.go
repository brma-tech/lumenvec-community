package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestSecretCacheAtomicLoadAndValidation(t *testing.T) {
	var cache SecretCache
	if err := cache.Load(SecretVersion{}); err == nil {
		t.Fatal("expected empty secret rejection")
	}
	if err := cache.Load(SecretVersion{Value: "old", Version: "1"}); err != nil {
		t.Fatal(err)
	}
	if got := cache.Current(); got.Value != "old" || got.Version != "1" {
		t.Fatalf("initial secret=%+v", got)
	}
	if err := cache.Load(SecretVersion{Value: "new", Version: "2"}); err != nil {
		t.Fatal(err)
	}
	if got := cache.Current(); got.Value != "new" || got.Version != "2" {
		t.Fatalf("rotated secret=%+v", got)
	}
}

func TestFileSecretProviderReadAndAtomicRotate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "secret", "api-key")
	p := FileSecretProvider{Path: path}
	if _, err := p.Rotate(context.Background(), "api", "first"); err != nil {
		t.Fatal(err)
	}
	got, err := p.Read(context.Background(), "api")
	if err != nil || got.Value != "first" || got.Version == "" {
		t.Fatalf("read=%+v err=%v", got, err)
	}
	if _, err := p.Rotate(context.Background(), "api", "second"); err != nil {
		t.Fatal(err)
	}
	got, err = p.Read(context.Background(), "api")
	if err != nil || got.Value != "second" {
		t.Fatalf("rotated=%+v err=%v", got, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatal(err)
	}
}

func TestServerLoadsAPIKeyFromProvider(t *testing.T) {
	s := NewServer("0")
	p := FileSecretProvider{Path: filepath.Join(t.TempDir(), "key")}
	if _, err := p.Rotate(context.Background(), "api", "provider-key"); err != nil {
		t.Fatal(err)
	}
	if err := s.LoadAPIKeyFromProvider(context.Background(), p, "api"); err != nil {
		t.Fatal(err)
	}
	if got := s.currentAPIKey(); got != "provider-key" {
		t.Fatalf("key=%q", got)
	}
}

func TestServerAPIKeyReload(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := NewServer("0")
	p := FileSecretProvider{Path: filepath.Join(t.TempDir(), "key")}
	if _, err := p.Rotate(context.Background(), "api", "first"); err != nil {
		t.Fatal(err)
	}
	s.StartAPIKeyReload(ctx, p, "api", 5*time.Millisecond)
	deadline := time.Now().Add(time.Second)
	for s.currentAPIKey() != "first" && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if s.currentAPIKey() != "first" {
		t.Fatalf("initial key=%q", s.currentAPIKey())
	}
	if _, err := p.Rotate(context.Background(), "api", "second"); err != nil {
		t.Fatal(err)
	}
	deadline = time.Now().Add(time.Second)
	for s.currentAPIKey() != "second" && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if s.currentAPIKey() != "second" {
		t.Fatalf("rotated key=%q", s.currentAPIKey())
	}
}

func TestHTTPSecretProviderReadAndRotate(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		if r.Method == http.MethodPut {
			var body map[string]string
			_ = json.NewDecoder(r.Body).Decode(&body)
			_ = json.NewEncoder(w).Encode(SecretVersion{Value: body["value"], Version: "v2"})
			return
		}
		_ = json.NewEncoder(w).Encode(SecretVersion{Value: "remote", Version: "v1"})
	}))
	defer srv.Close()
	p := HTTPSecretProvider{BaseURL: srv.URL, Token: "token"}
	v, err := p.Read(context.Background(), "api/key")
	if err != nil || v.Value != "remote" || v.Version != "v1" || gotAuth != "Bearer token" {
		t.Fatalf("read=%+v auth=%q err=%v", v, gotAuth, err)
	}
	v, err = p.Rotate(context.Background(), "api/key", "rotated")
	if err != nil || v.Value != "rotated" || v.Version != "v2" {
		t.Fatalf("rotate=%+v err=%v", v, err)
	}
}
