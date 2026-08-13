package storage

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHTTPBlobStoreOperations(t *testing.T) {
	data := map[string]string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.URL.Path, "/objects/")
		switch r.Method {
		case http.MethodPut:
			body, _ := io.ReadAll(r.Body)
			data[key] = string(body)
			w.WriteHeader(http.StatusCreated)
		case http.MethodGet:
			if value, ok := data[key]; ok {
				_, _ = w.Write([]byte(value))
				return
			}
			http.NotFound(w, r)
		case http.MethodDelete:
			delete(data, key)
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()
	store, err := NewHTTPBlobStore(server.URL+"/objects", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Put("a", strings.NewReader("payload")); err != nil {
		t.Fatal(err)
	}
	reader, err := store.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(reader)
	reader.Close()
	if string(got) != "payload" {
		t.Fatalf("got=%q", got)
	}
	if err := store.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Get("a"); err == nil {
		t.Fatal("expected missing object")
	}
	if err := store.Put("../escape", strings.NewReader("x")); err == nil {
		t.Fatal("expected invalid key")
	}
}

func TestHTTPBlobStoreDefaultClientHasTimeout(t *testing.T) {
	store, err := NewHTTPBlobStore("https://example.invalid/objects", nil)
	if err != nil {
		t.Fatal(err)
	}
	if store.client.Timeout <= 0 {
		t.Fatal("expected finite default timeout")
	}
}

func TestHTTPBlobStoreRequestHook(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusCreated)
	}))
	defer server.Close()
	store, err := NewHTTPBlobStoreWithHook(server.URL, nil, func(req *http.Request) error { req.Header.Set("Authorization", "Bearer test"); return nil })
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Put("signed", strings.NewReader("x")); err != nil {
		t.Fatal(err)
	}
}
