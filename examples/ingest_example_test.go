package main

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRun(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	})
	mux.HandleFunc("/vectors/search", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"doc-1","distance":0.1},{"id":"doc-2","distance":0.2}]`))
	})
	mux.HandleFunc("/vectors/search/batch", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"query-0","results":[{"id":"doc-1","distance":0.1},{"id":"doc-2","distance":0.2}]}]`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	var out bytes.Buffer
	if err := run(srv.URL, &out); err != nil {
		t.Fatalf("run() error = %v", err)
	}
	if out.Len() == 0 {
		t.Fatal("expected output")
	}
}

func TestRunErrors(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	var out bytes.Buffer
	if err := run(srv.URL, &out); err == nil {
		t.Fatal("expected run error")
	}
}

func TestMustRun(t *testing.T) {
	mustRun(func(string, io.Writer) error { return nil }, "x", &bytes.Buffer{})

	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	mustRun(func(string, io.Writer) error { return errors.New("boom") }, "x", &bytes.Buffer{})
}

func TestRunWriteFailures(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	})
	mux.HandleFunc("/vectors/search", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"doc-1","distance":0.1}]`))
	})
	mux.HandleFunc("/vectors/search/batch", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"query-0","results":[{"id":"doc-1","distance":0.1}]}]`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	if err := run(srv.URL, failWriter{}); err == nil {
		t.Fatal("expected writer failure")
	}
}

type failWriter struct{}

func (failWriter) Write([]byte) (int, error) {
	return 0, errors.New("write failed")
}
