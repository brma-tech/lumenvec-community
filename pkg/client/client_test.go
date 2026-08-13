package client

import (
	"bytes"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestClientBatchOperations(t *testing.T) {
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
		_, _ = w.Write([]byte(`[{"id":"q1","results":[{"id":"doc-1","distance":0.1}]}]`))
	})
	mux.HandleFunc("/vectors/doc-1", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		w.WriteHeader(http.StatusMethodNotAllowed)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := NewVectorClient(srv.URL)
	if err := c.AddVectorWithID("doc-1", []float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVectorWithID() error = %v", err)
	}
	results, err := c.SearchVector([]float64{1, 2, 3}, 1)
	if err != nil {
		t.Fatalf("SearchVector() error = %v", err)
	}
	if len(results) != 1 || results[0].ID != "doc-1" {
		t.Fatal("unexpected search results")
	}
	batchResults, err := c.SearchVectors([]BatchSearchQuery{{ID: "q1", Values: []float64{1, 2, 3}, K: 1}})
	if err != nil {
		t.Fatalf("SearchVectors() error = %v", err)
	}
	if len(batchResults) != 1 || batchResults[0].ID != "q1" || len(batchResults[0].Results) != 1 {
		t.Fatal("unexpected batch search results")
	}
	if err := c.DeleteVector("doc-1"); err != nil {
		t.Fatalf("DeleteVector() error = %v", err)
	}
}

func TestClientErrorPaths(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	mux.HandleFunc("/vectors/search/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{bad-json`))
	})
	mux.HandleFunc("/vectors/missing", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := NewVectorClient(srv.URL)
	if err := c.AddVectors([]VectorPayload{{ID: "a", Values: []float64{1}}}); err == nil {
		t.Fatal("expected AddVectors error")
	}
	if _, err := c.SearchVectors([]BatchSearchQuery{{ID: "q1", Values: []float64{1}, K: 1}}); err == nil {
		t.Fatal("expected SearchVectors decode error")
	}
	if err := c.DeleteVector("missing"); err == nil {
		t.Fatal("expected DeleteVector error")
	}
}

func TestClientAddVectorAutoID(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := NewVectorClient(srv.URL)
	if c.httpClient.Timeout != 10*time.Second {
		t.Fatal("expected default timeout")
	}
	if err := c.AddVector([]float64{1, 2, 3}); err != nil {
		t.Fatalf("AddVector() error = %v", err)
	}
}

func TestClientMarshalErrorsAndEmptySearchResult(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
	})
	mux.HandleFunc("/vectors/search", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[]`))
	})
	mux.HandleFunc("/vectors/search/batch", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[]`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := NewVectorClient(srv.URL)
	if err := c.AddVectors([]VectorPayload{{ID: "a", Values: []float64{math.NaN()}}}); err == nil {
		t.Fatal("expected AddVectors marshal error")
	}
	if _, err := c.SearchVectors([]BatchSearchQuery{{ID: "q", Values: []float64{math.NaN()}, K: 1}}); err == nil {
		t.Fatal("expected SearchVectors marshal error")
	}
	results, err := c.SearchVector([]float64{1, 2, 3}, 1)
	if err != nil || len(results) != 0 {
		t.Fatal("expected empty SearchVector result set")
	}
}

func TestClientSearchStatusAndTransportErrors(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/search", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	mux.HandleFunc("/vectors/search/batch", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := NewVectorClient(srv.URL)
	if _, err := c.SearchVector([]float64{1}, 1); err == nil {
		t.Fatal("expected search status error")
	}
	if _, err := c.SearchVectors([]BatchSearchQuery{{ID: "q", Values: []float64{1}, K: 1}}); err == nil {
		t.Fatal("expected search status error")
	}

	badClient := NewVectorClient("://bad-url")
	if _, err := badClient.SearchVector([]float64{1}, 1); err == nil {
		t.Fatal("expected search request creation error")
	}
	if _, err := badClient.SearchVectors([]BatchSearchQuery{{ID: "q", Values: []float64{1}, K: 1}}); err == nil {
		t.Fatal("expected batch search request creation error")
	}
	if err := badClient.AddVectorWithID("doc-1", []float64{1}); err == nil {
		t.Fatal("expected add request creation error")
	}
	if err := badClient.DeleteVector("doc-1"); err == nil {
		t.Fatal("expected delete transport error")
	}

	deleteClient := NewVectorClient(srv.URL)
	srv.Close()
	if err := deleteClient.DeleteVector("doc-1"); err == nil {
		t.Fatal("expected delete request transport error")
	}
}

func TestClientSearchDecodeAndTransportErrors(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/vectors/search", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{bad-json`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	c := NewVectorClient(srv.URL)
	if _, err := c.SearchVector([]float64{1}, 1); err == nil {
		t.Fatal("expected SearchVector decode error")
	}

	srv.Close()
	if _, err := c.SearchVector([]float64{1}, 1); err == nil {
		t.Fatal("expected SearchVector transport error")
	}
}

func TestPutRequestBodyBufferBranches(t *testing.T) {
	putRequestBodyBuffer(nil)

	buf := bytes.NewBuffer(make([]byte, 0, (1<<20)+1))
	putRequestBodyBuffer(buf)
	reused := getRequestBodyBuffer()
	if reused.Len() != 0 || reused.Cap() > 1<<20 {
		t.Fatalf("expected oversized buffer to be replaced, len=%d cap=%d", reused.Len(), reused.Cap())
	}
	putRequestBodyBuffer(reused)
}
