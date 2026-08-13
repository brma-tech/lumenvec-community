package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"testing"
	"time"

	"github.com/gorilla/mux"
)

func TestListVectorsCursorAndAfterAndIdsOnly(t *testing.T) {
	// reuse helper from server_test.go
	base := t.TempDir()
	server := NewServerWithOptions(ServerOptions{
		Port:          ":0",
		ReadTimeout:   5 * time.Second,
		WriteTimeout:  5 * time.Second,
		MaxBodyBytes:  1 << 20,
		MaxVectorDim:  16,
		MaxK:          5,
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 2,
		SearchMode:    "exact",
	})

	// add three vectors
	if err := server.service.AddVector("a", []float64{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	if err := server.service.AddVector("b", []float64{4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	if err := server.service.AddVector("c", []float64{7, 8, 9}); err != nil {
		t.Fatal(err)
	}

	// first page (limit=2)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/vectors?limit=2", nil)
	server.Router().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	var page struct {
		Vectors []struct {
			ID     string    `json:"id"`
			Values []float64 `json:"values"`
		} `json:"vectors"`
		NextCursor string `json:"next_cursor"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &page); err != nil {
		t.Fatalf("json.Unmarshal page1: %v", err)
	}
	if len(page.Vectors) != 2 {
		t.Fatalf("expected 2 vectors on page1, got %d", len(page.Vectors))
	}
	if page.NextCursor == "" {
		t.Fatalf("expected next_cursor on page1")
	}

	// second page using cursor
	rec2 := httptest.NewRecorder()
	q := url.Values{}
	q.Set("cursor", page.NextCursor)
	req2 := httptest.NewRequest(http.MethodGet, "/vectors?"+q.Encode(), nil)
	server.Router().ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec2.Code)
	}
	var page2 struct {
		Vectors []struct {
			ID string `json:"id"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(rec2.Body.Bytes(), &page2); err != nil {
		t.Fatalf("json.Unmarshal page2: %v", err)
	}
	if len(page2.Vectors) != 1 || page2.Vectors[0].ID != "c" {
		t.Fatalf("expected remaining vector 'c', got %+v", page2)
	}

	// using legacy 'after' raw id should behave the same
	rec3 := httptest.NewRecorder()
	req3 := httptest.NewRequest(http.MethodGet, "/vectors?after=b", nil)
	server.Router().ServeHTTP(rec3, req3)
	if rec3.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec3.Code)
	}
	var page3 struct {
		Vectors []struct {
			ID string `json:"id"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(rec3.Body.Bytes(), &page3); err != nil {
		t.Fatalf("json.Unmarshal page3: %v", err)
	}
	if len(page3.Vectors) != 1 || page3.Vectors[0].ID != "c" {
		t.Fatalf("expected remaining vector 'c' for after=b, got %+v", page3)
	}

	// ids_only should return only ids and not values
	rec4 := httptest.NewRecorder()
	req4 := httptest.NewRequest(http.MethodGet, "/vectors?ids_only=true", nil)
	server.Router().ServeHTTP(rec4, req4)
	if rec4.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec4.Code)
	}
	var idsOnly struct {
		Vectors []map[string]json.RawMessage `json:"vectors"`
	}
	if err := json.Unmarshal(rec4.Body.Bytes(), &idsOnly); err != nil {
		t.Fatalf("json.Unmarshal ids_only: %v", err)
	}
	if len(idsOnly.Vectors) != 3 {
		t.Fatalf("expected 3 ids for ids_only, got %d", len(idsOnly.Vectors))
	}
	if _, ok := idsOnly.Vectors[0]["values"]; ok {
		t.Fatalf("ids_only response included values: %s", rec4.Body.String())
	}

	// Cursor semantics should be lexicographic, not exact-id matching. A deleted
	// or otherwise missing cursor must not restart the listing from page one.
	rec5 := httptest.NewRecorder()
	req5 := httptest.NewRequest(http.MethodGet, "/vectors?after=b0", nil)
	server.Router().ServeHTTP(rec5, req5)
	if rec5.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec5.Code)
	}
	var page5 struct {
		Vectors []struct {
			ID string `json:"id"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(rec5.Body.Bytes(), &page5); err != nil {
		t.Fatalf("json.Unmarshal page5: %v", err)
	}
	if len(page5.Vectors) != 1 || page5.Vectors[0].ID != "c" {
		t.Fatalf("expected lexicographic page after b0 to start at c, got %+v", page5)
	}
}

func TestListVectorsRejectsBadPaginationInput(t *testing.T) {
	server := NewServerWithOptions(ServerOptions{})

	tests := []string{
		"/vectors?limit=0",
		"/vectors?limit=-1",
		"/vectors?limit=abc",
		"/vectors?cursor=!",
	}
	for _, path := range tests {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, path, nil)
		server.Router().ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("%s returned %d, want 400", path, rec.Code)
		}
	}
}

func TestParseTrustedProxiesAndIsTrustedProxy(t *testing.T) {
	prefixes := parseTrustedProxies([]string{"10.0.0.0/24", "127.0.0.1"})
	if len(prefixes) != 2 {
		t.Fatalf("expected 2 prefixes, got %d", len(prefixes))
	}
	s := NewServerWithOptions(ServerOptions{})
	s.trustedCIDRs = prefixes

	if !s.isTrustedProxy("10.0.0.5:1234") {
		t.Fatal("expected 10.0.0.5 to be trusted")
	}
	if s.isTrustedProxy("8.8.8.8:9999") {
		t.Fatal("did not expect 8.8.8.8 to be trusted")
	}
}

func TestUtilityHelpers(t *testing.T) {
	if got := firstNonEmpty("", "", "x"); got != "x" {
		t.Fatalf("expected x, got %q", got)
	}
	if got := normalizeServerProtocol("grpc"); got != "grpc" {
		t.Fatalf("expected grpc, got %q", got)
	}
	if got := normalizeServerProtocol("HTTP"); got != "http" {
		t.Fatalf("expected http, got %q", got)
	}

	// ensure router exists and handlers can be invoked directly
	srv := NewServerWithOptions(ServerOptions{})
	r := mux.NewRouter()
	srv.routes()
	if srv.Router() == nil || r == nil {
		t.Fatal("expected router to be set up")
	}
}
