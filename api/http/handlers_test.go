package http

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	api2 "lumenvec/internal/api"

	"github.com/gorilla/mux"
)

func testServer(t *testing.T) *api2.Server {
	t.Helper()
	base := t.TempDir()
	return api2.NewServerWithOptions(api2.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		SnapshotPath: filepath.Join(base, "snapshot.json"),
		WALPath:      filepath.Join(base, "wal.log"),
	})
}

func TestRegisterRoutesAndHealth(t *testing.T) {
	server := testServer(t)
	handlers := NewHandlers(server)
	r := mux.NewRouter()
	handlers.RegisterRoutes(r)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	r.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
}

func TestHandlerWrappers(t *testing.T) {
	server := testServer(t)
	handlers := NewHandlers(server)

	create := httptest.NewRecorder()
	createReq := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"a","values":[1,2,3]}`))
	handlers.AddVector(create, createReq)
	if create.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", create.Code)
	}

	list := httptest.NewRecorder()
	listReq := httptest.NewRequest(http.MethodGet, "/vectors", nil)
	handlers.ListVectors(list, listReq)
	if list.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", list.Code)
	}
	var payload struct {
		Vectors []struct {
			ID string `json:"id"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(list.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if len(payload.Vectors) != 1 || payload.Vectors[0].ID != "a" {
		t.Fatalf("unexpected list: %+v", payload)
	}

	search := httptest.NewRecorder()
	searchReq := httptest.NewRequest(http.MethodPost, "/vectors/search", bytes.NewBufferString(`{"values":[1,2,3],"k":1}`))
	handlers.SearchVectors(search, searchReq)
	if search.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", search.Code)
	}
	filtered := httptest.NewRecorder()
	filteredReq := httptest.NewRequest(http.MethodPost, "/vectors/search", bytes.NewBufferString(`{"values":[1,2,3],"k":1,"filter_ids":["a"],"text_query":"A","metric":"cosine"}`))
	handlers.SearchVectors(filtered, filteredReq)
	if filtered.Code != http.StatusOK {
		t.Fatalf("filtered search status=%d body=%s", filtered.Code, filtered.Body.String())
	}
	invalidMetric := httptest.NewRecorder()
	invalidReq := httptest.NewRequest(http.MethodPost, "/vectors/search", bytes.NewBufferString(`{"values":[1,2,3],"k":1,"text_query":"a","metric":"manhattan"}`))
	handlers.SearchVectors(invalidMetric, invalidReq)
	if invalidMetric.Code != http.StatusBadRequest {
		t.Fatalf("invalid metric status=%d", invalidMetric.Code)
	}

	get := httptest.NewRecorder()
	getReq := httptest.NewRequest(http.MethodGet, "/vectors/a", nil)
	getReq = mux.SetURLVars(getReq, map[string]string{"id": "a"})
	handlers.GetVector(get, getReq)
	if get.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", get.Code)
	}

	del := httptest.NewRecorder()
	delReq := httptest.NewRequest(http.MethodDelete, "/vectors/a", nil)
	delReq = mux.SetURLVars(delReq, map[string]string{"id": "a"})
	handlers.DeleteVector(del, delReq)
	if del.Code != http.StatusNoContent {
		t.Fatalf("expected 204, got %d", del.Code)
	}
}

func TestListVectorsPaginationAndIdsOnly(t *testing.T) {
	server := testServer(t)
	handlers := NewHandlers(server)

	// create two vectors
	create1 := httptest.NewRecorder()
	createReq1 := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"a","values":[1,2,3]}`))
	handlers.AddVector(create1, createReq1)
	if create1.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", create1.Code)
	}

	create2 := httptest.NewRecorder()
	createReq2 := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"b","values":[4,5,6]}`))
	handlers.AddVector(create2, createReq2)
	if create2.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d", create2.Code)
	}

	// limit=1 should return only one vector
	list1 := httptest.NewRecorder()
	listReq1 := httptest.NewRequest(http.MethodGet, "/vectors?limit=1", nil)
	handlers.ListVectors(list1, listReq1)
	if list1.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", list1.Code)
	}
	var payload1 struct {
		Vectors []struct {
			ID     string    `json:"id"`
			Values []float64 `json:"values"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(list1.Body.Bytes(), &payload1); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if len(payload1.Vectors) != 1 {
		t.Fatalf("expected 1 vector, got %d", len(payload1.Vectors))
	}

	// ids_only should return only ids
	list2 := httptest.NewRecorder()
	listReq2 := httptest.NewRequest(http.MethodGet, "/vectors?ids_only=true", nil)
	handlers.ListVectors(list2, listReq2)
	if list2.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", list2.Code)
	}
	var payload2 struct {
		Vectors []map[string]json.RawMessage `json:"vectors"`
	}
	if err := json.Unmarshal(list2.Body.Bytes(), &payload2); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	if len(payload2.Vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(payload2.Vectors))
	}
	if _, ok := payload2.Vectors[0]["values"]; ok {
		t.Fatalf("ids_only response included values: %s", list2.Body.String())
	}
}
