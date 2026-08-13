package integration_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"lumenvec/internal/api"
)

func newTestServer(t *testing.T) *api.Server {
	base := t.TempDir()
	return api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		SnapshotPath: filepath.Join(base, "snapshot.json"),
		WALPath:      filepath.Join(base, "wal.log"),
	})
}

func TestAPIHealth(t *testing.T) {
	server := newTestServer(t)
	req, err := http.NewRequest(http.MethodGet, "/health", nil)
	if err != nil {
		t.Fatal(err)
	}

	recorder := httptest.NewRecorder()
	server.Router().ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "ok", recorder.Body.String())
}

func TestAPIVectorLifecycle(t *testing.T) {
	server := newTestServer(t)

	payload := map[string]interface{}{
		"id":     "1",
		"values": []float64{1, 2, 3},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	createReq := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewReader(body))
	createRec := httptest.NewRecorder()
	server.Router().ServeHTTP(createRec, createReq)
	assert.Equal(t, http.StatusCreated, createRec.Code)

	getReq := httptest.NewRequest(http.MethodGet, "/vectors/1", nil)
	getRec := httptest.NewRecorder()
	server.Router().ServeHTTP(getRec, getReq)
	assert.Equal(t, http.StatusOK, getRec.Code)

	var got map[string]interface{}
	err = json.Unmarshal(getRec.Body.Bytes(), &got)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, "1", got["id"])

	searchPayload := map[string]interface{}{
		"values": []float64{1.0, 2.0, 3.0},
		"k":      1,
	}
	searchBody, err := json.Marshal(searchPayload)
	if err != nil {
		t.Fatal(err)
	}
	searchReq := httptest.NewRequest(http.MethodPost, "/vectors/search", bytes.NewReader(searchBody))
	searchRec := httptest.NewRecorder()
	server.Router().ServeHTTP(searchRec, searchReq)
	assert.Equal(t, http.StatusOK, searchRec.Code)

	var searchResults []map[string]interface{}
	err = json.Unmarshal(searchRec.Body.Bytes(), &searchResults)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, 1, len(searchResults))
	assert.Equal(t, "1", searchResults[0]["id"])

	deleteReq := httptest.NewRequest(http.MethodDelete, "/vectors/1", nil)
	deleteRec := httptest.NewRecorder()
	server.Router().ServeHTTP(deleteRec, deleteReq)
	assert.Equal(t, http.StatusNoContent, deleteRec.Code)

	getAfterDeleteReq := httptest.NewRequest(http.MethodGet, "/vectors/1", nil)
	getAfterDeleteRec := httptest.NewRecorder()
	server.Router().ServeHTTP(getAfterDeleteRec, getAfterDeleteReq)
	assert.Equal(t, http.StatusNotFound, getAfterDeleteRec.Code)
}

func TestAPIBatchVectorLifecycle(t *testing.T) {
	server := newTestServer(t)

	body := bytes.NewBufferString(`{"vectors":[{"id":"doc-1","values":[1,2,3]},{"id":"doc-2","values":[4,5,6]}]}`)
	createReq := httptest.NewRequest(http.MethodPost, "/vectors/batch", body)
	createRec := httptest.NewRecorder()
	server.Router().ServeHTTP(createRec, createReq)
	assert.Equal(t, http.StatusCreated, createRec.Code)

	searchReq := httptest.NewRequest(http.MethodPost, "/vectors/search/batch", bytes.NewBufferString(`{"queries":[{"id":"q1","values":[1,2,3.1],"k":1},{"id":"q2","values":[4,5,6.1],"k":1}]}`))
	searchRec := httptest.NewRecorder()
	server.Router().ServeHTTP(searchRec, searchReq)
	assert.Equal(t, http.StatusOK, searchRec.Code)

	var got []map[string]interface{}
	if err := json.Unmarshal(searchRec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, 2, len(got))
	assert.Equal(t, "q1", got[0]["id"])
	assert.Equal(t, "q2", got[1]["id"])

	resultsA := got[0]["results"].([]interface{})
	resultsB := got[1]["results"].([]interface{})
	assert.Equal(t, "doc-1", resultsA[0].(map[string]interface{})["id"])
	assert.Equal(t, "doc-2", resultsB[0].(map[string]interface{})["id"])
}

func TestAPIBatchInsertIsAtomicOnConflict(t *testing.T) {
	server := newTestServer(t)

	seedReq := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"existing","values":[1,1,1]}`))
	seedRec := httptest.NewRecorder()
	server.Router().ServeHTTP(seedRec, seedReq)
	assert.Equal(t, http.StatusCreated, seedRec.Code)

	batchReq := httptest.NewRequest(http.MethodPost, "/vectors/batch", bytes.NewBufferString(`{"vectors":[{"id":"fresh","values":[2,2,2]},{"id":"existing","values":[3,3,3]}]}`))
	batchRec := httptest.NewRecorder()
	server.Router().ServeHTTP(batchRec, batchReq)
	assert.Equal(t, http.StatusConflict, batchRec.Code)

	getReq := httptest.NewRequest(http.MethodGet, "/vectors/fresh", nil)
	getRec := httptest.NewRecorder()
	server.Router().ServeHTTP(getRec, getReq)
	assert.Equal(t, http.StatusNotFound, getRec.Code)
}

func TestAPILimitsValidation(t *testing.T) {
	server := api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		MaxBodyBytes: 128,
		MaxVectorDim: 2,
		MaxK:         1,
		SnapshotPath: filepath.Join(t.TempDir(), "snapshot.json"),
	})

	payload := map[string]interface{}{
		"id":     "v-lim",
		"values": []float64{1, 2, 3},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	createReq := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewReader(body))
	createRec := httptest.NewRecorder()
	server.Router().ServeHTTP(createRec, createReq)
	assert.Equal(t, http.StatusBadRequest, createRec.Code)

	searchPayload := map[string]interface{}{
		"values": []float64{1, 2},
		"k":      2,
	}
	searchBody, err := json.Marshal(searchPayload)
	if err != nil {
		t.Fatal(err)
	}
	searchReq := httptest.NewRequest(http.MethodPost, "/vectors/search", bytes.NewReader(searchBody))
	searchRec := httptest.NewRecorder()
	server.Router().ServeHTTP(searchRec, searchReq)
	assert.Equal(t, http.StatusBadRequest, searchRec.Code)

	batchReq := httptest.NewRequest(http.MethodPost, "/vectors/batch", bytes.NewBufferString(`{"vectors":[{"id":"a","values":[1,2,3]}]}`))
	batchRec := httptest.NewRecorder()
	server.Router().ServeHTTP(batchRec, batchReq)
	assert.Equal(t, http.StatusBadRequest, batchRec.Code)
}

func TestAPISnapshotRecovery(t *testing.T) {
	snapshotPath := filepath.Join(t.TempDir(), "snapshot.json")
	walPath := filepath.Join(t.TempDir(), "wal.log")

	serverA := api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		SnapshotPath: snapshotPath,
		WALPath:      walPath,
	})

	payload := map[string]interface{}{
		"id":     "persist-1",
		"values": []float64{0.1, 0.2, 0.3},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	createReq := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewReader(body))
	createRec := httptest.NewRecorder()
	serverA.Router().ServeHTTP(createRec, createReq)
	assert.Equal(t, http.StatusCreated, createRec.Code)

	serverB := api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		SnapshotPath: snapshotPath,
		WALPath:      walPath,
	})
	getReq := httptest.NewRequest(http.MethodGet, "/vectors/persist-1", nil)
	getRec := httptest.NewRecorder()
	serverB.Router().ServeHTTP(getRec, getReq)
	assert.Equal(t, http.StatusOK, getRec.Code)
}

func TestAPIWALRecoveryWithoutSnapshotFlush(t *testing.T) {
	tmp := t.TempDir()
	snapshotPath := filepath.Join(tmp, "snapshot.json")
	walPath := filepath.Join(tmp, "wal.log")

	serverA := api.NewServerWithOptions(api.ServerOptions{
		Port:          ":0",
		ReadTimeout:   5 * time.Second,
		WriteTimeout:  5 * time.Second,
		SnapshotPath:  snapshotPath,
		WALPath:       walPath,
		SnapshotEvery: 1000,
	})

	createReq := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"wal-1","values":[4,5,6]}`))
	createRec := httptest.NewRecorder()
	serverA.Router().ServeHTTP(createRec, createReq)
	assert.Equal(t, http.StatusCreated, createRec.Code)

	serverB := api.NewServerWithOptions(api.ServerOptions{
		Port:          ":0",
		ReadTimeout:   5 * time.Second,
		WriteTimeout:  5 * time.Second,
		SnapshotPath:  snapshotPath,
		WALPath:       walPath,
		SnapshotEvery: 1000,
	})

	getReq := httptest.NewRequest(http.MethodGet, "/vectors/wal-1", nil)
	getRec := httptest.NewRecorder()
	serverB.Router().ServeHTTP(getRec, getReq)
	assert.Equal(t, http.StatusOK, getRec.Code)
}

func TestAPIKeyAuth(t *testing.T) {
	dataDir := t.TempDir()
	server := api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		AuthEnabled:  true,
		AuthAPIKey:   "secret-key",
		SnapshotPath: filepath.Join(dataDir, "snapshot.json"),
		WALPath:      filepath.Join(dataDir, "wal.log"),
	})

	req := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"a","values":[1,2]}`))
	rec := httptest.NewRecorder()
	server.Router().ServeHTTP(rec, req)
	assert.Equal(t, http.StatusUnauthorized, rec.Code)

	reqAuth := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"a","values":[1,2]}`))
	reqAuth.Header.Set("X-API-Key", "secret-key")
	recAuth := httptest.NewRecorder()
	server.Router().ServeHTTP(recAuth, reqAuth)
	assert.Equal(t, http.StatusCreated, recAuth.Code)
}

func TestAPIRateLimit(t *testing.T) {
	server := api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		RateLimitRPS: 1,
		SnapshotPath: filepath.Join(t.TempDir(), "snapshot.json"),
	})

	req1 := httptest.NewRequest(http.MethodGet, "/vectors/not-found", nil)
	rec1 := httptest.NewRecorder()
	server.Router().ServeHTTP(rec1, req1)
	assert.NotEqual(t, http.StatusTooManyRequests, rec1.Code)

	req2 := httptest.NewRequest(http.MethodGet, "/vectors/not-found", nil)
	rec2 := httptest.NewRecorder()
	server.Router().ServeHTTP(rec2, req2)
	assert.Equal(t, http.StatusTooManyRequests, rec2.Code)
}

func TestAPISearchModeANN(t *testing.T) {
	server := api.NewServerWithOptions(api.ServerOptions{
		Port:         ":0",
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		SearchMode:   "ann",
		SnapshotPath: filepath.Join(t.TempDir(), "snapshot.json"),
		WALPath:      filepath.Join(t.TempDir(), "wal.log"),
	})

	reqA := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"a","values":[1,2,3]}`))
	recA := httptest.NewRecorder()
	server.Router().ServeHTTP(recA, reqA)
	assert.Equal(t, http.StatusCreated, recA.Code)

	reqB := httptest.NewRequest(http.MethodPost, "/vectors", bytes.NewBufferString(`{"id":"b","values":[8,9,10]}`))
	recB := httptest.NewRecorder()
	server.Router().ServeHTTP(recB, reqB)
	assert.Equal(t, http.StatusCreated, recB.Code)

	searchReq := httptest.NewRequest(http.MethodPost, "/vectors/search", bytes.NewBufferString(`{"values":[1,2,3.1],"k":1}`))
	searchRec := httptest.NewRecorder()
	server.Router().ServeHTTP(searchRec, searchReq)
	assert.Equal(t, http.StatusOK, searchRec.Code)

	var got []map[string]interface{}
	err := json.Unmarshal(searchRec.Body.Bytes(), &got)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, 1, len(got))
	assert.Equal(t, "a", got[0]["id"])
}
