package api

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
)

type usageObserverStub struct {
	searches atomic.Uint64
	ingested atomic.Uint64
}

func (o *usageObserverStub) ObserveSearchRequests(value uint64)  { o.searches.Add(value) }
func (o *usageObserverStub) ObserveIngestedVectors(value uint64) { o.ingested.Add(value) }

func TestHTTPUsageObserverCountsOnlySuccessfulOperations(t *testing.T) {
	observer := &usageObserverStub{}
	directory := t.TempDir()
	server := NewServerWithOptions(ServerOptions{Port: ":0", MaxVectorDim: 8, MaxK: 10, SearchMode: "exact", SnapshotPath: filepath.Join(directory, "snapshot.json"), WALPath: filepath.Join(directory, "wal.log"), UsageObserver: observer})
	defer server.Close()
	request := func(path, body string) int {
		r := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(body))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		server.Router().ServeHTTP(w, r)
		return w.Code
	}
	if status := request("/v1/vectors", `{"id":"one","values":[1,0]}`); status != http.StatusCreated {
		t.Fatalf("add status=%d", status)
	}
	if status := request("/v1/vectors", `{"id":"one","values":[1,0]}`); status != http.StatusConflict {
		t.Fatalf("duplicate status=%d", status)
	}
	if status := request("/v1/vectors/search", `{"values":[1,0],"k":1}`); status != http.StatusOK {
		t.Fatalf("search status=%d", status)
	}
	if status := request("/v1/vectors/search/batch", `{"queries":[{"id":"a","values":[1,0],"k":1},{"id":"b","values":[1,0],"k":1}]}`); status != http.StatusOK {
		t.Fatalf("batch status=%d", status)
	}
	if got := observer.ingested.Load(); got != 1 {
		t.Fatalf("ingested=%d", got)
	}
	if got := observer.searches.Load(); got != 3 {
		t.Fatalf("searches=%d", got)
	}
}
