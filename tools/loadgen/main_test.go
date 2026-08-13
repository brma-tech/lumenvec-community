package main

import (
	"sync/atomic"
	"testing"
	"time"

	clientpkg "lumenvec/pkg/client"
)

type fakeLoadClient struct {
	searches atomic.Int64
	ingested atomic.Int64
}

func (c *fakeLoadClient) AddVectors(vectors []clientpkg.VectorPayload) error {
	c.ingested.Add(int64(len(vectors)))
	time.Sleep(time.Microsecond)
	return nil
}

func TestRunMixedWorkload(t *testing.T) {
	client := &fakeLoadClient{}
	stats, err := runMixedWorkload(client, "mixed", 0, 50, 40, 8, 7, 5, 4)
	if err != nil {
		t.Fatal(err)
	}
	if got := client.ingested.Load(); got != 50 {
		t.Fatalf("ingested = %d, want 50", got)
	}
	if got := client.searches.Load(); got != 40 {
		t.Fatalf("searches = %d, want 40", got)
	}
	if stats.ingested != 50 || stats.search.count != 40 || stats.vectorsPerSecond() <= 0 {
		t.Fatalf("unexpected mixed stats: %+v", stats)
	}
}

func (c *fakeLoadClient) SearchVector([]float64, int) ([]clientpkg.SearchResult, error) {
	c.searches.Add(1)
	time.Sleep(time.Microsecond)
	return []clientpkg.SearchResult{{ID: "doc-1", Distance: 0}}, nil
}

func (c *fakeLoadClient) Close() error {
	return nil
}

func TestPercentileDuration(t *testing.T) {
	values := []time.Duration{
		1 * time.Millisecond,
		2 * time.Millisecond,
		3 * time.Millisecond,
		4 * time.Millisecond,
		5 * time.Millisecond,
	}

	if got := percentileDuration(values, 50); got != 3*time.Millisecond {
		t.Fatalf("p50 = %v, want 3ms", got)
	}
	if got := percentileDuration(values, 95); got != 5*time.Millisecond {
		t.Fatalf("p95 = %v, want 5ms", got)
	}
	if got := percentileDuration(values, 99); got != 5*time.Millisecond {
		t.Fatalf("p99 = %v, want 5ms", got)
	}
}

func TestRunSearchesConcurrentStats(t *testing.T) {
	client := &fakeLoadClient{}
	stats, err := runSearches(client, 0, 10, 25, 8, 5, 4)
	if err != nil {
		t.Fatal(err)
	}

	if got := int(client.searches.Load()); got != 25 {
		t.Fatalf("searches = %d, want 25", got)
	}
	if stats.count != 25 {
		t.Fatalf("stats.count = %d, want 25", stats.count)
	}
	if stats.sampleTop1 != "doc-1" {
		t.Fatalf("sampleTop1 = %q, want doc-1", stats.sampleTop1)
	}
	if stats.p50 == 0 || stats.p95 == 0 || stats.p99 == 0 || stats.max == 0 {
		t.Fatalf("expected latency percentiles to be populated: %+v", stats)
	}
}
