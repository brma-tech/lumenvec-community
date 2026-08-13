package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	clientpkg "lumenvec/pkg/client"
)

type vectorClient interface {
	AddVectors(vectors []clientpkg.VectorPayload) error
	SearchVector(vector []float64, k int) ([]clientpkg.SearchResult, error)
	Close() error
}

type httpLoadClient struct {
	client *clientpkg.VectorClient
}

func (c *httpLoadClient) AddVectors(vectors []clientpkg.VectorPayload) error {
	return c.client.AddVectors(vectors)
}

func (c *httpLoadClient) SearchVector(vector []float64, k int) ([]clientpkg.SearchResult, error) {
	return c.client.SearchVector(vector, k)
}

func (c *httpLoadClient) Close() error { return nil }

func main() {
	var (
		transport   = flag.String("transport", "http", "Transport to use: http or grpc")
		baseURL     = flag.String("base-url", "http://localhost:19190", "HTTP base URL of the LumenVec server")
		grpcAddr    = flag.String("grpc-addr", "localhost:19191", "gRPC address of the LumenVec server")
		prefix      = flag.String("prefix", "demo", "Prefix used for generated vector IDs")
		vectors     = flag.Int("vectors", 500, "Number of vectors to ingest")
		searches    = flag.Int("searches", 200, "Number of search requests to execute")
		dim         = flag.Int("dim", 8, "Vector dimension")
		batchSize   = flag.Int("batch-size", 100, "Batch size used for ingest")
		topK        = flag.Int("k", 5, "Top-k used for searches")
		concurrency = flag.Int("concurrency", 1, "Number of concurrent search workers")
		mixed       = flag.Bool("mixed", false, "Run ingest and search concurrently")
	)
	flag.Parse()

	if *vectors <= 0 || *searches < 0 || *dim <= 0 || *batchSize <= 0 || *topK <= 0 || *concurrency <= 0 {
		fmt.Fprintln(os.Stderr, "vectors, searches, dim, batch-size, k, and concurrency must be positive")
		os.Exit(1)
	}

	client, target, err := newLoadClient(strings.ToLower(strings.TrimSpace(*transport)), *baseURL, *grpcAddr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "client init failed: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	namespace := namespaceOffset(*prefix)
	if *mixed && *searches > 0 {
		fmt.Printf("Running mixed workload: ingest=%d batch=%d searches=%d concurrency=%d against %s via %s\n", *vectors, *batchSize, *searches, *concurrency, target, *transport)
		mixedStats, err := runMixedWorkload(client, *prefix, namespace, *vectors, *searches, *dim, *batchSize, *topK, *concurrency)
		if err != nil {
			fmt.Fprintf(os.Stderr, "mixed load failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Mixed ingest summary: vectors=%d duration_ms=%.2f vectors_per_second=%.2f\n",
			mixedStats.ingested, mixedStats.ingestDuration.Seconds()*1000, mixedStats.vectorsPerSecond())
		printSearchSummary("Mixed search summary", mixedStats.search)
		fmt.Println("Load generation completed.")
		return
	}

	fmt.Printf("Ingesting %d vectors in batches of %d against %s via %s\n", *vectors, *batchSize, target, *transport)
	if err := ingestVectors(client, *prefix, namespace, *vectors, *dim, *batchSize); err != nil {
		fmt.Fprintf(os.Stderr, "ingest failed: %v\n", err)
		os.Exit(1)
	}

	if *searches > 0 {
		fmt.Printf("Running %d search requests with k=%d concurrency=%d via %s\n", *searches, *topK, *concurrency, *transport)
		stats, err := runSearches(client, namespace, *vectors, *searches, *dim, *topK, *concurrency)
		if err != nil {
			fmt.Fprintf(os.Stderr, "search load failed: %v\n", err)
			os.Exit(1)
		}

		printSearchSummary("Search summary", stats)
	}

	fmt.Println("Load generation completed.")
}

func printSearchSummary(label string, stats searchStats) {
	fmt.Printf("%s: count=%d avg_ms=%.2f p50_ms=%.2f p95_ms=%.2f p99_ms=%.2f max_ms=%.2f sample_top1=%s\n",
		label,
		stats.count,
		stats.total.Seconds()*1000/float64(stats.count),
		stats.p50.Seconds()*1000,
		stats.p95.Seconds()*1000,
		stats.p99.Seconds()*1000,
		stats.max.Seconds()*1000,
		stats.sampleTop1,
	)
}

type mixedWorkloadStats struct {
	ingested       int
	ingestDuration time.Duration
	search         searchStats
}

func (s mixedWorkloadStats) vectorsPerSecond() float64 {
	if s.ingestDuration <= 0 {
		return 0
	}
	return float64(s.ingested) / s.ingestDuration.Seconds()
}

func runMixedWorkload(client vectorClient, prefix string, namespace float64, totalVectors, searches, dim, batchSize, topK, concurrency int) (mixedWorkloadStats, error) {
	type ingestResult struct {
		duration time.Duration
		err      error
	}
	ingestDone := make(chan ingestResult, 1)
	go func() {
		started := time.Now()
		err := ingestVectors(client, prefix, namespace, totalVectors, dim, batchSize)
		ingestDone <- ingestResult{duration: time.Since(started), err: err}
	}()

	search, searchErr := runSearches(client, namespace, totalVectors, searches, dim, topK, concurrency)
	ingest := <-ingestDone
	stats := mixedWorkloadStats{
		ingested:       totalVectors,
		ingestDuration: ingest.duration,
		search:         search,
	}
	if ingest.err != nil {
		return mixedWorkloadStats{}, fmt.Errorf("ingest: %w", ingest.err)
	}
	if searchErr != nil {
		return mixedWorkloadStats{}, fmt.Errorf("search: %w", searchErr)
	}
	return stats, nil
}

func newLoadClient(transport, baseURL, grpcAddr string) (vectorClient, string, error) {
	switch transport {
	case "http":
		base := strings.TrimRight(baseURL, "/")
		return &httpLoadClient{client: clientpkg.NewVectorClient(base)}, base, nil
	case "grpc":
		grpcClient, err := clientpkg.NewGRPCVectorClient(grpcAddr)
		if err != nil {
			return nil, "", err
		}
		return grpcClient, grpcAddr, nil
	default:
		return nil, "", fmt.Errorf("unsupported transport %q", transport)
	}
}

func ingestVectors(client vectorClient, prefix string, namespace float64, total, dim, batchSize int) error {
	for start := 0; start < total; start += batchSize {
		end := min(start+batchSize, total)
		vectors := make([]clientpkg.VectorPayload, 0, end-start)
		for i := start; i < end; i++ {
			vectors = append(vectors, clientpkg.VectorPayload{
				ID:     fmt.Sprintf("%s-%06d", prefix, i),
				Values: generatedVector(namespace, i, dim),
			})
		}

		if err := client.AddVectors(vectors); err != nil {
			return fmt.Errorf("batch [%d:%d): %w", start, end, err)
		}
	}
	return nil
}

type searchStats struct {
	count      int
	total      time.Duration
	max        time.Duration
	p50        time.Duration
	p95        time.Duration
	p99        time.Duration
	sampleTop1 string
}

func runSearches(client vectorClient, namespace float64, totalVectors, searches, dim, topK, concurrency int) (searchStats, error) {
	latencies := make([]time.Duration, searches)
	errCh := make(chan error, 1)
	var next atomic.Int64
	var sampleTop1 atomic.Value

	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= searches {
					return
				}
				target := i % totalVectors
				query := perturbVector(generatedVector(namespace, target, dim), i)

				start := time.Now()
				results, err := client.SearchVector(query, topK)
				elapsed := time.Since(start)
				if err != nil {
					select {
					case errCh <- fmt.Errorf("search %d: %w", i, err):
					default:
					}
					return
				}

				latencies[i] = elapsed
				if i == 0 && len(results) > 0 {
					sampleTop1.Store(results[0].ID)
				}
			}
		}()
	}
	wg.Wait()

	select {
	case err := <-errCh:
		return searchStats{}, err
	default:
	}

	var stats searchStats
	for _, elapsed := range latencies {
		stats.total += elapsed
		if elapsed > stats.max {
			stats.max = elapsed
		}
	}
	stats.count = searches
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	stats.p50 = percentileDuration(latencies, 50)
	stats.p95 = percentileDuration(latencies, 95)
	stats.p99 = percentileDuration(latencies, 99)
	if value, ok := sampleTop1.Load().(string); ok {
		stats.sampleTop1 = value
	}
	return stats, nil
}

func percentileDuration(values []time.Duration, percentile int) time.Duration {
	if len(values) == 0 {
		return 0
	}
	if percentile <= 0 {
		return values[0]
	}
	if percentile >= 100 {
		return values[len(values)-1]
	}
	idx := (len(values)*percentile + 99) / 100
	if idx < 1 {
		idx = 1
	}
	return values[idx-1]
}

func generatedVector(namespace float64, seed, dim int) []float64 {
	cluster := seed % 10
	values := make([]float64, dim)
	base := namespace + float64(cluster)*10
	for i := range values {
		values[i] = base + math.Sin(float64(seed+i))*0.25 + float64((seed+i)%7)/100
	}
	return values
}

func perturbVector(values []float64, seed int) []float64 {
	out := make([]float64, len(values))
	copy(out, values)
	for i := range out {
		out[i] += math.Cos(float64(seed+i)) * 0.01
	}
	return out
}

func namespaceOffset(prefix string) float64 {
	var sum int
	for _, ch := range prefix {
		sum += int(ch)
	}
	return float64(sum%1000) * 100
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
