package core

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

func BenchmarkReshardMigration(b *testing.B) {
	base := b.TempDir()
	source := NewShardedService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotEvery: 1001, VectorPath: filepath.Join(base, "source"), SnapshotPath: filepath.Join(base, "source-snapshot.json"), WALPath: filepath.Join(base, "source-wal.log")}, 1, 1)
	vectors := make([]index.Vector, 1000)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("v-%06d", i), Values: []float64{float64(i), 1, 2, 3}}
	}
	if err := source.AddVectors(vectors); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := NewShardedService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotEvery: 1001, VectorPath: filepath.Join(base, fmt.Sprintf("target-%d", i)), SnapshotPath: filepath.Join(base, fmt.Sprintf("target-%d-snapshot.json", i)), WALPath: filepath.Join(base, fmt.Sprintf("target-%d-wal.log", i))}, 4, 4)
		if err := source.MigrateTo(target); err != nil {
			b.Fatal(err)
		}
		if target.countVectors() != len(vectors) {
			b.Fatal("migration cardinality mismatch")
		}
		_ = target.Close()
	}
	_ = source.Close()
}

func BenchmarkReshardMigrationTargets(b *testing.B) {
	base := b.TempDir()
	count := 1000
	vectorStore := ""
	if os.Getenv("LUMENVEC_RESHARD_SEGMENTED") == "1" {
		vectorStore = "segment"
	}
	annOptions := ann.Options{}
	if raw := os.Getenv("LUMENVEC_RESHARD_M"); raw != "" {
		if m, err := strconv.Atoi(raw); err == nil && m > 0 {
			annOptions.M = m
		}
	}
	if raw := os.Getenv("LUMENVEC_RESHARD_EF_CONSTRUCTION"); raw != "" {
		if ef, err := strconv.Atoi(raw); err == nil && ef > 0 {
			annOptions.EfConstruction = ef
		}
	}
	if os.Getenv("LUMENVEC_RESHARD_SEGMENT_ROUTING") == "1" {
		annOptions.SegmentRouting = true
	}
	if raw := os.Getenv("LUMENVEC_RESHARD_VECTORS"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			count = parsed
		}
	}
	snapshotEvery := count + 1
	segmentMaxNodes := 10000
	if raw := os.Getenv("LUMENVEC_RESHARD_SEGMENT_NODES"); raw != "" {
		if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
			segmentMaxNodes = parsed
		}
	}
	if raw := os.Getenv("LUMENVEC_RESHARD_SNAPSHOT_EVERY"); raw != "" {
		if every, err := strconv.Atoi(raw); err == nil && every > 0 {
			snapshotEvery = every
		}
	}
	sourceShards := 1
	if raw := os.Getenv("LUMENVEC_RESHARD_SOURCE_SHARDS"); raw != "" {
		if configured, err := strconv.Atoi(raw); err == nil && configured > 0 {
			sourceShards = configured
		}
	}
	source := NewShardedService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SearchMode: "ann", VectorStore: vectorStore, ANNOptions: annOptions, ANNSegmentMaxNodes: segmentMaxNodes, SnapshotEvery: snapshotEvery, VectorPath: filepath.Join(base, "source"), SnapshotPath: filepath.Join(base, "source-snapshot.json"), WALPath: filepath.Join(base, "source-wal.log")}, sourceShards, sourceShards)
	const batchSize = 10000
	logReshardStage("source_ingest_start", 0)
	if os.Getenv("LUMENVEC_RESHARD_STREAMING") == "1" && os.Getenv("LUMENVEC_RESHARD_NATIVE_F32") == "1" {
		start := 0
		if err := source.AddVectorsStream32(func() ([]vectorBatch32, error) {
			if start >= count {
				return nil, nil
			}
			end := min(start+batchSize, count)
			batch := make([]vectorBatch32, end-start)
			for i := range batch {
				id := start + i
				batch[i] = vectorBatch32{ID: fmt.Sprintf("v-%06d", id), Values: reshardValues32(id)}
			}
			start = end
			return batch, nil
		}); err != nil {
			b.Fatal(err)
		}
	} else if os.Getenv("LUMENVEC_RESHARD_STREAMING") == "1" {
		start := 0
		if err := source.AddVectorsStream(func() ([]index.Vector, error) {
			if start >= count {
				return nil, nil
			}
			end := min(start+batchSize, count)
			batch := make([]index.Vector, end-start)
			for i := range batch {
				id := start + i
				batch[i] = index.Vector{ID: fmt.Sprintf("v-%06d", id), Values: reshardValues(id)}
			}
			start = end
			return batch, nil
		}); err != nil {
			b.Fatal(err)
		}
	} else {
		for start := 0; start < count; start += batchSize {
			end := min(start+batchSize, count)
			batch := make([]index.Vector, end-start)
			for i := range batch {
				id := start + i
				batch[i] = index.Vector{ID: fmt.Sprintf("v-%06d", id), Values: reshardValues(id)}
			}
			if err := source.AddVectors(batch); err != nil {
				b.Fatal(err)
			}
		}
	}
	maintenanceCtx, cancelMaintenance := context.WithTimeout(context.Background(), 2*time.Minute)
	if err := waitForReshardMaintenance(maintenanceCtx, source); err != nil {
		cancelMaintenance()
		b.Fatal(err)
	}
	cancelMaintenance()
	logReshardStage("source_ingest_complete", 0)
	logReshardMem("source_ready", 0, source.shards)
	defer source.Retire()
	for _, shards := range []int{2, 4, 8} {
		b.Run(fmt.Sprintf("target-%d", shards), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				target := NewShardedService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SearchMode: "ann", VectorStore: vectorStore, ANNOptions: annOptions, ANNSegmentMaxNodes: segmentMaxNodes, SnapshotEvery: snapshotEvery, VectorPath: filepath.Join(base, fmt.Sprintf("target-%d-%d", shards, i)), SnapshotPath: filepath.Join(base, fmt.Sprintf("target-%d-%d-snapshot.json", shards, i)), WALPath: filepath.Join(base, fmt.Sprintf("target-%d-%d-wal.log", shards, i))}, shards, shards)
				defer target.Close()
				logReshardStage("migration_start", shards)
				online := os.Getenv("LUMENVEC_RESHARD_ONLINE") == "1"
				onlineMetrics, err := migrateWithOnlineWorkload(source, target, shards, online)
				if err != nil {
					b.Fatal(err)
				}
				if online {
					fmt.Printf("RESHARD_ONLINE shards=%d searches=%d search_errors=%d baseline_p95_ms=%.6f search_p95_ms=%.6f search_p95_multiple=%.3f late_vectors=%d ingest_ms=%.3f\n",
						shards, onlineMetrics.searches, onlineMetrics.searchErrors,
						onlineMetrics.baselineP95MS, onlineMetrics.searchP95MS, onlineMetrics.searchP95Multiple,
						onlineMetrics.lateVectors, onlineMetrics.ingestMS)
				}
				logReshardStage("migration_complete", shards)
				logReshardMem("target_ready", shards, target.shards)
				expected := count + onlineMetrics.lateVectors
				if target.countVectors() != expected {
					b.Fatalf("migration cardinality mismatch: got %d want %d", target.countVectors(), expected)
				}
				logReshardStage("list_complete", shards)
				if b.N == 1 {
					sourceBytes := directoryBytes(filepath.Join(base, "source"))
					targetBytes := directoryBytes(filepath.Join(base, fmt.Sprintf("target-%d-%d", shards, i)))
					fmt.Printf("RESHARD_PERSISTENCE shards=%d source_bytes=%d target_bytes=%d total_bytes=%d vectors=%d\n", shards, sourceBytes, targetBytes, sourceBytes+targetBytes, count)
				}
				logReshardMem("target_closed", shards, target.shards)
			}
		})
	}
}

func waitForReshardMaintenance(ctx context.Context, service *ShardedService) error {
	for _, shard := range service.shards {
		svc, ok := shard.(*Service)
		if !ok {
			continue
		}
		index := svc.currentANNIndex()
		if maintenance, ok := index.(interface{ WaitForMaintenance(context.Context) error }); ok {
			if err := maintenance.WaitForMaintenance(ctx); err != nil {
				return err
			}
		}
	}
	return nil
}

type reshardOnlineMetrics struct {
	searches          int64
	searchErrors      int64
	searchP95MS       float64
	baselineP95MS     float64
	searchP95Multiple float64
	lateVectors       int
	ingestMS          float64
}

func migrateWithOnlineWorkload(source, target *ShardedService, shards int, enabled bool) (reshardOnlineMetrics, error) {
	if !enabled {
		return reshardOnlineMetrics{}, source.MigrateTo(target)
	}
	query := reshardValues(500000)
	baselineLatencies := make([]float64, 100)
	for i := range baselineLatencies {
		started := time.Now()
		if _, err := source.Search(query, 10); err != nil {
			return reshardOnlineMetrics{}, fmt.Errorf("baseline search: %w", err)
		}
		baselineLatencies[i] = float64(time.Since(started)) / float64(time.Millisecond)
	}
	sort.Float64s(baselineLatencies)
	baselineP95 := baselineLatencies[(len(baselineLatencies)-1)*95/100]
	lateVectors := 1000
	if raw := os.Getenv("LUMENVEC_RESHARD_ONLINE_VECTORS"); raw != "" {
		if configured, err := strconv.Atoi(raw); err == nil && configured > 0 {
			lateVectors = configured
		}
	}
	stop := make(chan struct{})
	var searches, searchErrors atomic.Int64
	latencies := make([]float64, 0, 4096)
	var latencyMu sync.Mutex
	var workers sync.WaitGroup
	workers.Add(1)
	go func() {
		defer workers.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			started := time.Now()
			if _, err := source.Search(query, 10); err != nil {
				searchErrors.Add(1)
			}
			elapsed := float64(time.Since(started)) / float64(time.Millisecond)
			searches.Add(1)
			latencyMu.Lock()
			if len(latencies) < cap(latencies) {
				latencies = append(latencies, elapsed)
			}
			latencyMu.Unlock()
		}
	}()
	ingestResult := make(chan struct {
		duration time.Duration
		err      error
	}, 1)
	go func() {
		started := time.Now()
		// Larger online batches keep the WAL/segment commit overhead visible in
		// the benchmark without turning each vector into a separate persistence
		// barrier. The migration semantics and late-vector count are unchanged.
		const batchSize = 1000
		nativeFloat32 := os.Getenv("LUMENVEC_RESHARD_NATIVE_F32") == "1"
		for start := 0; start < lateVectors; start += batchSize {
			end := min(start+batchSize, lateVectors)
			if nativeFloat32 {
				batch := make([]vectorBatch32, end-start)
				for i := range batch {
					id := start + i
					batch[i] = vectorBatch32{ID: fmt.Sprintf("late-%06d", id), Values: reshardValues32(1000000 + id)}
				}
				emitted := false
				if err := source.AddVectorsStream32(func() ([]vectorBatch32, error) {
					if emitted {
						return nil, nil
					}
					emitted = true
					return batch, nil
				}); err != nil {
					ingestResult <- struct {
						duration time.Duration
						err      error
					}{time.Since(started), err}
					return
				}
				continue
			}
			batch := make([]index.Vector, end-start)
			for i := range batch {
				id := start + i
				// Keep repeated benchmark subcases idempotent: the same late IDs
				// are upserted when the source is reused for the 2/4/8 matrix.
				batch[i] = index.Vector{ID: fmt.Sprintf("late-%06d", id), Values: reshardValues(1000000 + id)}
			}
			if err := source.AddVectors(batch); err != nil {
				ingestResult <- struct {
					duration time.Duration
					err      error
				}{time.Since(started), err}
				return
			}
		}
		ingestResult <- struct {
			duration time.Duration
			err      error
		}{duration: time.Since(started)}
	}()
	migrateErr := source.MigrateTo(target)
	ingest := <-ingestResult
	close(stop)
	workers.Wait()
	if migrateErr != nil {
		return reshardOnlineMetrics{}, migrateErr
	}
	if ingest.err != nil {
		return reshardOnlineMetrics{}, ingest.err
	}
	sort.Float64s(latencies)
	p95 := 0.0
	if len(latencies) > 0 {
		p95 = latencies[(len(latencies)-1)*95/100]
	}
	multiple := 0.0
	if baselineP95 > 0 {
		multiple = p95 / baselineP95
	}
	return reshardOnlineMetrics{
		searches: searches.Load(), searchErrors: searchErrors.Load(),
		searchP95MS: p95, baselineP95MS: baselineP95, searchP95Multiple: multiple, lateVectors: lateVectors,
		ingestMS: float64(ingest.duration) / float64(time.Millisecond),
	}, nil
}

func reshardValues(id int) []float64 {
	x := float64(id)
	cluster := id / 10000
	sign := func(bit uint) float64 {
		if cluster&(1<<bit) == 0 {
			return -1
		}
		return 1
	}
	return []float64{
		sign(0) + 0.05*math.Sin(x*0.017),
		sign(1) + 0.05*math.Cos(x*0.013),
		sign(2) + 0.05*math.Sin(x*0.007+1.3),
		sign(3) + 0.05*math.Cos(x*0.019-0.7),
	}
}

func reshardValues32(id int) []float32 {
	values := reshardValues(id)
	return []float32{float32(values[0]), float32(values[1]), float32(values[2]), float32(values[3])}
}

func logReshardMem(stage string, shards int, services []VectorService) {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	var annStructural, annVectors, annAdjacency uint64
	for _, shard := range services {
		if service, ok := shard.(*Service); ok {
			stats := service.Stats()
			annStructural += stats.ANNStructuralBytes
			annVectors += stats.ANNVectorBytes
			annAdjacency += stats.ANNAdjacencyBytes
		}
	}
	fmt.Printf("RESHARD_MEM stage=%s shards=%d heap_alloc=%d heap_inuse=%d heap_objects=%d total_alloc=%d num_gc=%d ann_structural=%d ann_vectors=%d ann_adjacency=%d\n", stage, shards, mem.HeapAlloc, mem.HeapInuse, mem.HeapObjects, mem.TotalAlloc, mem.NumGC, annStructural, annVectors, annAdjacency)
}

func logReshardStage(stage string, shards int) {
	fmt.Printf("RESHARD_STAGE stage=%s shards=%d at=%s\n", stage, shards, time.Now().UTC().Format(time.RFC3339Nano))
}

func directoryBytes(root string) int64 {
	var total int64
	_ = filepath.Walk(root, func(_ string, info os.FileInfo, err error) error {
		if err == nil && info != nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return total
}
