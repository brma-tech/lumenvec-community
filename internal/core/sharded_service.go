package core

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"lumenvec/internal/index"
)

// VectorService is the transport-facing contract implemented by a single
// shard and by the fan-out router.
type VectorService interface {
	AddVector(id string, values []float64) error
	AddVectors(vectors []index.Vector) error
	GetVector(id string) (index.Vector, error)
	ListVectors() []index.Vector
	ListVectorsPage(opts ListVectorsOptions) ListVectorsPage
	DeleteVector(id string) error
	Search(values []float64, k int) ([]SearchResult, error)
	SearchBatch(queries []BatchSearchQuery) ([]BatchSearchResult, error)
	Stats() ServiceStats
	Close() error
}

// metadataVectorService is an optional extension implemented by the built-in
// service. Keeping it out of VectorService preserves compatibility with
// lightweight transports and test doubles that only carry vector values.
type metadataVectorService interface {
	AddVectorWithMetadata(id string, values []float64, metadata map[string]string) error
}

type metadataVectorBatchService interface {
	AddVectorsWithMetadata(vectors []index.Vector, metadata []map[string]string) error
}

// vectorBatch32Service is an optional bulk-ingest path used by resharding.
// It lets a segment-backed source stream its native float32 payload without
// first materializing a float64 ListVectorsPage.
type vectorBatch32Service interface {
	AddVectors32([]vectorBatch32) error
}

type deltaReplayService interface {
	ReplayDeltasSince(uint64, func(DeltaRecord) error) error
}

type metadataPresence interface{ hasAnyMetadata() bool }

type VectorBatch32 struct {
	ID     string
	Values []float32
}

type vectorBatch32 = VectorBatch32

type vectorMetadataReader interface {
	VectorMetadata(id string) map[string]string
}

type vectorMetadataBatchReader interface {
	VectorMetadataBatch(ids []string) map[string]map[string]string
}

// checkedVectorPageReader is implemented by transports that can distinguish
// an empty page from a failed page request. Migration uses it when available
// so a network error cannot be mistaken for end-of-data.
type checkedVectorPageReader interface {
	ListVectorsPageChecked(ListVectorsOptions) (ListVectorsPage, error)
}

// migrationPageLimitProvider advertises a transport's maximum reliable page
// size. It lets the generic migration path respect API-level limits.
type migrationPageLimitProvider interface {
	MigrationPageLimit() int
}

type filteredVectorService interface {
	SearchFiltered(values []float64, k int, filter VectorFilter) ([]SearchResult, error)
}

type filteredMetricVectorService interface {
	SearchFilteredMetric([]float64, int, VectorFilter, DistanceMetric) ([]SearchResult, error)
}

type ShardedService struct {
	shards                 []VectorService
	shardIDs               []string
	shardHashes            []uint64
	routingMode            string
	fanoutConcurrency      int
	migrationAfterSnapshot func()
	migrationBeforeDrain   func()
	closeOnce              sync.Once
	closeErr               error
}

var errNativeMigrationUnsupported = errors.New("native float32 migration unsupported")

const (
	ShardRoutingModulo     = "modulo-v1"
	ShardRoutingRendezvous = "rendezvous-v1"
)

// MigrateTo copies a consistent vector snapshot into a target shard layout.
// It does not switch traffic or delete the source; the control plane can
// validate the target, drain the source, and then atomically replace the
// router. This makes resharding restartable and prevents partial migration
// from silently becoming the active topology.
func (s *ShardedService) MigrateTo(target *ShardedService) error {
	if s == nil || target == nil || len(target.shards) == 0 {
		return errors.New("source and target shard layouts are required")
	}
	if handled, err := s.migratePrebuiltRendezvous(target); handled {
		return err
	}
	checkpointMode := os.Getenv("LUMENVEC_RESHARD_CHECKPOINT_DIR") != ""
	if checkpointMode || len(target.ListVectorsPage(ListVectorsOptions{Limit: 1}).Vectors) == 0 {
		if err := s.migratePaged(target); err != nil {
			return err
		}
		count, err := s.countVectorsChecked()
		if err != nil {
			return err
		}
		targetCount, err := target.countVectorsChecked()
		if err != nil {
			return err
		}
		if count == targetCount {
			return nil
		}
	}
	for pass := 0; pass < 4; pass++ {
		vectors := s.ListVectors()
		got := target.ListVectors()
		enrichVectorMetadata(s, vectors)
		enrichVectorMetadata(target, got)
		existing := make(map[string]index.Vector, len(got))
		for _, vector := range got {
			existing[vector.ID] = vector
		}
		delta := make([]index.Vector, 0, len(vectors))
		for _, vector := range vectors {
			if old, ok := existing[vector.ID]; !ok || vectorChecksum([]index.Vector{old}) != vectorChecksum([]index.Vector{vector}) {
				delta = append(delta, vector)
			}
		}
		if len(delta) > 0 {
			if err := addVectorsPreservingMetadata(target, delta); err != nil {
				return fmt.Errorf("migrate vectors: %w", err)
			}
		}
		latest := s.ListVectors()
		final := target.ListVectors()
		if len(latest) == len(final) && vectorChecksum(latest) == vectorChecksum(final) {
			return nil
		}
		if pass == 3 {
			return errors.New("migration did not converge while source was changing")
		}
	}
	return errors.New("migration did not converge")
}

// DestinationPrebuiltBuilds reports successful ANN generations constructed
// by remote data nodes. It is benchmark evidence that graph construction was
// distributed rather than silently falling back to the coordinator.
func (s *ShardedService) DestinationPrebuiltBuilds() uint64 {
	if s == nil {
		return 0
	}
	var total uint64
	for _, shard := range s.shards {
		if measured, ok := shard.(interface{ DestinationPrebuiltBuilds() uint64 }); ok {
			total += measured.DestinationPrebuiltBuilds()
		}
	}
	return total
}

func (s *ShardedService) migratePaged(target *ShardedService) error {
	if err := s.migratePaged32(target); err == nil {
		return nil
	} else if !errors.Is(err, errNativeMigrationUnsupported) {
		return err
	}
	// Keep writes to each destination shard aligned with the immutable ANN
	// segment size. A source page is redistributed by the target router, so
	// sending it directly would turn a 10k page into (for example) eight 1.25k
	// writes. Those undersized writes trigger repeated compactions and rebuilds.
	const migrationSegmentSize = 10000
	migrationPageSize := s.migrationPageLimit(migrationSegmentSize)
	checkpointPath := ""
	var checkpoint ReshardCheckpoint
	if dir := os.Getenv("LUMENVEC_RESHARD_CHECKPOINT_DIR"); dir != "" {
		checkpointPath = filepath.Join(dir, "reshard-checkpoint.json")
		if saved, err := LoadReshardCheckpoint(checkpointPath); err == nil && saved.State != "complete" {
			checkpoint = saved
		}
	}
	if checkpoint.MigrationID == "" {
		checkpoint = ReshardCheckpoint{MigrationID: "paged-reshard", SourceGeneration: 1, TargetShard: 0, State: "copying"}
	}
	pending := make([][]index.Vector, len(target.shards))
	flush := func(fullOnly bool) error {
		batches := make([][]index.Vector, len(target.shards))
		for shardID := range pending {
			if len(pending[shardID]) == 0 || (fullOnly && len(pending[shardID]) < migrationSegmentSize) {
				continue
			}
			if fullOnly {
				batches[shardID] = pending[shardID][:migrationSegmentSize]
				pending[shardID] = pending[shardID][migrationSegmentSize:]
			} else {
				batches[shardID] = pending[shardID]
				pending[shardID] = nil
			}
		}
		// ANN construction is CPU and allocation intensive. Preserve the
		// configured fan-out for small layouts, but cap larger migrations to
		// avoid oversubscribing the host with N concurrent graph builders.
		migrationParallelism := target.fanoutConcurrency
		if migrationParallelism <= 0 {
			migrationParallelism = 1
		}
		if migrationParallelism > 4 {
			migrationParallelism = 2
		}
		if raw := os.Getenv("LUMENVEC_RESHARD_MIGRATION_PARALLELISM"); raw != "" {
			if configured, err := strconv.Atoi(raw); err == nil && configured > 0 {
				migrationParallelism = configured
			}
		}
		err := target.parallelShardsLimit(migrationParallelism, func(shardID int, shard VectorService) error {
			return addVectorsToShardPreservingMetadata(shard, batches[shardID])
		})
		// Release temporary page/builder allocations before the next migration
		// window. This is intentionally once per full window, not per segment.
		runtime.GC()
		return err
	}
	cursor := checkpoint.Cursor
	for {
		page, err := s.listVectorsPageChecked(ListVectorsOptions{AfterID: cursor, Limit: migrationPageSize})
		if err != nil {
			return fmt.Errorf("read migration page: %w", err)
		}
		if len(page.Vectors) == 0 {
			if err := flush(false); err != nil {
				return err
			}
			if checkpointPath != "" {
				checkpoint.Cursor, checkpoint.State = cursor, "complete"
				if err := SaveReshardCheckpoint(checkpointPath, checkpoint); err != nil {
					return err
				}
			}
			return nil
		}
		enrichVectorMetadata(s, page.Vectors)
		for _, vector := range page.Vectors {
			shardID := target.ShardForID(vector.ID)
			pending[shardID] = append(pending[shardID], vector)
		}
		if err := flush(true); err != nil {
			return fmt.Errorf("paged migration: %w", err)
		}
		if checkpointPath != "" {
			checkpoint.Cursor = page.NextCursor
			checkpoint.VectorsCopied += uint64(len(page.Vectors))
			if err := SaveReshardCheckpoint(checkpointPath, checkpoint); err != nil {
				return err
			}
		}
		if page.NextCursor == "" {
			if err := flush(false); err != nil {
				return err
			}
			if checkpointPath != "" {
				checkpoint.State = "complete"
				if err := SaveReshardCheckpoint(checkpointPath, checkpoint); err != nil {
					return err
				}
			}
			return nil
		}
		cursor = page.NextCursor
	}
}

// migratePaged32 is selected only when every source/destination shard is the
// built-in service and the source store can stream float32 values. Metadata
// vectors fall back to the compatibility path because their sidecar must be
// copied atomically with the vector write.
func (s *ShardedService) migratePaged32(target *ShardedService) error {
	sources := make([]rangeVector32Reader, len(s.shards))
	for i, shard := range s.shards {
		svc, ok := shard.(*Service)
		if !ok {
			return errNativeMigrationUnsupported
		}
		if svc.hasAnyMetadata() {
			return fmt.Errorf("%w: metadata requires compatibility migration", errNativeMigrationUnsupported)
		}
		reader, ok := svc.vectorStore.(rangeVector32Reader)
		if !ok {
			return fmt.Errorf("%w: source store does not expose float32 streaming", errNativeMigrationUnsupported)
		}
		sources[i] = reader
	}
	dests := make([]vectorBatch32Service, len(target.shards))
	for i, shard := range target.shards {
		dest, ok := shard.(vectorBatch32Service)
		if !ok {
			return fmt.Errorf("%w: destination does not expose float32 bulk ingest", errNativeMigrationUnsupported)
		}
		dests[i] = dest
	}
	snapshotOffsets := make([]uint64, len(sources))
	for i, shard := range s.shards {
		if svc, ok := shard.(*Service); ok {
			snapshotOffsets[i] = svc.CurrentDeltaOffset()
		}
	}
	migrateOne := func(sourceID int, reader rangeVector32Reader) error {
		sourceService := s.shards[sourceID].(*Service)
		batches := make([][]vectorBatch32, len(dests))
		arenas := make([][]float32, len(dests))
		// 20k is the default compromise measured on the 200k/4-shard gate;
		// operators can lower it when RSS is the tighter constraint.
		nativeMigrationWindow := 20000
		if raw := os.Getenv("LUMENVEC_RESHARD_NATIVE_WINDOW"); raw != "" {
			if configured, err := strconv.Atoi(raw); err == nil && configured > 0 {
				nativeMigrationWindow = configured
			}
		}
		var batchErr error
		offset := uint64(0)
		nativeCheckpointPath := ""
		if dir := os.Getenv("LUMENVEC_RESHARD_CHECKPOINT_DIR"); dir != "" {
			nativeCheckpointPath = filepath.Join(dir, fmt.Sprintf("native-source-%d.json", sourceID))
			if saved, err := LoadReshardCheckpoint(nativeCheckpointPath); err == nil {
				if saved.State == "complete" && len(sources) == 1 && uint64(target.countVectors()) == saved.VectorsCopied {
					return nil
				}
				offset = saved.VectorsCopied
			}
		}
		seen := offset
		windowCount := 0
		failAfterVectors := uint64(0)
		if raw := os.Getenv("LUMENVEC_RESHARD_FAIL_AFTER_VECTORS"); raw != "" {
			if configured, err := strconv.ParseUint(raw, 10, 64); err == nil {
				failAfterVectors = configured
			}
		}
		flushWindow := func() error {
			for shardID, batch := range batches {
				if len(batch) == 0 {
					continue
				}
				if err := dests[shardID].AddVectors32(batch); err != nil {
					return err
				}
				batches[shardID] = nil
				arenas[shardID] = nil
			}
			// The native migration window owns temporary ID/value batches. Force a
			// collection after publication so a long reshard does not retain every
			// completed window until the next GC cycle. This bounds RSS at the cost
			// of predictable, window-level GC work.
			runtime.GC()
			windowCount = 0
			if nativeCheckpointPath != "" {
				if err := SaveReshardCheckpoint(nativeCheckpointPath, ReshardCheckpoint{MigrationID: fmt.Sprintf("native-%d", sourceID), SourceGeneration: 1, TargetShard: 0, VectorsCopied: seen, State: "copying"}); err != nil {
					return err
				}
			}
			if failAfterVectors > 0 && seen >= failAfterVectors {
				return fmt.Errorf("injected reshard interruption after %d vectors", seen)
			}
			return nil
		}
		consume := func(id string, values []float32) bool {
			seen++
			windowCount++
			shardID := target.ShardForID(id)
			// Keep one reusable backing arena per destination window. Segment
			// readers reuse their decode buffer, so retaining values directly is
			// unsafe; the arena avoids one heap allocation per vector.
			arena := arenas[shardID]
			if len(arena) == 0 {
				arena = make([]float32, 0, len(values)*nativeMigrationWindow)
			}
			start := len(arena)
			arena = append(arena, values...)
			arenas[shardID] = arena
			copyValues := arena[start:len(arena)]
			batches[shardID] = append(batches[shardID], vectorBatch32{ID: id, Values: copyValues})
			if len(batches[shardID]) >= nativeMigrationWindow {
				if err := dests[shardID].AddVectors32(batches[shardID]); err != nil {
					batchErr = err
					return false
				}
				batches[shardID] = nil
				arenas[shardID] = nil
			}
			if windowCount >= nativeMigrationWindow*len(dests) {
				if err := flushWindow(); err != nil {
					batchErr = err
					return false
				}
			}
			return true
		}
		if offsetReader, ok := reader.(rangeVector32OffsetReader); ok {
			offsetReader.RangeVectors32From(offset, consume)
		} else {
			reader.RangeVectors32(func(id string, values []float32) bool {
				if seen < offset {
					seen++
					return true
				}
				return consume(id, values)
			})
		}
		if batchErr != nil {
			return batchErr
		}
		for shardID, batch := range batches {
			if len(batch) > 0 {
				if err := dests[shardID].AddVectors32(batch); err != nil {
					return err
				}
			}
		}
		// Drain mutations that arrived while this source was being copied.
		// Replaying from the last durable checkpoint is safe because destination
		// writes are upserts and deletes are idempotent.
		if snapshotOffsets[sourceID] > 0 {
			if replay, ok := interface{}(sourceService).(deltaReplayService); ok {
				offset := snapshotOffsets[sourceID]
				if nativeCheckpointPath != "" {
					if err := SaveReshardCheckpoint(nativeCheckpointPath, ReshardCheckpoint{MigrationID: fmt.Sprintf("native-%d", sourceID), SourceGeneration: 1, TargetShard: 0, AppliedWAL: offset, VectorsCopied: seen, State: "draining"}); err != nil {
						return err
					}
				}
				stableReads := 0
				for pass := 0; pass < 4 && stableReads < 2; pass++ {
					passStarted := time.Now()
					latest := make([]map[string]DeltaRecord, len(dests))
					if err := replay.ReplayDeltasSince(offset, func(delta DeltaRecord) error {
						shardID := target.ShardForID(delta.VectorID)
						if latest[shardID] == nil {
							latest[shardID] = make(map[string]DeltaRecord)
						}
						// Only the final mutation for an ID matters at this
						// cutover offset. Applying every intermediate WAL entry
						// would rebuild the ANN graph once per vector.
						latest[shardID][delta.VectorID] = delta
						return nil
					}); err != nil {
						return err
					}
					replayElapsed := time.Since(passStarted)
					upsertCounts := make([]int, len(dests))
					deleteCounts := make([]int, len(dests))
					applyErr := target.parallelShardsLimit(min(len(dests), 4), func(shardID int, shard VectorService) error {
						records := latest[shardID]
						if len(records) == 0 {
							return nil
						}
						upserts := make([]vectorBatch32, 0, len(records))
						for _, delta := range records {
							if delta.Deleted {
								deleteCounts[shardID]++
								if err := shard.DeleteVector(delta.VectorID); err != nil && !errors.Is(err, index.ErrVectorNotFound) {
									return err
								}
								continue
							}
							upsertCounts[shardID]++
							// The source snapshot may already have copied an
							// earlier value. Tombstone it before atomically
							// publishing the replacement generation.
							_ = shard.DeleteVector(delta.VectorID)
							upserts = append(upserts, vectorBatch32{ID: delta.VectorID, Values: delta.Values})
						}
						if len(upserts) == 0 {
							return nil
						}
						if os.Getenv("LUMENVEC_RESHARD_STAGED") == "1" {
							if staged, ok := shard.(*Service); ok {
								return staged.addVectors32Staged(upserts)
							}
							return errors.New("staged native migration requires a local service destination")
						}
						return dests[shardID].AddVectors32(upserts)
					})
					if applyErr != nil {
						return applyErr
					}
					if os.Getenv("LUMENVEC_RESHARD_TRACE") == "1" {
						upsertCount, deleteCount := 0, 0
						for shardID := range upsertCounts {
							upsertCount += upsertCounts[shardID]
							deleteCount += deleteCounts[shardID]
						}
						fmt.Printf("RESHARD_NATIVE_DELTA source=%d pass=%d upserts=%d deletes=%d replay_ms=%.3f apply_ms=%.3f\n",
							sourceID, pass, upsertCount, deleteCount,
							float64(replayElapsed)/float64(time.Millisecond),
							float64(time.Since(passStarted)-replayElapsed)/float64(time.Millisecond))
					}
					current := sourceService.CurrentDeltaOffset()
					if current == offset {
						stableReads++
					} else {
						stableReads = 0
						offset = current
					}
				}
				if nativeCheckpointPath != "" && stableReads >= 2 {
					if err := SaveReshardCheckpoint(nativeCheckpointPath, ReshardCheckpoint{MigrationID: fmt.Sprintf("native-%d", sourceID), SourceGeneration: 1, TargetShard: 0, AppliedWAL: offset, VectorsCopied: seen, State: "complete"}); err != nil {
						return err
					}
				}
			}
		}
		if nativeCheckpointPath != "" {
			state := "complete"
			if batchErr != nil {
				state = "copying"
			}
			if err := SaveReshardCheckpoint(nativeCheckpointPath, ReshardCheckpoint{MigrationID: fmt.Sprintf("native-%d", sourceID), SourceGeneration: 1, TargetShard: 0, VectorsCopied: seen, State: state}); err != nil {
				return err
			}
		}
		return nil
	}
	parallelism := 1 // preserve the historical bounded-RSS default.
	if raw := os.Getenv("LUMENVEC_RESHARD_SOURCE_PARALLELISM"); raw != "" {
		if configured, err := strconv.Atoi(raw); err == nil && configured > 0 {
			parallelism = min(configured, len(sources))
		}
	}
	return runNativeMigrationSources(sources, parallelism, migrateOne)
}

// runNativeMigrationSources copies independent source shards concurrently.
// Destination writes remain protected by their shard services; the explicit
// limit keeps ANN builders and window arenas within the host budget.
func runNativeMigrationSources(sources []rangeVector32Reader, parallelism int, migrate func(int, rangeVector32Reader) error) error {
	if parallelism <= 0 {
		parallelism = 1
	}
	parallelism = min(parallelism, len(sources))
	sem := make(chan struct{}, parallelism)
	errs := make(chan error, len(sources))
	var wg sync.WaitGroup
	for sourceID, reader := range sources {
		sourceID, reader := sourceID, reader
		wg.Add(1)
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			if err := migrate(sourceID, reader); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	return errors.Join(readErrors(errs)...)
}

func enrichVectorMetadata(source *ShardedService, vectors []index.Vector) {
	byShard := make(map[int][]int)
	for i := range vectors {
		shardID := source.ShardForID(vectors[i].ID)
		byShard[shardID] = append(byShard[shardID], i)
	}
	for shardID, positions := range byShard {
		shard := source.shards[shardID]
		if reader, ok := shard.(vectorMetadataBatchReader); ok {
			ids := make([]string, len(positions))
			positionByID := make(map[string]int, len(positions))
			for i, pos := range positions {
				ids[i] = vectors[pos].ID
				positionByID[vectors[pos].ID] = pos
			}
			for id, metadata := range reader.VectorMetadataBatch(ids) {
				if pos, ok := positionByID[id]; ok {
					vectors[pos].Metadata = metadata
				}
			}
			continue
		}
		if reader, ok := shard.(vectorMetadataReader); ok {
			for _, pos := range positions {
				vectors[pos].Metadata = reader.VectorMetadata(vectors[pos].ID)
			}
		}
	}
}

func addVectorsPreservingMetadata(target *ShardedService, vectors []index.Vector) error {
	if len(vectors) == 0 {
		return nil
	}
	// Grouping remains the fast path for vectors without metadata. Metadata
	// vectors use the explicit API so the sidecar and metadata WAL stay in sync.
	plain := make([]index.Vector, 0, len(vectors))
	for _, vector := range vectors {
		if len(vector.Metadata) == 0 {
			plain = append(plain, vector)
			continue
		}
		shard := target.shards[target.ShardForID(vector.ID)]
		if svc, ok := shard.(metadataVectorService); ok {
			if err := svc.AddVectorWithMetadata(vector.ID, vector.Values, vector.Metadata); err != nil {
				return err
			}
			continue
		}
		plain = append(plain, vector)
	}
	if len(plain) > 0 {
		return target.AddVectors(plain)
	}
	return nil
}

func addVectorsToShardPreservingMetadata(shard VectorService, vectors []index.Vector) error {
	if len(vectors) == 0 {
		return nil
	}
	plain := make([]index.Vector, 0, len(vectors))
	for _, vector := range vectors {
		if len(vector.Metadata) == 0 {
			plain = append(plain, vector)
			continue
		}
		if svc, ok := shard.(metadataVectorService); ok {
			if err := svc.AddVectorWithMetadata(vector.ID, vector.Values, vector.Metadata); err != nil {
				return err
			}
			continue
		}
		plain = append(plain, vector)
	}
	if len(plain) > 0 {
		return shard.AddVectors(plain)
	}
	return nil
}

func (s *ShardedService) countVectors() int {
	count, _ := s.countVectorsChecked()
	return count
}

func (s *ShardedService) countVectorsChecked() (int, error) {
	allCounted := len(s.shards) > 0
	for _, shard := range s.shards {
		if _, ok := shard.(interface {
			VectorCount() (int, error)
			VectorCountAuthoritative()
		}); !ok {
			allCounted = false
			break
		}
	}
	if allCounted {
		var total atomic.Int64
		if err := s.parallelShards(func(_ int, shard VectorService) error {
			count, err := shard.(interface{ VectorCount() (int, error) }).VectorCount()
			if err != nil {
				return err
			}
			total.Add(int64(count))
			return nil
		}); err != nil {
			return 0, err
		}
		if total.Load() > int64(^uint(0)>>1) {
			return 0, errors.New("sharded vector count exceeds platform int")
		}
		return int(total.Load()), nil
	}
	return s.countVectorsCheckedPaged()
}

func (s *ShardedService) countVectorsCheckedPaged() (int, error) {
	count := 0
	var cursor string
	pageLimit := s.migrationPageLimit(1024)
	for {
		page, err := s.listVectorsPageChecked(ListVectorsOptions{AfterID: cursor, Limit: pageLimit, IDsOnly: true})
		if err != nil {
			return 0, err
		}
		count += len(page.Vectors)
		if page.NextCursor == "" {
			return count, nil
		}
		cursor = page.NextCursor
	}
}

func (s *ShardedService) migrationPageLimit(requested int) int {
	limit := requested
	for _, shard := range s.shards {
		if provider, ok := shard.(migrationPageLimitProvider); ok && provider.MigrationPageLimit() > 0 {
			// ListVectorsPage reads one extra item from each shard to produce a
			// globally stable cursor, so keep one slot below the transport cap.
			limit = min(limit, max(1, provider.MigrationPageLimit()-1))
		}
	}
	return limit
}

// Retire closes a layout after a successful atomic cutover. Unlike Close it
// skips ANN checkpoint generation for built-in shards; the retired layout is
// no longer a recovery target and its canonical vector segments stay durable.
func (s *ShardedService) Retire() error {
	if s == nil {
		return nil
	}
	errCh := make(chan error, len(s.shards))
	var wg sync.WaitGroup
	for _, shard := range s.shards {
		shard := shard
		wg.Add(1)
		go func() {
			defer wg.Done()
			if service, ok := shard.(*Service); ok {
				errCh <- service.CloseRetired()
			} else {
				errCh <- shard.Close()
			}
		}()
	}
	wg.Wait()
	close(errCh)
	return errors.Join(readErrors(errCh)...)
}

func vectorChecksum(vectors []index.Vector) [32]byte {
	ordered := append([]index.Vector(nil), vectors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ID < ordered[j].ID })
	h := sha256.New()
	for _, v := range ordered {
		_, _ = h.Write([]byte(v.ID))
		var buf [8]byte
		for _, value := range v.Values {
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(value))
			_, _ = h.Write(buf[:])
		}
		keys := make([]string, 0, len(v.Metadata))
		for key := range v.Metadata {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			_, _ = h.Write([]byte(key))
			_, _ = h.Write([]byte{0})
			_, _ = h.Write([]byte(v.Metadata[key]))
			_, _ = h.Write([]byte{0})
		}
	}
	var out [32]byte
	copy(out[:], h.Sum(nil))
	return out
}

func NewShardedService(opts ServiceOptions, shardCount, fanoutConcurrency int) *ShardedService {
	if shardCount <= 0 {
		shardCount = 1
	}
	return newShardedService(opts, make([]string, shardCount), fanoutConcurrency, ShardRoutingModulo)
}

// NewShardedServiceWithIDs enables versioned rendezvous routing. Shard IDs
// are durable topology identities: scale-out callers preserve the existing
// IDs and append new ones, so only keys won by a new node move.
func NewShardedServiceWithIDs(opts ServiceOptions, shardIDs []string, fanoutConcurrency int) (*ShardedService, error) {
	if len(shardIDs) == 0 {
		return nil, errors.New("at least one durable shard ID is required")
	}
	if err := validateShardIDs(shardIDs); err != nil {
		return nil, err
	}
	return newShardedService(opts, append([]string(nil), shardIDs...), fanoutConcurrency, ShardRoutingRendezvous), nil
}

func newShardedService(opts ServiceOptions, shardIDs []string, fanoutConcurrency int, routingMode string) *ShardedService {
	shardCount := len(shardIDs)
	if fanoutConcurrency <= 0 || fanoutConcurrency > shardCount {
		fanoutConcurrency = shardCount
	}
	shards := make([]VectorService, shardCount)
	var wg sync.WaitGroup
	for i := range shards {
		i := i
		shardOpts := opts
		shardDir := "shard-" + formatShardID(i)
		if opts.VectorPath != "" {
			shardOpts.VectorPath = filepath.Join(opts.VectorPath, shardDir)
		}
		if opts.SnapshotPath != "" {
			shardOpts.SnapshotPath = filepath.Join(filepath.Dir(opts.SnapshotPath), shardDir, filepath.Base(opts.SnapshotPath))
		}
		if opts.WALPath != "" {
			shardOpts.WALPath = filepath.Join(filepath.Dir(opts.WALPath), shardDir, filepath.Base(opts.WALPath))
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			shards[i] = NewService(shardOpts)
		}()
	}
	wg.Wait()
	return &ShardedService{
		shards: shards, shardIDs: shardIDs, shardHashes: durableShardHashes(shardIDs, routingMode),
		routingMode: routingMode, fanoutConcurrency: fanoutConcurrency,
	}
}

func newShardedServiceFromShards(shards []VectorService, fanoutConcurrency int) *ShardedService {
	if fanoutConcurrency <= 0 || fanoutConcurrency > len(shards) {
		fanoutConcurrency = len(shards)
	}
	return &ShardedService{shards: shards, routingMode: ShardRoutingModulo, fanoutConcurrency: fanoutConcurrency}
}

func NewShardedServiceFromShards(shards []VectorService, fanoutConcurrency int) *ShardedService {
	return newShardedServiceFromShards(shards, fanoutConcurrency)
}

// NewShardedServiceFromShardsWithIDs is the remote/control-plane counterpart
// of NewShardedServiceWithIDs. IDs must remain stable across topology versions.
func NewShardedServiceFromShardsWithIDs(shards []VectorService, shardIDs []string, fanoutConcurrency int) (*ShardedService, error) {
	if len(shards) == 0 || len(shards) != len(shardIDs) {
		return nil, errors.New("shards and durable shard IDs must be non-empty and have equal length")
	}
	if err := validateShardIDs(shardIDs); err != nil {
		return nil, err
	}
	if fanoutConcurrency <= 0 || fanoutConcurrency > len(shards) {
		fanoutConcurrency = len(shards)
	}
	return &ShardedService{
		shards: append([]VectorService(nil), shards...), shardIDs: append([]string(nil), shardIDs...),
		shardHashes: durableShardHashes(shardIDs, ShardRoutingRendezvous),
		routingMode: ShardRoutingRendezvous, fanoutConcurrency: fanoutConcurrency,
	}, nil
}

func durableShardHashes(shardIDs []string, routingMode string) []uint64 {
	if routingMode != ShardRoutingRendezvous {
		return nil
	}
	hashes := make([]uint64, len(shardIDs))
	for i, id := range shardIDs {
		hashes[i] = stableFNV64(id)
	}
	return hashes
}

func validateShardIDs(shardIDs []string) error {
	seen := make(map[string]struct{}, len(shardIDs))
	for _, id := range shardIDs {
		if strings.TrimSpace(id) == "" {
			return errors.New("durable shard ID is required")
		}
		if _, exists := seen[id]; exists {
			return errors.New("durable shard IDs must be unique")
		}
		seen[id] = struct{}{}
	}
	return nil
}

func formatShardID(id int) string {
	return fmt.Sprintf("%04d", id)
}

func (s *ShardedService) ShardForID(id string) int {
	if len(s.shards) == 0 {
		return 0
	}
	if s.routingMode == ShardRoutingRendezvous && len(s.shardHashes) == len(s.shards) {
		keyHash := stableFNV64(id)
		winner, best := 0, rendezvousScoreHashes(keyHash, s.shardHashes[0])
		for shard := 1; shard < len(s.shardHashes); shard++ {
			if score := rendezvousScoreHashes(keyHash, s.shardHashes[shard]); score > best {
				winner, best = shard, score
			}
		}
		return winner
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(id))
	return int(h.Sum64() % uint64(len(s.shards))) // #nosec G115 -- modulo bounds result below len(shards)
}

func rendezvousScore(key, shardID string) uint64 {
	return rendezvousScoreHash(stableFNV64(key), shardID)
}

func rendezvousScoreHash(keyHash uint64, shardID string) uint64 {
	return rendezvousScoreHashes(keyHash, stableFNV64(shardID))
}

func rendezvousScoreHashes(keyHash, shardHash uint64) uint64 {
	// FNV supplies stable input hashes while the SplitMix64 finalizer removes
	// the strong cross-candidate correlation that plain concatenated FNV has
	// under highest-random-weight selection.
	value := keyHash ^ shardHash
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func stableFNV64(value string) uint64 {
	hash := uint64(14695981039346656037)
	for i := 0; i < len(value); i++ {
		hash ^= uint64(value[i])
		hash *= 1099511628211
	}
	return hash
}

func (s *ShardedService) AddVector(id string, values []float64) error {
	if len(s.shards) == 0 {
		return errors.New("no shards configured")
	}
	return s.shards[s.ShardForID(id)].AddVector(id, values)
}

// AddVectorWithMetadata routes a vector and its structured fields to the same
// shard. The method is intentionally additive so existing VectorService
// clients remain source-compatible.
func (s *ShardedService) AddVectorWithMetadata(id string, values []float64, metadata map[string]string) error {
	if len(s.shards) == 0 {
		return errors.New("no shards configured")
	}
	shard := s.shards[s.ShardForID(id)]
	if svc, ok := shard.(metadataVectorService); ok {
		return svc.AddVectorWithMetadata(id, values, metadata)
	}
	return shard.AddVector(id, values)
}

func (s *ShardedService) AddVectors(vectors []index.Vector) error {
	if len(vectors) == 0 {
		return ErrInvalidValues
	}
	groups := make([][]index.Vector, len(s.shards))
	metadataGroups := make([][]index.Vector, len(s.shards))
	for _, vec := range vectors {
		shardID := s.ShardForID(vec.ID)
		if len(vec.Metadata) > 0 {
			metadataGroups[shardID] = append(metadataGroups[shardID], vec)
		} else {
			groups[shardID] = append(groups[shardID], vec)
		}
	}
	succeeded := make([]bool, len(s.shards))
	err := s.parallelShards(func(shardID int, shard VectorService) error {
		if len(groups[shardID]) == 0 && len(metadataGroups[shardID]) == 0 {
			return nil
		}
		if len(groups[shardID]) > 0 {
			if err := shard.AddVectors(groups[shardID]); err != nil {
				return err
			}
		}
		if len(metadataGroups[shardID]) > 0 {
			if batchShard, ok := shard.(metadataVectorBatchService); ok {
				vectors := make([]index.Vector, len(metadataGroups[shardID]))
				metadata := make([]map[string]string, len(metadataGroups[shardID]))
				for i, vec := range metadataGroups[shardID] {
					vectors[i] = index.Vector{ID: vec.ID, Values: vec.Values}
					metadata[i] = vec.Metadata
				}
				if err := batchShard.AddVectorsWithMetadata(vectors, metadata); err != nil {
					return err
				}
				succeeded[shardID] = true
				return nil
			}
			metadataShard, ok := shard.(metadataVectorService)
			if !ok {
				return errors.New("shard does not support vector metadata")
			}
			for _, vec := range metadataGroups[shardID] {
				if err := metadataShard.AddVectorWithMetadata(vec.ID, vec.Values, vec.Metadata); err != nil {
					return err
				}
			}
		}
		succeeded[shardID] = true
		return nil
	})
	if err == nil {
		return nil
	}
	for shardID, ok := range succeeded {
		if !ok {
			continue
		}
		for _, vec := range append(groups[shardID], metadataGroups[shardID]...) {
			_ = s.shards[shardID].DeleteVector(vec.ID)
		}
	}
	return err
}

func (s *ShardedService) AddVectorsWithMetadata(vectors []index.Vector, metadata []map[string]string) error {
	if len(vectors) != len(metadata) {
		return fmt.Errorf("metadata count %d does not match vector count %d", len(metadata), len(vectors))
	}
	withMetadata := make([]index.Vector, len(vectors))
	for i, vector := range vectors {
		withMetadata[i] = index.Vector{ID: vector.ID, Values: vector.Values, Metadata: metadata[i]}
	}
	return s.AddVectors(withMetadata)
}

// AddVectorsStream forwards incremental batches to the sharded bulk path.
// Each batch is committed independently, allowing the producer to release it
// before the next batch is allocated.
func (s *ShardedService) AddVectorsStream(next func() ([]index.Vector, error)) error {
	if next == nil {
		return ErrInvalidValues
	}
	for {
		batch, err := next()
		if err != nil {
			return err
		}
		if batch == nil {
			return nil
		}
		if len(batch) == 0 {
			continue
		}
		if err := s.AddVectors(batch); err != nil {
			return err
		}
	}
}

// AddVectorsStream32 is the native float32 counterpart used by binary
// importers. Each batch is partitioned once and committed through the shard
// AddVectors32 fast path, avoiding float64 conversion and per-vector fallback.
func (s *ShardedService) AddVectorsStream32(next func() ([]vectorBatch32, error)) error {
	if next == nil {
		return ErrInvalidValues
	}
	for {
		batch, err := next()
		if err != nil {
			return err
		}
		if batch == nil {
			return nil
		}
		if len(batch) == 0 {
			continue
		}
		groups := make([][]vectorBatch32, len(s.shards))
		for _, vec := range batch {
			if len(s.shards) == 0 {
				return errors.New("no shards configured")
			}
			shardID := s.ShardForID(vec.ID)
			groups[shardID] = append(groups[shardID], vec)
		}
		if err := s.parallelShards(func(shardID int, shard VectorService) error {
			if len(groups[shardID]) == 0 {
				return nil
			}
			native, ok := shard.(interface{ AddVectors32([]vectorBatch32) error })
			if !ok {
				return errors.New("shard does not support float32 bulk ingestion")
			}
			return native.AddVectors32(groups[shardID])
		}); err != nil {
			return err
		}
	}
}

func (s *ShardedService) GetVector(id string) (index.Vector, error) {
	if len(s.shards) == 0 {
		return index.Vector{}, index.ErrVectorNotFound
	}
	return s.shards[s.ShardForID(id)].GetVector(id)
}

func (s *ShardedService) DeleteVector(id string) error {
	if len(s.shards) == 0 {
		return index.ErrVectorNotFound
	}
	return s.shards[s.ShardForID(id)].DeleteVector(id)
}

func (s *ShardedService) Search(values []float64, k int) ([]SearchResult, error) {
	return s.SearchContext(context.Background(), values, k)
}

func (s *ShardedService) SearchContext(ctx context.Context, values []float64, k int) ([]SearchResult, error) {
	perShard := make([][]SearchResult, len(s.shards))
	if err := s.parallelShards(func(shardID int, shard VectorService) error {
		results, err := SearchWithContext(ctx, shard, values, k)
		perShard[shardID] = results
		return err
	}); err != nil {
		return nil, err
	}
	return mergeSearchResults(perShard, k), nil
}

// SearchFiltered fans a structured predicate out to shards that support the
// optional filtered-search extension, then merges the globally best results.
// It fails closed when a shard cannot evaluate the predicate; returning
// unfiltered results would violate correctness.
func (s *ShardedService) SearchFiltered(values []float64, k int, filter VectorFilter) ([]SearchResult, error) {
	perShard := make([][]SearchResult, len(s.shards))
	if err := s.parallelShards(func(shardID int, shard VectorService) error {
		if filtered, ok := shard.(filteredVectorService); ok {
			results, err := filtered.SearchFiltered(values, k, filter)
			perShard[shardID] = results
			return err
		}
		if filter != nil {
			return ErrFilteredUnsupported
		}
		results, err := shard.Search(values, k)
		perShard[shardID] = results
		return err
	}); err != nil {
		return nil, err
	}
	return mergeSearchResults(perShard, k), nil
}

// SearchFilteredMetric preserves metric semantics across shards, including
// unfiltered cosine and inner-product queries. Each shard returns its bounded
// local top-k and the coordinator merges the globally comparable distances.
func (s *ShardedService) SearchFilteredMetric(values []float64, k int, filter VectorFilter, metric DistanceMetric) ([]SearchResult, error) {
	return s.SearchFilteredMetricContext(context.Background(), values, k, filter, metric)
}

func (s *ShardedService) SearchFilteredMetricContext(ctx context.Context, values []float64, k int, filter VectorFilter, metric DistanceMetric) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	perShard := make([][]SearchResult, len(s.shards))
	if err := s.parallelShards(func(shardID int, shard VectorService) error {
		if filtered, ok := shard.(filteredMetricVectorService); ok {
			results, err := SearchFilteredMetricWithContext(ctx, filtered, values, k, filter, metric)
			perShard[shardID] = results
			return err
		}
		if metric == MetricL2 && filter == nil {
			results, err := SearchWithContext(ctx, shard, values, k)
			perShard[shardID] = results
			return err
		}
		return ErrFilteredUnsupported
	}); err != nil {
		return nil, err
	}
	return mergeSearchResults(perShard, k), nil
}

func (s *ShardedService) WarmMetricANN(metrics []DistanceMetric) error {
	return s.parallelShards(func(_ int, shard VectorService) error {
		warmer, ok := shard.(interface {
			WarmMetricANN([]DistanceMetric) error
		})
		if !ok {
			return ErrFilteredUnsupported
		}
		return warmer.WarmMetricANN(metrics)
	})
}

// SearchStructured preserves shard-local candidate reduction for text
// queries, then merges the bounded top-k result from each shard.
func (s *ShardedService) SearchStructured(values []float64, k int, filter StructuredFilter, metric DistanceMetric) ([]SearchResult, error) {
	return s.SearchStructuredContext(context.Background(), values, k, filter, metric)
}

func (s *ShardedService) SearchStructuredContext(ctx context.Context, values []float64, k int, filter StructuredFilter, metric DistanceMetric) ([]SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	perShard := make([][]SearchResult, len(s.shards))
	if err := s.parallelShards(func(shardID int, shard VectorService) error {
		if structured, ok := shard.(interface {
			SearchStructured([]float64, int, StructuredFilter, DistanceMetric) ([]SearchResult, error)
		}); ok {
			results, err := SearchStructuredWithContext(ctx, structured, values, k, filter, metric)
			perShard[shardID] = results
			return err
		}
		if filtered, ok := shard.(filteredMetricVectorService); ok {
			results, err := SearchFilteredMetricWithContext(ctx, filtered, values, k, filter.Match, metric)
			perShard[shardID] = results
			return err
		}
		return ErrFilteredUnsupported
	}); err != nil {
		return nil, err
	}
	return mergeSearchResults(perShard, k), nil
}

func (s *ShardedService) SearchBatch(queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	return s.SearchBatchContext(context.Background(), queries)
}

func (s *ShardedService) SearchBatchContext(ctx context.Context, queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	if len(queries) == 0 {
		return nil, ErrInvalidValues
	}
	perShard := make([][]BatchSearchResult, len(s.shards))
	if err := s.parallelShards(func(shardID int, shard VectorService) error {
		results, err := SearchBatchWithContext(ctx, shard, queries)
		perShard[shardID] = results
		return err
	}); err != nil {
		return nil, err
	}
	out := make([]BatchSearchResult, len(queries))
	for queryIndex, query := range queries {
		groups := make([][]SearchResult, 0, len(s.shards))
		for shardID := range s.shards {
			if queryIndex < len(perShard[shardID]) {
				groups = append(groups, perShard[shardID][queryIndex].Results)
			}
		}
		id := query.ID
		if id == "" {
			id = perShard[0][queryIndex].ID
		}
		out[queryIndex] = BatchSearchResult{ID: id, Results: mergeSearchResults(groups, query.K)}
	}
	return out, nil
}

func mergeSearchResults(groups [][]SearchResult, k int) []SearchResult {
	merged := make([]SearchResult, 0, len(groups)*k)
	for _, group := range groups {
		merged = append(merged, group...)
	}
	sort.Slice(merged, func(i, j int) bool {
		if merged[i].Distance != merged[j].Distance {
			return merged[i].Distance < merged[j].Distance
		}
		return merged[i].ID < merged[j].ID
	})
	unique := merged[:0]
	seen := make(map[string]struct{}, len(merged))
	for _, result := range merged {
		if _, duplicate := seen[result.ID]; duplicate {
			continue
		}
		seen[result.ID] = struct{}{}
		unique = append(unique, result)
		if len(unique) == k {
			break
		}
	}
	return unique
}

func (s *ShardedService) ListVectors() []index.Vector {
	all := make([]index.Vector, 0)
	for shardID, shard := range s.shards {
		for _, vector := range shard.ListVectors() {
			// During online rendezvous cutover the previous owner may retain a
			// stale physical copy until asynchronous cleanup. Only the current
			// owner contributes to the logical collection.
			if s.ShardForID(vector.ID) == shardID {
				all = append(all, vector)
			}
		}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].ID < all[j].ID })
	enrichVectorMetadata(s, all)
	return all
}

func (s *ShardedService) ListVectorsPage(opts ListVectorsOptions) ListVectorsPage {
	page, _ := s.listVectorsPageChecked(opts)
	return page
}

func (s *ShardedService) listVectorsPageChecked(opts ListVectorsOptions) (ListVectorsPage, error) {
	if opts.Limit <= 0 {
		return ListVectorsPage{}, nil
	}
	// A page from each shard is independent. Fetch them concurrently so a
	// remote topology is bounded by the slowest shard rather than the sum of
	// every RPC latency. Ordering remains global below.
	pages := make([]ListVectorsPage, len(s.shards))
	if err := s.parallelShards(func(shardID int, shard VectorService) error {
		pageOpts := ListVectorsOptions{AfterID: opts.AfterID, Limit: opts.Limit + 1, IDsOnly: opts.IDsOnly}
		if checked, ok := shard.(checkedVectorPageReader); ok {
			page, err := checked.ListVectorsPageChecked(pageOpts)
			if err != nil {
				return err
			}
			pages[shardID] = page
			return nil
		}
		pages[shardID] = shard.ListVectorsPage(pageOpts)
		return nil
	}); err != nil {
		return ListVectorsPage{}, err
	}
	candidates := make([]index.Vector, 0, len(s.shards)*(opts.Limit+1))
	for shardID, page := range pages {
		for _, vector := range page.Vectors {
			if s.ShardForID(vector.ID) == shardID {
				candidates = append(candidates, vector)
			}
		}
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].ID < candidates[j].ID })
	if !opts.IDsOnly {
		enrichVectorMetadata(s, candidates)
	}
	// A transport can cap a request below opts.Limit (for example, to keep a
	// gRPC response below its byte budget). Its non-empty cursor is therefore
	// authoritative: using only candidate count would silently stop a remote
	// reshard after the first capped page.
	hasMore := len(candidates) > opts.Limit
	if !hasMore {
		for _, page := range pages {
			if page.NextCursor != "" {
				hasMore = true
				break
			}
		}
	}
	if len(candidates) > opts.Limit {
		candidates = candidates[:opts.Limit]
	}
	next := ""
	if hasMore {
		if len(candidates) > 0 {
			next = candidates[len(candidates)-1].ID
		} else {
			for _, page := range pages {
				if page.NextCursor != "" && page.NextCursor > opts.AfterID && (next == "" || page.NextCursor < next) {
					next = page.NextCursor
				}
			}
		}
	}
	return ListVectorsPage{Vectors: candidates, NextCursor: next}, nil
}

func (s *ShardedService) Stats() ServiceStats {
	stats := ServiceStats{ShardCount: len(s.shards), ANNCheckpointLoaded: len(s.shards) > 0}
	for i, shard := range s.shards {
		part := shard.Stats()
		stats.add(part)
		if i == 0 {
			stats.ANNProfile = part.ANNProfile
			stats.ANNM = part.ANNM
			stats.ANNEfConstruction = part.ANNEfConstruction
			stats.ANNEfSearch = part.ANNEfSearch
			stats.ANNSearchExecution = part.ANNSearchExecution
			stats.ANNEFBudgetMode = part.ANNEFBudgetMode
			stats.ANNEFGlobalPercent = part.ANNEFGlobalPercent
			stats.ANNPrimaryIndexEnabled = part.ANNPrimaryIndexEnabled
		}
		stats.ANNCheckpointLoaded = stats.ANNCheckpointLoaded && part.ANNCheckpointLoaded
	}
	return stats
}

func (s *ShardedService) parallelShards(fn func(int, VectorService) error) error {
	return s.parallelShardsLimit(s.fanoutConcurrency, fn)
}

func (s *ShardedService) parallelShardsLimit(limit int, fn func(int, VectorService) error) error {
	if len(s.shards) == 0 {
		return errors.New("no shards configured")
	}
	if limit <= 0 {
		limit = 1
	}
	errCh := make(chan error, len(s.shards))
	var wg sync.WaitGroup
	jobs := make(chan int)
	for worker := 0; worker < min(limit, len(s.shards)); worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for shardID := range jobs {
				if err := fn(shardID, s.shards[shardID]); err != nil {
					errCh <- err
				}
			}
		}()
	}
	for shardID := range s.shards {
		jobs <- shardID
	}
	close(jobs)
	wg.Wait()
	close(errCh)
	return errors.Join(readErrors(errCh)...)
}

func readErrors(errCh <-chan error) []error {
	var errs []error
	for err := range errCh {
		errs = append(errs, err)
	}
	return errs
}

func (s *ShardedService) Close() error {
	s.closeOnce.Do(func() {
		errCh := make(chan error, len(s.shards))
		var wg sync.WaitGroup
		for _, shard := range s.shards {
			shard := shard
			wg.Add(1)
			go func() {
				defer wg.Done()
				errCh <- shard.Close()
			}()
		}
		wg.Wait()
		close(errCh)
		s.closeErr = errors.Join(readErrors(errCh)...)
	})
	return s.closeErr
}

// Checkpoint persists independent shard recovery points concurrently.
func (s *ShardedService) Checkpoint() error {
	if s == nil {
		return nil
	}
	errCh := make(chan error, len(s.shards))
	var wg sync.WaitGroup
	for _, shard := range s.shards {
		checkpoint, ok := shard.(interface{ Checkpoint() error })
		if !ok {
			continue
		}
		wg.Add(1)
		go func() { defer wg.Done(); errCh <- checkpoint.Checkpoint() }()
	}
	wg.Wait()
	close(errCh)
	return errors.Join(readErrors(errCh)...)
}

func (s *ServiceStats) add(other ServiceStats) {
	s.SearchRequestsTotal += other.SearchRequestsTotal
	s.ExactSearchesTotal += other.ExactSearchesTotal
	s.ANNSearchesTotal += other.ANNSearchesTotal
	s.ANNSearchHitsTotal += other.ANNSearchHitsTotal
	s.ANNSearchFallbacks += other.ANNSearchFallbacks
	s.ANNSearchErrorsTotal += other.ANNSearchErrorsTotal
	s.ANNCandidatesReturned += other.ANNCandidatesReturned
	s.ANNEvalSamplesTotal += other.ANNEvalSamplesTotal
	s.ANNEvalTop1Matches += other.ANNEvalTop1Matches
	s.ANNEvalOverlapResults += other.ANNEvalOverlapResults
	s.ANNEvalComparedResults += other.ANNEvalComparedResults
	s.ANNNodes += other.ANNNodes
	s.ANNDeleted += other.ANNDeleted
	s.ANNSegments += other.ANNSegments
	s.ANNCompactionPending += other.ANNCompactionPending
	s.ANNCompacting += other.ANNCompacting
	s.ANNCompactionCompactable += other.ANNCompactionCompactable
	s.ANNReaders += other.ANNReaders
	s.ANNRetiredSegments += other.ANNRetiredSegments
	s.ANNReclaimedSegments += other.ANNReclaimedSegments
	s.ANNReclaimErrors += other.ANNReclaimErrors
	if other.ANNCompactionMemoryBudget > s.ANNCompactionMemoryBudget {
		s.ANNCompactionMemoryBudget = other.ANNCompactionMemoryBudget
	}
	s.ANNCompactionMemoryReserved += other.ANNCompactionMemoryReserved
	if other.ANNCompactionMemoryEstimate > s.ANNCompactionMemoryEstimate {
		s.ANNCompactionMemoryEstimate = other.ANNCompactionMemoryEstimate
	}
	s.ANNCompactionMemoryDeferred += other.ANNCompactionMemoryDeferred
	s.ANNPrimaryBuildActive = s.ANNPrimaryBuildActive || other.ANNPrimaryBuildActive
	s.ANNPrimaryBuildTotal += other.ANNPrimaryBuildTotal
	s.ANNPrimaryBuildDone += other.ANNPrimaryBuildDone
	if other.ANNPrimaryBuildDurationMs > s.ANNPrimaryBuildDurationMs {
		s.ANNPrimaryBuildDurationMs = other.ANNPrimaryBuildDurationMs
	}
	s.ANNPrimaryBuildFailures += other.ANNPrimaryBuildFailures
	s.ANNSearchBudgetQueries += other.ANNSearchBudgetQueries
	s.ANNSearchBudgetSegments += other.ANNSearchBudgetSegments
	s.ANNSearchEFBudget += other.ANNSearchEFBudget
	s.CacheHitsTotal += other.CacheHitsTotal
	s.CacheMissesTotal += other.CacheMissesTotal
	s.CacheEvictionsTotal += other.CacheEvictionsTotal
	s.CacheItems += other.CacheItems
	s.CacheBytes += other.CacheBytes
	s.DiskFileBytes += other.DiskFileBytes
	s.DiskRecords += other.DiskRecords
	s.DiskStaleRecords += other.DiskStaleRecords
	s.DiskCompactionsTotal += other.DiskCompactionsTotal
	s.DiskSegments += other.DiskSegments
	s.ReplicationCommitted += other.ReplicationCommitted
	s.ReplicationApplied += other.ReplicationApplied
	s.ReplicationPending += other.ReplicationPending
	s.ReplicationFailures += other.ReplicationFailures
	if other.ReplicationTerm > s.ReplicationTerm {
		s.ReplicationTerm = other.ReplicationTerm
	}
	s.ReplicationFailovers += other.ReplicationFailovers
}
