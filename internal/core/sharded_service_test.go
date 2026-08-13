package core

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

type concurrentVersionedExportTracker struct {
	active  atomic.Int32
	maximum atomic.Int32
	ready   chan struct{}
	once    sync.Once
	want    int32
}

func (t *concurrentVersionedExportTracker) enter() error {
	active := t.active.Add(1)
	defer t.active.Add(-1)
	for {
		maximum := t.maximum.Load()
		if active <= maximum || t.maximum.CompareAndSwap(maximum, active) {
			break
		}
	}
	if active >= t.want {
		t.once.Do(func() { close(t.ready) })
	}
	select {
	case <-t.ready:
		return nil
	case <-time.After(2 * time.Second):
		return errors.New("versioned sources were not exported concurrently")
	}
}

type concurrentVersionedSource struct {
	VectorService
	tracker *concurrentVersionedExportTracker
	record  PrebuiltANNRecord
}

type identifiedVectorService struct {
	VectorService
	identity string
}

func (s *identifiedVectorService) DataNodeIdentity() string { return s.identity }

func TestSameDataNodeUsesStableRemoteIdentity(t *testing.T) {
	first := &identifiedVectorService{identity: "node-a"}
	second := &identifiedVectorService{identity: "node-a"}
	third := &identifiedVectorService{identity: "node-b"}
	unknown := &identifiedVectorService{}

	if !sameDataNode(first, second) {
		t.Fatal("different adapters for the same physical node must match")
	}
	if sameDataNode(first, third) {
		t.Fatal("different physical nodes must not match")
	}
	if sameDataNode(first, unknown) {
		t.Fatal("an empty identity must not alias a physical node")
	}
	if !layoutsShareDataNodes(
		&ShardedService{shards: []VectorService{first}},
		&ShardedService{shards: []VectorService{third, second}},
	) {
		t.Fatal("expected overlapping layouts to be detected")
	}
}

func (s *concurrentVersionedSource) ExportANNState(visit func(PrebuiltANNStateEvent) error) error {
	if err := s.tracker.enter(); err != nil {
		return err
	}
	if err := visit(PrebuiltANNStateEvent{Kind: 0, DeltaOffset: 1}); err != nil {
		return err
	}
	if err := visit(PrebuiltANNStateEvent{Kind: 1, Record: s.record}); err != nil {
		return err
	}
	return visit(PrebuiltANNStateEvent{Kind: 4, Cardinality: 1})
}

func (s *concurrentVersionedSource) VisitSealedANNRecords(func(int, PrebuiltANNReplay) error) error {
	return nil
}
func (s *concurrentVersionedSource) SnapshotMutableANNRecords() ([]PrebuiltANNRecord, error) {
	return nil, nil
}
func (s *concurrentVersionedSource) ReplayDeltasSince(uint64, func(DeltaRecord) error) error {
	return nil
}
func (s *concurrentVersionedSource) CurrentDeltaOffsetChecked() (uint64, error) { return 1, nil }
func (s *concurrentVersionedSource) PrebuiltANNMigrationConfig() (string, ann.Options) {
	return "", ann.Options{M: 4, EfConstruction: 8, EfSearch: 8}
}
func (s *concurrentVersionedSource) ListVectorsPage(opts ListVectorsOptions) ListVectorsPage {
	if opts.AfterID != "" {
		return ListVectorsPage{}
	}
	return ListVectorsPage{Vectors: []index.Vector{{ID: s.record.ID, Values: s.record.Values}}}
}

func TestRendezvousMigrationExportsIndependentVersionedSourcesConcurrently(t *testing.T) {
	base := t.TempDir()
	tracker := &concurrentVersionedExportTracker{ready: make(chan struct{}), want: 2}
	first := &concurrentVersionedSource{tracker: tracker, record: PrebuiltANNRecord{ID: "parallel-source-a", InternalID: 1, Values: []float64{1, 0}}}
	second := &concurrentVersionedSource{tracker: tracker, record: PrebuiltANNRecord{ID: "parallel-source-b", InternalID: 1, Values: []float64{0, 1}}}
	source, err := NewShardedServiceFromShardsWithIDs([]VectorService{first, second}, []string{"source-a", "source-b"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	target, err := NewShardedServiceWithIDs(ServiceOptions{
		MaxVectorDim: 4, MaxK: 4, SearchMode: "ann", VectorStore: "segment",
		VectorPath: filepath.Join(base, "target"), SnapshotPath: filepath.Join(base, "snapshot.json"),
		WALPath: filepath.Join(base, "wal.log"), ANNSegmentMaxNodes: 8,
	}, []string{"target-a", "target-b"}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if tracker.maximum.Load() != 2 {
		t.Fatalf("maximum concurrent exports=%d want 2", tracker.maximum.Load())
	}
	if got := target.ListVectors(); len(got) != 2 {
		t.Fatalf("target vectors=%+v", got)
	}
}

type bulkVersionedSource struct{ VectorService }

func (s *bulkVersionedSource) ExportANNState(visit func(PrebuiltANNStateEvent) error) error {
	if err := visit(PrebuiltANNStateEvent{Kind: 0, DeltaOffset: 1}); err != nil {
		return err
	}
	for i := 0; i < 20000; i++ {
		if err := visit(PrebuiltANNStateEvent{
			Kind: 1,
			Record: PrebuiltANNRecord{
				ID: fmt.Sprintf("balanced-worker-%05d", i), InternalID: i + 1,
				Values: []float64{float64(i), 1},
			},
		}); err != nil {
			return err
		}
	}
	return visit(PrebuiltANNStateEvent{Kind: 4, Cardinality: 20000})
}

func (s *bulkVersionedSource) VisitSealedANNRecords(func(int, PrebuiltANNReplay) error) error {
	return nil
}
func (s *bulkVersionedSource) SnapshotMutableANNRecords() ([]PrebuiltANNRecord, error) {
	return nil, nil
}
func (s *bulkVersionedSource) ReplayDeltasSince(uint64, func(DeltaRecord) error) error {
	return nil
}
func (s *bulkVersionedSource) CurrentDeltaOffsetChecked() (uint64, error) { return 1, nil }

type concurrentPrebuiltDestination struct {
	VectorService
	tracker *concurrentVersionedExportTracker
}

func (d *concurrentPrebuiltDestination) ReservePrebuiltIDs([]string) ([]IDMappingEntry, error) {
	return nil, nil
}
func (d *concurrentPrebuiltDestination) UploadPrebuiltSnapshot(ann.SegmentManifest, string) (string, error) {
	return "", nil
}
func (d *concurrentPrebuiltDestination) ApplyPrebuiltSegment(ann.SegmentManifest, PrebuiltANNReplay) error {
	return nil
}
func (d *concurrentPrebuiltDestination) BuildPrebuiltSegment(segmentID, metric string, options ann.Options, replay PrebuiltANNReplay) (ann.SegmentManifest, error) {
	if err := d.tracker.enter(); err != nil {
		return ann.SegmentManifest{}, err
	}
	count := 0
	if err := replay(func(PrebuiltANNRecord) error { count++; return nil }); err != nil {
		return ann.SegmentManifest{}, err
	}
	return ann.SegmentManifest{SegmentID: segmentID, Nodes: count, Metric: metric, M: options.M, EfConstruction: options.EfConstruction}, nil
}

func TestVersionedTransferUsesConfiguredWorkersPerDestination(t *testing.T) {
	t.Setenv("LUMENVEC_RESHARD_TARGET_WORKERS", "2")
	t.Setenv("LUMENVEC_RESHARD_TARGET_QUEUE_DEPTH", "1")
	tracker := &concurrentVersionedExportTracker{ready: make(chan struct{}), want: 2}
	destination := &concurrentPrebuiltDestination{tracker: tracker}
	target, err := NewShardedServiceFromShardsWithIDs(
		[]VectorService{destination}, []string{"target-only"}, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	source := &bulkVersionedSource{}
	sourceRouter, err := NewShardedServiceFromShardsWithIDs(
		[]VectorService{source}, []string{"source-only"}, 1,
	)
	if err != nil {
		t.Fatal(err)
	}
	offsets := []uint64{1}
	deduper := &prebuiltMigrationDeduper{seen: make(map[string]PrebuiltANNRecord)}
	if err := sourceRouter.migrateVersionedSourceSnapshot(
		target, 0, source, source,
		[]PrebuiltANNPartitionDestination{destination}, offsets, deduper,
	); err != nil {
		t.Fatal(err)
	}
	if got := tracker.maximum.Load(); got != 2 {
		t.Fatalf("concurrent destination workers=%d, want 2", got)
	}
}

func TestSegmentedMigrationKeepsQueriesAvailable(t *testing.T) {
	base := t.TempDir()
	options := func(path string) ServiceOptions {
		return ServiceOptions{MaxVectorDim: 8, MaxK: 10, SearchMode: "exact", VectorStore: "segment", ANNSegmentMaxNodes: 2000, VectorPath: path}
	}
	source := NewShardedService(options(filepath.Join(base, "source")), 1, 1)
	target := NewShardedService(options(filepath.Join(base, "target")), 4, 4)
	defer source.Close()
	defer target.Close()
	const count = 20000
	vectors := make([]index.Vector, count)
	for n := range vectors {
		vectors[n] = index.Vector{ID: fmt.Sprintf("v-%05d", n), Values: []float64{float64(n), 1, 2, 3}}
	}
	if err := source.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	var searchErr error
	var searchMu sync.Mutex
	var latencies []time.Duration
	wg.Add(1)
	go func() {
		defer wg.Done()
		query := []float64{100, 1, 2, 3}
		for {
			select {
			case <-stop:
				return
			default:
			}
			started := time.Now()
			if _, err := source.Search(query, 10); err != nil {
				searchMu.Lock()
				searchErr = err
				searchMu.Unlock()
				return
			}
			searchMu.Lock()
			latencies = append(latencies, time.Since(started))
			searchMu.Unlock()
		}
	}()
	start := time.Now()
	err := source.MigrateTo(target)
	close(stop)
	wg.Wait()
	if err != nil {
		t.Fatal(err)
	}
	searchMu.Lock()
	err = searchErr
	searchMu.Unlock()
	if err != nil {
		t.Fatalf("concurrent search failed during migration: %v", err)
	}
	searchMu.Lock()
	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	if len(latencies) < 5 {
		searchMu.Unlock()
		t.Fatalf("insufficient concurrent search samples: %d", len(latencies))
	}
	p95 := latencies[(len(latencies)*95+99)/100-1]
	searchMu.Unlock()
	if p95 > 2*time.Second {
		t.Fatalf("concurrent search p95 exceeded SLO: %s", p95)
	}
	t.Logf("concurrent search samples=%d p95=%s", len(latencies), p95)
	if elapsed := time.Since(start); elapsed > 2*time.Minute {
		t.Fatalf("migration exceeded test bound: %s", elapsed)
	}
	if got, err := target.Search([]float64{100, 1, 2, 3}, 10); err != nil || len(got) == 0 {
		t.Fatalf("target unavailable after migration: results=%d err=%v", len(got), err)
	}
}

func TestShardedServiceRoutesSearchesPagesAndRecovers(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{
		MaxVectorDim:  8,
		MaxK:          10,
		SearchMode:    "exact",
		VectorStore:   "segment",
		VectorPath:    filepath.Join(base, "vectors"),
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 100,
	}
	svc := NewShardedService(opts, 4, 2)
	vectors := make([]index.Vector, 0, 20)
	for i := 0; i < 20; i++ {
		vectors = append(vectors, index.Vector{ID: fmt.Sprintf("v-%02d", i), Values: []float64{float64(i), 0}})
	}
	if err := svc.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	for _, vec := range vectors {
		got, err := svc.GetVector(vec.ID)
		if err != nil || got.Values[0] != vec.Values[0] {
			t.Fatalf("get %s: got=%+v err=%v", vec.ID, got, err)
		}
	}
	results, err := svc.Search([]float64{19.1, 0}, 3)
	if err != nil || len(results) != 3 || results[0].ID != "v-19" {
		t.Fatalf("search=%+v err=%v", results, err)
	}
	batch, err := svc.SearchBatch([]BatchSearchQuery{
		{ID: "high", Values: []float64{18.9, 0}, K: 2},
		{Values: []float64{0.1, 0}, K: 2},
	})
	if err != nil || len(batch) != 2 || batch[0].Results[0].ID != "v-19" || batch[1].ID != "query-1" {
		t.Fatalf("batch=%+v err=%v", batch, err)
	}
	page1 := svc.ListVectorsPage(ListVectorsOptions{Limit: 7, IDsOnly: true})
	page2 := svc.ListVectorsPage(ListVectorsOptions{AfterID: page1.NextCursor, Limit: 7, IDsOnly: true})
	if len(page1.Vectors) != 7 || page1.NextCursor == "" || len(page2.Vectors) != 7 || page2.Vectors[0].ID <= page1.NextCursor {
		t.Fatalf("pages=%+v %+v", page1, page2)
	}
	if all := svc.ListVectors(); len(all) != 20 || all[0].ID != "v-00" || all[19].ID != "v-19" {
		t.Fatalf("all vectors=%d", len(all))
	}
	if stats := svc.Stats(); stats.ShardCount != 4 || stats.DiskRecords != 20 {
		t.Fatalf("stats=%+v", stats)
	}
	if err := svc.DeleteVector("v-10"); err != nil {
		t.Fatal(err)
	}
	if _, err := svc.GetVector("v-10"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("deleted get error=%v", err)
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}
	if err := svc.Close(); err != nil {
		t.Fatal(err)
	}

	restored := NewShardedService(opts, 4, 4)
	t.Cleanup(func() { _ = restored.Close() })
	if all := restored.ListVectors(); len(all) != 19 {
		t.Fatalf("restored vectors=%d", len(all))
	}
	for shardID := 0; shardID < 4; shardID++ {
		path := filepath.Join(opts.VectorPath, "shard-"+formatShardID(shardID), "manifest.json")
		if _, err := os.Stat(path); err != nil {
			t.Fatal(err)
		}
	}
}

func TestShardedServicePagesShardsConcurrently(t *testing.T) {
	base := t.TempDir()
	const delay = 75 * time.Millisecond
	services := make([]*Service, 0, 4)
	shards := make([]VectorService, 0, 4)
	for i := 0; i < 4; i++ {
		service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 2, SnapshotPath: filepath.Join(base, fmt.Sprintf("%d.json", i)), WALPath: filepath.Join(base, fmt.Sprintf("%d.wal", i))})
		services = append(services, service)
		shards = append(shards, delayedPageService{VectorService: service, delay: delay})
	}
	router := NewShardedServiceFromShards(shards, len(shards))
	defer router.Close()
	for i := 0; i < 4; i++ {
		id := fmt.Sprintf("v-%d", i)
		if err := services[router.ShardForID(id)].AddVector(id, []float64{float64(i), 0}); err != nil {
			t.Fatal(err)
		}
	}
	started := time.Now()
	page := router.ListVectorsPage(ListVectorsOptions{Limit: 4, IDsOnly: true})
	if elapsed := time.Since(started); elapsed >= 2*delay {
		t.Fatalf("page fan-out was serial: %s", elapsed)
	}
	if len(page.Vectors) != 4 {
		t.Fatalf("page vectors=%d want 4", len(page.Vectors))
	}
}

func TestShardedServicePagePreservesTransportContinuationCursor(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 2, SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal")})
	defer service.Close()
	for _, id := range []string{"a", "b", "c"} {
		if err := service.AddVector(id, []float64{1, 0}); err != nil {
			t.Fatal(err)
		}
	}
	router := NewShardedServiceFromShards([]VectorService{clampedPageService{VectorService: service, limit: 1}}, 1)
	page := router.ListVectorsPage(ListVectorsOptions{Limit: 10, IDsOnly: true})
	if len(page.Vectors) != 1 || page.NextCursor != "a" {
		t.Fatalf("capped page=%+v; want first item and continuation cursor", page)
	}
}

func TestShardedServiceMigrationFailsOnCheckedPageRead(t *testing.T) {
	base := t.TempDir()
	service := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 2, SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal")})
	if err := service.AddVector("v", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	source := NewShardedServiceFromShards([]VectorService{checkedPageErrorService{VectorService: service, err: errors.New("remote page unavailable")}}, 1)
	target := NewShardedService(ServiceOptions{MaxVectorDim: 2, MaxK: 2, SnapshotPath: filepath.Join(base, "target.json"), WALPath: filepath.Join(base, "target.wal")}, 1, 1)
	defer source.Close()
	defer target.Close()
	if err := source.MigrateTo(target); err == nil || !strings.Contains(err.Error(), "remote page unavailable") {
		t.Fatalf("migration error=%v", err)
	}
	if got := target.countVectors(); got != 0 {
		t.Fatalf("target was modified after failed page read: %d", got)
	}
}

func TestRunNativeMigrationSourcesHonorsParallelism(t *testing.T) {
	const delay = 60 * time.Millisecond
	sources := []rangeVector32Reader{delayedRangeReader{delay: delay}, delayedRangeReader{delay: delay}, delayedRangeReader{delay: delay}}
	started := time.Now()
	err := runNativeMigrationSources(sources, len(sources), func(_ int, reader rangeVector32Reader) error {
		reader.RangeVectors32(func(string, []float32) bool { return true })
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed >= 2*delay {
		t.Fatalf("native source migration was serial: %s", elapsed)
	}
}

func TestShardedServiceMigrationHonorsTransportPageLimit(t *testing.T) {
	base := t.TempDir()
	backing := NewService(ServiceOptions{MaxVectorDim: 2, MaxK: 2, SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal")})
	vectors := make([]index.Vector, 1500)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("v-%04d", i), Values: []float64{float64(i), 0}}
	}
	if err := backing.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	limited := &limitedPageService{VectorService: backing, limit: 1000}
	source := NewShardedServiceFromShards([]VectorService{limited}, 1)
	target := NewShardedService(ServiceOptions{MaxVectorDim: 2, MaxK: 2, SnapshotPath: filepath.Join(base, "target.json"), WALPath: filepath.Join(base, "target.wal")}, 1, 1)
	defer source.Close()
	defer target.Close()
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if limited.maxRequested > limited.limit {
		t.Fatalf("page request %d exceeded transport limit %d", limited.maxRequested, limited.limit)
	}
	if got := target.countVectors(); got != len(vectors) {
		t.Fatalf("target count=%d want %d", got, len(vectors))
	}
}

func TestShardedServiceMigrateToNewLayout(t *testing.T) {
	if err := (*ShardedService)(nil).MigrateTo(nil); err == nil {
		t.Fatal("expected invalid migration error")
	}
	base := t.TempDir()
	opts := ServiceOptions{MaxVectorDim: 8, MaxK: 10, SearchMode: "exact", SnapshotEvery: 100, SnapshotPath: filepath.Join(base, "source.json"), WALPath: filepath.Join(base, "source.wal")}
	source := NewShardedService(opts, 2, 2)
	target := NewShardedService(ServiceOptions{MaxVectorDim: 8, MaxK: 10, SearchMode: "exact", SnapshotEvery: 100, SnapshotPath: filepath.Join(base, "target.json"), WALPath: filepath.Join(base, "target.wal")}, 3, 3)
	t.Cleanup(func() { _ = source.Close(); _ = target.Close() })
	if err := source.AddVectors([]index.Vector{{ID: "a", Values: []float64{1}}, {ID: "b", Values: []float64{2}}}); err != nil {
		t.Fatal(err)
	}
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if got := target.ListVectors(); len(got) != 2 {
		t.Fatalf("target vectors=%+v", got)
	}
}

func TestShardedServiceMigrationPreservesMetadata(t *testing.T) {
	base := t.TempDir()
	newOpts := func(name string) ServiceOptions {
		return ServiceOptions{MaxVectorDim: 8, MaxK: 10, SnapshotEvery: 100,
			SnapshotPath: filepath.Join(base, name+".json"), WALPath: filepath.Join(base, name+".wal")}
	}
	source := NewShardedService(newOpts("meta-source"), 2, 2)
	target := NewShardedService(newOpts("meta-target"), 3, 3)
	t.Cleanup(func() { _ = source.Close(); _ = target.Close() })
	shard := source.shards[source.ShardForID("doc")]
	metaSvc, ok := shard.(metadataVectorService)
	if !ok {
		t.Fatal("built-in shard must support metadata")
	}
	if err := metaSvc.AddVectorWithMetadata("doc", []float64{1, 0}, map[string]string{"tenant": "acme", "title": "raft"}); err != nil {
		t.Fatal(err)
	}
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	reader, ok := target.shards[target.ShardForID("doc")].(vectorMetadataReader)
	if !ok {
		t.Fatal("built-in target shard must expose metadata")
	}
	got := reader.VectorMetadata("doc")
	if got["tenant"] != "acme" || got["title"] != "raft" {
		t.Fatalf("metadata lost during migration: %#v", got)
	}
}

func TestShardedServiceRoutesMetadataAndFilteredSearch(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{MaxVectorDim: 4, MaxK: 10, SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log")}
	svc := NewShardedService(opts, 3, 3)
	t.Cleanup(func() { _ = svc.Close() })
	if err := svc.AddVectorWithMetadata("acme-1", []float64{1, 0}, map[string]string{"tenant": "acme"}); err != nil {
		t.Fatal(err)
	}
	if err := svc.AddVectorWithMetadata("other-1", []float64{0.99, 0}, map[string]string{"tenant": "other"}); err != nil {
		t.Fatal(err)
	}
	results, err := svc.SearchFiltered([]float64{1, 0}, 5, StructuredFilter{Metadata: map[string]string{"tenant": "acme"}}.Match)
	if err != nil || len(results) != 1 || results[0].ID != "acme-1" {
		t.Fatalf("filtered results=%+v err=%v", results, err)
	}
}

func TestShardedServiceBulkAddPreservesMetadata(t *testing.T) {
	base := t.TempDir()
	opts := ServiceOptions{MaxVectorDim: 4, MaxK: 10, SnapshotPath: filepath.Join(base, "snapshot.json"), WALPath: filepath.Join(base, "wal.log")}
	svc := NewShardedService(opts, 3, 3)
	t.Cleanup(func() { _ = svc.Close() })
	err := svc.AddVectors([]index.Vector{
		{ID: "bulk-a", Values: []float64{1, 0}, Metadata: map[string]string{"tenant": "a"}},
		{ID: "bulk-b", Values: []float64{0, 1}},
	})
	if err != nil {
		t.Fatal(err)
	}
	reader, ok := svc.shards[svc.ShardForID("bulk-a")].(vectorMetadataReader)
	if !ok || reader.VectorMetadata("bulk-a")["tenant"] != "a" {
		t.Fatalf("bulk metadata was not persisted")
	}
	listed := svc.ListVectors()
	for _, vector := range listed {
		if vector.ID == "bulk-a" && vector.Metadata["tenant"] != "a" {
			t.Fatalf("list metadata was not returned: %#v", vector.Metadata)
		}
	}
	page := svc.ListVectorsPage(ListVectorsOptions{Limit: 10})
	for _, vector := range page.Vectors {
		if vector.ID == "bulk-a" && vector.Metadata["tenant"] != "a" {
			t.Fatalf("paged metadata was not returned: %#v", vector.Metadata)
		}
	}
}

func TestShardedServiceMigrationAllowsConcurrentSearch(t *testing.T) {
	base := t.TempDir()
	newOpts := func(name string) ServiceOptions {
		return ServiceOptions{MaxVectorDim: 4, MaxK: 10, SnapshotPath: filepath.Join(base, name+".json"), WALPath: filepath.Join(base, name+".wal"), VectorPath: filepath.Join(base, name)}
	}
	source := NewShardedService(newOpts("source-online"), 1, 1)
	target := NewShardedService(newOpts("target-online"), 8, 8)
	defer source.Close()
	defer target.Close()
	vectors := make([]index.Vector, 10000)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("v-%06d", i), Values: []float64{float64(i), 1}}
	}
	if err := source.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	var searchErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			if _, err := source.Search([]float64{5000, 1}, 10); err != nil {
				searchErr = err
				return
			}
		}
	}()
	var ingestErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		batch := make([]index.Vector, 100)
		for i := range batch {
			batch[i] = index.Vector{ID: fmt.Sprintf("late-%03d", i), Values: []float64{float64(i), 2}}
		}
		ingestErr = source.AddVectors(batch)
	}()
	err := source.MigrateTo(target)
	close(stop)
	wg.Wait()
	if err != nil || searchErr != nil || ingestErr != nil {
		t.Fatalf("migration=%v concurrent search=%v ingest=%v", err, searchErr, ingestErr)
	}
	if got := len(target.ListVectors()); got != len(vectors)+100 {
		t.Fatalf("target vectors=%d want %d", got, len(vectors)+100)
	}
}

func TestShardedServicePersistentMigrationResumesAfterTargetRestart(t *testing.T) {
	base := t.TempDir()
	checkpointDir := filepath.Join(base, "migration-state")
	t.Setenv("LUMENVEC_RESHARD_CHECKPOINT_DIR", checkpointDir)
	t.Setenv("LUMENVEC_RESHARD_NATIVE_WINDOW", "16")
	newOpts := func(name string) ServiceOptions {
		return ServiceOptions{
			MaxVectorDim: 4, MaxK: 10, SearchMode: "exact", VectorStore: "segment",
			ANNSegmentMaxNodes: 32, VectorPath: filepath.Join(base, name, "vectors"),
			SnapshotPath: filepath.Join(base, name, "snapshot.json"),
			WALPath:      filepath.Join(base, name, "wal.log"), SnapshotEvery: 32,
		}
	}
	source := NewShardedService(newOpts("source"), 1, 1)
	target := NewShardedService(newOpts("target"), 3, 3)
	vectors := make([]index.Vector, 160)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("persistent-%03d", i), Values: []float64{float64(i), 1, 2, 3}}
	}
	if err := source.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if err := target.Close(); err != nil {
		t.Fatal(err)
	}
	// Reopen the same destination paths and run the migration again. The
	// durable manifest/checkpoint must make this a no-op, without duplicates.
	target = NewShardedService(newOpts("target"), 3, 3)
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if got := target.ListVectors(); len(got) != len(vectors) {
		t.Fatalf("reopened target vectors=%d want %d", len(got), len(vectors))
	}
	if _, err := LoadReshardCheckpoint(filepath.Join(checkpointDir, "native-source-0.json")); err != nil {
		t.Fatalf("durable migration checkpoint missing: %v", err)
	}
	_ = target.Close()
	_ = source.Close()
}

func TestShardedServicePersistentMigrationResumesFromInterruptedWindow(t *testing.T) {
	base := t.TempDir()
	checkpointDir := filepath.Join(base, "migration-state")
	t.Setenv("LUMENVEC_RESHARD_CHECKPOINT_DIR", checkpointDir)
	t.Setenv("LUMENVEC_RESHARD_NATIVE_WINDOW", "8")
	newOpts := func(name string) ServiceOptions {
		return ServiceOptions{
			MaxVectorDim: 4, MaxK: 10, SearchMode: "exact", VectorStore: "segment",
			ANNSegmentMaxNodes: 16, VectorPath: filepath.Join(base, name, "vectors"),
			SnapshotPath: filepath.Join(base, name, "snapshot.json"),
			WALPath:      filepath.Join(base, name, "wal.log"), SnapshotEvery: 200,
		}
	}
	source := NewShardedService(newOpts("source-interrupted"), 1, 1)
	target := NewShardedService(newOpts("target-interrupted"), 3, 3)
	vectors := make([]index.Vector, 96)
	for i := range vectors {
		vectors[i] = index.Vector{ID: fmt.Sprintf("resume-%03d", i), Values: []float64{float64(i), 1, 2, 3}}
	}
	if err := source.AddVectors(vectors); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LUMENVEC_RESHARD_FAIL_AFTER_VECTORS", "24")
	if err := source.MigrateTo(target); err == nil {
		t.Fatal("expected injected migration interruption")
	}
	checkpointPath := filepath.Join(checkpointDir, "native-source-0.json")
	interrupted, err := LoadReshardCheckpoint(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	if interrupted.State != "copying" || interrupted.VectorsCopied < 24 || interrupted.VectorsCopied >= uint64(len(vectors)) {
		t.Fatalf("unexpected interrupted checkpoint: %+v", interrupted)
	}
	if err := target.Close(); err != nil {
		t.Fatal(err)
	}

	target = NewShardedService(newOpts("target-interrupted"), 3, 3)
	t.Setenv("LUMENVEC_RESHARD_FAIL_AFTER_VECTORS", "")
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if got := target.countVectors(); got != len(vectors) {
		t.Fatalf("resumed target vectors=%d want %d", got, len(vectors))
	}
	complete, err := LoadReshardCheckpoint(checkpointPath)
	if err != nil {
		t.Fatal(err)
	}
	if complete.State != "complete" || complete.VectorsCopied != uint64(len(vectors)) {
		t.Fatalf("unexpected completed checkpoint: %+v", complete)
	}
	_ = target.Close()
	_ = source.Close()
}

type addFailShard struct {
	VectorService
	err error
}

func (s addFailShard) AddVectors([]index.Vector) error { return s.err }

func TestShardedServiceBatchRollbackAndEmptyTopology(t *testing.T) {
	newShard := func(name string) *Service {
		base := filepath.Join(t.TempDir(), name)
		return NewService(ServiceOptions{
			MaxVectorDim: 8, MaxK: 10, SnapshotEvery: 100,
			SnapshotPath: filepath.Join(base, "snapshot.json"),
			WALPath:      filepath.Join(base, "wal.log"),
		})
	}
	left := newShard("left")
	rightBase := newShard("right")
	t.Cleanup(func() { _ = left.Close(); _ = rightBase.Close() })
	svc := newShardedServiceFromShards([]VectorService{left, addFailShard{VectorService: rightBase, err: errors.New("write failed")}}, 2)

	ids := make([]string, 2)
	for i := 0; ids[0] == "" || ids[1] == ""; i++ {
		id := fmt.Sprintf("id-%d", i)
		ids[svc.ShardForID(id)] = id
	}
	err := svc.AddVectors([]index.Vector{
		{ID: ids[0], Values: []float64{1}},
		{ID: ids[1], Values: []float64{2}},
	})
	if err == nil {
		t.Fatal("expected distributed batch error")
	}
	if _, err := left.GetVector(ids[0]); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("successful shard was not rolled back: %v", err)
	}

	empty := newShardedServiceFromShards(nil, 0)
	if empty.ShardForID("x") != 0 {
		t.Fatal("empty topology shard must be zero")
	}
	if err := empty.AddVector("x", []float64{1}); err == nil {
		t.Fatal("expected empty topology add error")
	}
	if _, err := empty.GetVector("x"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("empty get error=%v", err)
	}
	if err := empty.DeleteVector("x"); !errors.Is(err, index.ErrVectorNotFound) {
		t.Fatalf("empty delete error=%v", err)
	}
	if _, err := empty.Search([]float64{1}, 1); err == nil {
		t.Fatal("expected empty search error")
	}
	if _, err := empty.SearchBatch(nil); !errors.Is(err, ErrInvalidValues) {
		t.Fatalf("empty batch error=%v", err)
	}
	if page := empty.ListVectorsPage(ListVectorsOptions{}); len(page.Vectors) != 0 {
		t.Fatalf("empty page=%+v", page)
	}
}

func TestShardedServiceFilteredSearchFailsClosedForLegacyShard(t *testing.T) {
	base := t.TempDir()
	modern := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 4, SnapshotPath: filepath.Join(base, "modern.json"), WALPath: filepath.Join(base, "modern.wal")})
	legacyBase := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 4, SnapshotPath: filepath.Join(base, "legacy.json"), WALPath: filepath.Join(base, "legacy.wal")})
	t.Cleanup(func() { _ = modern.Close(); _ = legacyBase.Close() })
	svc := newShardedServiceFromShards([]VectorService{modern, legacyVectorService{VectorService: legacyBase}}, 2)
	_, err := svc.SearchFiltered([]float64{1}, 1, StructuredFilter{TextQuery: "tenant"}.Match)
	if !errors.Is(err, ErrFilteredUnsupported) {
		t.Fatalf("filtered legacy shard error=%v, want %v", err, ErrFilteredUnsupported)
	}
}

func TestShardedServiceStructuredTextSearchUsesShardIndexes(t *testing.T) {
	base := t.TempDir()
	left := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 4, SnapshotPath: filepath.Join(base, "left.json"), WALPath: filepath.Join(base, "left.wal")})
	right := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 4, SnapshotPath: filepath.Join(base, "right.json"), WALPath: filepath.Join(base, "right.wal")})
	t.Cleanup(func() { _ = left.Close(); _ = right.Close() })
	if err := left.AddVector("article-left", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	if err := right.AddVector("image-right", []float64{1, 0}); err != nil {
		t.Fatal(err)
	}
	svc := newShardedServiceFromShards([]VectorService{left, right}, 2)
	got, err := svc.SearchStructured([]float64{1, 0}, 4, StructuredFilter{TextQuery: "article"}, MetricCosine)
	if err != nil || len(got) != 1 || got[0].ID != "article-left" {
		t.Fatalf("results=%+v err=%v", got, err)
	}
}

func TestShardedServiceMetricSearchWithoutFilter(t *testing.T) {
	base := t.TempDir()
	left := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 4, SnapshotPath: filepath.Join(base, "left.json"), WALPath: filepath.Join(base, "left.wal")})
	right := NewService(ServiceOptions{MaxVectorDim: 4, MaxK: 4, SnapshotPath: filepath.Join(base, "right.json"), WALPath: filepath.Join(base, "right.wal")})
	t.Cleanup(func() { _ = left.Close(); _ = right.Close() })
	if err := left.AddVector("cosine-and-dot", []float64{100, 1}); err != nil {
		t.Fatal(err)
	}
	if err := right.AddVector("l2", []float64{1, 1}); err != nil {
		t.Fatal(err)
	}
	svc := newShardedServiceFromShards([]VectorService{left, right}, 2)
	for _, metric := range []DistanceMetric{MetricCosine, MetricInnerProduct} {
		got, err := svc.SearchFilteredMetric([]float64{1, 0}, 1, nil, metric)
		if err != nil || len(got) != 1 || got[0].ID != "cosine-and-dot" {
			t.Fatalf("metric=%s results=%+v err=%v", metric, got, err)
		}
	}
}

// legacyVectorService intentionally exposes only the original VectorService
// contract, ensuring the router fails closed when filtering is unavailable.
type legacyVectorService struct{ VectorService }

type delayedPageService struct {
	VectorService
	delay time.Duration
}

type clampedPageService struct {
	VectorService
	limit int
}

func (s clampedPageService) ListVectorsPage(opts ListVectorsOptions) ListVectorsPage {
	if opts.Limit > s.limit {
		opts.Limit = s.limit
	}
	return s.VectorService.ListVectorsPage(opts)
}

func (s delayedPageService) ListVectorsPage(opts ListVectorsOptions) ListVectorsPage {
	time.Sleep(s.delay)
	return s.VectorService.ListVectorsPage(opts)
}

type checkedPageErrorService struct {
	VectorService
	err error
}

func (s checkedPageErrorService) ListVectorsPageChecked(ListVectorsOptions) (ListVectorsPage, error) {
	return ListVectorsPage{}, s.err
}

type delayedRangeReader struct{ delay time.Duration }

func (r delayedRangeReader) RangeVectors32(fn func(string, []float32) bool) {
	time.Sleep(r.delay)
	fn("v", []float32{1})
}

type limitedPageService struct {
	VectorService
	limit        int
	maxRequested int
}

func (s *limitedPageService) MigrationPageLimit() int { return s.limit }

func (s *limitedPageService) ListVectorsPage(opts ListVectorsOptions) ListVectorsPage {
	if opts.Limit > s.maxRequested {
		s.maxRequested = opts.Limit
	}
	if opts.Limit > s.limit {
		return ListVectorsPage{}
	}
	return s.VectorService.ListVectorsPage(opts)
}

type listCountingService struct {
	*Service
	listCalls int
}

func (s *listCountingService) ListVectors() []index.Vector {
	s.listCalls++
	return s.Service.ListVectors()
}

func TestShardedStructuredSearchDoesNotMaterializeAllVectors(t *testing.T) {
	dir := t.TempDir()
	shard := &listCountingService{Service: NewService(ServiceOptions{
		MaxVectorDim: 2, MaxK: 10, SnapshotPath: filepath.Join(dir, "snapshot.json"), WALPath: filepath.Join(dir, "wal.log"),
	})}
	defer shard.Close()
	if err := shard.AddVectorWithMetadata("doc", []float64{1, 0}, map[string]string{"tenant": "acme"}); err != nil {
		t.Fatal(err)
	}
	router := NewShardedServiceFromShards([]VectorService{shard}, 1)
	results, err := router.SearchStructured([]float64{1, 0}, 1, StructuredFilter{Metadata: map[string]string{"tenant": "acme"}}, MetricL2)
	if err != nil || len(results) != 1 {
		t.Fatalf("results=%v err=%v", results, err)
	}
	if shard.listCalls != 0 {
		t.Fatalf("structured search materialized the shard %d times", shard.listCalls)
	}
}

func TestMergeSearchResultsOrdering(t *testing.T) {
	got := mergeSearchResults([][]SearchResult{
		{{ID: "b", Distance: 1}, {ID: "c", Distance: 3}},
		{{ID: "a", Distance: 1}, {ID: "d", Distance: 2}},
	}, 3)
	if len(got) != 3 || got[0].ID != "a" || got[1].ID != "b" || got[2].ID != "d" {
		t.Fatalf("merged=%+v", got)
	}
	deduplicated := mergeSearchResults([][]SearchResult{
		{{ID: "same", Distance: 2}, {ID: "other", Distance: 3}},
		{{ID: "same", Distance: 1}},
	}, 3)
	if len(deduplicated) != 2 || deduplicated[0].ID != "same" || deduplicated[0].Distance != 1 {
		t.Fatalf("deduplicated=%+v", deduplicated)
	}
}

func TestRendezvousRoutingMovesOnlyKeysOwnedByTopologyChange(t *testing.T) {
	three, err := NewShardedServiceFromShardsWithIDs(
		[]VectorService{nil, nil, nil}, []string{"node-a", "node-b", "node-c"}, 3,
	)
	if err != nil {
		t.Fatal(err)
	}
	four, err := NewShardedServiceFromShardsWithIDs(
		[]VectorService{nil, nil, nil, nil}, []string{"node-a", "node-b", "node-c", "node-d"}, 4,
	)
	if err != nil {
		t.Fatal(err)
	}
	moved := 0
	const keys = 100000
	for i := 0; i < keys; i++ {
		id := fmt.Sprintf("routing-key-%d", i)
		before, after := three.ShardForID(id), four.ShardForID(id)
		if before != after {
			moved++
			if after != 3 {
				t.Fatalf("key %q moved between existing nodes: %d -> %d", id, before, after)
			}
		}
	}
	ratio := float64(moved) / keys
	if ratio < 0.20 || ratio > 0.30 {
		t.Fatalf("scale-out movement ratio=%f want approximately 0.25", ratio)
	}
	for i := 0; i < keys; i++ {
		id := fmt.Sprintf("routing-key-%d", i)
		before, after := four.ShardForID(id), three.ShardForID(id)
		if before != after && before != 3 {
			t.Fatalf("removal moved key not owned by removed node: %q %d -> %d", id, before, after)
		}
	}
}

func TestRendezvousRoutingRejectsInvalidDurableShardIDs(t *testing.T) {
	if _, err := NewShardedServiceFromShardsWithIDs([]VectorService{nil, nil}, []string{"same", "same"}, 2); err == nil {
		t.Fatal("expected duplicate durable shard IDs to fail")
	}
	if _, err := NewShardedServiceFromShardsWithIDs([]VectorService{nil}, nil, 1); err == nil {
		t.Fatal("expected mismatched shard IDs to fail")
	}
}

func TestRendezvousMigrateToTransfersOnlyChangedOwnersWithPrebuiltANN(t *testing.T) {
	base := t.TempDir()
	newShard := func(name string) *Service {
		return NewService(ServiceOptions{
			MaxVectorDim: 4, MaxK: 5, SearchMode: "ann",
			VectorStore: "segment", VectorPath: filepath.Join(base, name, "vectors"),
			SnapshotPath: filepath.Join(base, name, "snapshot.json"), WALPath: filepath.Join(base, name, "wal.log"),
			ANNSegmentMaxNodes: 4,
		})
	}
	nodeA, nodeB, nodeC := newShard("a"), newShard("b"), newShard("c")
	t.Cleanup(func() {
		_ = nodeA.Close()
		_ = nodeB.Close()
		_ = nodeC.Close()
	})
	source, err := NewShardedServiceFromShardsWithIDs(
		[]VectorService{nodeA, nodeB}, []string{"node-a", "node-b"}, 2,
	)
	if err != nil {
		t.Fatal(err)
	}
	target, err := NewShardedServiceFromShardsWithIDs(
		[]VectorService{nodeA, nodeB, nodeC}, []string{"node-a", "node-b", "node-c"}, 3,
	)
	if err != nil {
		t.Fatal(err)
	}
	const vectors = 120
	for i := 0; i < vectors; i++ {
		id := fmt.Sprintf("rendezvous-vector-%04d", i)
		if err := source.AddVector(id, []float64{float64(i), 1}); err != nil {
			t.Fatal(err)
		}
	}
	expectedMoved := 0
	updatedID := ""
	for _, vector := range source.ListVectors() {
		if target.ShardForID(vector.ID) == 2 {
			expectedMoved++
			if updatedID == "" {
				updatedID = vector.ID
			}
		}
	}
	if expectedMoved == 0 || expectedMoved >= vectors || updatedID == "" {
		t.Fatalf("unexpected moved vectors=%d", expectedMoved)
	}
	var hookErr error
	source.migrationBeforeDrain = func() {
		if err := source.DeleteVector(updatedID); err != nil {
			hookErr = err
			return
		}
		hookErr = source.AddVector(updatedID, []float64{9999, 1})
	}
	if err := source.MigrateTo(target); err != nil {
		t.Fatal(err)
	}
	if hookErr != nil {
		t.Fatal(hookErr)
	}
	if got := len(target.ListVectors()); got != vectors {
		t.Fatalf("target logical vectors=%d want %d", got, vectors)
	}
	if got := len(nodeC.ListVectors()); got != expectedMoved {
		t.Fatalf("new owner vectors=%d want %d", got, expectedMoved)
	}
	if _, err := os.Stat(filepath.Join(nodeC.vectorPath, "prebuilt-ann-catalog.json")); err != nil {
		t.Fatalf("prebuilt path was not used: %v", err)
	}
	updated, err := target.GetVector(updatedID)
	if err != nil || updated.Values[0] != 9999 {
		t.Fatalf("delta WAL update=%+v err=%v", updated, err)
	}
	results, err := target.Search([]float64{42, 1}, 5)
	if err != nil {
		t.Fatal(err)
	}
	seen := make(map[string]struct{}, len(results))
	for _, result := range results {
		if _, duplicate := seen[result.ID]; duplicate {
			t.Fatalf("search returned duplicate ID during stale-owner cleanup window: %+v", results)
		}
		seen[result.ID] = struct{}{}
	}
}
