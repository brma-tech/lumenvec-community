package core

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"
)

type prebuiltANNMigrationSource interface {
	VisitSealedANNRecords(func(int, PrebuiltANNReplay) error) error
	SnapshotMutableANNRecords() ([]PrebuiltANNRecord, error)
	ReplayDeltasSince(uint64, func(DeltaRecord) error) error
}

type prebuiltANNVersionedSource interface {
	ExportANNState(func(PrebuiltANNStateEvent) error) error
}

type prebuiltANNMigrationConfig interface {
	PrebuiltANNMigrationConfig() (string, ann.Options)
}

type dataNodeIdentityProvider interface {
	DataNodeIdentity() string
}

func sameDataNode(left, right VectorService) bool {
	if left == right {
		return true
	}
	leftIdentity, leftOK := left.(dataNodeIdentityProvider)
	rightIdentity, rightOK := right.(dataNodeIdentityProvider)
	if !leftOK || !rightOK {
		return false
	}
	leftID := leftIdentity.DataNodeIdentity()
	return leftID != "" && leftID == rightIdentity.DataNodeIdentity()
}

func layoutsShareDataNodes(source, target *ShardedService) bool {
	for _, sourceShard := range source.shards {
		for _, targetShard := range target.shards {
			if sameDataNode(sourceShard, targetShard) {
				return true
			}
		}
	}
	return false
}

type versionedPrebuiltTransferJob struct {
	targetIndex int
	segmentID   string
	records     []PrebuiltANNRecord
}

const versionedPrebuiltTransferBatch = 10000

func migrationSourceOffset(source prebuiltANNMigrationSource) (uint64, error) {
	if checked, ok := source.(interface{ CurrentDeltaOffsetChecked() (uint64, error) }); ok {
		return checked.CurrentDeltaOffsetChecked()
	}
	if local, ok := source.(interface{ CurrentDeltaOffset() uint64 }); ok {
		return local.CurrentDeltaOffset(), nil
	}
	return 0, errors.New("prebuilt ANN source does not expose a delta offset")
}

type prebuiltMigrationDeduper struct {
	mu   sync.Mutex
	seen map[string]PrebuiltANNRecord
}

func (d *prebuiltMigrationDeduper) accept(sourceIndex int, record PrebuiltANNRecord) (bool, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	previous, duplicate := d.seen[record.ID]
	if !duplicate {
		d.seen[record.ID] = record
		return true, nil
	}
	if vectorChecksum([]index.Vector{{ID: record.ID, Values: previous.Values}}) != vectorChecksum([]index.Vector{{ID: record.ID, Values: record.Values}}) {
		return false, fmt.Errorf("versioned ANN source=%d has conflicting duplicate vector %q", sourceIndex, record.ID)
	}
	return false, nil
}

func (s *ShardedService) migrateVersionedSourceSnapshot(
	target *ShardedService,
	sourceIndex int,
	source prebuiltANNMigrationSource,
	versioned prebuiltANNVersionedSource,
	destinations []PrebuiltANNPartitionDestination,
	snapshotOffsets []uint64,
	deduper *prebuiltMigrationDeduper,
) error {
	buildOptions := ann.Options{M: 16, EfConstruction: 64, EfSearch: 64}
	if configured, ok := source.(prebuiltANNMigrationConfig); ok {
		_, buildOptions = configured.PrebuiltANNMigrationConfig()
	}
	seenBegin := false
	pending := make(map[int][]PrebuiltANNRecord)
	batchSequence := 0
	jobs := make([]chan versionedPrebuiltTransferJob, len(destinations))
	queueDepth := 4
	if raw := os.Getenv("LUMENVEC_RESHARD_TARGET_QUEUE_DEPTH"); raw != "" {
		if configured, parseErr := strconv.Atoi(raw); parseErr == nil && configured > 0 {
			queueDepth = min(configured, 32)
		}
	}
	workersPerTarget := 1
	if raw := os.Getenv("LUMENVEC_RESHARD_TARGET_WORKERS"); raw != "" {
		if configured, parseErr := strconv.Atoi(raw); parseErr == nil && configured > 0 {
			workersPerTarget = min(configured, 4)
		}
	}
	queueWaitNS := make([]atomic.Int64, len(destinations))
	transferNS := make([]atomic.Int64, len(destinations))
	transferJobs := make([]atomic.Int64, len(destinations))
	transferRecords := make([]atomic.Int64, len(destinations))
	exportStarted := time.Now()
	var transferWG sync.WaitGroup
	var transferErr error
	var transferErrMu sync.Mutex
	setTransferErr := func(err error) {
		if err == nil {
			return
		}
		transferErrMu.Lock()
		if transferErr == nil {
			transferErr = err
		}
		transferErrMu.Unlock()
	}
	getTransferErr := func() error {
		transferErrMu.Lock()
		defer transferErrMu.Unlock()
		return transferErr
	}
	for targetIndex := range destinations {
		targetIndex := targetIndex
		jobs[targetIndex] = make(chan versionedPrebuiltTransferJob, queueDepth)
		for worker := 0; worker < workersPerTarget; worker++ {
			transferWG.Add(1)
			go func() {
				defer transferWG.Done()
				for job := range jobs[targetIndex] {
					if getTransferErr() != nil {
						continue
					}
					transferStarted := time.Now()
					dir, err := os.MkdirTemp("", "lumenvec-versioned-export-")
					if err != nil {
						setTransferErr(err)
						continue
					}
					_, err = TransferPrebuiltANNPartition(dir, job.segmentID, "l2", buildOptions, func(yield func(PrebuiltANNRecord) error) error {
						for _, record := range job.records {
							if err := yield(record); err != nil {
								return err
							}
						}
						return nil
					}, destinations[targetIndex])
					_ = os.RemoveAll(dir)
					transferNS[targetIndex].Add(time.Since(transferStarted).Nanoseconds())
					transferJobs[targetIndex].Add(1)
					transferRecords[targetIndex].Add(int64(len(job.records)))
					if err != nil {
						setTransferErr(fmt.Errorf("versioned ANN source=%d target=%d: %w", sourceIndex, targetIndex, err))
					}
				}
			}()
		}
	}
	finishTransfers := func() error {
		for _, queue := range jobs {
			close(queue)
		}
		transferWG.Wait()
		return getTransferErr()
	}
	flush := func(targetIndex int) error {
		records := pending[targetIndex]
		if len(records) == 0 {
			return nil
		}
		batchSequence++
		segmentID := fmt.Sprintf("versioned-%016x-s%04d-t%04d-%d", snapshotOffsets[sourceIndex], sourceIndex, targetIndex, batchSequence)
		delete(pending, targetIndex)
		if err := getTransferErr(); err != nil {
			return err
		}
		queueStarted := time.Now()
		jobs[targetIndex] <- versionedPrebuiltTransferJob{targetIndex: targetIndex, segmentID: segmentID, records: records}
		queueWaitNS[targetIndex].Add(time.Since(queueStarted).Nanoseconds())
		return nil
	}
	exportErr := versioned.ExportANNState(func(event PrebuiltANNStateEvent) error {
		switch event.Kind {
		case 0:
			seenBegin = true
			snapshotOffsets[sourceIndex] = event.DeltaOffset
		case 1, 3:
			if !seenBegin {
				return errors.New("versioned ANN export missing begin frame")
			}
			targetIndex := target.ShardForID(event.Record.ID)
			if sameDataNode(s.shards[sourceIndex], target.shards[targetIndex]) {
				return nil
			}
			accepted, err := deduper.accept(sourceIndex, event.Record)
			if err != nil {
				return err
			}
			if !accepted {
				return nil
			}
			pending[targetIndex] = append(pending[targetIndex], event.Record)
			if len(pending[targetIndex]) >= versionedPrebuiltTransferBatch {
				if err := flush(targetIndex); err != nil {
					return fmt.Errorf("versioned ANN source=%d target=%d: %w", sourceIndex, targetIndex, err)
				}
			}
		case 4:
			if !seenBegin {
				return errors.New("versioned ANN export missing begin frame")
			}
			for targetIndex := range pending {
				if err := flush(targetIndex); err != nil {
					return err
				}
			}
		}
		return nil
	})
	if err := errors.Join(exportErr, finishTransfers()); err != nil {
		return fmt.Errorf("versioned ANN export source=%d: %w", sourceIndex, err)
	}
	if !seenBegin {
		return fmt.Errorf("versioned ANN export source=%d returned no begin frame", sourceIndex)
	}
	if os.Getenv("LUMENVEC_RESHARD_TRACE") == "1" {
		for targetIndex := range destinations {
			fmt.Printf("RESHARD_VERSIONED source=%d target=%d jobs=%d records=%d queue_wait_ms=%.3f transfer_ms=%.3f export_total_ms=%.3f queue_depth=%d workers=%d\n",
				sourceIndex, targetIndex, transferJobs[targetIndex].Load(), transferRecords[targetIndex].Load(),
				float64(queueWaitNS[targetIndex].Load())/float64(time.Millisecond),
				float64(transferNS[targetIndex].Load())/float64(time.Millisecond),
				float64(time.Since(exportStarted))/float64(time.Millisecond), queueDepth, workersPerTarget)
		}
	}
	return nil
}

// migratePrebuiltRendezvous handles the incremental local-source path. It
// returns handled=false before mutation when any capability is unavailable,
// allowing the compatibility migration to remain the fallback.
func (s *ShardedService) migratePrebuiltRendezvous(target *ShardedService) (handled bool, err error) {
	if s.routingMode != ShardRoutingRendezvous || target.routingMode != ShardRoutingRendezvous ||
		len(s.shardIDs) != len(s.shards) || len(target.shardIDs) != len(target.shards) {
		return false, nil
	}
	sources := make([]prebuiltANNMigrationSource, len(s.shards))
	for i, shard := range s.shards {
		source, ok := shard.(prebuiltANNMigrationSource)
		if !ok {
			return false, nil
		}
		if service, local := shard.(*Service); local && (!service.annSegmented || service.hasAnyMetadata() || service.vectorPath == "") {
			return false, nil
		}
		offset, err := migrationSourceOffset(source)
		if err != nil || source.ReplayDeltasSince(offset, func(DeltaRecord) error { return nil }) != nil {
			return false, nil
		}
		sources[i] = source
	}
	destinations := make([]PrebuiltANNPartitionDestination, len(target.shards))
	for i, shard := range target.shards {
		destination, ok := shard.(PrebuiltANNPartitionDestination)
		if !ok {
			return false, nil
		}
		destinations[i] = destination
	}
	snapshotOffsets := make([]uint64, len(sources))
	for i, source := range sources {
		snapshotOffsets[i], err = migrationSourceOffset(source)
		if err != nil {
			return true, err
		}
	}
	if s.migrationAfterSnapshot != nil {
		s.migrationAfterSnapshot()
	}
	topologyHash := stableFNV64(target.routingMode)
	for _, id := range target.shardIDs {
		topologyHash ^= stableFNV64(id)
		topologyHash *= 1099511628211
	}
	// Remote versioned sources are independent data nodes. Export them in
	// parallel while each source retains ordered per-destination queues.
	// Deduplication remains global because topology transitions may briefly
	// expose the same external ID from two physical sources.
	deduper := &prebuiltMigrationDeduper{seen: make(map[string]PrebuiltANNRecord)}
	versionedHandled := make([]bool, len(sources))
	sourceParallelism := min(len(sources), 4)
	if raw := os.Getenv("LUMENVEC_RESHARD_SOURCE_PARALLELISM"); raw != "" {
		if configured, parseErr := strconv.Atoi(raw); parseErr == nil && configured > 0 {
			sourceParallelism = min(configured, len(sources))
		}
	}
	sourceSem := make(chan struct{}, max(1, sourceParallelism))
	sourceErrs := make(chan error, len(sources))
	var sourceWG sync.WaitGroup
	for sourceIndex, source := range sources {
		versioned, ok := source.(prebuiltANNVersionedSource)
		if !ok {
			continue
		}
		if _, local := s.shards[sourceIndex].(*Service); local {
			continue
		}
		versionedHandled[sourceIndex] = true
		sourceIndex, source, versioned := sourceIndex, source, versioned
		sourceWG.Add(1)
		go func() {
			defer sourceWG.Done()
			sourceSem <- struct{}{}
			defer func() { <-sourceSem }()
			if err := s.migrateVersionedSourceSnapshot(target, sourceIndex, source, versioned, destinations, snapshotOffsets, deduper); err != nil {
				sourceErrs <- err
			}
		}()
	}
	sourceWG.Wait()
	close(sourceErrs)
	if joined := errors.Join(readErrors(sourceErrs)...); joined != nil {
		return true, joined
	}

	// Local and legacy sources keep the artifact fast path below.
	seenMigrationIDs := make(map[string]PrebuiltANNRecord)
	for sourceIndex, source := range sources {
		if versionedHandled[sourceIndex] {
			continue
		}
		sourceShard := s.shards[sourceIndex]
		// Prefer the versioned export whenever the source advertises it. The
		// stream is consumed record-by-record, so no segment-sized in-memory
		// snapshot is created; its epoch/offset also define the delta baseline.
		if versioned, ok := source.(prebuiltANNVersionedSource); ok {
			// Keep the local prebuilt fast path (segment artifacts + replay)
			// until its destination-side atomic artifact commit is available;
			// remote nodes use the bounded versioned stream below.
			if _, local := sourceShard.(*Service); local {
				goto legacyExport
			}
			buildOptions := ann.Options{M: 16, EfConstruction: 64, EfSearch: 64}
			if configured, ok := source.(prebuiltANNMigrationConfig); ok {
				_, buildOptions = configured.PrebuiltANNMigrationConfig()
			}
			seenBegin := false
			pending := make(map[int][]PrebuiltANNRecord)
			batchSequence := 0
			jobs := make([]chan versionedPrebuiltTransferJob, len(destinations))
			var transferWG sync.WaitGroup
			var transferErr error
			var transferErrMu sync.Mutex
			setTransferErr := func(err error) {
				if err == nil {
					return
				}
				transferErrMu.Lock()
				if transferErr == nil {
					transferErr = err
				}
				transferErrMu.Unlock()
			}
			getTransferErr := func() error {
				transferErrMu.Lock()
				defer transferErrMu.Unlock()
				return transferErr
			}
			for targetIndex := range destinations {
				targetIndex := targetIndex
				jobs[targetIndex] = make(chan versionedPrebuiltTransferJob, 1)
				transferWG.Add(1)
				go func() {
					defer transferWG.Done()
					for job := range jobs[targetIndex] {
						if getTransferErr() != nil {
							continue
						}
						dir, err := os.MkdirTemp("", "lumenvec-versioned-export-")
						if err != nil {
							setTransferErr(err)
							continue
						}
						_, err = TransferPrebuiltANNPartition(dir, job.segmentID, "l2", buildOptions, func(yield func(PrebuiltANNRecord) error) error {
							for _, record := range job.records {
								if err := yield(record); err != nil {
									return err
								}
							}
							return nil
						}, destinations[targetIndex])
						_ = os.RemoveAll(dir)
						if err != nil {
							setTransferErr(fmt.Errorf("versioned ANN source=%d target=%d: %w", sourceIndex, targetIndex, err))
						}
					}
				}()
			}
			finishTransfers := func() error {
				for _, queue := range jobs {
					close(queue)
				}
				transferWG.Wait()
				transferErrMu.Lock()
				defer transferErrMu.Unlock()
				return transferErr
			}
			flush := func(targetIndex int) error {
				records := pending[targetIndex]
				if len(records) == 0 {
					return nil
				}
				batchSequence++
				segmentID := fmt.Sprintf("versioned-%016x-%04d-%d", snapshotOffsets[sourceIndex], targetIndex, batchSequence)
				delete(pending, targetIndex)
				if err := getTransferErr(); err != nil {
					return err
				}
				jobs[targetIndex] <- versionedPrebuiltTransferJob{targetIndex: targetIndex, segmentID: segmentID, records: records}
				return nil
			}
			exportErr := versioned.ExportANNState(func(event PrebuiltANNStateEvent) error {
				switch event.Kind {
				case 0:
					seenBegin = true
					snapshotOffsets[sourceIndex] = event.DeltaOffset
				case 1, 3:
					if !seenBegin {
						return errors.New("versioned ANN export missing begin frame")
					}
					targetIndex := target.ShardForID(event.Record.ID)
					if sameDataNode(sourceShard, target.shards[targetIndex]) {
						return nil
					}
					if previous, duplicate := seenMigrationIDs[event.Record.ID]; duplicate {
						if vectorChecksum([]index.Vector{{ID: event.Record.ID, Values: previous.Values}}) != vectorChecksum([]index.Vector{{ID: event.Record.ID, Values: event.Record.Values}}) {
							return fmt.Errorf("versioned ANN source=%d has conflicting duplicate vector %q", sourceIndex, event.Record.ID)
						}
						return nil
					}
					seenMigrationIDs[event.Record.ID] = event.Record
					pending[targetIndex] = append(pending[targetIndex], event.Record)
					if len(pending[targetIndex]) >= versionedPrebuiltTransferBatch {
						if err := flush(targetIndex); err != nil {
							return fmt.Errorf("versioned ANN source=%d target=%d: %w", sourceIndex, targetIndex, err)
						}
					}
				case 4:
					if !seenBegin {
						return errors.New("versioned ANN export missing begin frame")
					}
					for targetIndex := range pending {
						if err := flush(targetIndex); err != nil {
							return err
						}
					}
				}
				return nil
			})
			transferErr = errors.Join(exportErr, finishTransfers())
			if transferErr != nil {
				return true, fmt.Errorf("versioned ANN export source=%d: %w", sourceIndex, transferErr)
			}
			if !seenBegin {
				return true, fmt.Errorf("versioned ANN export source=%d returned no begin frame", sourceIndex)
			}
			continue
		}
	legacyExport:
		outgoingDir := ""
		options := ann.Options{M: 16, EfConstruction: 64, EfSearch: 64}
		if configured, ok := source.(prebuiltANNMigrationConfig); ok {
			outgoingDir, options = configured.PrebuiltANNMigrationConfig()
		}
		if outgoingDir == "" {
			outgoingDir, err = os.MkdirTemp("", "lumenvec-reshard-outgoing-")
			if err != nil {
				return true, err
			}
			defer os.RemoveAll(outgoingDir)
		}
		if err := os.MkdirAll(outgoingDir, 0o755); err != nil {
			return true, err
		}
		if err := source.VisitSealedANNRecords(func(segmentIndex int, replay PrebuiltANNReplay) error {
			for targetIndex, destination := range destinations {
				// A local fast path may share the existing shard object with the
				// target topology. Remote shards never share process memory, even
				// when their durable route IDs are equal, so their records must be
				// transferred into the independent destination.
				if sameDataNode(sourceShard, target.shards[targetIndex]) {
					continue
				}
				filtered := FilterPrebuiltANNReplay(replay, func(id string) bool {
					return target.ShardForID(id) == targetIndex
				})
				segmentID := fmt.Sprintf("reshard-%016x-s%04d-g%06d-t%04d", topologyHash, sourceIndex, segmentIndex, targetIndex)
				_, err := TransferPrebuiltANNPartition(outgoingDir, segmentID, "l2", options, filtered, destination)
				if errors.Is(err, ErrEmptyPrebuiltANNPartition) {
					continue
				}
				if err != nil {
					return fmt.Errorf("sealed source=%d segment=%d target=%d: %w", sourceIndex, segmentIndex, targetIndex, err)
				}
			}
			return nil
		}); err != nil {
			return true, fmt.Errorf("sealed export source=%d: %w", sourceIndex, err)
		}
		tail, err := source.SnapshotMutableANNRecords()
		if err != nil {
			return true, fmt.Errorf("mutable tail snapshot source=%d: %w", sourceIndex, err)
		}
		for _, record := range tail {
			targetIndex := target.ShardForID(record.ID)
			if sameDataNode(sourceShard, target.shards[targetIndex]) {
				continue
			}
			if err := applyReshardVector(target.shards[targetIndex], record.ID, record.Values); err != nil {
				return true, fmt.Errorf("mutable tail source=%d target=%d vector=%q: %w", sourceIndex, targetIndex, record.ID, err)
			}
		}
	}
	if s.migrationBeforeDrain != nil {
		s.migrationBeforeDrain()
	}
	var sourceCount, targetCount int
	for convergenceRound := 0; convergenceRound < 8; convergenceRound++ {
		for sourceIndex, source := range sources {
			sourceShard := s.shards[sourceIndex]
			offset := snapshotOffsets[sourceIndex]
			stableReads := 0
			for pass := 0; pass < 8 && stableReads < 2; pass++ {
				passStarted := time.Now()
				latest := make(map[int]map[string]DeltaRecord)
				if err := source.ReplayDeltasSince(offset, func(delta DeltaRecord) error {
					targetIndex := target.ShardForID(delta.VectorID)
					if !sameDataNode(sourceShard, target.shards[targetIndex]) {
						if latest[targetIndex] == nil {
							latest[targetIndex] = make(map[string]DeltaRecord)
						}
						latest[targetIndex][delta.VectorID] = delta
					}
					return nil
				}); err != nil {
					return true, fmt.Errorf("delta replay source=%d: %w", sourceIndex, err)
				}
				replayElapsed := time.Since(passStarted)
				upsertCount, deleteCount := 0, 0
				var stagedWG sync.WaitGroup
				var stagedErrCh chan error
				if os.Getenv("LUMENVEC_RESHARD_STAGED") == "1" {
					stagedErrCh = make(chan error, len(latest))
				}
				for targetIndex, records := range latest {
					upserts := make([]vectorBatch32, 0, len(records))
					deletes := make([]DeltaRecord, 0)
					for _, delta := range records {
						if delta.Deleted {
							deletes = append(deletes, delta)
							deleteCount++
							continue
						}
						upserts = append(upserts, vectorBatch32{ID: delta.VectorID, Values: delta.Values})
						upsertCount++
					}
					if len(upserts) > 0 {
						if os.Getenv("LUMENVEC_RESHARD_STAGED") == "1" {
							if staged, ok := target.shards[targetIndex].(*Service); ok {
								stagedWG.Add(1)
								go func(targetIndex int, staged *Service, upserts []vectorBatch32) {
									defer stagedWG.Done()
									if err := staged.addVectors32Staged(upserts); err != nil {
										stagedErrCh <- fmt.Errorf("staged delta publication source=%d target=%d: %w", sourceIndex, targetIndex, err)
									}
								}(targetIndex, staged, upserts)
							} else {
								return true, fmt.Errorf("staged delta publication target=%d is not a local service", targetIndex)
							}
						} else if batcher, ok := target.shards[targetIndex].(interface{ AddVectors32([]vectorBatch32) error }); ok {
							if err := batcher.AddVectors32(upserts); err != nil {
								return true, fmt.Errorf("delta batch upsert source=%d target=%d: %w", sourceIndex, targetIndex, err)
							}
						} else {
							for _, upsert := range upserts {
								values := make([]float64, len(upsert.Values))
								for i, value := range upsert.Values {
									values[i] = float64(value)
								}
								if err := applyReshardVector(target.shards[targetIndex], upsert.ID, values); err != nil {
									return true, err
								}
							}
						}
					}
					for _, delta := range deletes {
						err := target.shards[targetIndex].DeleteVector(delta.VectorID)
						if err != nil && !errors.Is(err, index.ErrVectorNotFound) {
							return true, fmt.Errorf("delta delete source=%d target=%d vector=%q: %w", sourceIndex, targetIndex, delta.VectorID, err)
						}
					}
				}
				if stagedErrCh != nil {
					stagedWG.Wait()
					close(stagedErrCh)
					for stagedErr := range stagedErrCh {
						if stagedErr != nil {
							return true, stagedErr
						}
					}
				}
				if os.Getenv("LUMENVEC_RESHARD_TRACE") == "1" {
					fmt.Printf("RESHARD_DELTA source=%d pass=%d upserts=%d deletes=%d replay_ms=%.3f apply_ms=%.3f\n",
						sourceIndex, pass, upsertCount, deleteCount,
						float64(replayElapsed)/float64(time.Millisecond),
						float64(time.Since(passStarted)-replayElapsed)/float64(time.Millisecond))
				}
				current, err := migrationSourceOffset(source)
				if err != nil {
					return true, err
				}
				if current == offset {
					// Do not treat two back-to-back reads between client batches as
					// a quiescent source. Give a producer that just received its
					// previous commit enough time to enter the next persistence
					// transaction; CurrentDeltaOffset then waits on that barrier.
					time.Sleep(50 * time.Millisecond)
					stableReads++
				} else {
					offset = current
					stableReads = 0
				}
			}
			if stableReads < 2 {
				return true, errors.New("prebuilt reshard delta WAL did not reach a stable cutover offset")
			}
			snapshotOffsets[sourceIndex] = offset
		}
		if layoutsShareDataNodes(s, target) {
			// Overlapping topologies can temporarily contain stale copies on
			// retained nodes. Sum-of-node counts would double count them.
			sourceCount, err = s.countVectorsCheckedPaged()
		} else {
			sourceCount, err = s.countVectorsChecked()
		}
		if err != nil {
			return true, err
		}
		if layoutsShareDataNodes(s, target) {
			targetCount, err = target.countVectorsCheckedPaged()
		} else {
			targetCount, err = target.countVectorsChecked()
		}
		if err != nil {
			return true, err
		}
		if sourceCount == targetCount {
			return true, nil
		}
		// A client batch may have started just after a stable offset sample.
		// Preserve the last applied offsets and resume another idempotent drain
		// round rather than accepting or immediately failing a partial cutover.
		time.Sleep(100 * time.Millisecond)
	}
	return true, fmt.Errorf("prebuilt reshard cardinality did not converge: source=%d target=%d", sourceCount, targetCount)
}

func applyReshardVector(destination VectorService, id string, values []float64) error {
	if existing, err := destination.GetVector(id); err == nil {
		if vectorChecksum([]index.Vector{existing}) == vectorChecksum([]index.Vector{{ID: id, Values: values}}) {
			return nil
		}
		if err := destination.DeleteVector(id); err != nil && !errors.Is(err, index.ErrVectorNotFound) {
			return err
		}
	} else if !errors.Is(err, index.ErrVectorNotFound) {
		return err
	}
	return destination.AddVector(id, values)
}
