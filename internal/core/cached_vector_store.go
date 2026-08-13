package core

import (
	"container/list"
	"sync"
	"sync/atomic"
	"time"

	"lumenvec/internal/index"
	"lumenvec/internal/vector"
)

type CacheOptions struct {
	Enabled  bool
	MaxBytes int64
	MaxItems int
	TTL      time.Duration
}

type cachedVectorStore struct {
	backend  VectorStore
	maxBytes int64
	maxItems int
	ttl      time.Duration

	mu           sync.Mutex
	entries      map[string]*list.Element
	order        *list.List
	currentBytes int64
	stats        cacheStoreStats
}

type cacheEntry struct {
	id        string
	values    []float32
	expiresAt time.Time
	sizeBytes int64
}

type CacheStats struct {
	Hits      uint64 `json:"hits"`
	Misses    uint64 `json:"misses"`
	Evictions uint64 `json:"evictions"`
	Items     uint64 `json:"items"`
	Bytes     uint64 `json:"bytes"`
}

type cacheStoreStats struct {
	hits      atomic.Uint64
	misses    atomic.Uint64
	evictions atomic.Uint64
}

func newCachedVectorStore(backend VectorStore, opts CacheOptions) VectorStore {
	if backend == nil {
		backend = newMemoryVectorStore()
	}
	if !opts.Enabled {
		return backend
	}
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = 8 << 20
	}
	if opts.MaxItems <= 0 {
		opts.MaxItems = 1024
	}
	return &cachedVectorStore{
		backend:  backend,
		maxBytes: opts.MaxBytes,
		maxItems: opts.MaxItems,
		ttl:      opts.TTL,
		entries:  make(map[string]*list.Element),
		order:    list.New(),
	}
}

func (s *cachedVectorStore) UpsertVector(vec index.Vector) error {
	return s.UpsertVectors([]index.Vector{vec})
}

func (s *cachedVectorStore) UpsertVectors(vectors []index.Vector) error {
	if len(vectors) == 0 {
		return nil
	}
	if writer, ok := s.backend.(batchVectorStore); ok {
		if err := writer.UpsertVectors(vectors); err != nil {
			return err
		}
	} else {
		for _, vec := range vectors {
			if err := s.backend.UpsertVector(vec); err != nil {
				return err
			}
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, vec := range vectors {
		s.putLocked(vec)
	}
	return nil
}

func (s *cachedVectorStore) GetVector(id string) (index.Vector, error) {
	if vec, ok := s.getCached(id); ok {
		return vec, nil
	}
	s.stats.misses.Add(1)
	vec, err := s.backend.GetVector(id)
	if err != nil {
		return index.Vector{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.putLocked(vec)
	return index.Vector{ID: vec.ID, Values: cloneVectorValues(vec.Values)}, nil
}

func (s *cachedVectorStore) GetVectorReadOnly(id string) (index.Vector, error) {
	if vec, ok := s.getCachedReadOnly(id); ok {
		return vec, nil
	}
	s.stats.misses.Add(1)

	if reader, ok := s.backend.(readOnlyVectorReader); ok {
		vec, err := reader.GetVectorReadOnly(id)
		if err != nil {
			return index.Vector{}, err
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.putLocked(vec)
		return index.Vector{ID: vec.ID, Values: cloneVectorValues(vec.Values)}, nil
	}

	vec, err := s.backend.GetVector(id)
	if err != nil {
		return index.Vector{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.putLocked(vec)
	return vec, nil
}

func (s *cachedVectorStore) GetVectorReadOnly32(id string) ([]float32, error) {
	if values, ok := s.getCachedReadOnly32(id); ok {
		return values, nil
	}
	s.stats.misses.Add(1)

	if reader, ok := s.backend.(readOnlyVector32Reader); ok {
		values, err := reader.GetVectorReadOnly32(id)
		if err != nil {
			return nil, err
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		s.putLocked32(id, values)
		return values, nil
	}

	var vec index.Vector
	var err error
	if reader, ok := s.backend.(readOnlyVectorReader); ok {
		vec, err = reader.GetVectorReadOnly(id)
	} else {
		vec, err = s.backend.GetVector(id)
	}
	if err != nil {
		return nil, err
	}
	values := vector.ToFloat32(vec.Values)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.putLocked32(vec.ID, values)
	return values, nil
}

func (s *cachedVectorStore) DeleteVector(id string) error {
	if err := s.backend.DeleteVector(id); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.removeLocked(id)
	return nil
}

func (s *cachedVectorStore) ListVectors() []index.Vector {
	return s.backend.ListVectors()
}

func (s *cachedVectorStore) RangeVectorIDs(fn func(string) bool) {
	if ranger, ok := s.backend.(vectorIDRanger); ok {
		ranger.RangeVectorIDs(fn)
		return
	}
	for _, vec := range s.backend.ListVectors() {
		if !fn(vec.ID) {
			return
		}
	}
}

func (s *cachedVectorStore) PageVectorIDs(afterID string, limit int) []string {
	if pager, ok := s.backend.(vectorIDPager); ok {
		return pager.PageVectorIDs(afterID, limit)
	}
	return selectPageIDsFromRange(afterID, limit, s.RangeVectorIDs)
}

func (s *cachedVectorStore) RangeSelectedVectors32(candidates map[string]struct{}, fn func(string, []float32) bool) {
	if reader, ok := s.backend.(selectedVector32Reader); ok {
		reader.RangeSelectedVectors32(candidates, fn)
		return
	}
	for id := range candidates {
		values, err := s.GetVectorReadOnly32(id)
		if err != nil || !fn(id, values) {
			return
		}
	}
}

func (s *cachedVectorStore) RangeSelectedVectorIDs32(candidates []string, fn func(string, []float32) bool) {
	if reader, ok := s.backend.(selectedVectorIDs32Reader); ok {
		reader.RangeSelectedVectorIDs32(candidates, fn)
		return
	}
	for _, id := range candidates {
		values, err := s.GetVectorReadOnly32(id)
		if err != nil || !fn(id, values) {
			return
		}
	}
}

func (s *cachedVectorStore) IsPersistent() bool {
	persistent, ok := s.backend.(persistentVectorStore)
	return ok && persistent.IsPersistent()
}

func (s *cachedVectorStore) Close() error {
	closer, ok := s.backend.(interface{ Close() error })
	if !ok {
		return nil
	}
	return closer.Close()
}

func (s *cachedVectorStore) DiskStats() DiskStoreStats {
	reader, ok := s.backend.(diskStatsReader)
	if !ok {
		return DiskStoreStats{}
	}
	return reader.DiskStats()
}

func (s *cachedVectorStore) SaveRecoveryIndex() error {
	writer, ok := s.backend.(interface{ SaveRecoveryIndex() error })
	if !ok {
		return nil
	}
	return writer.SaveRecoveryIndex()
}

func (s *cachedVectorStore) RecoveryIndexLoaded() bool {
	reader, ok := s.backend.(interface{ RecoveryIndexLoaded() bool })
	return ok && reader.RecoveryIndexLoaded()
}

func (s *cachedVectorStore) RebuildRecoveryIndex() error {
	rebuilder, ok := s.backend.(interface{ RebuildRecoveryIndex() error })
	if !ok {
		return nil
	}
	return rebuilder.RebuildRecoveryIndex()
}

func (s *cachedVectorStore) getCached(id string) (index.Vector, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	elem, ok := s.entries[id]
	if !ok {
		return index.Vector{}, false
	}
	entry := elem.Value.(*cacheEntry)
	if s.isExpired(entry) {
		s.stats.misses.Add(1)
		s.removeElementLocked(elem)
		return index.Vector{}, false
	}
	s.stats.hits.Add(1)
	s.order.MoveToFront(elem)
	return index.Vector{ID: entry.id, Values: vector.ToFloat64(entry.values)}, true
}

func (s *cachedVectorStore) getCachedReadOnly(id string) (index.Vector, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	elem, ok := s.entries[id]
	if !ok {
		return index.Vector{}, false
	}
	entry := elem.Value.(*cacheEntry)
	if s.isExpired(entry) {
		s.stats.misses.Add(1)
		s.removeElementLocked(elem)
		return index.Vector{}, false
	}
	s.stats.hits.Add(1)
	s.order.MoveToFront(elem)
	return index.Vector{ID: entry.id, Values: vector.ToFloat64(entry.values)}, true
}

func (s *cachedVectorStore) getCachedReadOnly32(id string) ([]float32, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	elem, ok := s.entries[id]
	if !ok {
		return nil, false
	}
	entry := elem.Value.(*cacheEntry)
	if s.isExpired(entry) {
		s.stats.misses.Add(1)
		s.removeElementLocked(elem)
		return nil, false
	}
	s.stats.hits.Add(1)
	s.order.MoveToFront(elem)
	return entry.values, true
}

func (s *cachedVectorStore) putLocked(vec index.Vector) {
	s.putLocked32(vec.ID, vector.ToFloat32(vec.Values))
}

func (s *cachedVectorStore) putLocked32(id string, values []float32) {
	valuesCopy := cloneFloat32Values(values)
	sizeBytes := estimateVectorSizeBytes32(id, valuesCopy)
	if elem, ok := s.entries[id]; ok {
		entry := elem.Value.(*cacheEntry)
		s.currentBytes -= entry.sizeBytes
		entry.values = valuesCopy
		entry.expiresAt = s.nextExpiry()
		entry.sizeBytes = sizeBytes
		s.currentBytes += sizeBytes
		s.order.MoveToFront(elem)
		s.evictLocked()
		return
	}
	entry := &cacheEntry{
		id:        id,
		values:    valuesCopy,
		expiresAt: s.nextExpiry(),
		sizeBytes: sizeBytes,
	}
	elem := s.order.PushFront(entry)
	s.entries[id] = elem
	s.currentBytes += sizeBytes
	s.evictLocked()
}

func (s *cachedVectorStore) removeLocked(id string) {
	if elem, ok := s.entries[id]; ok {
		s.removeElementLocked(elem)
	}
}

func (s *cachedVectorStore) removeElementLocked(elem *list.Element) {
	if elem == nil {
		return
	}
	entry := elem.Value.(*cacheEntry)
	delete(s.entries, entry.id)
	s.currentBytes -= entry.sizeBytes
	s.order.Remove(elem)
}

func (s *cachedVectorStore) nextExpiry() time.Time {
	if s.ttl <= 0 {
		return time.Time{}
	}
	return time.Now().Add(s.ttl)
}

func (s *cachedVectorStore) isExpired(entry *cacheEntry) bool {
	return !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt)
}

func (s *cachedVectorStore) Stats() CacheStats {
	s.mu.Lock()
	defer s.mu.Unlock()

	return CacheStats{
		Hits:      s.stats.hits.Load(),
		Misses:    s.stats.misses.Load(),
		Evictions: s.stats.evictions.Load(),
		Items:     uint64(len(s.entries)),
		Bytes:     nonNegativeUint64(s.currentBytes),
	}
}

func (s *cachedVectorStore) evictLocked() {
	for len(s.entries) > s.maxItems || s.currentBytes > s.maxBytes {
		s.stats.evictions.Add(1)
		s.removeElementLocked(s.order.Back())
	}
}

func estimateVectorSizeBytes(vec index.Vector) int64 {
	return int64(len(vec.ID)) + int64(len(vec.Values))*4
}

func estimateVectorSizeBytes32(id string, values []float32) int64 {
	return int64(len(id)) + int64(len(values))*4
}

func cloneFloat32Values(values []float32) []float32 {
	out := make([]float32, len(values))
	copy(out, values)
	return out
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
