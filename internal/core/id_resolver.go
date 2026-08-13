package core

import (
	"errors"
	"math"
	"sort"
	"sync"
	"unsafe"

	"github.com/cespare/xxhash/v2"
)

const reverseIDPageSize = 4096

type idResolverEntry struct {
	id         string
	internalID int
}

// IDMappingEntry is the stable external-to-internal identity carried with a
// prebuilt ANN segment. Preserving it lets a destination publish the graph
// without changing the IDs returned to clients.
type IDMappingEntry struct {
	ID         string
	InternalID int
}

type mergeIDMappingResolver interface {
	MergeIDMappings([]IDMappingEntry) error
}

type trackedMergeIDMappingResolver interface {
	// MergeIDMappingsTracked returns only the external IDs created by this
	// call. Callers can therefore roll back a failed reshard window without
	// deleting compatible mappings that predated the window.
	MergeIDMappingsTracked([]IDMappingEntry) ([]string, error)
}

type externalIDLookupResolver interface {
	LookupExternalID(string) (int, bool, error)
}

// checkpointIDResolver separates checkpoint persistence from the online
// resolver API. Disk-backed resolvers can implement this contract without
// forcing ANN checkpoint code to depend on the in-memory implementation.
type checkpointIDResolver interface {
	IDResolver
	snapshotSorted() []idResolverEntry
	Restore(map[string]int) error
	restoreEntries([]idResolverEntry) error
}

type errorAwareCheckpointIDResolver interface {
	checkpointIDResolver
	snapshotSortedE() ([]idResolverEntry, error)
}

type memoryIDResolver struct {
	mu sync.RWMutex
	// String ownership lives in internalToID. The direct index stores only a
	// 64-bit fingerprint and verifies the original string before returning,
	// so hash collisions cannot alias user IDs.
	hashToInternal map[uint64]int
	hashCollisions map[uint64][]int
	hashID         func(string) uint64
	// Internal IDs are monotonic and mostly dense. A paged slice avoids the
	// bucket and hashing overhead of map[int]string while retaining O(1)
	// lookup and without allocating storage for deleted/sparse pages.
	internalToID [][]string
	nextInternal int
	idCount      int
}

type idResolverMemoryStats struct {
	IDs          int
	ReversePages int
	ReverseBytes uint64
}

func newMemoryIDResolver() *memoryIDResolver {
	return &memoryIDResolver{
		hashToInternal: make(map[uint64]int),
		hashCollisions: make(map[uint64][]int),
		hashID:         xxhash.Sum64String,
	}
}

func (r *memoryIDResolver) setReverse(internalID int, id string) {
	pageIndex := internalID / reverseIDPageSize
	slot := internalID % reverseIDPageSize
	for len(r.internalToID) <= pageIndex {
		r.internalToID = append(r.internalToID, nil)
	}
	if r.internalToID[pageIndex] == nil {
		r.internalToID[pageIndex] = make([]string, reverseIDPageSize)
	}
	r.internalToID[pageIndex][slot] = id
}

func (r *memoryIDResolver) reverse(internalID int) (string, bool) {
	if internalID <= 0 {
		return "", false
	}
	pageIndex := internalID / reverseIDPageSize
	slot := internalID % reverseIDPageSize
	if pageIndex >= len(r.internalToID) || r.internalToID[pageIndex] == nil {
		return "", false
	}
	id := r.internalToID[pageIndex][slot]
	return id, id != ""
}

func (r *memoryIDResolver) find(id string) (int, bool) {
	hash := r.hashID(id)
	if internalID, ok := r.hashToInternal[hash]; ok {
		if stored, live := r.reverse(internalID); live && stored == id {
			return internalID, true
		}
	}
	for _, internalID := range r.hashCollisions[hash] {
		if stored, live := r.reverse(internalID); live && stored == id {
			return internalID, true
		}
	}
	return 0, false
}

func (r *memoryIDResolver) addDirect(id string, internalID int) {
	hash := r.hashID(id)
	if _, exists := r.hashToInternal[hash]; !exists {
		r.hashToInternal[hash] = internalID
		return
	}
	r.hashCollisions[hash] = append(r.hashCollisions[hash], internalID)
}

func (r *memoryIDResolver) Assign(id string) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	if internalID, ok := r.find(id); ok {
		return internalID
	}
	r.nextInternal++
	internalID := r.nextInternal
	r.setReverse(internalID, id)
	r.addDirect(id, internalID)
	r.idCount++
	return internalID
}

func (r *memoryIDResolver) AssignID(id string) (int, error) {
	return r.Assign(id), nil
}

func (r *memoryIDResolver) Lookup(internalID int) (string, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	return r.reverse(internalID)
}

func (r *memoryIDResolver) LookupID(internalID int) (string, bool, error) {
	id, ok := r.Lookup(internalID)
	return id, ok, nil
}

func (r *memoryIDResolver) LookupExternalID(id string) (int, bool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	internalID, ok := r.find(id)
	return internalID, ok, nil
}

func (r *memoryIDResolver) Remove(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	internalID, ok := r.find(id)
	if !ok {
		return
	}
	hash := r.hashID(id)
	if r.hashToInternal[hash] == internalID {
		collisions := r.hashCollisions[hash]
		if len(collisions) == 0 {
			delete(r.hashToInternal, hash)
		} else {
			r.hashToInternal[hash] = collisions[0]
			if len(collisions) == 1 {
				delete(r.hashCollisions, hash)
			} else {
				r.hashCollisions[hash] = collisions[1:]
			}
		}
	} else {
		collisions := r.hashCollisions[hash]
		for n, candidate := range collisions {
			if candidate == internalID {
				collisions[n] = collisions[len(collisions)-1]
				collisions = collisions[:len(collisions)-1]
				break
			}
		}
		if len(collisions) == 0 {
			delete(r.hashCollisions, hash)
		} else {
			r.hashCollisions[hash] = collisions
		}
	}
	pageIndex := internalID / reverseIDPageSize
	slot := internalID % reverseIDPageSize
	if pageIndex < len(r.internalToID) && r.internalToID[pageIndex] != nil {
		r.internalToID[pageIndex][slot] = ""
	}
	r.idCount--
}

func (r *memoryIDResolver) RemoveID(id string) error {
	r.Remove(id)
	return nil
}

func (r *memoryIDResolver) Snapshot() map[string]int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]int, r.idCount)
	for pageIndex, page := range r.internalToID {
		for slot, id := range page {
			if id != "" {
				out[id] = pageIndex*reverseIDPageSize + slot
			}
		}
	}
	return out
}

func (r *memoryIDResolver) snapshotSorted() []idResolverEntry {
	r.mu.RLock()
	entries := make([]idResolverEntry, 0, r.idCount)
	for pageIndex, page := range r.internalToID {
		for slot, id := range page {
			if id != "" {
				entries = append(entries, idResolverEntry{id: id, internalID: pageIndex*reverseIDPageSize + slot})
			}
		}
	}
	r.mu.RUnlock()
	sort.Slice(entries, func(i, j int) bool { return entries[i].id < entries[j].id })
	return entries
}

func (r *memoryIDResolver) snapshotSortedE() ([]idResolverEntry, error) {
	return r.snapshotSorted(), nil
}

func (r *memoryIDResolver) MemoryStats() idResolverMemoryStats {
	r.mu.RLock()
	defer r.mu.RUnlock()
	pages := 0
	for _, page := range r.internalToID {
		if page != nil {
			pages++
		}
	}
	stringBytes := uint64(unsafe.Sizeof(""))
	pointerBytes := uint64(unsafe.Sizeof(uintptr(0)))
	return idResolverMemoryStats{
		IDs:          r.idCount,
		ReversePages: pages,
		ReverseBytes: uint64(pages*reverseIDPageSize)*stringBytes + uint64(cap(r.internalToID))*pointerBytes,
	}
}

func (r *memoryIDResolver) Restore(entries map[string]int) error {
	ordered := make([]idResolverEntry, 0, len(entries))
	for id, internalID := range entries {
		ordered = append(ordered, idResolverEntry{id: id, internalID: internalID})
	}
	return r.restoreEntries(ordered)
}

// MergeIDMappings atomically adds mappings while rejecting both external and
// internal identity conflicts. The all-or-nothing validation is essential for
// reshard retries: a repeated window is harmless, a mismatched window cannot
// silently corrupt result translation.
func (r *memoryIDResolver) MergeIDMappings(entries []IDMappingEntry) error {
	_, err := r.MergeIDMappingsTracked(entries)
	return err
}

func (r *memoryIDResolver) MergeIDMappingsTracked(entries []IDMappingEntry) ([]string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	incomingByExternal := make(map[string]int, len(entries))
	incomingByInternal := make(map[int]string, len(entries))
	for _, entry := range entries {
		if entry.ID == "" || entry.InternalID <= 0 || entry.InternalID > math.MaxInt32 {
			return nil, errors.New("invalid ID mapping entry")
		}
		if previous, exists := incomingByExternal[entry.ID]; exists && previous != entry.InternalID {
			return nil, errors.New("duplicate external ID in mapping window")
		}
		if previous, exists := incomingByInternal[entry.InternalID]; exists && previous != entry.ID {
			return nil, errors.New("duplicate internal ID in mapping window")
		}
		incomingByExternal[entry.ID] = entry.InternalID
		incomingByInternal[entry.InternalID] = entry.ID
		if existing, ok := r.find(entry.ID); ok && existing != entry.InternalID {
			return nil, errors.New("external ID maps to a different internal ID")
		}
		if existing, ok := r.reverse(entry.InternalID); ok && existing != entry.ID {
			return nil, errors.New("internal ID maps to a different external ID")
		}
	}
	added := make([]string, 0, len(entries))
	for _, entry := range entries {
		if _, exists := r.find(entry.ID); exists {
			continue // idempotent retry of the same mapping.
		}
		r.setReverse(entry.InternalID, entry.ID)
		r.addDirect(entry.ID, entry.InternalID)
		r.idCount++
		added = append(added, entry.ID)
		if entry.InternalID > r.nextInternal {
			r.nextInternal = entry.InternalID
		}
	}
	return added, nil
}

func (r *memoryIDResolver) restoreEntries(entries []idResolverEntry) error {
	var internalToID [][]string
	hashToInternal := make(map[uint64]int, len(entries))
	hashCollisions := make(map[uint64][]int)
	maxID := 0
	for _, entry := range entries {
		id, internalID := entry.id, entry.internalID
		if id == "" || internalID <= 0 || internalID > math.MaxInt32 {
			return errors.New("invalid ID resolver checkpoint entry")
		}
		pageIndex := internalID / reverseIDPageSize
		slot := internalID % reverseIDPageSize
		for len(internalToID) <= pageIndex {
			internalToID = append(internalToID, nil)
		}
		if internalToID[pageIndex] == nil {
			internalToID[pageIndex] = make([]string, reverseIDPageSize)
		}
		if internalToID[pageIndex][slot] != "" {
			return errors.New("duplicate internal ID in resolver checkpoint")
		}
		internalToID[pageIndex][slot] = id
		hash := r.hashID(id)
		if _, exists := hashToInternal[hash]; exists {
			hashCollisions[hash] = append(hashCollisions[hash], internalID)
		} else {
			hashToInternal[hash] = internalID
		}
		if internalID > maxID {
			maxID = internalID
		}
	}
	r.mu.Lock()
	r.hashToInternal = hashToInternal
	r.hashCollisions = hashCollisions
	r.internalToID = internalToID
	r.nextInternal = maxID
	r.idCount = len(entries)
	r.mu.Unlock()
	return nil
}
