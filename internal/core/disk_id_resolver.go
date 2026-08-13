package core

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
)

// diskIDResolver keeps both directions of the public-ID mapping outside the
// Go heap. The forward exact hash owns complete ID bytes; reverse slots point
// back into that key log. Both indexes are derived from the durable ANN
// checkpoint and can therefore be rebuilt atomically during restore.
type diskIDResolver struct {
	mu       sync.RWMutex
	forward  *diskLocationIndex
	reverse  *diskLocationIndex
	capacity uint64
	next     int
	count    int
	closed   bool
}

func openDiskIDResolver(directory string, capacity uint64) (*diskIDResolver, error) {
	if capacity < 8 {
		return nil, errors.New("disk ID resolver capacity must be at least 8")
	}
	forward, err := openDiskLocationIndex(
		filepath.Join(directory, "ids.forward.hash"),
		filepath.Join(directory, "ids.forward.keys"),
		capacity,
	)
	if err != nil {
		return nil, err
	}
	reverse, err := openDiskLocationIndex(
		filepath.Join(directory, "ids.reverse.hash"),
		filepath.Join(directory, "ids.reverse.keys"),
		capacity,
	)
	if err != nil {
		_ = forward.Close()
		return nil, err
	}
	resolver := &diskIDResolver{forward: forward, reverse: reverse, capacity: capacity}
	if err := resolver.validateExisting(); err != nil {
		_ = resolver.Close()
		return nil, err
	}
	return resolver, nil
}

func (r *diskIDResolver) validateExisting() error {
	if r.forward.Count() != r.reverse.Count() {
		return errors.New("disk ID resolver direction counts differ")
	}
	maxID, count := 0, 0
	err := r.reverse.Range(func(rawInternal string, value diskLocationValue) bool {
		internalID, parseErr := strconv.Atoi(rawInternal)
		if parseErr != nil || internalID <= 0 || internalID > math.MaxInt32 ||
			value.RecordOffset < 0 {
			maxID = -1
			return false
		}
		id, readErr := r.forward.ReadKeyReference(diskKeyReference{
			Offset: uint64(value.RecordOffset),
			Length: value.RecordLength,
		})
		forwardValue, found, getErr := r.forward.Get(id)
		if readErr != nil || getErr != nil || !found || int(forwardValue.SegmentGeneration) != internalID {
			maxID = -1
			return false
		}
		if internalID > maxID {
			maxID = internalID
		}
		count++
		return true
	})
	if err != nil {
		return err
	}
	if maxID < 0 || count != int(r.forward.Count()) {
		return errors.New("disk ID resolver is inconsistent")
	}
	r.next, r.count = maxID, count
	return nil
}

func (r *diskIDResolver) Assign(id string) int {
	internalID, _ := r.AssignID(id)
	return internalID
}

func (r *diskIDResolver) AssignID(id string) (int, error) {
	if id == "" {
		return 0, errors.New("vector ID must not be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return 0, errors.New("disk ID resolver is closed")
	}
	value, found, err := r.forward.Get(id)
	if err != nil {
		return 0, err
	}
	if found {
		internalID := int(value.SegmentGeneration)
		if internalID <= 0 || uint64(internalID) != value.SegmentGeneration {
			return 0, errors.New("disk ID resolver contains an invalid internal ID")
		}
		return internalID, nil
	}
	if r.count >= int(r.capacity) || r.next >= math.MaxInt32 {
		return 0, ErrLocationIndexFull
	}
	internalID := r.next + 1
	forwardValue := diskLocationValue{SegmentGeneration: uint64(internalID), RecordLength: 1}
	if err := r.forward.Put(id, forwardValue); err != nil {
		return 0, err
	}
	reference, found, err := r.forward.KeyReference(id)
	if err != nil || !found || reference.Offset > math.MaxInt64 {
		_, _ = r.forward.Delete(id)
		if err == nil {
			err = errors.New("disk ID resolver key reference is invalid")
		}
		return 0, err
	}
	reverseValue := diskLocationValue{
		RecordOffset: int64(reference.Offset),
		RecordLength: reference.Length,
	}
	if err := r.reverse.Put(strconv.Itoa(internalID), reverseValue); err != nil {
		_, _ = r.forward.Delete(id)
		return 0, err
	}
	r.next, r.count = internalID, r.count+1
	return internalID, nil
}

func (r *diskIDResolver) Lookup(internalID int) (string, bool) {
	id, found, _ := r.LookupID(internalID)
	return id, found
}

func (r *diskIDResolver) LookupID(internalID int) (string, bool, error) {
	if internalID <= 0 {
		return "", false, nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return "", false, errors.New("disk ID resolver is closed")
	}
	value, found, err := r.reverse.Get(strconv.Itoa(internalID))
	if err != nil || !found {
		return "", false, err
	}
	if value.RecordOffset < 0 {
		return "", false, errors.New("disk ID resolver contains an invalid key offset")
	}
	id, err := r.forward.ReadKeyReference(diskKeyReference{
		Offset: uint64(value.RecordOffset),
		Length: value.RecordLength,
	})
	if err != nil {
		return "", false, err
	}
	forwardValue, found, err := r.forward.Get(id)
	if err != nil || !found {
		return "", false, err
	}
	if forwardValue.SegmentGeneration != uint64(internalID) {
		return "", false, errors.New("disk ID resolver directions disagree")
	}
	return id, true, nil
}

func (r *diskIDResolver) LookupExternalID(id string) (int, bool, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return 0, false, errors.New("disk ID resolver is closed")
	}
	value, found, err := r.forward.Get(id)
	if err != nil || !found {
		return 0, false, err
	}
	if value.SegmentGeneration == 0 || value.SegmentGeneration > math.MaxInt32 {
		return 0, false, errors.New("disk ID resolver contains an invalid internal ID")
	}
	return int(value.SegmentGeneration), true, nil
}

func (r *diskIDResolver) Remove(id string) {
	_ = r.RemoveID(id)
}

func (r *diskIDResolver) RemoveID(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return errors.New("disk ID resolver is closed")
	}
	value, found, err := r.forward.Get(id)
	if err != nil || !found {
		return err
	}
	internalID := int(value.SegmentGeneration)
	deleted, err := r.forward.Delete(id)
	if err != nil || !deleted {
		return err
	}
	if _, err := r.reverse.Delete(strconv.Itoa(internalID)); err != nil {
		// Restore the forward direction before reporting the failed mutation.
		_ = r.forward.Put(id, value)
		return err
	}
	r.count--
	return nil
}

func (r *diskIDResolver) snapshotSorted() []idResolverEntry {
	entries, _ := r.snapshotSortedE()
	return entries
}

func (r *diskIDResolver) snapshotSortedE() ([]idResolverEntry, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.closed {
		return nil, errors.New("disk ID resolver is closed")
	}
	entries := make([]idResolverEntry, 0, r.count)
	if err := r.forward.Range(func(id string, value diskLocationValue) bool {
		entries = append(entries, idResolverEntry{id: id, internalID: int(value.SegmentGeneration)})
		return true
	}); err != nil {
		return nil, err
	}
	if len(entries) != r.count {
		return nil, errors.New("disk ID resolver snapshot count mismatch")
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].id < entries[j].id })
	return entries, nil
}

func (r *diskIDResolver) Restore(entries map[string]int) error {
	ordered := make([]idResolverEntry, 0, len(entries))
	for id, internalID := range entries {
		ordered = append(ordered, idResolverEntry{id: id, internalID: internalID})
	}
	return r.restoreEntries(ordered)
}

func (r *diskIDResolver) restoreEntries(entries []idResolverEntry) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return errors.New("disk ID resolver is closed")
	}
	if uint64(len(entries)) > r.capacity {
		return ErrLocationIndexFull
	}
	if r.count == len(entries) {
		matches := true
		maxID := 0
		for _, entry := range entries {
			value, found, err := r.forward.Get(entry.id)
			if err != nil {
				return err
			}
			if !found || value.SegmentGeneration != uint64(entry.internalID) {
				matches = false
				break
			}
			if entry.internalID > maxID {
				maxID = entry.internalID
			}
		}
		if matches {
			r.next = maxID
			return nil
		}
	}
	if err := r.forward.Reset(); err != nil {
		return err
	}
	if err := r.reverse.Reset(); err != nil {
		return err
	}
	maxID := 0
	for _, entry := range entries {
		if entry.id == "" || entry.internalID <= 0 || entry.internalID > math.MaxInt32 {
			_ = r.forward.Reset()
			_ = r.reverse.Reset()
			return errors.New("invalid disk ID resolver checkpoint entry")
		}
		if _, found, err := r.reverse.Get(strconv.Itoa(entry.internalID)); err != nil || found {
			_ = r.forward.Reset()
			_ = r.reverse.Reset()
			if err != nil {
				return err
			}
			return errors.New("duplicate internal ID in disk resolver checkpoint")
		}
		if _, found, err := r.forward.Get(entry.id); err != nil || found {
			_ = r.forward.Reset()
			_ = r.reverse.Reset()
			if err != nil {
				return err
			}
			return errors.New("duplicate public ID in disk resolver checkpoint")
		}
		if err := r.forward.Put(entry.id, diskLocationValue{SegmentGeneration: uint64(entry.internalID), RecordLength: 1}); err != nil {
			_ = r.forward.Reset()
			_ = r.reverse.Reset()
			return err
		}
		reference, found, err := r.forward.KeyReference(entry.id)
		if err != nil || !found || reference.Offset > math.MaxInt64 {
			_ = r.forward.Reset()
			_ = r.reverse.Reset()
			if err != nil {
				return fmt.Errorf("resolve restored ID reference: %w", err)
			}
			return errors.New("restored disk ID reference is invalid")
		}
		if err := r.reverse.Put(strconv.Itoa(entry.internalID), diskLocationValue{
			RecordOffset: int64(reference.Offset),
			RecordLength: reference.Length,
		}); err != nil {
			_ = r.forward.Reset()
			_ = r.reverse.Reset()
			return err
		}
		if entry.internalID > maxID {
			maxID = entry.internalID
		}
	}
	if err := errors.Join(r.forward.Sync(), r.reverse.Sync()); err != nil {
		return err
	}
	r.next, r.count = maxID, len(entries)
	return nil
}

func (r *diskIDResolver) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	r.closed = true
	return errors.Join(r.forward.Close(), r.reverse.Close())
}
