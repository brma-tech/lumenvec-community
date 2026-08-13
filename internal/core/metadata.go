package core

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"lumenvec/internal/index"
)

// AddVectorsWithMetadata preserves the batch ingest fast path and persists the
// metadata sidecar once per batch instead of once per vector.
func (s *Service) AddVectorsWithMetadata(vectors []index.Vector, metadata []map[string]string) error {
	if len(vectors) != len(metadata) {
		return fmt.Errorf("metadata count %d does not match vector count %d", len(metadata), len(vectors))
	}
	if err := s.AddVectors(vectors); err != nil {
		return err
	}
	s.metadataMu.Lock()
	if s.metadata == nil {
		s.metadata = make(map[string]map[string]string)
	}
	for i, vector := range vectors {
		s.metadata[vector.ID] = cloneMetadata(metadata[i])
		s.textIndex.addMetadata(vector.ID, metadata[i])
		s.addFullTextMetadataIfReady(vector.ID, metadata[i])
	}
	s.metadataMu.Unlock()
	entries := make([]walMetadataEntry, len(vectors))
	for i, vector := range vectors {
		entries[i] = walMetadataEntry{ID: vector.ID, Metadata: cloneMetadata(metadata[i])}
	}
	return s.appendWAL(walOp{Op: "metadata_batch", MetadataEntries: entries})
}

// AddVectorWithMetadata adds a vector and associates structured fields with
// it. Metadata is kept separate from the vector index so existing persistence
// formats remain compatible while the metadata index evolves.
func (s *Service) AddVectorWithMetadata(id string, values []float64, metadata map[string]string) error {
	if err := s.AddVector(id, values); err != nil {
		return err
	}
	s.metadataMu.Lock()
	if s.metadata == nil {
		s.metadata = make(map[string]map[string]string)
	}
	s.metadata[id] = cloneMetadata(metadata)
	s.textIndex.addMetadata(id, metadata)
	s.addFullTextMetadataIfReady(id, metadata)
	s.metadataMu.Unlock()
	// Keep metadata replayable alongside the vector WAL. The sidecar remains
	// for fast startup, while this record covers replica/recovery replay.
	if err := s.appendWAL(walOp{Op: "metadata", ID: id, Metadata: cloneMetadata(metadata)}); err != nil {
		return err
	}
	return nil
}

func (s *Service) persistMetadata() error {
	data, err := json.Marshal(s.metadata)
	if err != nil {
		return err
	}
	// A shard can be empty when a topology is closed or retired. In that case
	// no vector/WAL write has created its directory yet, but the empty metadata
	// sidecar must still be persisted atomically.
	if err := os.MkdirAll(filepath.Dir(s.snapshotPath), 0o755); err != nil {
		return err
	}
	tmpPath := s.snapshotPath + ".metadata.tmp"
	f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return os.Rename(tmpPath, s.snapshotPath+".metadata.json")
}

func cloneMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func (s *Service) addFullTextMetadataIfReady(id string, metadata map[string]string) {
	if !s.textIndexReady.Load() {
		return
	}
	values := []string{id}
	for key, value := range metadata {
		values = append(values, key, value)
	}
	s.textIndex.add(id, values...)
}

func (s *Service) metadataFor(id string) map[string]string {
	_ = s.loadMetadataSidecar()
	s.metadataMu.RLock()
	defer s.metadataMu.RUnlock()
	return cloneMetadata(s.metadata[id])
}

func (s *Service) loadMetadataSidecar() error {
	s.metadataMu.Lock()
	if s.metadata != nil || s.snapshotPath == "" {
		s.metadataMu.Unlock()
		return nil
	}
	data, err := os.ReadFile(s.snapshotPath + ".metadata.json")
	if errors.Is(err, os.ErrNotExist) {
		s.metadataMu.Unlock()
		return nil
	}
	if err != nil {
		s.metadataMu.Unlock()
		return err
	}
	metadata := make(map[string]map[string]string)
	if err := json.Unmarshal(data, &metadata); err != nil {
		s.metadataMu.Unlock()
		return fmt.Errorf("decode metadata sidecar: %w", err)
	}
	s.metadata = metadata
	s.metadataMu.Unlock()

	// Exact postings are part of the structured-query recovery state. The
	// expensive n-gram/full-text index remains lazy until a text query.
	for id, values := range metadata {
		s.textIndex.addMetadata(id, values)
	}
	return nil
}

func (s *Service) withMetadataReadOnly(id string, fn func(map[string]string) bool) bool {
	s.metadataMu.RLock()
	defer s.metadataMu.RUnlock()
	return fn(s.metadata[id])
}

func (s *Service) removeMetadata(id string) {
	s.metadataMu.Lock()
	delete(s.metadata, id)
	s.metadataMu.Unlock()
	s.textIndex.remove(id)
}

// VectorMetadata returns a defensive copy of the structured fields associated
// with a vector. It is used by control-plane operations such as resharding.
func (s *Service) VectorMetadata(id string) map[string]string {
	return s.metadataFor(id)
}

func (s *Service) VectorMetadataBatch(ids []string) map[string]map[string]string {
	out := make(map[string]map[string]string)
	if len(ids) > 0 && s.metadata == nil && s.snapshotPath != "" {
		_ = s.metadataFor(ids[0])
	}
	s.metadataMu.RLock()
	for _, id := range ids {
		if metadata, ok := s.metadata[id]; ok {
			out[id] = cloneMetadata(metadata)
		}
	}
	s.metadataMu.RUnlock()
	return out
}
