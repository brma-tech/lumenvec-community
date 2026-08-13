package storage

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
)

type LevelDBStore struct {
	mu    sync.RWMutex
	store map[string][]byte
}

func NewLevelDBStore(path string) (*LevelDBStore, error) {
	// In-memory store for portability in examples/tests.
	return &LevelDBStore{store: make(map[string][]byte)}, nil
}

func (s *LevelDBStore) Put(key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}
	s.mu.Lock()
	s.store[key] = data
	s.mu.Unlock()
	return nil
}

func (s *LevelDBStore) Get(key string, value interface{}) error {
	s.mu.RLock()
	data, ok := s.store[key]
	s.mu.RUnlock()
	if !ok {
		return nil
	}
	return json.Unmarshal(data, value)
}

func (s *LevelDBStore) Delete(key string) error {
	s.mu.Lock()
	delete(s.store, key)
	s.mu.Unlock()
	return nil
}

func (s *LevelDBStore) Close() error {
	// nothing to close for in-memory store
	return nil
}

func (s *LevelDBStore) Iterate(prefix string, handler func(key string, value interface{}) bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for k, v := range s.store {
		if len(k) >= len(prefix) && k[:len(prefix)] == prefix {
			var value interface{}
			if err := json.Unmarshal(v, &value); err != nil {
				log.Printf("failed to unmarshal value for key %s: %v", k, err)
				continue
			}
			if !handler(k, value) {
				break
			}
		}
	}
	return nil
}
