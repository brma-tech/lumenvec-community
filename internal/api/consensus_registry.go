package api

import (
	"fmt"
	"path/filepath"
	"sync"

	"lumenvec/internal/edition"
)

type consensusLogRegistry struct {
	mu     sync.Mutex
	dir    string
	opener func(string) (edition.ConsensusLog, error)
	logs   map[uint32]edition.ConsensusLog
}

func newConsensusLogRegistry(dir string, opener func(string) (edition.ConsensusLog, error)) *consensusLogRegistry {
	return &consensusLogRegistry{dir: dir, opener: opener, logs: make(map[uint32]edition.ConsensusLog)}
}

func (r *consensusLogRegistry) Get(shardID uint32) (edition.ConsensusLog, error) {
	if r == nil {
		return nil, fmt.Errorf("consensus log registry is disabled")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if log := r.logs[shardID]; log != nil {
		return log, nil
	}
	if r.opener == nil {
		return nil, fmt.Errorf("consensus log opener is disabled")
	}
	log, err := r.opener(filepath.Join(r.dir, fmt.Sprintf("consensus-shard-%04d.log", shardID)))
	if err != nil {
		return nil, err
	}
	r.logs[shardID] = log
	return log, nil
}

func (r *consensusLogRegistry) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var errs []error
	for _, log := range r.logs {
		errs = append(errs, log.Close())
	}
	r.logs = nil
	return errorsJoin(errs)
}

func errorsJoin(errs []error) error {
	var out error
	for _, err := range errs {
		if err != nil {
			if out == nil {
				out = err
			} else {
				out = fmt.Errorf("%v; %w", out, err)
			}
		}
	}
	return out
}
