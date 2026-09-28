package config

import (
	"path/filepath"
	"testing"
)

func TestLoadAcceptsHierarchicalHNSWBackend(t *testing.T) {
	t.Setenv("VECTOR_DB_ANN_BACKEND", "hierarchical-hnsw")
	cfg, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Search.ANNBackend != "hierarchical-hnsw" {
		t.Fatalf("ANNBackend = %q, want hierarchical-hnsw", cfg.Search.ANNBackend)
	}
}

func TestLoadDefaultsToHierarchicalHNSWBackend(t *testing.T) {
	t.Setenv("VECTOR_DB_ANN_BACKEND", "")
	cfg, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if cfg.Search.ANNBackend != "hierarchical-hnsw" {
		t.Fatalf("ANNBackend = %q, want hierarchical-hnsw", cfg.Search.ANNBackend)
	}
}
