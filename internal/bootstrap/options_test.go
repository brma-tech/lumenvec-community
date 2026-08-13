package bootstrap

import (
	"errors"
	"testing"

	"lumenvec/internal/config"
)

func TestServerOptionsDoesNotInjectEditionAdapters(t *testing.T) {
	options := ServerOptions(config.Config{})
	if options.DistributedRuntimeFactory != nil {
		t.Fatal("Community bootstrap injected distributed runtime")
	}
	if options.AdminConfig != nil || options.AdminCluster != nil {
		t.Fatal("Community bootstrap injected Business administration")
	}
}

func TestCommunityServerOptionsRejectsDistributedConfiguration(t *testing.T) {
	cfg := config.Config{}
	cfg.Cluster.NodeID = "node-a"
	_, err := CommunityServerOptions(cfg)
	if !errors.Is(err, ErrBusinessFeaturesRequired) {
		t.Fatalf("expected Business feature error, got %v", err)
	}
}
