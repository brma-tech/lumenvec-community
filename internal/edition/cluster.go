// Package edition defines neutral extension contracts shared by product
// editions. The Community API may depend on these contracts, but never on a
// proprietary implementation.
package edition

import (
	"errors"
	"time"

	"lumenvec/internal/core"
)

var (
	ErrElectionQuorumUnavailable = errors.New("leader election quorum unavailable")
	ErrElectionUnknownMember     = errors.New("unknown election member")
	ErrLeadershipLost            = errors.New("leadership lease is not held by this node")
	ErrStaleLeaderTerm           = errors.New("leader term is stale")

	ErrConsensusLogConflict = errors.New("consensus log conflict")
	ErrConsensusCommit      = errors.New("consensus commit index is invalid")
	ErrConsensusStaleTerm   = errors.New("consensus term is stale")
)

type ElectionResult struct {
	Term     uint64
	LeaderID string
	Votes    int
	Quorum   int
}

type ElectionCoordinator interface {
	Campaign(candidateID string, lastOffset uint64) (ElectionResult, error)
	EnsureLeader(candidateID string, term uint64) error
	Close() error
}

type ConsensusEntry struct {
	Index       uint64 `json:"index"`
	Term        uint64 `json:"term"`
	OperationID string `json:"operation_id"`
	Payload     []byte `json:"payload,omitempty"`
}

type ConsensusLog interface {
	AppendEntries(
		term uint64,
		leaderID string,
		prevIndex uint64,
		prevTerm uint64,
		entries []ConsensusEntry,
		leaderCommit uint64,
	) (uint64, error)
	Close() error
}

type Closer interface {
	Close() error
}

type ConsensusLogProvider interface {
	Get(shardID uint32) (ConsensusLog, error)
	Close() error
}

type DistributedRuntimeOptions struct {
	TopologyPath      string
	FanoutConcurrency int

	ReplicationMode              string
	ReplicationQueueSize         int
	ReplicationWriteQuorum       int
	ReplicationLogPath           string
	ReplicationFencePath         string
	ReplicationHeartbeatInterval time.Duration
	ReplicationElectionTimeout   time.Duration

	NodeID   string
	Members  []string
	StateDir string
	WALPath  string

	TLSEnabled  bool
	TLSCertFile string
}

type DistributedRuntime struct {
	Service           core.VectorService
	Election          ElectionCoordinator
	ElectionTransport Closer
	ConsensusLog      ConsensusLog
	ConsensusLogs     ConsensusLogProvider
}

type DistributedRuntimeFactory func(DistributedRuntimeOptions) (DistributedRuntime, error)
