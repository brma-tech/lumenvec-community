package storage

import (
	"context"
	"fmt"
	"time"
)

type TierMover interface {
	Move(TierMove) error
}

// TierCatalog supplies the current tier metadata to the lifecycle worker.
// Implementations may read this from a manifest, a metastore, or an object
// store index; the worker never performs data-plane reads.
type TierCatalog interface {
	ListTierObjects(context.Context) ([]TierObject, error)
}

// TierLifecycle periodically reconciles object placement with a policy.
// RunOnce is intentionally idempotent: a failed batch can be retried safely
// by the caller after the catalog records successful moves.
type TierLifecycle struct {
	Policy  TierPolicy
	Catalog TierCatalog
	Mover   TierMover
	Now     func() time.Time
}

func (l TierLifecycle) RunOnce(ctx context.Context) error {
	if ctx == nil {
		return fmt.Errorf("context is required")
	}
	if l.Catalog == nil {
		return fmt.Errorf("tier catalog is required")
	}
	if l.Mover == nil {
		return fmt.Errorf("tier mover is required")
	}
	now := time.Now
	if l.Now != nil {
		now = l.Now
	}
	objects, err := l.Catalog.ListTierObjects(ctx)
	if err != nil {
		return err
	}
	return ExecuteMoves(l.Mover, l.Policy.PlanMoves(now(), objects))
}

// Run reconciles until cancellation. A non-positive interval is rejected to
// avoid a hot loop that could starve ingestion and queries.
func (l TierLifecycle) Run(ctx context.Context, interval time.Duration) error {
	if ctx == nil {
		return fmt.Errorf("context is required")
	}
	if interval <= 0 {
		return fmt.Errorf("lifecycle interval must be positive")
	}
	if err := l.RunOnce(ctx); err != nil {
		return err
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := l.RunOnce(ctx); err != nil {
				return err
			}
		}
	}
}

// StorageTier describes the intended durability/latency class of an object.
type StorageTier string

const (
	TierHot  StorageTier = "hot"
	TierWarm StorageTier = "warm"
	TierCold StorageTier = "cold"
)

// TierPolicy classifies data by last access time. The policy is deliberately
// provider-neutral; a lifecycle worker can use the result to move a segment
// between local disk, warm volumes, and object storage.
type TierPolicy struct {
	HotAfter  time.Duration
	WarmAfter time.Duration
}

type TierObject struct {
	Key        string
	LastAccess time.Time
	Tier       StorageTier
}

type TierMove struct {
	Key  string
	From StorageTier
	To   StorageTier
}

func DefaultTierPolicy() TierPolicy {
	return TierPolicy{HotAfter: 24 * time.Hour, WarmAfter: 7 * 24 * time.Hour}
}

func (p TierPolicy) Classify(now, lastAccess time.Time) StorageTier {
	if now.Before(lastAccess) {
		return TierHot
	}
	age := now.Sub(lastAccess)
	if age <= p.HotAfter {
		return TierHot
	}
	if age <= p.WarmAfter {
		return TierWarm
	}
	return TierCold
}

// PlanMoves returns only transitions required by the current policy. It is
// side-effect free so callers can persist/execute moves transactionally.
func (p TierPolicy) PlanMoves(now time.Time, objects []TierObject) []TierMove {
	moves := make([]TierMove, 0)
	for _, object := range objects {
		want := p.Classify(now, object.LastAccess)
		if object.Tier != want {
			moves = append(moves, TierMove{Key: object.Key, From: object.Tier, To: want})
		}
	}
	return moves
}

// ExecuteMoves applies a previously planned batch in order. It stops on the
// first error so callers can persist a checkpoint and retry remaining moves.
func ExecuteMoves(mover TierMover, moves []TierMove) error {
	if mover == nil {
		return fmt.Errorf("tier mover is required")
	}
	for _, move := range moves {
		if err := mover.Move(move); err != nil {
			return err
		}
	}
	return nil
}
