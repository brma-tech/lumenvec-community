package storage

import (
	"context"
	"errors"
	"testing"
	"time"
)

type recordingTierCatalog struct{ objects []TierObject }

func (c recordingTierCatalog) ListTierObjects(context.Context) ([]TierObject, error) {
	return c.objects, nil
}

func TestTierPolicyClassifiesAccessAge(t *testing.T) {
	now := time.Date(2026, 7, 12, 0, 0, 0, 0, time.UTC)
	policy := TierPolicy{HotAfter: time.Hour, WarmAfter: 24 * time.Hour}
	for _, tc := range []struct {
		name string
		age  time.Duration
		want StorageTier
	}{
		{"future", -time.Minute, TierHot},
		{"hot", 30 * time.Minute, TierHot},
		{"warm", 2 * time.Hour, TierWarm},
		{"cold", 48 * time.Hour, TierCold},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := policy.Classify(now, now.Add(-tc.age)); got != tc.want {
				t.Fatalf("tier=%s want %s", got, tc.want)
			}
		})
	}
}

func TestDefaultTierPolicy(t *testing.T) {
	p := DefaultTierPolicy()
	if p.HotAfter <= 0 || p.WarmAfter <= p.HotAfter {
		t.Fatalf("invalid default policy: %+v", p)
	}
}

func TestTierPolicyPlansOnlyRequiredMoves(t *testing.T) {
	now := time.Date(2026, 7, 12, 0, 0, 0, 0, time.UTC)
	p := TierPolicy{HotAfter: time.Hour, WarmAfter: 24 * time.Hour}
	moves := p.PlanMoves(now, []TierObject{
		{Key: "hot", LastAccess: now.Add(-30 * time.Minute), Tier: TierHot},
		{Key: "warm", LastAccess: now.Add(-2 * time.Hour), Tier: TierHot},
		{Key: "cold", LastAccess: now.Add(-48 * time.Hour), Tier: TierWarm},
	})
	if len(moves) != 2 || moves[0] != (TierMove{Key: "warm", From: TierHot, To: TierWarm}) || moves[1] != (TierMove{Key: "cold", From: TierWarm, To: TierCold}) {
		t.Fatalf("moves=%+v", moves)
	}
}

type recordingTierMover struct {
	moves  []TierMove
	failAt int
}

func (m *recordingTierMover) Move(move TierMove) error {
	if m.failAt >= 0 && len(m.moves) == m.failAt {
		return errors.New("move failed")
	}
	m.moves = append(m.moves, move)
	return nil
}

func TestExecuteMovesStopsAtFirstError(t *testing.T) {
	mover := &recordingTierMover{failAt: 1}
	moves := []TierMove{{Key: "a", From: TierHot, To: TierWarm}, {Key: "b", From: TierWarm, To: TierCold}}
	if err := ExecuteMoves(mover, moves); err == nil {
		t.Fatal("expected move error")
	}
	if len(mover.moves) != 1 || mover.moves[0].Key != "a" {
		t.Fatalf("moves=%+v", mover.moves)
	}
}

func TestExecuteMovesRejectsNilMover(t *testing.T) {
	if err := ExecuteMoves(nil, nil); err == nil {
		t.Fatal("expected nil mover error")
	}
}

func TestTierLifecycleRunOncePlansAndExecutes(t *testing.T) {
	now := time.Date(2026, 7, 12, 0, 0, 0, 0, time.UTC)
	mover := &recordingTierMover{failAt: -1}
	lifecycle := TierLifecycle{
		Policy:  TierPolicy{HotAfter: time.Hour, WarmAfter: 24 * time.Hour},
		Catalog: recordingTierCatalog{objects: []TierObject{{Key: "old", LastAccess: now.Add(-2 * time.Hour), Tier: TierHot}}},
		Mover:   mover,
		Now:     func() time.Time { return now },
	}
	if err := lifecycle.RunOnce(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(mover.moves) != 1 || mover.moves[0].To != TierWarm {
		t.Fatalf("moves=%+v", mover.moves)
	}
}

func TestTierLifecycleRejectsInvalidRun(t *testing.T) {
	lifecycle := TierLifecycle{}
	if err := lifecycle.Run(context.Background(), 0); err == nil {
		t.Fatal("expected interval error")
	}
}
