package core

import "github.com/google/btree"

const orderedIDTreeDegree = 32

type orderedIDIndex struct {
	tree *btree.BTreeG[string]
}

func newOrderedIDIndex() *orderedIDIndex {
	return &orderedIDIndex{
		tree: btree.NewG(orderedIDTreeDegree, func(a, b string) bool { return a < b }),
	}
}

func newOrderedIDIndexFromMap[T any](items map[string]T) *orderedIDIndex {
	idx := newOrderedIDIndex()
	for id := range items {
		idx.Upsert(id)
	}
	return idx
}

func (i *orderedIDIndex) Upsert(id string) {
	i.ensure()
	i.tree.ReplaceOrInsert(id)
}

func (i *orderedIDIndex) Delete(id string) {
	i.ensure()
	i.tree.Delete(id)
}

func (i *orderedIDIndex) Range(fn func(string) bool) {
	i.ensure()
	i.tree.Ascend(fn)
}

func (i *orderedIDIndex) PageAfter(afterID string, limit int) []string {
	if limit <= 0 {
		return nil
	}
	i.ensure()

	out := make([]string, 0, limit)
	visit := func(id string) bool {
		out = append(out, id)
		return len(out) < limit
	}
	if afterID == "" {
		i.tree.Ascend(visit)
		return out
	}
	i.tree.AscendGreaterOrEqual(afterID, func(id string) bool {
		if id <= afterID {
			return true
		}
		return visit(id)
	})
	return out
}

func (i *orderedIDIndex) ensure() {
	if i.tree == nil {
		i.tree = btree.NewG(orderedIDTreeDegree, func(a, b string) bool { return a < b })
	}
}
