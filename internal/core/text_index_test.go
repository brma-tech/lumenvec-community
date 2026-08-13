package core

import "testing"

func TestFullTextIndexSubstringAndReplacement(t *testing.T) {
	var idx fullTextIndex
	idx.add("article-1", "article-1", "title", "Distributed systems")
	idx.add("article-2", "article-2", "title", "Vector search")
	if got := idx.search("tribut"); len(got) != 1 {
		t.Fatalf("substring search returned %v, want one document", got)
	}
	idx.add("article-1", "article-1", "title", "Storage")
	if got := idx.search("distributed"); len(got) != 0 {
		t.Fatalf("replacement retained stale terms: %v", got)
	}
	idx.remove("article-2")
	if got := idx.search("vector"); len(got) != 0 {
		t.Fatalf("remove retained document: %v", got)
	}
}
