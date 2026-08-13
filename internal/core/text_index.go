package core

import (
	"regexp"
	"strings"
	"sync"
)

// fullTextIndex is an in-memory inverted index for metadata and IDs. It is
// deliberately additive: the existing predicate remains the source of truth,
// while this index narrows large collections before vector distance evaluation.
type fullTextIndex struct {
	mu    sync.RWMutex
	terms map[string]map[string]struct{}
	docs  map[string]map[string]struct{}
	// exact stores postings for structured key/value predicates. Keeping this
	// separate from token n-grams avoids false candidate intersections (for
	// example, status=ready must not match status=not-ready).
	exact     map[string]map[string]struct{}
	exactDocs map[string]map[string]struct{}
}

var textToken = regexp.MustCompile(`[\pL\pN]+`)

func tokenizeText(value string) []string {
	parts := textToken.FindAllString(strings.ToLower(value), -1)
	seen := make(map[string]struct{}, len(parts))
	for _, part := range parts {
		seen[part] = struct{}{}
		runes := []rune(part)
		// Store bounded character n-grams so the legacy substring semantics
		// remain exact while queries avoid scanning unrelated documents.
		for size := 2; size <= len(runes) && size <= 32; size++ {
			for i := 0; i+size <= len(runes); i++ {
				seen[string(runes[i:i+size])] = struct{}{}
			}
		}
	}
	out := make([]string, 0, len(seen))
	for token := range seen {
		out = append(out, token)
	}
	return out
}

func (x *fullTextIndex) add(id string, values ...string) {
	if x == nil || id == "" {
		return
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.terms == nil {
		x.terms = make(map[string]map[string]struct{})
	}
	if x.docs == nil {
		x.docs = make(map[string]map[string]struct{})
	}
	if old := x.docs[id]; old != nil {
		for token := range old {
			delete(x.terms[token], id)
		}
	}
	doc := make(map[string]struct{})
	for _, value := range values {
		for _, token := range tokenizeText(value) {
			doc[token] = struct{}{}
		}
	}
	x.docs[id] = doc
	for token := range doc {
		if x.terms[token] == nil {
			x.terms[token] = make(map[string]struct{})
		}
		x.terms[token][id] = struct{}{}
	}
}

func (x *fullTextIndex) addMetadata(id string, metadata map[string]string) {
	if x == nil || id == "" {
		return
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	if x.exact == nil {
		x.exact = make(map[string]map[string]struct{})
	}
	if x.exactDocs == nil {
		x.exactDocs = make(map[string]map[string]struct{})
	}
	for term := range x.exactDocs[id] {
		delete(x.exact[term], id)
	}
	doc := make(map[string]struct{}, len(metadata))
	for key, value := range metadata {
		term := key + "\x00" + value
		doc[term] = struct{}{}
		if x.exact[term] == nil {
			x.exact[term] = make(map[string]struct{})
		}
		x.exact[term][id] = struct{}{}
	}
	x.exactDocs[id] = doc
}

func (x *fullTextIndex) searchMetadata(key, value string) map[string]struct{} {
	if x == nil {
		return nil
	}
	x.mu.RLock()
	defer x.mu.RUnlock()
	postings := x.exact[key+"\x00"+value]
	if postings == nil {
		return map[string]struct{}{}
	}
	result := make(map[string]struct{}, len(postings))
	for id := range postings {
		result[id] = struct{}{}
	}
	return result
}

// searchMetadataIDs returns a compact snapshot of an exact posting. A slice
// avoids the per-entry bucket overhead of cloning a large map for the common
// single-predicate query plan while still releasing the index lock before I/O.
func (x *fullTextIndex) searchMetadataIDs(key, value string) []string {
	if x == nil {
		return nil
	}
	x.mu.RLock()
	defer x.mu.RUnlock()
	postings := x.exact[key+"\x00"+value]
	ids := make([]string, 0, len(postings))
	for id := range postings {
		ids = append(ids, id)
	}
	return ids
}

func (x *fullTextIndex) remove(id string) {
	if x == nil {
		return
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	for token := range x.docs[id] {
		delete(x.terms[token], id)
	}
	delete(x.docs, id)
	for term := range x.exactDocs[id] {
		delete(x.exact[term], id)
	}
	delete(x.exactDocs, id)
}

func (x *fullTextIndex) search(query string) map[string]struct{} {
	if x == nil {
		return nil
	}
	tokens := tokenizeText(query)
	if len(tokens) == 0 {
		return nil
	}
	x.mu.RLock()
	defer x.mu.RUnlock()
	var result map[string]struct{}
	for _, token := range tokens {
		postings := x.terms[token]
		if len(postings) == 0 {
			return map[string]struct{}{}
		}
		if result == nil {
			result = make(map[string]struct{}, len(postings))
			for id := range postings {
				result[id] = struct{}{}
			}
		} else {
			for id := range result {
				if _, ok := postings[id]; !ok {
					delete(result, id)
				}
			}
		}
	}
	return result
}
