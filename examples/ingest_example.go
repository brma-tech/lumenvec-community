package main

import (
	"fmt"
	"io"
	"os"

	"lumenvec/pkg/client"
)

func main() {
	mustRun(run, "http://localhost:19190", os.Stdout)
}

func run(baseURL string, out io.Writer) error {
	c := client.NewVectorClient(baseURL)

	vectors := map[string][]float64{
		"doc-1": {1.0, 2.0, 3.0},
		"doc-2": {1.1, 2.1, 2.9},
		"doc-3": {9.0, 8.5, 7.5},
	}

	for id, v := range vectors {
		if err := c.AddVectorWithID(id, v); err != nil {
			return fmt.Errorf("failed to ingest vector %s: %w", id, err)
		}
	}

	results, err := c.SearchVector([]float64{1.0, 2.0, 3.1}, 2)
	if err != nil {
		return fmt.Errorf("failed to search vectors: %w", err)
	}

	if _, err := fmt.Fprintln(out, "Top 2 nearest vectors:"); err != nil {
		return err
	}
	for _, r := range results {
		if _, err := fmt.Fprintf(out, "- %s (distance=%.4f)\n", r.ID, r.Distance); err != nil {
			return err
		}
	}
	return nil
}

func mustRun(runFn func(string, io.Writer) error, baseURL string, out io.Writer) {
	if err := runFn(baseURL, out); err != nil {
		panic(err)
	}
}
