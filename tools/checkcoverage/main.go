package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

var packages = []string{
	"./api/http",
	"./cmd/server",
	"./internal/api",
	"./internal/cluster",
	"./internal/config",
	"./internal/core",
	"./internal/index",
	"./internal/index/ann",
	"./internal/storage",
	"./internal/util",
	"./internal/vector",
	"./pkg/client",
}

func main() {
	rootDir, err := findModuleRoot()
	if err != nil {
		fmt.Fprintf(os.Stderr, "find module root: %v\n", err)
		os.Exit(1)
	}

	threshold := 90.0
	if raw := strings.TrimSpace(os.Getenv("COVERAGE_THRESHOLD")); raw != "" {
		value, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "invalid COVERAGE_THRESHOLD %q: %v\n", raw, err)
			os.Exit(1)
		}
		threshold = value
	}

	tmpDir, err := os.MkdirTemp("", "lumenvec-coverage-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "mktemp: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	fmt.Printf("Coverage threshold: %.1f%%\n", threshold)

	failed := false
	for _, pkg := range packages {
		coverFile := filepath.Join(tmpDir, strings.NewReplacer("/", "_", ".", "_").Replace(pkg)+".out")
		if err := run(rootDir, "go", "test", "-coverprofile="+coverFile, pkg); err != nil {
			fmt.Fprintf(os.Stderr, "go test failed for %s: %v\n", pkg, err)
			os.Exit(1)
		}

		total, err := coverageTotal(rootDir, coverFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "read coverage failed for %s: %v\n", pkg, err)
			os.Exit(1)
		}

		if total < threshold {
			fmt.Printf("FAIL  %-24s %6.1f%% (threshold %.1f%%)\n", pkg, total, threshold)
			failed = true
		} else {
			fmt.Printf("PASS  %-24s %6.1f%% (threshold %.1f%%)\n", pkg, total, threshold)
		}
	}

	if failed {
		fmt.Fprintln(os.Stderr, "Coverage check failed.")
		os.Exit(1)
	}

	fmt.Println("Coverage check passed.")
}

func run(dir, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout = ioDiscard{}
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func coverageTotal(dir, coverFile string) (float64, error) {
	cmd := exec.Command("go", "tool", "cover", "-func="+coverFile)
	cmd.Dir = dir
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return 0, err
	}

	lines := strings.Split(strings.TrimSpace(stdout.String()), "\n")
	for _, line := range lines {
		if !strings.HasPrefix(line, "total:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			return 0, fmt.Errorf("unexpected total line: %q", line)
		}
		raw := strings.TrimSuffix(fields[len(fields)-1], "%")
		return strconv.ParseFloat(raw, 64)
	}
	return 0, fmt.Errorf("missing total line")
}

func findModuleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found from %s", dir)
		}
		dir = parent
	}
}

type ioDiscard struct{}

func (ioDiscard) Write(p []byte) (int, error) { return len(p), nil }
