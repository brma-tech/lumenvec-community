package core

import (
	"os"
	"strconv"
	"strings"
)

type StorageSecurityOptions struct {
	StrictFilePermissions bool
	DirMode               os.FileMode
	FileMode              os.FileMode
}

func DefaultStorageSecurityOptions() StorageSecurityOptions {
	return StorageSecurityOptions{
		StrictFilePermissions: false,
		DirMode:               0o755,
		FileMode:              0o644,
	}
}

func StrictStorageSecurityOptions() StorageSecurityOptions {
	return StorageSecurityOptions{
		StrictFilePermissions: true,
		DirMode:               0o700,
		FileMode:              0o600,
	}
}

func ParseFileMode(raw string, fallback os.FileMode) os.FileMode {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback
	}
	value, err := strconv.ParseUint(raw, 8, 32)
	if err != nil {
		return fallback
	}
	return os.FileMode(value)
}

func normalizeStorageSecurityOptions(opts StorageSecurityOptions) StorageSecurityOptions {
	if opts.DirMode == 0 {
		opts.DirMode = 0o755
	}
	if opts.FileMode == 0 {
		opts.FileMode = 0o644
	}
	return opts
}
