//go:build !windows

package ann

import (
	"errors"
	"os"

	"golang.org/x/sys/unix"
)

type mappedSnapshot struct {
	data []byte
}

func mapSnapshotFile(path string) (*mappedSnapshot, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if info.Size() <= 0 || uint64(info.Size()) > uint64(^uint(0)>>1) {
		return nil, errors.New("invalid ANN snapshot file size")
	}
	data, err := unix.Mmap(int(file.Fd()), 0, int(info.Size()), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	return &mappedSnapshot{data: data}, nil
}

func (m *mappedSnapshot) Close() error {
	if len(m.data) == 0 {
		return nil
	}
	err := unix.Munmap(m.data)
	m.data = nil
	return err
}
