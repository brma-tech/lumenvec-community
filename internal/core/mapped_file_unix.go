//go:build !windows

package core

import (
	"os"

	"golang.org/x/sys/unix"
)

type mappedFile struct {
	data []byte
}

func mapReadWriteFile(file *os.File, size int) (*mappedFile, error) {
	data, err := unix.Mmap(int(file.Fd()), 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		return nil, err
	}
	return &mappedFile{data: data}, nil
}

func (m *mappedFile) Flush() error {
	return unix.Msync(m.data, unix.MS_SYNC)
}

func (m *mappedFile) Close() error {
	err := unix.Munmap(m.data)
	m.data = nil
	return err
}
