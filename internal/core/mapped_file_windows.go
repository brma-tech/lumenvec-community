//go:build windows

package core

import (
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

type mappedFile struct {
	data    []byte
	address uintptr
	mapping windows.Handle
}

//go:nocheckptr
func mapReadWriteFile(file *os.File, size int) (*mappedFile, error) {
	mapping, err := windows.CreateFileMapping(windows.Handle(file.Fd()), nil, windows.PAGE_READWRITE, 0, 0, nil)
	if err != nil {
		return nil, err
	}
	address, err := windows.MapViewOfFile(mapping, windows.FILE_MAP_WRITE|windows.FILE_MAP_READ, 0, 0, uintptr(size))
	if err != nil {
		_ = windows.CloseHandle(mapping)
		return nil, err
	}
	data := unsafe.Slice((*byte)(unsafe.Add(nil, address)), size)
	return &mappedFile{
		data:    data,
		address: address,
		mapping: mapping,
	}, nil
}

func (m *mappedFile) Flush() error {
	return windows.FlushViewOfFile(m.address, uintptr(len(m.data)))
}

func (m *mappedFile) Close() error {
	unmapErr := windows.UnmapViewOfFile(m.address)
	closeErr := windows.CloseHandle(m.mapping)
	m.data = nil
	m.address = 0
	m.mapping = 0
	if unmapErr != nil {
		return unmapErr
	}
	return closeErr
}
