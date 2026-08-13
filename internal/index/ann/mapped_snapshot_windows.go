//go:build windows

package ann

import (
	"errors"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

type mappedSnapshot struct {
	data    []byte
	address uintptr
	mapping windows.Handle
}

//go:nocheckptr
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
	mapping, err := windows.CreateFileMapping(windows.Handle(file.Fd()), nil, windows.PAGE_READONLY, 0, 0, nil)
	if err != nil {
		return nil, err
	}
	address, err := windows.MapViewOfFile(mapping, windows.FILE_MAP_READ, 0, 0, uintptr(info.Size()))
	if err != nil {
		_ = windows.CloseHandle(mapping)
		return nil, err
	}
	data := unsafe.Slice((*byte)(unsafe.Add(nil, address)), int(info.Size()))
	return &mappedSnapshot{
		data:    data,
		address: address,
		mapping: mapping,
	}, nil
}

func (m *mappedSnapshot) Close() error {
	if m.address == 0 {
		return nil
	}
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
