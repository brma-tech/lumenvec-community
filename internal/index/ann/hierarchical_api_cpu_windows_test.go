//go:build annapiaudit && windows

package ann

import "golang.org/x/sys/windows"

func auditProcessCPU() float64 {
	var creation, exit, kernel, user windows.Filetime
	if windows.GetProcessTimes(windows.CurrentProcess(), &creation, &exit, &kernel, &user) != nil {
		return -1
	}
	ticks := func(f windows.Filetime) uint64 { return uint64(f.HighDateTime)<<32 | uint64(f.LowDateTime) }
	return float64(ticks(kernel)+ticks(user)) / 1e7
}
