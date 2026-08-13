package util

import (
	"bytes"
	"strings"
	"testing"
)

func TestInfoAndError(t *testing.T) {
	var infoBuf, errBuf bytes.Buffer
	infoLogger.SetOutput(&infoBuf)
	errorLogger.SetOutput(&errBuf)
	t.Cleanup(func() {
		infoLogger.SetOutput(nil)
		errorLogger.SetOutput(nil)
	})

	Info("hello")
	Error("boom")

	if !strings.Contains(infoBuf.String(), "hello") {
		t.Fatal("expected info output")
	}
	if !strings.Contains(errBuf.String(), "boom") {
		t.Fatal("expected error output")
	}
}
