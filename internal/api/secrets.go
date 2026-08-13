package api

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// SecretVersion is the provider-neutral representation returned by a secret
// manager. Providers can map their native version/ETag into Version.
type SecretVersion struct {
	Value   string
	Version string
}

// SecretProvider supplies credentials without coupling the server to a cloud
// vendor. Implementations must return an immutable snapshot for each read.
type SecretProvider interface {
	Read(ctx context.Context, name string) (SecretVersion, error)
	Rotate(ctx context.Context, name, value string) (SecretVersion, error)
}

// HTTPSecretProvider adapts a Secret Manager HTTP gateway. The gateway must
// return {"value":"...","version":"..."}; writes use the same contract
// with a JSON {"value":"..."} body. Tokens are sent only in Authorization.
type HTTPSecretProvider struct {
	BaseURL string
	Token   string
	Client  *http.Client
}

func (p HTTPSecretProvider) Read(ctx context.Context, name string) (SecretVersion, error) {
	return p.request(ctx, http.MethodGet, name, nil)
}

func (p HTTPSecretProvider) Rotate(ctx context.Context, name, value string) (SecretVersion, error) {
	if strings.TrimSpace(value) == "" {
		return SecretVersion{}, errors.New("secret value is required")
	}
	body, err := json.Marshal(map[string]string{"value": value})
	if err != nil {
		return SecretVersion{}, err
	}
	return p.request(ctx, http.MethodPut, name, strings.NewReader(string(body)))
}

func (p HTTPSecretProvider) request(ctx context.Context, method, name string, body io.Reader) (SecretVersion, error) {
	base, err := url.Parse(strings.TrimRight(p.BaseURL, "/"))
	if err != nil || base.Scheme == "" || base.Host == "" {
		return SecretVersion{}, errors.New("invalid secret provider URL")
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/" + url.PathEscape(name)
	req, err := http.NewRequestWithContext(ctx, method, base.String(), body)
	if err != nil {
		return SecretVersion{}, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if p.Token != "" {
		req.Header.Set("Authorization", "Bearer "+p.Token)
	}
	client := p.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return SecretVersion{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return SecretVersion{}, fmt.Errorf("secret provider returned %s", resp.Status)
	}
	var out SecretVersion
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&out); err != nil {
		return SecretVersion{}, err
	}
	if strings.TrimSpace(out.Value) == "" {
		return SecretVersion{}, errors.New("secret provider returned empty value")
	}
	if out.Version == "" {
		out.Version = fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(out.Value)))
	}
	return out, nil
}

// FileSecretProvider is a reference provider for Kubernetes Secret volumes
// and local deployments. Rotation uses a sibling file plus rename.
type FileSecretProvider struct{ Path string }

func (p FileSecretProvider) Read(_ context.Context, _ string) (SecretVersion, error) {
	data, err := os.ReadFile(p.Path)
	if err != nil {
		return SecretVersion{}, err
	}
	return fileSecretVersion(data), nil
}

func (p FileSecretProvider) Rotate(_ context.Context, _ string, value string) (SecretVersion, error) {
	if strings.TrimSpace(value) == "" {
		return SecretVersion{}, errors.New("secret value is required")
	}
	if err := os.MkdirAll(filepath.Dir(p.Path), 0o750); err != nil {
		return SecretVersion{}, err
	}
	tmp, err := os.CreateTemp(filepath.Dir(p.Path), ".secret-*")
	if err != nil {
		return SecretVersion{}, err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.WriteString(value + "\n"); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return SecretVersion{}, err
	}
	if err := renameSecretFile(tmpName, p.Path); err != nil {
		return SecretVersion{}, err
	}
	return fileSecretVersion([]byte(value)), nil
}

func renameSecretFile(from, to string) error {
	if runtime.GOOS != "windows" {
		return os.Rename(from, to)
	}
	var err error
	for attempt := 0; attempt < 5; attempt++ {
		if err = os.Rename(from, to); err == nil {
			return nil
		}
		time.Sleep(time.Duration(attempt+1) * 10 * time.Millisecond)
	}
	return err
}

func fileSecretVersion(data []byte) SecretVersion {
	value := strings.TrimSpace(string(data))
	return SecretVersion{Value: value, Version: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(value)))}
}

// SecretCache keeps the last successfully loaded value and makes rotation
// atomic for concurrent HTTP and gRPC authentication.
type SecretCache struct {
	mu      sync.RWMutex
	value   string
	version string
}

func (c *SecretCache) Load(v SecretVersion) error {
	if strings.TrimSpace(v.Value) == "" {
		return errors.New("secret value is required")
	}
	c.mu.Lock()
	c.value, c.version = v.Value, v.Version
	c.mu.Unlock()
	return nil
}

func (c *SecretCache) Current() SecretVersion {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return SecretVersion{Value: c.value, Version: c.version}
}

func (c *SecretCache) Clear() {
	c.mu.Lock()
	c.value, c.version = "", ""
	c.mu.Unlock()
}
