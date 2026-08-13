package storage

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// HTTPBlobStore targets an object-storage gateway exposing key-addressable
// HTTP resources. Authentication, signing and TLS policy are supplied by the
// injected client/transport.
type HTTPRequestHook func(*http.Request) error

type HTTPBlobStore struct {
	base   *url.URL
	client *http.Client
	hook   HTTPRequestHook
}

func NewHTTPBlobStore(baseURL string, client *http.Client) (*HTTPBlobStore, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, fmt.Errorf("valid blob store URL is required")
	}
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	return &HTTPBlobStore{base: parsed, client: client}, nil
}

func NewHTTPBlobStoreWithHook(baseURL string, client *http.Client, hook HTTPRequestHook) (*HTTPBlobStore, error) {
	store, err := NewHTTPBlobStore(baseURL, client)
	if err != nil {
		return nil, err
	}
	store.hook = hook
	return store, nil
}

func (s *HTTPBlobStore) endpoint(key string) (*url.URL, error) {
	if s == nil || s.base == nil {
		return nil, fmt.Errorf("blob store is nil")
	}
	clean := strings.TrimPrefix(key, "/")
	if clean == "" || strings.Contains(clean, "..") {
		return nil, fmt.Errorf("invalid blob key")
	}
	u := *s.base
	u.Path = strings.TrimRight(s.base.Path, "/") + "/" + strings.TrimLeft(clean, "/")
	return &u, nil
}
func (s *HTTPBlobStore) Put(key string, body io.Reader) error {
	u, err := s.endpoint(key)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPut, u.String(), body)
	if err != nil {
		return err
	}
	if s.hook != nil {
		if err := s.hook(req); err != nil {
			return err
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("blob PUT returned %s", resp.Status)
	}
	return nil
}
func (s *HTTPBlobStore) Get(key string) (io.ReadCloser, error) {
	u, err := s.endpoint(key)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	if s.hook != nil {
		if err := s.hook(req); err != nil {
			return nil, err
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("blob GET returned %s", resp.Status)
	}
	return resp.Body, nil
}
func (s *HTTPBlobStore) Delete(key string) error {
	u, err := s.endpoint(key)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodDelete, u.String(), nil)
	if err != nil {
		return err
	}
	if s.hook != nil {
		if err := s.hook(req); err != nil {
			return err
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return fmt.Errorf("blob DELETE returned %s", resp.Status)
	}
	return nil
}
