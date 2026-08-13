package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	defaultHTTPClientTimeout         = 10 * time.Second
	defaultHTTPMaxIdleConns          = 256
	defaultHTTPMaxIdleConnsPerHost   = 256
	defaultHTTPIdleConnTimeout       = 90 * time.Second
	defaultHTTPTLSHandshakeTimeout   = 10 * time.Second
	defaultHTTPExpectContinueTimeout = 1 * time.Second
)

var requestBodyBufferPool = sync.Pool{
	New: func() any {
		buf := bytes.NewBuffer(make([]byte, 0, 4096))
		return buf
	},
}

type VectorClient struct {
	baseURL    string
	httpClient *http.Client
}

type SearchResult struct {
	ID       string  `json:"id"`
	Distance float64 `json:"distance"`
}

type VectorPayload struct {
	ID        string            `json:"id"`
	Values    []float64         `json:"values"`
	Values32  []float32         `json:"values_f32,omitempty"`
	Namespace string            `json:"namespace,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type BatchSearchQuery struct {
	ID       string    `json:"id"`
	Values   []float64 `json:"values,omitempty"`
	Values32 []float32 `json:"values_f32,omitempty"`
	K        int       `json:"k"`
}

type BatchSearchResult struct {
	ID      string         `json:"id"`
	Results []SearchResult `json:"results"`
}

type addVectorsRequest struct {
	Vectors []VectorPayload `json:"vectors"`
}

type searchVectorRequest struct {
	Values    []float64 `json:"values"`
	K         int       `json:"k"`
	Namespace string    `json:"namespace,omitempty"`
}

type searchVectorsRequest struct {
	Queries []BatchSearchQuery `json:"queries"`
}

type ListVectorsOptions struct {
	Limit   int
	Cursor  string
	IDsOnly bool
}

type ListVectorsPage struct {
	Vectors    []VectorPayload
	NextCursor string
}

func NewVectorClient(baseURL string) *VectorClient {
	return NewVectorClientWithTimeout(baseURL, defaultHTTPClientTimeout)
}

func NewVectorClientWithTimeout(baseURL string, timeout time.Duration) *VectorClient {
	if timeout <= 0 {
		timeout = defaultHTTPClientTimeout
	}
	return &VectorClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				Proxy:                 http.ProxyFromEnvironment,
				MaxIdleConns:          defaultHTTPMaxIdleConns,
				MaxIdleConnsPerHost:   defaultHTTPMaxIdleConnsPerHost,
				IdleConnTimeout:       defaultHTTPIdleConnTimeout,
				TLSHandshakeTimeout:   defaultHTTPTLSHandshakeTimeout,
				ExpectContinueTimeout: defaultHTTPExpectContinueTimeout,
				DisableCompression:    true,
			},
		},
	}
}

func (vc *VectorClient) AddVector(vector []float64) error {
	return vc.AddVectorWithID(fmt.Sprintf("vec-%d", time.Now().UnixNano()), vector)
}

func (vc *VectorClient) AddVectorWithID(id string, vector []float64) error {
	return vc.AddVectors([]VectorPayload{{ID: id, Values: vector}})
}

// AddVectorWithNamespace preserves the original AddVector API while exposing
// explicit tenant isolation for new callers.
func (vc *VectorClient) AddVectorWithNamespace(id string, vector []float64, namespace string) error {
	return vc.AddVectors([]VectorPayload{{ID: id, Values: vector, Namespace: namespace}})
}

func (vc *VectorClient) AddVectors(vectors []VectorPayload) error {
	resp, bodyBuf, err := vc.postJSON("/vectors/batch", addVectorsRequest{Vectors: vectors})
	if err != nil {
		return err
	}
	defer putRequestBodyBuffer(bodyBuf)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("failed to add vectors: %s", resp.Status)
	}
	return nil
}

// AddVectorsStream uploads batches on demand. Returning a nil batch signals
// end of input; each non-empty batch is committed independently.
func (vc *VectorClient) AddVectorsStream(next func() ([]VectorPayload, error)) error {
	if next == nil {
		return fmt.Errorf("vector batch source is nil")
	}
	for {
		batch, err := next()
		if err != nil {
			return err
		}
		if batch == nil {
			return nil
		}
		if len(batch) == 0 {
			continue
		}
		if err := vc.AddVectors(batch); err != nil {
			return err
		}
	}
}

// StreamVectors uploads newline-delimited vector JSON to the versioned
// streaming endpoint. The reader is not buffered in memory by the client.
func (vc *VectorClient) StreamVectors(body io.Reader) error {
	req, err := http.NewRequest(http.MethodPost, vc.baseURL+"/vectors/stream", body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-ndjson")
	resp, err := vc.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("failed to stream vectors: %s", resp.Status)
	}
	return nil
}

func (vc *VectorClient) SearchVector(vector []float64, k int) ([]SearchResult, error) {
	resp, bodyBuf, err := vc.postJSON("/vectors/search", searchVectorRequest{Values: vector, K: k})
	if err != nil {
		return nil, err
	}
	defer putRequestBodyBuffer(bodyBuf)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to search vector: %s", resp.Status)
	}

	var results []SearchResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

func (vc *VectorClient) SearchVectorInNamespace(vector []float64, k int, namespace string) ([]SearchResult, error) {
	resp, bodyBuf, err := vc.postJSON("/vectors/search", searchVectorRequest{Values: vector, K: k, Namespace: namespace})
	if err != nil {
		return nil, err
	}
	defer putRequestBodyBuffer(bodyBuf)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to search vector: %s", resp.Status)
	}
	var results []SearchResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

func (vc *VectorClient) SearchVectors(queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	resp, bodyBuf, err := vc.postJSON("/vectors/search/batch", searchVectorsRequest{Queries: queries})
	if err != nil {
		return nil, err
	}
	defer putRequestBodyBuffer(bodyBuf)
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to search vectors: %s", resp.Status)
	}

	var results []BatchSearchResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, err
	}
	return results, nil
}

func (vc *VectorClient) postJSON(path string, payload any) (*http.Response, *bytes.Buffer, error) {
	bodyBuf := getRequestBodyBuffer()
	enc := json.NewEncoder(bodyBuf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(payload); err != nil {
		putRequestBodyBuffer(bodyBuf)
		return nil, nil, err
	}

	req, err := http.NewRequest(http.MethodPost, vc.baseURL+path, bytes.NewReader(bodyBuf.Bytes()))
	if err != nil {
		putRequestBodyBuffer(bodyBuf)
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := vc.httpClient.Do(req)
	if err != nil {
		putRequestBodyBuffer(bodyBuf)
		return nil, nil, err
	}
	return resp, bodyBuf, nil
}

func getRequestBodyBuffer() *bytes.Buffer {
	buf := requestBodyBufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	return buf
}

func putRequestBodyBuffer(buf *bytes.Buffer) {
	if buf == nil {
		return
	}
	if buf.Cap() > 1<<20 {
		*buf = *bytes.NewBuffer(make([]byte, 0, 4096))
	} else {
		buf.Reset()
	}
	requestBodyBufferPool.Put(buf)
}

func (vc *VectorClient) DeleteVector(id string) error {
	url := fmt.Sprintf("%s/vectors/%s", vc.baseURL, id)
	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return err
	}

	resp, err := vc.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("failed to delete vector: %s", resp.Status)
	}
	return nil
}
