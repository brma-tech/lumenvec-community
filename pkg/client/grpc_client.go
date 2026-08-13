package client

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	lumenvecpb "lumenvec/api/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type GRPCVectorClient struct {
	conn    *grpc.ClientConn
	client  lumenvecpb.VectorServiceClient
	timeout time.Duration
}

const defaultGRPCListVectorsLimit = 1000
const prebuiltSnapshotChunkSize = 256 << 10
const defaultPrebuiltTransferTimeout = 30 * time.Minute

// SnapshotManifest describes an immutable ANN artifact accepted by the
// internal DataPlaneTransfer service. The payload itself is supplied through
// an io.Reader so callers do not need to allocate it as one byte slice.
type SnapshotManifest struct {
	SegmentID      string
	FirstID        int32
	LastID         int32
	Nodes          int32
	Dimension      int32
	Metric         string
	M              int32
	EfConstruction int32
	PayloadBytes   int64
	PayloadSHA256  string
}

type PrebuiltVectorRecord struct {
	ID         string
	InternalID int32
	Values     []float32
}

type DeltaMutation struct {
	Offset   uint64
	VectorID string
	Deleted  bool
	Values   []float32
}

type ANNStateEvent struct {
	Kind         int
	Epoch        uint64
	DeltaOffset  uint64
	SegmentIndex int
	Record       *PrebuiltVectorRecord
	Cardinality  uint64
}

type ANNProfile struct {
	M, EfConstruction, EfSearch int
	SegmentRouting              bool
}

func (c *GRPCVectorClient) GetANNProfile() (ANNProfile, error) {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	response, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).GetANNProfile(ctx, &lumenvecpb.GetANNProfileRequest{})
	if err != nil {
		return ANNProfile{}, err
	}
	return ANNProfile{M: int(response.GetM()), EfConstruction: int(response.GetEfConstruction()), EfSearch: int(response.GetEfSearch()), SegmentRouting: response.GetSegmentRouting()}, nil
}

func (c *GRPCVectorClient) ExportSealedANNRecords(visit func(int, *PrebuiltVectorRecord, bool) error) error {
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	stream, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).ExportSealedANNRecords(ctx, &lumenvecpb.ExportSealedANNRecordsRequest{})
	if err != nil {
		return err
	}
	for {
		message, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		var exported *PrebuiltVectorRecord
		if record := message.GetRecord(); record != nil {
			value := PrebuiltVectorRecord{ID: record.GetId(), InternalID: record.GetInternalId(), Values: append([]float32(nil), record.GetValues()...)}
			exported = &value
		}
		if err := visit(int(message.GetSegmentIndex()), exported, message.GetEndSegment()); err != nil {
			return err
		}
	}
}

// ExportANNState streams the source's atomically selected ANN view.
func (c *GRPCVectorClient) ExportANNState(visit func(ANNStateEvent) error) error {
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	stream, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).ExportANNState(ctx, &lumenvecpb.ExportANNStateRequest{})
	if err != nil {
		return err
	}
	for {
		message, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		var record *PrebuiltVectorRecord
		if raw := message.GetRecord(); raw != nil {
			record = &PrebuiltVectorRecord{ID: raw.GetId(), InternalID: raw.GetInternalId(), Values: append([]float32(nil), raw.GetValues()...)}
		}
		if err := visit(ANNStateEvent{Kind: int(message.GetKind()), Epoch: message.GetEpoch(), DeltaOffset: message.GetDeltaOffset(), SegmentIndex: int(message.GetSegmentIndex()), Record: record, Cardinality: message.GetCardinality()}); err != nil {
			return err
		}
	}
}

func (c *GRPCVectorClient) SnapshotMutableANNRecords() ([]PrebuiltVectorRecord, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	response, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).SnapshotMutableANNRecords(ctx, &lumenvecpb.SnapshotMutableANNRecordsRequest{})
	if err != nil {
		return nil, err
	}
	out := make([]PrebuiltVectorRecord, len(response.GetRecords()))
	for i, record := range response.GetRecords() {
		out[i] = PrebuiltVectorRecord{ID: record.GetId(), InternalID: record.GetInternalId(), Values: append([]float32(nil), record.GetValues()...)}
	}
	return out, nil
}

func (c *GRPCVectorClient) CurrentDeltaOffset() (uint64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()
	response, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).GetDeltaOffset(ctx, &lumenvecpb.GetDeltaOffsetRequest{})
	if err != nil {
		return 0, err
	}
	return response.GetOffset(), nil
}

func (c *GRPCVectorClient) VectorCount() (int, error) {
	ctx, cancel := c.context()
	defer cancel()
	response, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).GetVectorCount(ctx, &lumenvecpb.GetVectorCountRequest{})
	if err != nil {
		return 0, err
	}
	if response.GetCount() > uint64(^uint(0)>>1) {
		return 0, errors.New("remote vector count exceeds platform int")
	}
	return int(response.GetCount()), nil
}

func (c *GRPCVectorClient) ReplayDeltasSince(offset uint64, yield func(DeltaMutation) error) error {
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	stream, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).ReplayDeltas(ctx, &lumenvecpb.ReplayDeltasRequest{AfterOffset: offset})
	if err != nil {
		return err
	}
	for {
		record, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := yield(DeltaMutation{Offset: record.GetOffset(), VectorID: record.GetVectorId(), Deleted: record.GetDeleted(), Values: append([]float32(nil), record.GetValues()...)}); err != nil {
			return err
		}
	}
}

type PrebuiltIDMapping struct {
	ID         string
	InternalID int32
}

type PrebuiltVectorSource func(yield func(PrebuiltVectorRecord) error) error

var newGRPCClient = grpc.NewClient

func NewGRPCVectorClient(address string) (*GRPCVectorClient, error) {
	return NewGRPCVectorClientWithDialer(address, nil)
}

func NewGRPCVectorClientWithDialer(address string, dialOptions []grpc.DialOption) (*GRPCVectorClient, error) {
	if strings.TrimSpace(address) == "" {
		return nil, fmt.Errorf("grpc address is required")
	}
	options := append([]grpc.DialOption{}, dialOptions...)
	if len(options) == 0 {
		options = append(options, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}
	conn, err := newGRPCClient(address, options...)
	if err != nil {
		return nil, err
	}
	return &GRPCVectorClient{
		conn:    conn,
		client:  lumenvecpb.NewVectorServiceClient(conn),
		timeout: 10 * time.Second,
	}, nil
}

func (c *GRPCVectorClient) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

func (c *GRPCVectorClient) AddVector(vector []float64) error {
	return c.AddVectorWithID(fmt.Sprintf("vec-%d", time.Now().UnixNano()), vector)
}

func (c *GRPCVectorClient) AddVectorWithID(id string, vector []float64) error {
	ctx, cancel := c.context()
	defer cancel()
	_, err := c.client.AddVector(ctx, &lumenvecpb.AddVectorRequest{
		Id:     id,
		Values: vector,
	})
	return err
}

func (c *GRPCVectorClient) AddVectors(vectors []VectorPayload) error {
	items := make([]*lumenvecpb.Vector, 0, len(vectors))
	for _, vec := range vectors {
		items = append(items, &lumenvecpb.Vector{
			Id:        vec.ID,
			Values:    vec.Values,
			ValuesF32: vec.Values32,
		})
	}
	ctx, cancel := c.context()
	defer cancel()
	_, err := c.client.AddVectorsBatch(ctx, &lumenvecpb.AddVectorsBatchRequest{Vectors: items})
	return err
}

// UploadPrebuiltSnapshot streams an immutable ANN artifact to a cluster node.
// It sends the manifest first and then bounded chunks read from source. The
// destination performs durable checksum validation before exposing the file.
func (c *GRPCVectorClient) UploadPrebuiltSnapshot(manifest SnapshotManifest, source io.Reader) (string, error) {
	if source == nil {
		return "", fmt.Errorf("snapshot source is required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	stream, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).UploadPrebuiltSnapshot(ctx)
	if err != nil {
		return "", err
	}
	if err := stream.Send(&lumenvecpb.UploadPrebuiltSnapshotRequest{Body: &lumenvecpb.UploadPrebuiltSnapshotRequest_Manifest{Manifest: snapshotManifestProto(manifest)}}); err != nil {
		return "", err
	}
	buffer := make([]byte, prebuiltSnapshotChunkSize)
	for {
		n, readErr := source.Read(buffer)
		if n > 0 {
			chunk := append([]byte(nil), buffer[:n]...)
			if err := stream.Send(&lumenvecpb.UploadPrebuiltSnapshotRequest{Body: &lumenvecpb.UploadPrebuiltSnapshotRequest_Chunk{Chunk: chunk}}); err != nil {
				return "", err
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return "", readErr
		}
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		return "", err
	}
	return response.GetArtifactPath(), nil
}

func (c *GRPCVectorClient) ReservePrebuiltIDs(ids []string) ([]PrebuiltIDMapping, error) {
	if len(ids) == 0 {
		return nil, fmt.Errorf("prebuilt IDs are required")
	}
	ctx, cancel := c.context()
	defer cancel()
	response, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).ReservePrebuiltIDs(ctx, &lumenvecpb.ReservePrebuiltIDsRequest{Ids: ids})
	if err != nil {
		return nil, err
	}
	out := make([]PrebuiltIDMapping, len(response.GetMappings()))
	for i, mapping := range response.GetMappings() {
		out[i] = PrebuiltIDMapping{ID: mapping.GetId(), InternalID: mapping.GetInternalId()}
	}
	return out, nil
}

// ApplyPrebuiltSegment streams stable IDs and canonical float32 vectors for a
// previously uploaded snapshot. The server spools the records to disk and
// commits mappings, vector storage and ANN publication as one reshard window.
func (c *GRPCVectorClient) ApplyPrebuiltSegment(manifest SnapshotManifest, source PrebuiltVectorSource) error {
	if source == nil {
		return fmt.Errorf("prebuilt vector source is required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	stream, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).ApplyPrebuiltSegment(ctx)
	if err != nil {
		return err
	}
	if err := stream.Send(&lumenvecpb.ApplyPrebuiltSegmentRequest{Body: &lumenvecpb.ApplyPrebuiltSegmentRequest_Manifest{Manifest: snapshotManifestProto(manifest)}}); err != nil {
		return err
	}
	sent := int32(0)
	if err := source(func(record PrebuiltVectorRecord) error {
		if err := stream.Send(&lumenvecpb.ApplyPrebuiltSegmentRequest{Body: &lumenvecpb.ApplyPrebuiltSegmentRequest_Record{Record: &lumenvecpb.PrebuiltVectorRecord{Id: record.ID, InternalId: record.InternalID, Values: record.Values}}}); err != nil {
			return err
		}
		sent++
		return nil
	}); err != nil {
		return err
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		return err
	}
	if response.GetSegmentId() != manifest.SegmentID || response.GetVectorsCommitted() != sent || sent != manifest.Nodes {
		return fmt.Errorf("prebuilt segment commit acknowledgement mismatch")
	}
	return nil
}

// BuildPrebuiltSegment streams canonical vectors to the destination node.
// Internal IDs in source records are intentionally ignored by the server,
// which reserves mappings and builds the graph in its own data plane.
func (c *GRPCVectorClient) BuildPrebuiltSegment(manifest SnapshotManifest, source PrebuiltVectorSource) (SnapshotManifest, error) {
	if source == nil || manifest.SegmentID == "" || manifest.Nodes <= 0 || manifest.Dimension <= 0 {
		return SnapshotManifest{}, fmt.Errorf("destination build manifest and vector source are required")
	}
	ctx, cancel := context.WithTimeout(context.Background(), defaultPrebuiltTransferTimeout)
	defer cancel()
	stream, err := lumenvecpb.NewDataPlaneTransferClient(c.conn).BuildPrebuiltSegment(ctx)
	if err != nil {
		return SnapshotManifest{}, err
	}
	if err := stream.Send(&lumenvecpb.BuildPrebuiltSegmentRequest{Body: &lumenvecpb.BuildPrebuiltSegmentRequest_Manifest{Manifest: snapshotManifestProto(manifest)}}); err != nil {
		return SnapshotManifest{}, err
	}
	sent := int32(0)
	if err := source(func(record PrebuiltVectorRecord) error {
		if err := stream.Send(&lumenvecpb.BuildPrebuiltSegmentRequest{Body: &lumenvecpb.BuildPrebuiltSegmentRequest_Record{Record: &lumenvecpb.PrebuiltVectorRecord{Id: record.ID, Values: record.Values}}}); err != nil {
			return err
		}
		sent++
		return nil
	}); err != nil {
		return SnapshotManifest{}, err
	}
	response, err := stream.CloseAndRecv()
	if err != nil {
		return SnapshotManifest{}, err
	}
	responseManifest := response.GetManifest()
	out := SnapshotManifest{
		SegmentID: responseManifest.GetSegmentId(), FirstID: responseManifest.GetFirstId(), LastID: responseManifest.GetLastId(),
		Nodes: responseManifest.GetNodes(), Dimension: responseManifest.GetDimension(), Metric: responseManifest.GetMetric(),
		M: responseManifest.GetM(), EfConstruction: responseManifest.GetEfConstruction(),
		PayloadBytes: responseManifest.GetPayloadBytes(), PayloadSHA256: responseManifest.GetPayloadSha256(),
	}
	if out.SegmentID != manifest.SegmentID || out.Nodes != sent || sent != manifest.Nodes || out.Dimension != manifest.Dimension {
		return SnapshotManifest{}, fmt.Errorf("destination build acknowledgement mismatch")
	}
	return out, nil
}

func snapshotManifestProto(manifest SnapshotManifest) *lumenvecpb.SnapshotManifest {
	return &lumenvecpb.SnapshotManifest{
		SegmentId: manifest.SegmentID, FirstId: manifest.FirstID, LastId: manifest.LastID,
		Nodes: manifest.Nodes, Dimension: manifest.Dimension, Metric: manifest.Metric,
		M: manifest.M, EfConstruction: manifest.EfConstruction,
		PayloadBytes: manifest.PayloadBytes, PayloadSha256: manifest.PayloadSHA256,
	}
}

// AddVectorsStream uploads batches on demand. Returning a nil batch signals
// end of input; each non-empty batch is committed independently.
func (c *GRPCVectorClient) AddVectorsStream(next func() ([]VectorPayload, error)) error {
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
		if err := c.AddVectors(batch); err != nil {
			return err
		}
	}
}

func (c *GRPCVectorClient) ListVectors() ([]VectorPayload, error) {
	out := make([]VectorPayload, 0)
	cursor := ""
	for {
		page, err := c.ListVectorsPage(ListVectorsOptions{
			Limit:  defaultGRPCListVectorsLimit,
			Cursor: cursor,
		})
		if err != nil {
			return nil, err
		}
		out = append(out, page.Vectors...)
		if page.NextCursor == "" {
			return out, nil
		}
		if page.NextCursor == cursor {
			return nil, fmt.Errorf("list vectors cursor did not advance")
		}
		cursor = page.NextCursor
	}
}

func (c *GRPCVectorClient) ListVectorsPage(opts ListVectorsOptions) (ListVectorsPage, error) {
	ctx, cancel := c.context()
	defer cancel()
	resp, err := c.client.ListVectors(ctx, &lumenvecpb.ListVectorsRequest{
		Limit:   grpcListVectorsRequestLimit(opts.Limit),
		Cursor:  opts.Cursor,
		IdsOnly: opts.IDsOnly,
	})
	if err != nil {
		return ListVectorsPage{}, err
	}
	out := make([]VectorPayload, 0, len(resp.GetVectors()))
	for _, vec := range resp.GetVectors() {
		if vec == nil {
			continue
		}
		out = append(out, VectorPayload{
			ID:     vec.GetId(),
			Values: vec.GetValues(),
		})
	}
	return ListVectorsPage{Vectors: out, NextCursor: resp.GetNextCursor()}, nil
}

func grpcListVectorsRequestLimit(limit int) int32 {
	if limit < 0 {
		return -1
	}
	if limit > defaultGRPCListVectorsLimit {
		return defaultGRPCListVectorsLimit
	}
	return int32(limit)
}

func (c *GRPCVectorClient) GetVector(id string) (*VectorPayload, error) {
	ctx, cancel := c.context()
	defer cancel()
	resp, err := c.client.GetVector(ctx, &lumenvecpb.GetVectorRequest{Id: id})
	if err != nil {
		return nil, err
	}
	vec := resp.GetVector()
	if vec == nil {
		return nil, nil
	}
	return &VectorPayload{
		ID:     vec.GetId(),
		Values: vec.GetValues(),
	}, nil
}

func (c *GRPCVectorClient) SearchVector(vector []float64, k int) ([]SearchResult, error) {
	return c.SearchVectorMetric(vector, k, "")
}

// SearchVectorMetric executes an unfiltered search using the requested public
// distance metric. An empty metric preserves the historical L2 behavior.
func (c *GRPCVectorClient) SearchVectorMetric(vector []float64, k int, metric string) ([]SearchResult, error) {
	if k <= 0 || int64(k) > int64(^uint32(0)>>1) {
		return nil, fmt.Errorf("k out of int32 range: %d", k)
	}
	ctx, cancel := c.context()
	defer cancel()
	resp, err := c.client.Search(ctx, &lumenvecpb.SearchRequest{
		Values: vector,
		TopK:   boundedTopK(k),
		Metric: metric,
	})
	if err != nil {
		return nil, err
	}
	return fromProtoSearchResults(resp.GetResults()), nil
}

func (c *GRPCVectorClient) SearchVectors(queries []BatchSearchQuery) ([]BatchSearchResult, error) {
	items := make([]*lumenvecpb.SearchBatchQuery, 0, len(queries))
	for _, query := range queries {
		if query.K <= 0 || int64(query.K) > int64(^uint32(0)>>1) {
			return nil, fmt.Errorf("k out of int32 range: %d", query.K)
		}
		items = append(items, &lumenvecpb.SearchBatchQuery{
			Id:        query.ID,
			Values:    query.Values,
			ValuesF32: query.Values32,
			TopK:      boundedTopK(query.K),
		})
	}
	ctx, cancel := c.context()
	defer cancel()
	resp, err := c.client.SearchBatch(ctx, &lumenvecpb.SearchBatchRequest{Queries: items})
	if err != nil {
		return nil, err
	}
	results := make([]BatchSearchResult, 0, len(resp.GetResults()))
	for _, item := range resp.GetResults() {
		results = append(results, BatchSearchResult{
			ID:      item.GetId(),
			Results: fromProtoSearchResults(item.GetResults()),
		})
	}
	return results, nil
}

func boundedTopK(value int) int32 {
	return int32(value) // #nosec G115 -- callers validate the int32 range
}

func (c *GRPCVectorClient) DeleteVector(id string) error {
	ctx, cancel := c.context()
	defer cancel()
	_, err := c.client.DeleteVector(ctx, &lumenvecpb.DeleteVectorRequest{Id: id})
	return err
}

func (c *GRPCVectorClient) Health() (string, error) {
	ctx, cancel := c.context()
	defer cancel()
	resp, err := c.client.Health(ctx, &lumenvecpb.HealthRequest{})
	if err != nil {
		return "", err
	}
	return resp.GetStatus(), nil
}

func (c *GRPCVectorClient) context() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), c.timeout)
}

func fromProtoSearchResults(results []*lumenvecpb.SearchResult) []SearchResult {
	out := make([]SearchResult, 0, len(results))
	for _, result := range results {
		out = append(out, SearchResult{
			ID:       result.GetId(),
			Distance: result.GetDistance(),
		})
	}
	return out
}
