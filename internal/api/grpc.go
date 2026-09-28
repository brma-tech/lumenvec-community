package api

import (
	"context"
	"errors"
	"io"
	"net"
	"path/filepath"
	"strings"

	lumenvecpb "lumenvec/api/proto"
	"lumenvec/internal/core"
	"lumenvec/internal/edition"
	"lumenvec/internal/index"
	"lumenvec/internal/index/ann"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

var (
	grpcListenFunc = func(network, address string) (net.Listener, error) { return net.Listen(network, address) }
	grpcServeFunc  = func(server *grpc.Server, listener net.Listener) error { return server.Serve(listener) }
)

type grpcHandler struct {
	lumenvecpb.UnimplementedVectorServiceServer
	service          core.VectorService
	listVectorsLimit int
	usageObserver    UsageObserver
}

type clusterControlHandler struct {
	lumenvecpb.UnimplementedClusterControlServer
	coordinator edition.ElectionCoordinator
	log         edition.ConsensusLog
	logs        edition.ConsensusLogProvider
}

type snapshotTransferHandler struct {
	lumenvecpb.UnimplementedDataPlaneTransferServer
	dir     string
	service prebuiltANNService
}

func (h *snapshotTransferHandler) UploadPrebuiltSnapshot(stream grpc.ClientStreamingServer[lumenvecpb.UploadPrebuiltSnapshotRequest, lumenvecpb.UploadPrebuiltSnapshotResponse]) error {
	var stager *ann.SnapshotStager
	var manifest ann.SegmentManifest
	committed := false
	defer func() {
		if !committed && stager != nil {
			stager.Abort()
		}
	}()
	for {
		req, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if stager != nil {
				stager.Abort()
			}
			return err
		}
		if incoming := req.GetManifest(); incoming != nil {
			if stager != nil {
				return status.Error(codes.InvalidArgument, "snapshot manifest must be the first and only manifest message")
			}
			if !safeArtifactName(incoming.GetSegmentId()) {
				return status.Error(codes.InvalidArgument, "invalid snapshot segment ID")
			}
			manifest = snapshotManifestFromProto(incoming)
			path := filepath.Join(h.dir, "incoming-ann", manifest.SegmentID+".snapshot")
			stager, err = ann.NewSnapshotStager(path, manifest)
			if err != nil {
				return status.Error(codes.InvalidArgument, err.Error())
			}
			continue
		}
		if stager == nil {
			return status.Error(codes.InvalidArgument, "snapshot manifest is required before chunks")
		}
		if chunk := req.GetChunk(); len(chunk) > 0 {
			if _, err := stager.Write(chunk); err != nil {
				stager.Abort()
				return status.Error(codes.InvalidArgument, err.Error())
			}
		}
	}
	if stager == nil {
		return status.Error(codes.InvalidArgument, "snapshot manifest is required")
	}
	if err := stager.Commit(); err != nil {
		return status.Error(codes.InvalidArgument, err.Error())
	}
	committed = true
	path := filepath.Join(h.dir, "incoming-ann", manifest.SegmentID+".snapshot")
	return stream.SendAndClose(&lumenvecpb.UploadPrebuiltSnapshotResponse{ArtifactPath: path, BytesReceived: manifest.PayloadBytes})
}

func safeArtifactName(value string) bool {
	if strings.TrimSpace(value) == "" || filepath.Base(value) != value || strings.ContainsAny(value, `\\/:`) {
		return false
	}
	for _, r := range value {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' || r == '.') {
			return false
		}
	}
	return true
}

func (h *clusterControlHandler) AppendEntries(_ context.Context, req *lumenvecpb.AppendEntriesRequest) (*lumenvecpb.AppendEntriesResponse, error) {
	if h.coordinator == nil || (h.log == nil && h.logs == nil) {
		return nil, status.Error(codes.Unavailable, "cluster control plane is disabled")
	}
	if err := h.coordinator.EnsureLeader(req.GetLeaderId(), req.GetTerm()); err != nil {
		return nil, clusterControlStatus(err)
	}
	log := h.log
	if h.logs != nil {
		var err error
		log, err = h.logs.Get(req.GetShardId())
		if err != nil {
			return nil, status.Error(codes.Internal, err.Error())
		}
	}
	if log == nil {
		return nil, status.Error(codes.Unavailable, "consensus log is disabled")
	}
	entries := make([]edition.ConsensusEntry, 0, len(req.GetEntries()))
	for _, entry := range req.GetEntries() {
		entries = append(entries, edition.ConsensusEntry{Index: entry.GetIndex(), Term: entry.GetTerm(), OperationID: entry.GetOperationId(), Payload: append([]byte(nil), entry.GetPayload()...)})
	}
	match, err := log.AppendEntries(req.GetTerm(), req.GetLeaderId(), req.GetPrevIndex(), req.GetPrevTerm(), entries, req.GetLeaderCommit())
	if err != nil {
		return nil, consensusLogStatus(err)
	}
	return &lumenvecpb.AppendEntriesResponse{Accepted: true, Term: req.GetTerm(), MatchIndex: match}, nil
}

func (h *clusterControlHandler) Campaign(_ context.Context, req *lumenvecpb.CampaignRequest) (*lumenvecpb.CampaignResponse, error) {
	if h.coordinator == nil {
		return nil, status.Error(codes.Unavailable, "cluster control plane is disabled")
	}
	result, err := h.coordinator.Campaign(req.GetCandidateId(), req.GetLastOffset())
	if err != nil {
		return nil, clusterControlStatus(err)
	}
	return &lumenvecpb.CampaignResponse{Term: result.Term, LeaderId: result.LeaderID, Votes: boundedCount(result.Votes), Quorum: boundedCount(result.Quorum)}, nil
}

func boundedCount(value int) uint32 {
	if value <= 0 {
		return 0
	}
	if uint64(value) > uint64(^uint32(0)) {
		return ^uint32(0)
	}
	return uint32(value) // #nosec G115 -- upper-bounded above
}

func (h *clusterControlHandler) EnsureLeader(_ context.Context, req *lumenvecpb.EnsureLeaderRequest) (*lumenvecpb.EnsureLeaderResponse, error) {
	if h.coordinator == nil {
		return nil, status.Error(codes.Unavailable, "cluster control plane is disabled")
	}
	if err := h.coordinator.EnsureLeader(req.GetCandidateId(), req.GetTerm()); err != nil {
		return nil, clusterControlStatus(err)
	}
	return &lumenvecpb.EnsureLeaderResponse{Accepted: true}, nil
}

func clusterControlStatus(err error) error {
	switch {
	case errors.Is(err, edition.ErrElectionQuorumUnavailable):
		return status.Error(codes.Unavailable, err.Error())
	case errors.Is(err, edition.ErrLeadershipLost), errors.Is(err, edition.ErrStaleLeaderTerm):
		return status.Error(codes.FailedPrecondition, err.Error())
	case errors.Is(err, edition.ErrElectionUnknownMember):
		return status.Error(codes.InvalidArgument, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}

func consensusLogStatus(err error) error {
	switch {
	case errors.Is(err, edition.ErrConsensusStaleTerm):
		return status.Error(codes.FailedPrecondition, err.Error())
	case errors.Is(err, edition.ErrConsensusLogConflict), errors.Is(err, edition.ErrConsensusCommit):
		return status.Error(codes.Aborted, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}

func (h *grpcHandler) Health(context.Context, *lumenvecpb.HealthRequest) (*lumenvecpb.HealthResponse, error) {
	return &lumenvecpb.HealthResponse{Status: "ok"}, nil
}

func (h *grpcHandler) ListVectors(_ context.Context, req *lumenvecpb.ListVectorsRequest) (*lumenvecpb.ListVectorsResponse, error) {
	limit, err := listVectorsGRPCLimitWithCap(req.GetLimit(), h.effectiveListVectorsLimit())
	if err != nil {
		return nil, err
	}

	afterID := ""
	if cursor := strings.TrimSpace(req.GetCursor()); cursor != "" {
		decoded, err := decodeListVectorsCursor(cursor)
		if err != nil {
			return nil, status.Error(codes.InvalidArgument, "cursor must be a valid list cursor")
		}
		afterID = decoded
	}

	page := h.service.ListVectorsPage(core.ListVectorsOptions{
		AfterID: afterID,
		Limit:   limit,
		IDsOnly: req.GetIdsOnly(),
	})
	out := make([]*lumenvecpb.Vector, 0, len(page.Vectors))
	for _, vec := range page.Vectors {
		out = append(out, &lumenvecpb.Vector{Id: vec.ID, Values: vec.Values})
	}

	resp := &lumenvecpb.ListVectorsResponse{Vectors: out}
	if page.NextCursor != "" {
		resp.NextCursor = encodeListVectorsCursor(page.NextCursor)
	}
	return resp, nil
}

func listVectorsGRPCLimit(limit int32) (int, error) {
	return listVectorsGRPCLimitWithCap(limit, maxListVectorsLimit)
}

func listVectorsGRPCLimitWithCap(limit int32, cap int) (int, error) {
	if limit < 0 {
		return 0, status.Error(codes.InvalidArgument, "limit must be a positive integer")
	}
	if limit == 0 {
		return defaultListVectorsLimit, nil
	}
	if cap < 1 {
		cap = maxListVectorsLimit
	}
	if limit > int32(cap) {
		return cap, nil
	}
	return int(limit), nil
}

func (h *grpcHandler) effectiveListVectorsLimit() int {
	if h.listVectorsLimit > 0 {
		return h.listVectorsLimit
	}
	return maxListVectorsLimit
}

func (h *grpcHandler) AddVector(_ context.Context, req *lumenvecpb.AddVectorRequest) (*lumenvecpb.AddVectorResponse, error) {
	var err error
	if len(req.GetMetadata()) > 0 {
		if enriched, ok := h.service.(interface {
			AddVectorWithMetadata(string, []float64, map[string]string) error
		}); ok {
			err = enriched.AddVectorWithMetadata(req.GetId(), req.GetValues(), req.GetMetadata())
		} else {
			return nil, status.Error(codes.Unimplemented, "metadata ingestion is not supported")
		}
	} else {
		err = h.service.AddVector(req.GetId(), req.GetValues())
	}
	if err != nil {
		return nil, grpcStatusFromError(err)
	}
	if h.usageObserver != nil {
		h.usageObserver.ObserveIngestedVectors(1)
	}
	return &lumenvecpb.AddVectorResponse{Success: true}, nil
}

func (h *grpcHandler) AddVectorsBatch(_ context.Context, req *lumenvecpb.AddVectorsBatchRequest) (*lumenvecpb.AddVectorsBatchResponse, error) {
	vectors := make([]index.Vector, 0, len(req.GetVectors()))
	hasMetadata := false
	allFloat32 := len(req.GetVectors()) > 0
	for _, vec := range req.GetVectors() {
		vectors = append(vectors, index.Vector{ID: vec.GetId(), Values: vec.GetValues()})
		allFloat32 = allFloat32 && len(vec.GetValuesF32()) > 0 && len(vec.GetValues()) == 0
		if len(vec.GetMetadata()) > 0 {
			hasMetadata = true
		}
	}
	if allFloat32 && !hasMetadata {
		if native, ok := h.service.(interface {
			AddVectors32([]core.VectorBatch32) error
		}); ok {
			batch := make([]core.VectorBatch32, len(req.GetVectors()))
			for i, vec := range req.GetVectors() {
				batch[i] = core.VectorBatch32{ID: vec.GetId(), Values: vec.GetValuesF32()}
			}
			if err := native.AddVectors32(batch); err != nil {
				return nil, grpcStatusFromError(err)
			}
			if h.usageObserver != nil {
				h.usageObserver.ObserveIngestedVectors(uint64(len(batch)))
			}
			return &lumenvecpb.AddVectorsBatchResponse{Success: true}, nil
		}
	}
	if hasMetadata {
		if enriched, ok := h.service.(interface {
			AddVectorWithMetadata(string, []float64, map[string]string) error
		}); ok {
			for _, vec := range req.GetVectors() {
				if err := enriched.AddVectorWithMetadata(vec.GetId(), vec.GetValues(), vec.GetMetadata()); err != nil {
					return nil, grpcStatusFromError(err)
				}
				if h.usageObserver != nil {
					h.usageObserver.ObserveIngestedVectors(1)
				}
			}
			return &lumenvecpb.AddVectorsBatchResponse{Success: true}, nil
		}
	}
	if err := h.service.AddVectors(vectors); err != nil {
		return nil, grpcStatusFromError(err)
	}
	if h.usageObserver != nil {
		h.usageObserver.ObserveIngestedVectors(uint64(len(vectors)))
	}
	return &lumenvecpb.AddVectorsBatchResponse{Success: true}, nil
}

func (h *grpcHandler) GetVector(_ context.Context, req *lumenvecpb.GetVectorRequest) (*lumenvecpb.GetVectorResponse, error) {
	vec, err := h.service.GetVector(req.GetId())
	if err != nil {
		return nil, grpcStatusFromError(err)
	}
	return &lumenvecpb.GetVectorResponse{
		Vector: &lumenvecpb.Vector{Id: vec.ID, Values: vec.Values},
	}, nil
}

func (h *grpcHandler) Search(ctx context.Context, req *lumenvecpb.SearchRequest) (*lumenvecpb.SearchResponse, error) {
	var results []core.SearchResult
	var err error
	metric := core.DistanceMetric(strings.ToLower(req.GetMetric()))
	if metric == "" {
		metric = core.MetricL2
	}
	if metric != core.MetricL2 && metric != core.MetricCosine && metric != core.MetricInnerProduct {
		return nil, status.Error(codes.InvalidArgument, "unsupported search metric")
	}
	if len(req.GetFilterIds()) > 0 || req.GetTextQuery() != "" || len(req.GetMetadata()) > 0 {
		allowed := make(map[string]struct{}, len(req.GetFilterIds()))
		for _, id := range req.GetFilterIds() {
			allowed[id] = struct{}{}
		}
		structured := core.StructuredFilter{IDs: allowed, TextQuery: req.GetTextQuery(), Metadata: req.GetMetadata()}
		if structuredSearch, ok := h.service.(interface {
			SearchStructured([]float64, int, core.StructuredFilter, core.DistanceMetric) ([]core.SearchResult, error)
		}); ok {
			results, err = core.SearchStructuredWithContext(ctx, structuredSearch, req.GetValues(), int(req.GetTopK()), structured, metric)
		} else if filtered, ok := h.service.(interface {
			SearchFilteredMetric([]float64, int, core.VectorFilter, core.DistanceMetric) ([]core.SearchResult, error)
		}); ok {
			results, err = core.SearchFilteredMetricWithContext(ctx, filtered, req.GetValues(), int(req.GetTopK()), func(v index.Vector) bool {
				return structured.Match(v)
			}, metric)
		} else {
			return nil, status.Error(codes.Unimplemented, "filtered search is not supported by this service")
		}
	} else if metric == core.MetricL2 {
		results, err = core.SearchWithContext(ctx, h.service, req.GetValues(), int(req.GetTopK()))
	} else if metricSearch, ok := h.service.(interface {
		SearchFilteredMetric([]float64, int, core.VectorFilter, core.DistanceMetric) ([]core.SearchResult, error)
	}); ok {
		results, err = core.SearchFilteredMetricWithContext(ctx, metricSearch, req.GetValues(), int(req.GetTopK()), nil, metric)
	} else {
		return nil, status.Error(codes.Unimplemented, "metric search is not supported by this service")
	}
	if err != nil {
		return nil, grpcStatusFromError(err)
	}
	if h.usageObserver != nil {
		h.usageObserver.ObserveSearchRequests(1)
	}
	return &lumenvecpb.SearchResponse{Results: toProtoSearchResults(results)}, nil
}

func (h *grpcHandler) SearchBatch(ctx context.Context, req *lumenvecpb.SearchBatchRequest) (*lumenvecpb.SearchBatchResponse, error) {
	queries := make([]core.BatchSearchQuery, 0, len(req.GetQueries()))
	for _, query := range req.GetQueries() {
		queries = append(queries, core.BatchSearchQuery{
			ID:       query.GetId(),
			Values:   query.GetValues(),
			Values32: query.GetValuesF32(),
			K:        int(query.GetTopK()),
		})
	}
	results, err := core.SearchBatchWithContext(ctx, h.service, queries)
	if err != nil {
		return nil, grpcStatusFromError(err)
	}
	if h.usageObserver != nil {
		h.usageObserver.ObserveSearchRequests(uint64(len(queries)))
	}
	out := make([]*lumenvecpb.SearchBatchResult, len(results))
	for i, result := range results {
		out[i] = &lumenvecpb.SearchBatchResult{
			Id:      result.ID,
			Results: toProtoSearchResults(result.Results),
		}
	}
	return &lumenvecpb.SearchBatchResponse{Results: out}, nil
}

func (h *grpcHandler) DeleteVector(_ context.Context, req *lumenvecpb.DeleteVectorRequest) (*lumenvecpb.DeleteVectorResponse, error) {
	if err := h.service.DeleteVector(req.GetId()); err != nil {
		return nil, grpcStatusFromError(err)
	}
	return &lumenvecpb.DeleteVectorResponse{Success: true}, nil
}

func (s *Server) grpcServer() (*grpc.Server, error) {
	options := []grpc.ServerOption{
		grpc.UnaryInterceptor(s.grpcAuthInterceptor()),
		grpc.StreamInterceptor(s.grpcStreamAuthInterceptor()),
	}
	if s.tlsEnabled {
		configuration, err := s.serverTLSConfig()
		if err != nil {
			return nil, err
		}
		options = append(options, grpc.Creds(credentials.NewTLS(configuration)))
	}
	server := grpc.NewServer(options...)
	lumenvecpb.RegisterVectorServiceServer(server, &grpcHandler{service: s.service, listVectorsLimit: s.effectiveListVectorsLimit(), usageObserver: s.usageObserver})
	lumenvecpb.RegisterClusterControlServer(server, &clusterControlHandler{coordinator: s.election, log: s.consensusLog, logs: s.consensusLogs})
	prebuiltService, _ := s.service.(prebuiltANNService)
	lumenvecpb.RegisterDataPlaneTransferServer(server, &snapshotTransferHandler{dir: s.vectorPath, service: prebuiltService})
	return server, nil
}

func (s *Server) grpcListener() (net.Listener, error) {
	return grpcListenFunc("tcp", s.grpcPort)
}

func (s *Server) serveGRPC(listener net.Listener) error {
	server, err := s.grpcServer()
	if err != nil {
		return err
	}
	return grpcServeFunc(server, listener)
}

func (s *Server) grpcAuthInterceptor() grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req interface{},
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (interface{}, error) {
		if err := s.authorizeGRPC(ctx, info.FullMethod); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

func (s *Server) grpcStreamAuthInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := s.authorizeGRPC(stream.Context(), info.FullMethod); err != nil {
			return err
		}
		return handler(srv, stream)
	}
}

func (s *Server) authorizeGRPC(ctx context.Context, fullMethod string) error {
	apiKey := s.currentAPIKey()
	if !s.grpcAuth || !s.authEnabled || strings.TrimSpace(apiKey) == "" || fullMethod == "/lumenvec.VectorService/Health" {
		return nil
	}
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return status.Error(codes.Unauthenticated, "missing credentials")
	}
	if !validateAPIKey(grpcAuthKeyFromMetadata(md), apiKey) {
		return status.Error(codes.Unauthenticated, "unauthorized")
	}
	return nil
}

func grpcAuthKeyFromMetadata(md metadata.MD) string {
	if values := md.Get("x-api-key"); len(values) > 0 {
		return strings.TrimSpace(values[0])
	}
	if values := md.Get("authorization"); len(values) > 0 {
		auth := strings.TrimSpace(values[0])
		if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
			return strings.TrimSpace(auth[7:])
		}
	}
	return ""
}

func toProtoSearchResults(results []core.SearchResult) []*lumenvecpb.SearchResult {
	out := make([]*lumenvecpb.SearchResult, len(results))
	for i, result := range results {
		out[i] = &lumenvecpb.SearchResult{
			Id:       result.ID,
			Distance: result.Distance,
		}
	}
	return out
}

func grpcStatusFromError(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, context.Canceled):
		return status.Error(codes.Canceled, err.Error())
	case errors.Is(err, context.DeadlineExceeded):
		return status.Error(codes.DeadlineExceeded, err.Error())
	case grpcCodeFromError(err) == codes.AlreadyExists:
		return status.Error(codes.AlreadyExists, err.Error())
	case grpcCodeFromError(err) == codes.NotFound:
		return status.Error(codes.NotFound, err.Error())
	case grpcCodeFromError(err) == codes.InvalidArgument:
		return status.Error(codes.InvalidArgument, err.Error())
	case grpcCodeFromError(err) == codes.Unimplemented:
		return status.Error(codes.Unimplemented, err.Error())
	default:
		return status.Error(codes.Internal, err.Error())
	}
}

func grpcCodeFromError(err error) codes.Code {
	switch {
	case errors.Is(err, index.ErrVectorExists):
		return codes.AlreadyExists
	case errors.Is(err, index.ErrVectorNotFound):
		return codes.NotFound
	case errors.Is(err, core.ErrInvalidID),
		errors.Is(err, core.ErrInvalidValues),
		errors.Is(err, core.ErrInvalidK),
		errors.Is(err, core.ErrVectorDimTooHigh),
		errors.Is(err, core.ErrKTooHigh):
		return codes.InvalidArgument
	case errors.Is(err, core.ErrFilteredUnsupported):
		return codes.Unimplemented
	default:
		return codes.Internal
	}
}
