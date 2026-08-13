package api

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	lumenvecpb "lumenvec/api/proto"
	"lumenvec/internal/core"
	"lumenvec/internal/index/ann"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type prebuiltANNService interface {
	ReserveIDMappings([]string) ([]core.IDMappingEntry, error)
	ReservePrebuiltIDs([]string) ([]core.IDMappingEntry, error)
	UploadPrebuiltSnapshot(ann.SegmentManifest, string) (string, error)
	ApplyPrebuiltSegment(ann.SegmentManifest, core.PrebuiltANNReplay) error
	ApplyPrebuiltANNSnapshotFile(string, ann.SegmentManifest, core.PrebuiltANNReplay) error
	ApplyPrebuiltANNSnapshotSpoolFile(string, ann.SegmentManifest, string) error
}

// BuildPrebuiltSegment receives canonical records and deliberately ignores
// source-side internal IDs. The data node reserves its own mappings, builds
// the HNSW graph locally and commits through the existing durable journal.
func (h *snapshotTransferHandler) BuildPrebuiltSegment(stream lumenvecpb.DataPlaneTransfer_BuildPrebuiltSegmentServer) error {
	if h.service == nil {
		return status.Error(codes.Unavailable, "prebuilt ANN application is disabled")
	}
	var header *lumenvecpb.SnapshotManifest
	var spool *core.PrebuiltANNRecordSpool
	receiveStarted := time.Now()
	defer func() {
		if spool != nil {
			spool.Discard()
		}
	}()
	for {
		request, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if incoming := request.GetManifest(); incoming != nil {
			if header != nil || !safeArtifactName(incoming.GetSegmentId()) {
				return status.Error(codes.InvalidArgument, "manifest must be the first message with a safe segment ID")
			}
			if incoming.GetNodes() <= 0 || incoming.GetNodes() > 10000 || incoming.GetDimension() <= 0 ||
				incoming.GetM() <= 0 || incoming.GetEfConstruction() <= 0 || strings.TrimSpace(incoming.GetMetric()) == "" {
				return status.Error(codes.InvalidArgument, "invalid destination build manifest")
			}
			header = incoming
			stagingManifest := ann.SegmentManifest{
				Version: 1, SegmentID: incoming.GetSegmentId(), State: ann.SegmentSealed,
				FirstID: 1, LastID: int(incoming.GetNodes()), Nodes: int(incoming.GetNodes()),
				Dimension: int(incoming.GetDimension()), Metric: incoming.GetMetric(),
				M: int(incoming.GetM()), EfConstruction: int(incoming.GetEfConstruction()),
				PayloadBytes: 1, PayloadSHA256: strings.Repeat("0", 64),
			}
			dir, err := os.MkdirTemp(h.dir, ".destination-ann-build-")
			if err != nil {
				return status.Error(codes.Internal, err.Error())
			}
			defer os.RemoveAll(dir)
			spool, err = core.NewPrebuiltANNRecordSpool(dir, stagingManifest)
			if err != nil {
				return status.Error(codes.InvalidArgument, err.Error())
			}
			continue
		}
		if header == nil || spool == nil || request.GetRecord() == nil {
			return status.Error(codes.InvalidArgument, "manifest is required before vector records")
		}
		record := request.GetRecord()
		if spool.Count() >= int(header.GetNodes()) {
			return status.Error(codes.InvalidArgument, "destination build received too many records")
		}
		if err := spool.Append32(record.GetId(), spool.Count()+1, record.GetValues()); err != nil {
			return status.Error(codes.InvalidArgument, err.Error())
		}
	}
	if header == nil || spool == nil || spool.Count() != int(header.GetNodes()) {
		return status.Error(codes.InvalidArgument, "destination build record cardinality mismatch")
	}
	if err := spool.Seal(); err != nil {
		return status.Error(codes.Internal, err.Error())
	}
	buildDir := filepath.Dir(spool.Path())
	options := ann.Options{M: int(header.GetM()), EfConstruction: int(header.GetEfConstruction())}
	buildStarted := time.Now()
	manifest, err := core.TransferPrebuiltANNPartition(buildDir, header.GetSegmentId(), header.GetMetric(), options, spool.Replay, h.service)
	if err != nil {
		return status.Error(codes.FailedPrecondition, err.Error())
	}
	if os.Getenv("LUMENVEC_RESHARD_TRACE") == "1" {
		fmt.Printf("RESHARD_DESTINATION segment=%s vectors=%d receive_ms=%.3f build_commit_ms=%.3f total_ms=%.3f\n",
			header.GetSegmentId(), spool.Count(),
			float64(buildStarted.Sub(receiveStarted))/float64(time.Millisecond),
			float64(time.Since(buildStarted))/float64(time.Millisecond),
			float64(time.Since(receiveStarted))/float64(time.Millisecond))
	}
	return stream.SendAndClose(&lumenvecpb.BuildPrebuiltSegmentResponse{Manifest: &lumenvecpb.SnapshotManifest{
		SegmentId: manifest.SegmentID, FirstId: int32(manifest.FirstID), LastId: int32(manifest.LastID), // #nosec G115 -- ANN resolver IDs are int32-bounded
		Nodes: int32(manifest.Nodes), Dimension: int32(manifest.Dimension), Metric: manifest.Metric, // #nosec G115 -- transfer batches are capped at 10k
		M: int32(manifest.M), EfConstruction: int32(manifest.EfConstruction), // #nosec G115 -- ANN options are validated positive ints
		PayloadBytes: manifest.PayloadBytes, PayloadSha256: manifest.PayloadSHA256,
	}})
}

type prebuiltANNExportService interface {
	ExportANNState(func(core.PrebuiltANNStateEvent) error) error
	VisitSealedANNRecords(func(int, core.PrebuiltANNReplay) error) error
	SnapshotMutableANNRecords() ([]core.PrebuiltANNRecord, error)
	ReplayDeltasSince(uint64, func(core.DeltaRecord) error) error
	CurrentDeltaOffset() uint64
	ANNProfile() ann.Options
}

func (h *snapshotTransferHandler) GetANNProfile(context.Context, *lumenvecpb.GetANNProfileRequest) (*lumenvecpb.GetANNProfileResponse, error) {
	service, err := h.exportService()
	if err != nil {
		return nil, err
	}
	profile := service.ANNProfile()
	return &lumenvecpb.GetANNProfileResponse{M: int32(profile.M), EfConstruction: int32(profile.EfConstruction), EfSearch: int32(profile.EfSearch), SegmentRouting: profile.SegmentRouting}, nil
}

func prebuiltRecordProto(record core.PrebuiltANNRecord) *lumenvecpb.PrebuiltVectorRecord {
	values := make([]float32, len(record.Values))
	for i, value := range record.Values {
		values[i] = float32(value)
	}
	return &lumenvecpb.PrebuiltVectorRecord{Id: record.ID, InternalId: int32(record.InternalID), Values: values} // #nosec G115 -- resolver IDs are int32-bounded
}

func (h *snapshotTransferHandler) exportService() (prebuiltANNExportService, error) {
	service, ok := h.service.(prebuiltANNExportService)
	if !ok {
		return nil, status.Error(codes.Unavailable, "prebuilt ANN export is disabled")
	}
	return service, nil
}

func (h *snapshotTransferHandler) ExportSealedANNRecords(_ *lumenvecpb.ExportSealedANNRecordsRequest, stream lumenvecpb.DataPlaneTransfer_ExportSealedANNRecordsServer) error {
	service, err := h.exportService()
	if err != nil {
		return err
	}
	return service.VisitSealedANNRecords(func(segmentIndex int, replay core.PrebuiltANNReplay) error {
		if err := replay(func(record core.PrebuiltANNRecord) error {
			return stream.Send(&lumenvecpb.ExportSealedANNRecordsResponse{SegmentIndex: int32(segmentIndex), Record: prebuiltRecordProto(record)})
		}); err != nil {
			return err
		}
		return stream.Send(&lumenvecpb.ExportSealedANNRecordsResponse{SegmentIndex: int32(segmentIndex), EndSegment: true})
	})
}

// ExportANNState is the versioned, single-view export. Legacy clients may keep
// using the independent RPCs, while new coordinators get epoch and WAL offset
// on every frame and can reject an inconsistent transfer.
func (h *snapshotTransferHandler) ExportANNState(_ *lumenvecpb.ExportANNStateRequest, stream lumenvecpb.DataPlaneTransfer_ExportANNStateServer) error {
	service, err := h.exportService()
	if err != nil {
		return err
	}
	return service.ExportANNState(func(event core.PrebuiltANNStateEvent) error {
		response := &lumenvecpb.ExportANNStateResponse{Kind: lumenvecpb.ExportANNStateResponse_Kind(event.Kind), Epoch: event.Epoch, DeltaOffset: event.DeltaOffset, SegmentIndex: int32(event.SegmentIndex), Cardinality: event.Cardinality} // #nosec G115 -- segment indexes are bounded by process memory
		if event.Kind == 1 || event.Kind == 3 {
			response.Record = prebuiltRecordProto(event.Record)
		}
		return stream.Send(response)
	})
}

func (h *snapshotTransferHandler) SnapshotMutableANNRecords(context.Context, *lumenvecpb.SnapshotMutableANNRecordsRequest) (*lumenvecpb.SnapshotMutableANNRecordsResponse, error) {
	service, err := h.exportService()
	if err != nil {
		return nil, err
	}
	records, err := service.SnapshotMutableANNRecords()
	if err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	response := &lumenvecpb.SnapshotMutableANNRecordsResponse{Records: make([]*lumenvecpb.PrebuiltVectorRecord, len(records))}
	for i, record := range records {
		response.Records[i] = prebuiltRecordProto(record)
	}
	return response, nil
}

func (h *snapshotTransferHandler) GetDeltaOffset(context.Context, *lumenvecpb.GetDeltaOffsetRequest) (*lumenvecpb.GetDeltaOffsetResponse, error) {
	service, err := h.exportService()
	if err != nil {
		return nil, err
	}
	return &lumenvecpb.GetDeltaOffsetResponse{Offset: service.CurrentDeltaOffset()}, nil
}

func (h *snapshotTransferHandler) GetVectorCount(context.Context, *lumenvecpb.GetVectorCountRequest) (*lumenvecpb.GetVectorCountResponse, error) {
	if h.service == nil {
		return nil, status.Error(codes.Unavailable, "vector count is unavailable")
	}
	counter, ok := h.service.(interface{ VectorCount() (int, error) })
	if !ok {
		return nil, status.Error(codes.Unimplemented, "vector count is unsupported")
	}
	count, err := counter.VectorCount()
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &lumenvecpb.GetVectorCountResponse{Count: uint64(count)}, nil
}

func (h *snapshotTransferHandler) ReplayDeltas(request *lumenvecpb.ReplayDeltasRequest, stream lumenvecpb.DataPlaneTransfer_ReplayDeltasServer) error {
	service, err := h.exportService()
	if err != nil {
		return err
	}
	return service.ReplayDeltasSince(request.GetAfterOffset(), func(delta core.DeltaRecord) error {
		return stream.Send(&lumenvecpb.DeltaMutation{Offset: delta.Offset, VectorId: delta.VectorID, Deleted: delta.Deleted, Values: delta.Values})
	})
}

func (h *snapshotTransferHandler) ReservePrebuiltIDs(_ context.Context, request *lumenvecpb.ReservePrebuiltIDsRequest) (*lumenvecpb.ReservePrebuiltIDsResponse, error) {
	if h.service == nil {
		return nil, status.Error(codes.Unavailable, "prebuilt ANN application is disabled")
	}
	if len(request.GetIds()) == 0 || len(request.GetIds()) > 10000 {
		return nil, status.Error(codes.InvalidArgument, "prebuilt ID reservation requires 1..10000 IDs")
	}
	mappings, err := h.service.ReserveIDMappings(request.GetIds())
	if err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	response := &lumenvecpb.ReservePrebuiltIDsResponse{Mappings: make([]*lumenvecpb.IDMapping, len(mappings))}
	for i, mapping := range mappings {
		response.Mappings[i] = &lumenvecpb.IDMapping{Id: mapping.ID, InternalId: int32(mapping.InternalID)} // #nosec G115 -- resolver bounds IDs to int32
	}
	return response, nil
}

func snapshotManifestFromProto(incoming *lumenvecpb.SnapshotManifest) ann.SegmentManifest {
	return ann.SegmentManifest{
		Version: 1, SegmentID: incoming.GetSegmentId(), State: ann.SegmentSealed,
		FirstID: int(incoming.GetFirstId()), LastID: int(incoming.GetLastId()),
		Nodes: int(incoming.GetNodes()), Dimension: int(incoming.GetDimension()),
		Metric: incoming.GetMetric(), M: int(incoming.GetM()),
		EfConstruction: int(incoming.GetEfConstruction()), PayloadBytes: incoming.GetPayloadBytes(),
		PayloadSHA256: incoming.GetPayloadSha256(),
	}
}

func (h *snapshotTransferHandler) ApplyPrebuiltSegment(stream lumenvecpb.DataPlaneTransfer_ApplyPrebuiltSegmentServer) error {
	if h.service == nil {
		return status.Error(codes.Unavailable, "prebuilt ANN application is disabled")
	}
	var manifest ann.SegmentManifest
	var spool *core.PrebuiltANNRecordSpool
	defer func() {
		if spool != nil {
			spool.Abort()
		}
	}()
	for {
		request, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		if incoming := request.GetManifest(); incoming != nil {
			if spool != nil || !safeArtifactName(incoming.GetSegmentId()) {
				return status.Error(codes.InvalidArgument, "manifest must be the first message with a safe segment ID")
			}
			manifest = snapshotManifestFromProto(incoming)
			if err := manifest.Validate(); err != nil {
				return status.Error(codes.InvalidArgument, err.Error())
			}
			spool, err = core.NewPrebuiltANNRecordSpool(filepath.Join(h.dir, "incoming-ann"), manifest)
			if err != nil {
				return status.Error(codes.Internal, err.Error())
			}
			continue
		}
		if spool == nil || request.GetRecord() == nil {
			return status.Error(codes.InvalidArgument, "manifest is required before vector records")
		}
		record := request.GetRecord()
		if err := spool.Append32(record.GetId(), int(record.GetInternalId()), record.GetValues()); err != nil {
			return status.Error(codes.InvalidArgument, err.Error())
		}
	}
	if spool == nil {
		return status.Error(codes.InvalidArgument, "manifest is required")
	}
	if err := spool.Seal(); err != nil {
		return status.Error(codes.Internal, err.Error())
	}
	artifactPath := filepath.Join(h.dir, "incoming-ann", manifest.SegmentID+".snapshot")
	if err := h.service.ApplyPrebuiltANNSnapshotSpoolFile(artifactPath, manifest, spool.Path()); err != nil {
		return status.Error(codes.FailedPrecondition, err.Error())
	}
	count := spool.Count()
	spool.Discard()
	return stream.SendAndClose(&lumenvecpb.ApplyPrebuiltSegmentResponse{SegmentId: manifest.SegmentID, VectorsCommitted: int32(count)})
}
