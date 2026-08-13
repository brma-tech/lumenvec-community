package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	lumenvecpb "lumenvec/api/proto"
	"lumenvec/internal/core"
	"lumenvec/internal/index"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

func BenchmarkTransportSearch(b *testing.B) {
	svc := benchmarkTransportService(b)
	loadBenchmarkVectors(b, svc, 512, 256)

	httpServer := benchmarkHTTPServer(svc)
	defer httpServer.Close()

	grpcConn := benchmarkGRPCConn(b, svc)
	defer func() { _ = grpcConn.Close() }()
	grpcClient := lumenvecpb.NewVectorServiceClient(grpcConn)

	b.Run("http_search", func(b *testing.B) {
		payload := map[string]any{"values": benchmarkVector(256, 3), "k": 10}
		body, err := json.Marshal(payload)
		if err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			resp, err := http.Post(httpServer.URL+"/vectors/search", "application/json", bytes.NewReader(body))
			if err != nil {
				b.Fatal(err)
			}
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				b.Fatalf("unexpected status %d", resp.StatusCode)
			}
		}
	})

	b.Run("grpc_search", func(b *testing.B) {
		req := &lumenvecpb.SearchRequest{Values: benchmarkVector(256, 3), TopK: 10}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_, err := grpcClient.Search(ctx, req)
			cancel()
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkTransportBatchSearch(b *testing.B) {
	svc := benchmarkTransportService(b)
	loadBenchmarkVectors(b, svc, 512, 256)

	httpServer := benchmarkHTTPServer(svc)
	defer httpServer.Close()

	grpcConn := benchmarkGRPCConn(b, svc)
	defer func() { _ = grpcConn.Close() }()
	grpcClient := lumenvecpb.NewVectorServiceClient(grpcConn)

	httpQueries := make([]map[string]any, 0, 16)
	grpcQueriesLegacy := make([]*lumenvecpb.SearchBatchQuery, 0, 16)
	grpcQueriesCompact := make([]*lumenvecpb.SearchBatchQuery, 0, 16)
	for i := 0; i < 16; i++ {
		values := benchmarkVector(256, float64(i))
		httpQueries = append(httpQueries, map[string]any{"id": "q", "values": values, "k": 10})
		values32 := make([]float32, len(values))
		for j, value := range values {
			values32[j] = float32(value)
		}
		grpcQueriesLegacy = append(grpcQueriesLegacy, &lumenvecpb.SearchBatchQuery{Id: "q", Values: values, TopK: 10})
		grpcQueriesCompact = append(grpcQueriesCompact, &lumenvecpb.SearchBatchQuery{Id: "q", ValuesF32: values32, TopK: 10})
	}

	b.Run("http_batch_search", func(b *testing.B) {
		body, err := json.Marshal(map[string]any{"queries": httpQueries})
		if err != nil {
			b.Fatal(err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			resp, err := http.Post(httpServer.URL+"/vectors/search/batch", "application/json", bytes.NewReader(body))
			if err != nil {
				b.Fatal(err)
			}
			_ = resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				b.Fatalf("unexpected status %d", resp.StatusCode)
			}
		}
	})

	benchGRPCBatch := func(b *testing.B, queries []*lumenvecpb.SearchBatchQuery) {
		req := &lumenvecpb.SearchBatchRequest{Queries: queries}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			_, err := grpcClient.SearchBatch(ctx, req)
			cancel()
			if err != nil {
				b.Fatal(err)
			}
		}
	}
	b.Run("grpc_batch_search_legacy", func(b *testing.B) { benchGRPCBatch(b, grpcQueriesLegacy) })
	b.Run("grpc_batch_search_f32", func(b *testing.B) { benchGRPCBatch(b, grpcQueriesCompact) })
}

func benchmarkTransportService(tb testing.TB) *core.Service {
	tb.Helper()
	base := tb.TempDir()
	return core.NewService(core.ServiceOptions{
		MaxVectorDim:  1024,
		MaxK:          64,
		SnapshotPath:  filepath.Join(base, "snapshot.json"),
		WALPath:       filepath.Join(base, "wal.log"),
		SnapshotEvery: 1 << 30,
		SearchMode:    "ann",
	})
}

func loadBenchmarkVectors(tb testing.TB, svc *core.Service, total, dim int) {
	tb.Helper()
	vectors := make([]index.Vector, 0, total)
	for i := 0; i < total; i++ {
		vectors = append(vectors, index.Vector{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: benchmarkVector(dim, float64(i%17)),
		})
	}
	if err := svc.AddVectors(vectors); err != nil {
		tb.Fatal(err)
	}
}

func benchmarkHTTPServer(svc *core.Service) *httptest.Server {
	server := &Server{
		port:         ":0",
		readTimeout:  5 * time.Second,
		writeTimeout: 5 * time.Second,
		service:      svc,
		maxBodyBytes: 1 << 20,
		rateLimiter:  nil,
	}
	server.requestTotal, server.requestDuration, server.metricsRegistry = newMetricsRegistry(server.service)
	server.routes()
	return httptest.NewServer(server.Router())
}

func benchmarkGRPCConn(tb testing.TB, svc *core.Service) *grpc.ClientConn {
	tb.Helper()
	listener := bufconn.Listen(1024 * 1024)
	server := grpc.NewServer()
	lumenvecpb.RegisterVectorServiceServer(server, &grpcHandler{service: svc})
	tb.Cleanup(server.Stop)

	go func() {
		_ = server.Serve(listener)
	}()

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) {
			return listener.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		tb.Fatal(err)
	}
	return conn
}

func benchmarkVector(dim int, seed float64) []float64 {
	values := make([]float64, dim)
	for i := range values {
		values[i] = seed + float64(i%7)*0.125
	}
	return values
}
