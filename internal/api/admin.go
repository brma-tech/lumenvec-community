package api

import (
	"context"
	"embed"
	"errors"
	"math"
	"net/http"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"lumenvec/internal/core"
)

//go:embed adminui/*
var adminUI embed.FS

type AdminConfigField struct {
	Path            string `json:"path"`
	Category        string `json:"category"`
	Value           any    `json:"value"`
	RestartRequired bool   `json:"restart_required"`
}

type AdminConfigController interface {
	Snapshot(context.Context) ([]AdminConfigField, error)
	Update(context.Context, map[string]any) ([]AdminConfigField, error)
}

type AdminClusterStatus struct {
	Provider     string `json:"provider"`
	Name         string `json:"name,omitempty"`
	Namespace    string `json:"namespace,omitempty"`
	DesiredNodes int32  `json:"desired_nodes"`
	ReadyNodes   int32  `json:"ready_nodes"`
	Phase        string `json:"phase,omitempty"`
	CanScale     bool   `json:"can_scale"`
	Unavailable  string `json:"unavailable_reason,omitempty"`
}

type AdminClusterController interface {
	Status(context.Context) (AdminClusterStatus, error)
	Scale(context.Context, int32) (AdminClusterStatus, error)
}

type adminMemory struct {
	HeapBytes  uint64 `json:"heap_bytes"`
	AllocBytes uint64 `json:"alloc_bytes"`
	SysBytes   uint64 `json:"sys_bytes"`
}

type adminOverview struct {
	Status      string             `json:"status"`
	Uptime      int64              `json:"uptime_seconds"`
	VectorCount int                `json:"vector_count"`
	GoRoutines  int                `json:"goroutines"`
	Memory      adminMemory        `json:"memory"`
	Stats       core.ServiceStats  `json:"stats"`
	Cluster     AdminClusterStatus `json:"cluster"`
}

type vectorMapPoint struct {
	ID     string    `json:"id"`
	X      float64   `json:"x"`
	Y      float64   `json:"y"`
	Values []float64 `json:"values"`
}

type vectorMapEdge struct {
	Source int     `json:"source"`
	Target int     `json:"target"`
	Score  float64 `json:"score"`
}

type vectorMapResponse struct {
	Points    []vectorMapPoint `json:"points"`
	Edges     []vectorMapEdge  `json:"edges"`
	Dimension int              `json:"dimension"`
	Sampled   int              `json:"sampled"`
}

func (s *Server) registerAdminRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/admin", s.adminIndexHandler)
	mux.HandleFunc("/admin/", s.adminAssetHandler)
	mux.HandleFunc("/v1/admin/overview", methodHandler(http.MethodGet, s.AdminOverviewHandler))
	mux.HandleFunc("/v1/admin/vectors/map", methodHandler(http.MethodGet, s.AdminVectorMapHandler))
	mux.HandleFunc("/v1/admin/config", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodPatch {
			methodNotAllowed(w, r)
			return
		}
		s.AdminConfigHandler(w, r)
	})
	mux.HandleFunc("/v1/admin/cluster", methodHandler(http.MethodGet, s.AdminClusterHandler))
	mux.HandleFunc("/v1/admin/cluster/scale", methodHandler(http.MethodPost, s.AdminClusterScaleHandler))
}

func (s *Server) adminIndexHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/admin" {
		http.NotFound(w, r)
		return
	}
	http.Redirect(w, r, "/admin/", http.StatusTemporaryRedirect)
}

func (s *Server) adminAssetHandler(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/admin/")
	if path == "" {
		path = "index.html"
	}
	if strings.Contains(path, "..") {
		http.NotFound(w, r)
		return
	}
	data, err := adminUI.ReadFile("adminui/" + path)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	switch {
	case strings.HasSuffix(path, ".css"):
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
	case strings.HasSuffix(path, ".js"):
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	default:
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
	}
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(data)
}

func (s *Server) requireAdminSecurity(w http.ResponseWriter, r *http.Request) bool {
	if !s.authEnabled || strings.TrimSpace(s.currentAPIKey()) == "" {
		writeError(w, r, http.StatusServiceUnavailable, "admin_security_required", "admin API requires authentication to be enabled")
		return false
	}
	return true
}

func (s *Server) AdminOverviewHandler(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdminSecurity(w, r) {
		return
	}
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	stats := core.ServiceStats{}
	if provider, ok := s.service.(interface{ Stats() core.ServiceStats }); ok {
		stats = provider.Stats()
	}
	count := stats.ANNNodes - stats.ANNDeleted
	if counter, ok := s.service.(interface{ VectorCount() (int, error) }); ok {
		if current, err := counter.VectorCount(); err == nil {
			count = current
		}
	}
	clusterStatus := AdminClusterStatus{Provider: "standalone", DesiredNodes: 1, ReadyNodes: 1, CanScale: false, Unavailable: "Kubernetes control plane is not configured"}
	if s.adminCluster != nil {
		if current, err := s.adminCluster.Status(r.Context()); err == nil {
			clusterStatus = current
		} else {
			clusterStatus.Unavailable = err.Error()
		}
	}
	writeJSON(w, http.StatusOK, adminOverview{
		Status: "healthy", Uptime: int64(time.Since(s.adminStartedAt).Seconds()), VectorCount: count,
		GoRoutines: runtime.NumGoroutine(),
		Memory:     adminMemory{HeapBytes: memory.HeapInuse, AllocBytes: memory.Alloc, SysBytes: memory.Sys},
		Stats:      stats, Cluster: clusterStatus,
	})
}

func (s *Server) AdminVectorMapHandler(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdminSecurity(w, r) {
		return
	}
	limit := 180
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil || parsed < 10 || parsed > 500 {
			writeError(w, r, http.StatusBadRequest, "invalid_argument", "limit must be between 10 and 500")
			return
		}
		limit = parsed
	}
	page := s.service.ListVectorsPage(core.ListVectorsOptions{Limit: limit})
	writeJSON(w, http.StatusOK, projectVectors(page))
}

func projectVectors(page core.ListVectorsPage) vectorMapResponse {
	if len(page.Vectors) == 0 {
		return vectorMapResponse{Points: []vectorMapPoint{}, Edges: []vectorMapEdge{}}
	}
	dimension := len(page.Vectors[0].Values)
	raw := make([][2]float64, len(page.Vectors))
	for i, vector := range page.Vectors {
		for d, value := range vector.Values {
			raw[i][0] += value * projectionWeight(d, 0)
			raw[i][1] += value * projectionWeight(d, 1)
		}
	}
	minX, maxX, minY, maxY := raw[0][0], raw[0][0], raw[0][1], raw[0][1]
	for _, point := range raw[1:] {
		minX, maxX = math.Min(minX, point[0]), math.Max(maxX, point[0])
		minY, maxY = math.Min(minY, point[1]), math.Max(maxY, point[1])
	}
	points := make([]vectorMapPoint, len(raw))
	for i, point := range raw {
		points[i] = vectorMapPoint{
			ID: page.Vectors[i].ID, X: normalizeProjection(point[0], minX, maxX),
			Y: normalizeProjection(point[1], minY, maxY), Values: append([]float64(nil), page.Vectors[i].Values...),
		}
	}
	return vectorMapResponse{Points: points, Edges: nearestEdges(raw), Dimension: dimension, Sampled: len(points)}
}

func projectionWeight(dimension, axis int) float64 {
	x := uint64(dimension+1)*0x9e3779b97f4a7c15 + uint64(axis+1)*0xbf58476d1ce4e5b9
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return (float64(x&0xffff)/32767.5 - 1) / math.Sqrt(float64(dimension+1))
}

func normalizeProjection(value, min, max float64) float64 {
	if max == min {
		return .5
	}
	return (value - min) / (max - min)
}

func nearestEdges(points [][2]float64) []vectorMapEdge {
	if len(points) < 2 {
		return []vectorMapEdge{}
	}
	edges := make([]vectorMapEdge, 0, len(points))
	for i := range points {
		best, bestDistance := -1, math.MaxFloat64
		for j := range points {
			if i == j {
				continue
			}
			dx, dy := points[i][0]-points[j][0], points[i][1]-points[j][1]
			distance := dx*dx + dy*dy
			if distance < bestDistance {
				best, bestDistance = j, distance
			}
		}
		if best >= 0 && i < best {
			edges = append(edges, vectorMapEdge{Source: i, Target: best, Score: 1 / (1 + math.Sqrt(bestDistance))})
		}
	}
	sort.Slice(edges, func(i, j int) bool { return edges[i].Score > edges[j].Score })
	return edges
}

func (s *Server) AdminConfigHandler(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdminSecurity(w, r) {
		return
	}
	if s.adminConfig == nil {
		writeError(w, r, http.StatusNotImplemented, "not_configured", "configuration control is not configured")
		return
	}
	var fields []AdminConfigField
	var err error
	if r.Method == http.MethodPatch {
		var request struct {
			Changes map[string]any `json:"changes"`
		}
		if !s.readJSON(w, r, &request) {
			return
		}
		if len(request.Changes) == 0 {
			writeError(w, r, http.StatusBadRequest, "invalid_argument", "at least one change is required")
			return
		}
		fields, err = s.adminConfig.Update(r.Context(), request.Changes)
	} else {
		fields, err = s.adminConfig.Snapshot(r.Context())
	}
	if err != nil {
		writeError(w, r, http.StatusBadRequest, "config_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"fields": fields, "restart_required": r.Method == http.MethodPatch})
}

func (s *Server) AdminClusterHandler(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdminSecurity(w, r) {
		return
	}
	if s.adminCluster == nil {
		writeJSON(w, http.StatusOK, AdminClusterStatus{Provider: "standalone", DesiredNodes: 1, ReadyNodes: 1, CanScale: false, Unavailable: "Kubernetes control plane is not configured"})
		return
	}
	status, err := s.adminCluster.Status(r.Context())
	if err != nil {
		writeError(w, r, http.StatusBadGateway, "cluster_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, status)
}

func (s *Server) AdminClusterScaleHandler(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdminSecurity(w, r) {
		return
	}
	if s.adminCluster == nil {
		writeError(w, r, http.StatusNotImplemented, "not_configured", "Kubernetes control plane is not configured")
		return
	}
	var request struct {
		Delta int32 `json:"delta"`
	}
	if !s.readJSON(w, r, &request) {
		return
	}
	if request.Delta != 1 && request.Delta != -1 {
		writeError(w, r, http.StatusBadRequest, "invalid_argument", "delta must be 1 or -1")
		return
	}
	status, err := s.adminCluster.Scale(r.Context(), request.Delta)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			writeError(w, r, http.StatusRequestTimeout, "cancelled", err.Error())
			return
		}
		writeError(w, r, http.StatusBadGateway, "scale_failed", err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, status)
}
