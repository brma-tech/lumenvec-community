package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"lumenvec/internal/core"

	"github.com/prometheus/client_golang/prometheus"
)

const requestIDHeader = "X-Request-ID"

type requestIDContextKey struct{}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

type rateLimiter struct {
	mu      sync.Mutex
	buckets map[string]tokenBucket
	limit   float64
	window  time.Duration
	burst   float64
}

type tokenBucket struct {
	tokens float64
	last   time.Time
}

func newRateLimiter(limit int, window time.Duration) *rateLimiter {
	if limit <= 0 {
		return nil
	}
	return &rateLimiter{
		buckets: make(map[string]tokenBucket),
		limit:   float64(limit),
		window:  window,
		burst:   float64(limit),
	}
}

func (rl *rateLimiter) allow(key string) bool {
	now := time.Now()

	rl.mu.Lock()
	defer rl.mu.Unlock()

	bucket, ok := rl.buckets[key]
	if !ok {
		rl.buckets[key] = tokenBucket{
			tokens: rl.burst - 1,
			last:   now,
		}
		return true
	}

	elapsed := now.Sub(bucket.last)
	if elapsed > 0 {
		bucket.tokens += elapsed.Seconds() * (rl.limit / rl.window.Seconds())
		if bucket.tokens > rl.burst {
			bucket.tokens = rl.burst
		}
		bucket.last = now
	}

	if bucket.tokens < 1 {
		rl.buckets[key] = bucket
		return false
	}

	bucket.tokens--
	rl.buckets[key] = bucket
	return true
}

func (s *Server) getClientIP(r *http.Request) string {
	if s != nil && s.trustXFF && len(s.trustedCIDRs) > 0 && s.isTrustedProxy(r.RemoteAddr) {
		if xff := strings.TrimSpace(r.Header.Get("X-Forwarded-For")); xff != "" {
			parts := strings.Split(xff, ",")
			return strings.TrimSpace(parts[0])
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func getClientIP(r *http.Request) string {
	if xff := strings.TrimSpace(r.Header.Get("X-Forwarded-For")); xff != "" {
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		apiKey := s.currentAPIKey()
		if !s.authEnabled || apiKey == "" || isPublicOperationalPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}

		key := authKeyFromHTTPRequest(r)

		if !validateAPIKey(key, apiKey) {
			writeError(w, r, http.StatusUnauthorized, "unauthorized", "unauthorized")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.rateLimiter == nil || isPublicOperationalPath(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		if !s.rateLimiter.allow(s.rateLimitKey(r)) {
			writeError(w, r, http.StatusTooManyRequests, "rate_limited", "rate limit exceeded")
			return
		}
		next.ServeHTTP(w, r)
	})
}

const namespaceQuotaHeader = "X-LumenVec-Namespace"

func (s *Server) rateLimitKey(r *http.Request) string {
	key := s.getClientIP(r)
	namespace := strings.TrimSpace(r.Header.Get(namespaceQuotaHeader))
	if namespace != "" && len(namespace) <= 128 {
		return key + "\x00" + namespace
	}
	return key
}

func requestIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	id, _ := ctx.Value(requestIDContextKey{}).(string)
	return id
}

func requestIDFromRequest(r *http.Request) string {
	id := strings.TrimSpace(r.Header.Get(requestIDHeader))
	if id == "" || len(id) > 128 {
		return ""
	}
	for _, ch := range id {
		switch {
		case ch >= 'a' && ch <= 'z':
		case ch >= 'A' && ch <= 'Z':
		case ch >= '0' && ch <= '9':
		case ch == '-', ch == '_', ch == '.', ch == ':', ch == '/':
		default:
			return ""
		}
	}
	return id
}

func newRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	return hex.EncodeToString(b[:])
}

func (s *Server) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := requestIDFromRequest(r)
		if requestID == "" {
			requestID = newRequestID()
		}
		w.Header().Set(requestIDHeader, requestID)
		ctx := context.WithValue(r.Context(), requestIDContextKey{}, requestID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func isPublicOperationalPath(path string) bool {
	if path == "/admin" || strings.HasPrefix(path, "/admin/") {
		return true
	}
	switch path {
	case "/health", "/livez", "/readyz", "/v1/health", "/v1/livez", "/v1/readyz", "/metrics":
		return true
	default:
		return false
	}
}

func (s *Server) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		duration := time.Since(start).Seconds()

		if s.requestTotal != nil {
			s.requestTotal.WithLabelValues(r.Method, r.URL.Path, http.StatusText(rec.status)).Inc()
		}
		if s.requestDuration != nil {
			s.requestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration)
		}
	})
}

func (s *Server) accessLogMiddleware(next http.Handler) http.Handler {
	if !s.accessLog {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		s.writeAccessLog(r, rec.status, time.Since(start))
	})
}

func (s *Server) writeAccessLog(r *http.Request, status int, duration time.Duration) {
	payload := struct {
		Event      string `json:"event"`
		RequestID  string `json:"request_id"`
		Method     string `json:"method"`
		Path       string `json:"path"`
		Status     int    `json:"status"`
		DurationMS int64  `json:"duration_ms"`
		ClientAddr string `json:"client_addr"`
		Namespace  string `json:"namespace,omitempty"`
	}{
		Event:      "http_request",
		RequestID:  requestIDFromContext(r.Context()),
		Method:     r.Method,
		Path:       r.URL.Path,
		Status:     status,
		DurationMS: duration.Milliseconds(),
		ClientAddr: getClientIP(r),
		Namespace:  strings.TrimSpace(r.Header.Get(namespaceQuotaHeader)),
	}
	out, err := json.Marshal(payload)
	if err != nil {
		logPrintfAPI(`{"event":"http_request","request_id":"%s","method":"%s","path":"%s","status":%d,"duration_ms":%d,"client_addr":"%s"}`,
			payload.RequestID, payload.Method, payload.Path, payload.Status, payload.DurationMS, payload.ClientAddr)
		return
	}
	logPrintfAPI("%s", out)
	if s.auditLogPath != "" {
		s.auditLogMu.Lock()
		defer s.auditLogMu.Unlock()
		if file, err := os.OpenFile(s.auditLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600); err == nil {
			_, _ = file.Write(append(out, '\n'))
			_ = file.Close()
		}
	}
}

type coreMetricsCollector struct {
	service core.VectorService

	searchRequestsDesc              *prometheus.Desc
	exactSearchesDesc               *prometheus.Desc
	annSearchesDesc                 *prometheus.Desc
	annSearchHitsDesc               *prometheus.Desc
	annSearchFallbacks              *prometheus.Desc
	annSearchErrorsDesc             *prometheus.Desc
	annCandidatesDesc               *prometheus.Desc
	annSegmentsDesc                 *prometheus.Desc
	annCompactionPendingDesc        *prometheus.Desc
	annCompactingDesc               *prometheus.Desc
	annCompactableDesc              *prometheus.Desc
	annReadersDesc                  *prometheus.Desc
	annRetiredDesc                  *prometheus.Desc
	annReclaimedDesc                *prometheus.Desc
	annReclaimErrorsDesc            *prometheus.Desc
	annCompactionMemoryBudgetDesc   *prometheus.Desc
	annCompactionMemoryReservedDesc *prometheus.Desc
	annCompactionMemoryEstimateDesc *prometheus.Desc
	annCompactionMemoryDeferredDesc *prometheus.Desc
	annPrimaryBuildActiveDesc       *prometheus.Desc
	annPrimaryBuildTotalDesc        *prometheus.Desc
	annPrimaryBuildDoneDesc         *prometheus.Desc
	annPrimaryBuildDurationDesc     *prometheus.Desc
	annPrimaryBuildFailuresDesc     *prometheus.Desc
	annSearchBudgetQueriesDesc      *prometheus.Desc
	annSearchBudgetSegmentsDesc     *prometheus.Desc
	annSearchEFBudgetDesc           *prometheus.Desc
	annCheckpointLoadedDesc         *prometheus.Desc
	shardCountDesc                  *prometheus.Desc
	replicationCommittedDesc        *prometheus.Desc
	replicationAppliedDesc          *prometheus.Desc
	replicationPendingDesc          *prometheus.Desc
	replicationFailuresDesc         *prometheus.Desc
	replicationTermDesc             *prometheus.Desc
	replicationFailoversDesc        *prometheus.Desc
	annEvalSamplesDesc              *prometheus.Desc
	annEvalTop1MatchesDesc          *prometheus.Desc
	annEvalOverlapDesc              *prometheus.Desc
	annEvalComparedDesc             *prometheus.Desc
	cacheHitsDesc                   *prometheus.Desc
	cacheMissesDesc                 *prometheus.Desc
	cacheEvictionsDesc              *prometheus.Desc
	cacheItemsDesc                  *prometheus.Desc
	cacheBytesDesc                  *prometheus.Desc
	diskFileBytesDesc               *prometheus.Desc
	diskRecordsDesc                 *prometheus.Desc
	diskStaleRecordsDesc            *prometheus.Desc
	diskCompactionsDesc             *prometheus.Desc
	diskCompactionActiveDesc        *prometheus.Desc
	diskSegmentsDesc                *prometheus.Desc
	annConfigInfoDesc               *prometheus.Desc
}

func newCoreMetricsCollector(service core.VectorService) *coreMetricsCollector {
	return &coreMetricsCollector{
		service: service,
		searchRequestsDesc: prometheus.NewDesc(
			"lumenvec_core_search_requests_total",
			"Total search requests handled by the core service.",
			nil,
			nil,
		),
		exactSearchesDesc: prometheus.NewDesc(
			"lumenvec_core_exact_searches_total",
			"Total exact searches executed by the core service.",
			nil,
			nil,
		),
		annSearchesDesc: prometheus.NewDesc(
			"lumenvec_core_ann_searches_total",
			"Total ANN searches attempted by the core service.",
			nil,
			nil,
		),
		annSearchHitsDesc: prometheus.NewDesc(
			"lumenvec_core_ann_search_hits_total",
			"Total ANN searches that returned candidates successfully.",
			nil,
			nil,
		),
		annSearchFallbacks: prometheus.NewDesc(
			"lumenvec_core_ann_fallbacks_total",
			"Total times the core service fell back from ANN to exact search.",
			nil,
			nil,
		),
		annSearchErrorsDesc: prometheus.NewDesc(
			"lumenvec_core_ann_errors_total",
			"Total ANN search errors observed by the core service.",
			nil,
			nil,
		),
		annCandidatesDesc: prometheus.NewDesc(
			"lumenvec_core_ann_candidates_returned_total",
			"Total ANN candidates returned before final re-scoring.",
			nil,
			nil,
		),
		annSegmentsDesc: prometheus.NewDesc(
			"lumenvec_core_ann_segments",
			"Current number of ANN segments including the active delta.",
			nil,
			nil,
		),
		annCompactionPendingDesc: prometheus.NewDesc(
			"lumenvec_core_ann_compaction_pending",
			"Number of shards with queued ANN consolidation work.", nil, nil,
		),
		annCompactingDesc: prometheus.NewDesc(
			"lumenvec_core_ann_compacting",
			"Number of shards actively rebuilding an ANN generation.", nil, nil,
		),
		annCompactableDesc: prometheus.NewDesc(
			"lumenvec_core_ann_compaction_compactable",
			"Number of shards whose leading ANN generations can still be consolidated.", nil, nil,
		),
		annReadersDesc: prometheus.NewDesc(
			"lumenvec_core_ann_readers",
			"Current readers pinning ANN generations.", nil, nil,
		),
		annRetiredDesc: prometheus.NewDesc(
			"lumenvec_core_ann_retired_segments",
			"Retired ANN segments waiting for their final reader.", nil, nil,
		),
		annReclaimedDesc: prometheus.NewDesc(
			"lumenvec_core_ann_reclaimed_segments_total",
			"ANN segments closed after reader-safe retirement.", nil, nil,
		),
		annReclaimErrorsDesc: prometheus.NewDesc(
			"lumenvec_core_ann_reclaim_errors_total",
			"Errors while closing reader-safe retired ANN segments.", nil, nil,
		),
		annCompactionMemoryBudgetDesc:   prometheus.NewDesc("lumenvec_core_ann_compaction_memory_budget_bytes", "Process-wide ANN compaction build-memory budget; zero means observation mode.", nil, nil),
		annCompactionMemoryReservedDesc: prometheus.NewDesc("lumenvec_core_ann_compaction_memory_reserved_bytes", "ANN compaction build memory currently reserved by shards.", nil, nil),
		annCompactionMemoryEstimateDesc: prometheus.NewDesc("lumenvec_core_ann_compaction_memory_estimate_bytes", "Largest most recent per-shard ANN compaction build-memory estimate.", nil, nil),
		annCompactionMemoryDeferredDesc: prometheus.NewDesc("lumenvec_core_ann_compaction_memory_deferred_total", "ANN compactions deferred because the process-wide memory budget was unavailable.", nil, nil),
		annPrimaryBuildActiveDesc:       prometheus.NewDesc("lumenvec_core_ann_primary_build_active", "Whether an ANN primary generation build is active.", nil, nil),
		annPrimaryBuildTotalDesc:        prometheus.NewDesc("lumenvec_core_ann_primary_build_vectors_total", "Vectors in the current ANN primary generation build.", nil, nil),
		annPrimaryBuildDoneDesc:         prometheus.NewDesc("lumenvec_core_ann_primary_build_vectors_done", "Vectors processed by the current ANN primary generation build.", nil, nil),
		annPrimaryBuildDurationDesc:     prometheus.NewDesc("lumenvec_core_ann_primary_build_duration_ms", "Duration of the most recent ANN primary generation build in milliseconds.", nil, nil),
		annPrimaryBuildFailuresDesc:     prometheus.NewDesc("lumenvec_core_ann_primary_build_failures_total", "Failed ANN primary generation builds.", nil, nil),
		annSearchBudgetQueriesDesc:      prometheus.NewDesc("lumenvec_core_ann_search_budget_queries_total", "ANN queries included in fanout and exploration-budget accounting.", nil, nil),
		annSearchBudgetSegmentsDesc:     prometheus.NewDesc("lumenvec_core_ann_search_budget_segments_total", "ANN segment searches executed by budget-accounted queries.", nil, nil),
		annSearchEFBudgetDesc:           prometheus.NewDesc("lumenvec_core_ann_search_ef_budget_total", "Sum of effective efSearch budgets assigned across ANN segment searches.", nil, nil),
		annCheckpointLoadedDesc: prometheus.NewDesc(
			"lumenvec_core_ann_checkpoint_loaded",
			"Whether the ANN index was restored from a validated checkpoint.",
			nil,
			nil,
		),
		shardCountDesc: prometheus.NewDesc(
			"lumenvec_core_shards",
			"Number of logical shards served by this process.",
			nil,
			nil,
		),
		replicationCommittedDesc: prometheus.NewDesc("lumenvec_core_replication_committed_offset", "Latest replication offset committed by shard leaders.", nil, nil),
		replicationAppliedDesc:   prometheus.NewDesc("lumenvec_core_replication_applied_offset", "Latest replication offset applied by replicas.", nil, nil),
		replicationPendingDesc:   prometheus.NewDesc("lumenvec_core_replication_pending", "Number of committed replication operations not yet applied.", nil, nil),
		replicationFailuresDesc:  prometheus.NewDesc("lumenvec_core_replication_failures_total", "Total replica apply failures.", nil, nil),
		replicationTermDesc:      prometheus.NewDesc("lumenvec_core_replication_term", "Current leader fencing term.", nil, nil),
		replicationFailoversDesc: prometheus.NewDesc("lumenvec_core_replication_failovers_total", "Total automatic leader promotions.", nil, nil),
		annEvalSamplesDesc: prometheus.NewDesc(
			"lumenvec_core_ann_eval_samples_total",
			"Total sampled ANN queries evaluated against exact search.",
			nil,
			nil,
		),
		annEvalTop1MatchesDesc: prometheus.NewDesc(
			"lumenvec_core_ann_eval_top1_matches_total",
			"Total sampled ANN queries whose top-1 result matched exact search.",
			nil,
			nil,
		),
		annEvalOverlapDesc: prometheus.NewDesc(
			"lumenvec_core_ann_eval_overlap_results_total",
			"Total overlapping result IDs between sampled ANN and exact results.",
			nil,
			nil,
		),
		annEvalComparedDesc: prometheus.NewDesc(
			"lumenvec_core_ann_eval_compared_results_total",
			"Total result slots compared between sampled ANN and exact results.",
			nil,
			nil,
		),
		cacheHitsDesc: prometheus.NewDesc(
			"lumenvec_core_cache_hits_total",
			"Total vector store cache hits.",
			nil,
			nil,
		),
		cacheMissesDesc: prometheus.NewDesc(
			"lumenvec_core_cache_misses_total",
			"Total vector store cache misses.",
			nil,
			nil,
		),
		cacheEvictionsDesc: prometheus.NewDesc(
			"lumenvec_core_cache_evictions_total",
			"Total vector store cache evictions.",
			nil,
			nil,
		),
		cacheItemsDesc: prometheus.NewDesc(
			"lumenvec_core_cache_items",
			"Current number of items stored in the vector cache.",
			nil,
			nil,
		),
		cacheBytesDesc: prometheus.NewDesc(
			"lumenvec_core_cache_bytes",
			"Current approximate number of bytes stored in the vector cache.",
			nil,
			nil,
		),
		diskFileBytesDesc: prometheus.NewDesc(
			"lumenvec_core_disk_file_bytes",
			"Current size in bytes of the disk-backed vector store file.",
			nil,
			nil,
		),
		diskRecordsDesc: prometheus.NewDesc(
			"lumenvec_core_disk_records",
			"Current number of live records in the disk-backed vector store.",
			nil,
			nil,
		),
		diskStaleRecordsDesc: prometheus.NewDesc(
			"lumenvec_core_disk_stale_records",
			"Current number of stale records waiting for or surviving compaction in the disk-backed vector store.",
			nil,
			nil,
		),
		diskCompactionsDesc: prometheus.NewDesc(
			"lumenvec_core_disk_compactions_total",
			"Total disk-backed vector store compactions.",
			nil,
			nil,
		),
		diskCompactionActiveDesc: prometheus.NewDesc("lumenvec_core_disk_compaction_active", "Whether a disk-store compaction is currently running.", nil, nil),
		diskSegmentsDesc: prometheus.NewDesc(
			"lumenvec_core_disk_segments",
			"Current number of active immutable storage segments.",
			nil,
			nil,
		),
		annConfigInfoDesc: prometheus.NewDesc(
			"lumenvec_core_ann_config_info",
			"Effective ANN configuration exposed as labels.",
			[]string{"profile", "m", "ef_construction", "ef_search", "search_execution", "ef_budget", "ef_global_percent", "primary_index"},
			nil,
		),
	}
}

func (c *coreMetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.searchRequestsDesc
	ch <- c.exactSearchesDesc
	ch <- c.annSearchesDesc
	ch <- c.annSearchHitsDesc
	ch <- c.annSearchFallbacks
	ch <- c.annSearchErrorsDesc
	ch <- c.annCandidatesDesc
	ch <- c.annSegmentsDesc
	ch <- c.annCompactionPendingDesc
	ch <- c.annCompactingDesc
	ch <- c.annCompactableDesc
	ch <- c.annReadersDesc
	ch <- c.annRetiredDesc
	ch <- c.annReclaimedDesc
	ch <- c.annReclaimErrorsDesc
	ch <- c.annCompactionMemoryBudgetDesc
	ch <- c.annCompactionMemoryReservedDesc
	ch <- c.annCompactionMemoryEstimateDesc
	ch <- c.annCompactionMemoryDeferredDesc
	ch <- c.annPrimaryBuildActiveDesc
	ch <- c.annPrimaryBuildTotalDesc
	ch <- c.annPrimaryBuildDoneDesc
	ch <- c.annPrimaryBuildDurationDesc
	ch <- c.annPrimaryBuildFailuresDesc
	ch <- c.annSearchBudgetQueriesDesc
	ch <- c.annSearchBudgetSegmentsDesc
	ch <- c.annSearchEFBudgetDesc
	ch <- c.annCheckpointLoadedDesc
	ch <- c.shardCountDesc
	ch <- c.replicationCommittedDesc
	ch <- c.replicationAppliedDesc
	ch <- c.replicationPendingDesc
	ch <- c.replicationFailuresDesc
	ch <- c.replicationTermDesc
	ch <- c.replicationFailoversDesc
	ch <- c.annEvalSamplesDesc
	ch <- c.annEvalTop1MatchesDesc
	ch <- c.annEvalOverlapDesc
	ch <- c.annEvalComparedDesc
	ch <- c.cacheHitsDesc
	ch <- c.cacheMissesDesc
	ch <- c.cacheEvictionsDesc
	ch <- c.cacheItemsDesc
	ch <- c.cacheBytesDesc
	ch <- c.diskFileBytesDesc
	ch <- c.diskRecordsDesc
	ch <- c.diskStaleRecordsDesc
	ch <- c.diskCompactionsDesc
	ch <- c.diskCompactionActiveDesc
	ch <- c.diskSegmentsDesc
	ch <- c.annConfigInfoDesc
}

func (c *coreMetricsCollector) Collect(ch chan<- prometheus.Metric) {
	stats := c.service.Stats()
	ch <- prometheus.MustNewConstMetric(c.searchRequestsDesc, prometheus.CounterValue, float64(stats.SearchRequestsTotal))
	ch <- prometheus.MustNewConstMetric(c.exactSearchesDesc, prometheus.CounterValue, float64(stats.ExactSearchesTotal))
	ch <- prometheus.MustNewConstMetric(c.annSearchesDesc, prometheus.CounterValue, float64(stats.ANNSearchesTotal))
	ch <- prometheus.MustNewConstMetric(c.annSearchHitsDesc, prometheus.CounterValue, float64(stats.ANNSearchHitsTotal))
	ch <- prometheus.MustNewConstMetric(c.annSearchFallbacks, prometheus.CounterValue, float64(stats.ANNSearchFallbacks))
	ch <- prometheus.MustNewConstMetric(c.annSearchErrorsDesc, prometheus.CounterValue, float64(stats.ANNSearchErrorsTotal))
	ch <- prometheus.MustNewConstMetric(c.annCandidatesDesc, prometheus.CounterValue, float64(stats.ANNCandidatesReturned))
	ch <- prometheus.MustNewConstMetric(c.annSegmentsDesc, prometheus.GaugeValue, float64(stats.ANNSegments))
	ch <- prometheus.MustNewConstMetric(c.annCompactionPendingDesc, prometheus.GaugeValue, float64(stats.ANNCompactionPending))
	ch <- prometheus.MustNewConstMetric(c.annCompactingDesc, prometheus.GaugeValue, float64(stats.ANNCompacting))
	ch <- prometheus.MustNewConstMetric(c.annCompactableDesc, prometheus.GaugeValue, float64(stats.ANNCompactionCompactable))
	ch <- prometheus.MustNewConstMetric(c.annReadersDesc, prometheus.GaugeValue, float64(stats.ANNReaders))
	ch <- prometheus.MustNewConstMetric(c.annRetiredDesc, prometheus.GaugeValue, float64(stats.ANNRetiredSegments))
	ch <- prometheus.MustNewConstMetric(c.annReclaimedDesc, prometheus.CounterValue, float64(stats.ANNReclaimedSegments))
	ch <- prometheus.MustNewConstMetric(c.annReclaimErrorsDesc, prometheus.CounterValue, float64(stats.ANNReclaimErrors))
	ch <- prometheus.MustNewConstMetric(c.annCompactionMemoryBudgetDesc, prometheus.GaugeValue, float64(stats.ANNCompactionMemoryBudget))
	ch <- prometheus.MustNewConstMetric(c.annCompactionMemoryReservedDesc, prometheus.GaugeValue, float64(stats.ANNCompactionMemoryReserved))
	ch <- prometheus.MustNewConstMetric(c.annCompactionMemoryEstimateDesc, prometheus.GaugeValue, float64(stats.ANNCompactionMemoryEstimate))
	ch <- prometheus.MustNewConstMetric(c.annCompactionMemoryDeferredDesc, prometheus.CounterValue, float64(stats.ANNCompactionMemoryDeferred))
	primaryActive := 0.0
	if stats.ANNPrimaryBuildActive {
		primaryActive = 1
	}
	ch <- prometheus.MustNewConstMetric(c.annPrimaryBuildActiveDesc, prometheus.GaugeValue, primaryActive)
	ch <- prometheus.MustNewConstMetric(c.annPrimaryBuildTotalDesc, prometheus.GaugeValue, float64(stats.ANNPrimaryBuildTotal))
	ch <- prometheus.MustNewConstMetric(c.annPrimaryBuildDoneDesc, prometheus.GaugeValue, float64(stats.ANNPrimaryBuildDone))
	ch <- prometheus.MustNewConstMetric(c.annPrimaryBuildDurationDesc, prometheus.GaugeValue, float64(stats.ANNPrimaryBuildDurationMs))
	ch <- prometheus.MustNewConstMetric(c.annPrimaryBuildFailuresDesc, prometheus.CounterValue, float64(stats.ANNPrimaryBuildFailures))
	ch <- prometheus.MustNewConstMetric(c.annSearchBudgetQueriesDesc, prometheus.CounterValue, float64(stats.ANNSearchBudgetQueries))
	ch <- prometheus.MustNewConstMetric(c.annSearchBudgetSegmentsDesc, prometheus.CounterValue, float64(stats.ANNSearchBudgetSegments))
	ch <- prometheus.MustNewConstMetric(c.annSearchEFBudgetDesc, prometheus.CounterValue, float64(stats.ANNSearchEFBudget))
	checkpointLoaded := 0.0
	if stats.ANNCheckpointLoaded {
		checkpointLoaded = 1
	}
	ch <- prometheus.MustNewConstMetric(c.annCheckpointLoadedDesc, prometheus.GaugeValue, checkpointLoaded)
	shardCount := stats.ShardCount
	if shardCount <= 0 {
		shardCount = 1
	}
	ch <- prometheus.MustNewConstMetric(c.shardCountDesc, prometheus.GaugeValue, float64(shardCount))
	ch <- prometheus.MustNewConstMetric(c.replicationCommittedDesc, prometheus.GaugeValue, float64(stats.ReplicationCommitted))
	ch <- prometheus.MustNewConstMetric(c.replicationAppliedDesc, prometheus.GaugeValue, float64(stats.ReplicationApplied))
	ch <- prometheus.MustNewConstMetric(c.replicationPendingDesc, prometheus.GaugeValue, float64(stats.ReplicationPending))
	ch <- prometheus.MustNewConstMetric(c.replicationFailuresDesc, prometheus.CounterValue, float64(stats.ReplicationFailures))
	ch <- prometheus.MustNewConstMetric(c.replicationTermDesc, prometheus.GaugeValue, float64(stats.ReplicationTerm))
	ch <- prometheus.MustNewConstMetric(c.replicationFailoversDesc, prometheus.CounterValue, float64(stats.ReplicationFailovers))
	ch <- prometheus.MustNewConstMetric(c.annEvalSamplesDesc, prometheus.CounterValue, float64(stats.ANNEvalSamplesTotal))
	ch <- prometheus.MustNewConstMetric(c.annEvalTop1MatchesDesc, prometheus.CounterValue, float64(stats.ANNEvalTop1Matches))
	ch <- prometheus.MustNewConstMetric(c.annEvalOverlapDesc, prometheus.CounterValue, float64(stats.ANNEvalOverlapResults))
	ch <- prometheus.MustNewConstMetric(c.annEvalComparedDesc, prometheus.CounterValue, float64(stats.ANNEvalComparedResults))
	ch <- prometheus.MustNewConstMetric(c.cacheHitsDesc, prometheus.CounterValue, float64(stats.CacheHitsTotal))
	ch <- prometheus.MustNewConstMetric(c.cacheMissesDesc, prometheus.CounterValue, float64(stats.CacheMissesTotal))
	ch <- prometheus.MustNewConstMetric(c.cacheEvictionsDesc, prometheus.CounterValue, float64(stats.CacheEvictionsTotal))
	ch <- prometheus.MustNewConstMetric(c.cacheItemsDesc, prometheus.GaugeValue, float64(stats.CacheItems))
	ch <- prometheus.MustNewConstMetric(c.cacheBytesDesc, prometheus.GaugeValue, float64(stats.CacheBytes))
	ch <- prometheus.MustNewConstMetric(c.diskFileBytesDesc, prometheus.GaugeValue, float64(stats.DiskFileBytes))
	ch <- prometheus.MustNewConstMetric(c.diskRecordsDesc, prometheus.GaugeValue, float64(stats.DiskRecords))
	ch <- prometheus.MustNewConstMetric(c.diskStaleRecordsDesc, prometheus.GaugeValue, float64(stats.DiskStaleRecords))
	ch <- prometheus.MustNewConstMetric(c.diskCompactionsDesc, prometheus.CounterValue, float64(stats.DiskCompactionsTotal))
	active := 0.0
	if stats.DiskCompactionActive {
		active = 1
	}
	ch <- prometheus.MustNewConstMetric(c.diskCompactionActiveDesc, prometheus.GaugeValue, active)
	ch <- prometheus.MustNewConstMetric(c.diskSegmentsDesc, prometheus.GaugeValue, float64(stats.DiskSegments))
	ch <- prometheus.MustNewConstMetric(
		c.annConfigInfoDesc,
		prometheus.GaugeValue,
		1,
		stats.ANNProfile,
		strconv.Itoa(stats.ANNM),
		strconv.Itoa(stats.ANNEfConstruction),
		strconv.Itoa(stats.ANNEfSearch),
		stats.ANNSearchExecution,
		stats.ANNEFBudgetMode,
		strconv.Itoa(stats.ANNEFGlobalPercent),
		strconv.FormatBool(stats.ANNPrimaryIndexEnabled),
	)
}

func newMetricsRegistry(service core.VectorService) (*prometheus.CounterVec, *prometheus.HistogramVec, *prometheus.Registry) {
	requestTotal := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "lumenvec_http_requests_total",
			Help: "Total HTTP requests by method, route, and status.",
		},
		[]string{"method", "route", "status"},
	)

	requestDuration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "lumenvec_http_request_duration_seconds",
			Help:    "HTTP request latency in seconds.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "route"},
	)

	registry := prometheus.NewRegistry()
	registry.MustRegister(requestTotal, requestDuration)
	if service != nil {
		registry.MustRegister(newCoreMetricsCollector(service))
	}
	return requestTotal, requestDuration, registry
}
