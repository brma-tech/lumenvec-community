package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"lumenvec/internal/core"
	"lumenvec/internal/index"
)

type fakeAdminConfig struct{ updated map[string]any }

func (f *fakeAdminConfig) Snapshot(context.Context) ([]AdminConfigField, error) {
	return []AdminConfigField{{Path: "search.ann_ef_search", Category: "search", Value: 64, RestartRequired: true}}, nil
}
func (f *fakeAdminConfig) Update(_ context.Context, changes map[string]any) ([]AdminConfigField, error) {
	f.updated = changes
	return f.Snapshot(context.Background())
}

func TestAdminRequiresOptInAndAuthentication(t *testing.T) {
	disabled := NewServerWithOptions(ServerOptions{DisableRateLimit: true})
	response := httptest.NewRecorder()
	disabled.Router().ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/admin/", nil))
	if response.Code != http.StatusNotFound {
		t.Fatalf("disabled admin status = %d", response.Code)
	}

	enabled := NewServerWithOptions(ServerOptions{AdminEnabled: true, AuthEnabled: true, AuthAPIKey: "secret", DisableRateLimit: true})
	response = httptest.NewRecorder()
	enabled.Router().ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/admin/", nil))
	if response.Code != http.StatusOK {
		t.Fatalf("admin UI status = %d", response.Code)
	}
	response = httptest.NewRecorder()
	enabled.Router().ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/admin/overview", nil))
	if response.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated API status = %d", response.Code)
	}
}

func TestAdminConfigUpdateAndVectorProjection(t *testing.T) {
	controller := &fakeAdminConfig{}
	server := NewServerWithOptions(ServerOptions{
		AdminEnabled: true, AuthEnabled: true, AuthAPIKey: "secret",
		AdminConfig: controller, DisableRateLimit: true,
	})
	request := httptest.NewRequest(http.MethodPatch, "/v1/admin/config", bytes.NewBufferString(`{"changes":{"search.ann_ef_search":96}}`))
	request.Header.Set("X-API-Key", "secret")
	response := httptest.NewRecorder()
	server.Router().ServeHTTP(response, request)
	if response.Code != http.StatusOK || controller.updated["search.ann_ef_search"] != float64(96) {
		t.Fatalf("config update status=%d changes=%v", response.Code, controller.updated)
	}

	page := core.ListVectorsPage{Vectors: []index.Vector{
		{ID: "a", Values: []float64{1, 0, 2}},
		{ID: "b", Values: []float64{0, 1, 2}},
		{ID: "c", Values: []float64{2, 1, 0}},
	}}
	projected := projectVectors(page)
	if projected.Dimension != 3 || projected.Sampled != 3 || len(projected.Points) != 3 {
		t.Fatalf("projection = %+v", projected)
	}
	payload, err := json.Marshal(projected)
	if err != nil || !json.Valid(payload) {
		t.Fatalf("projection JSON: %v", err)
	}
	for _, point := range projected.Points {
		if point.X < 0 || point.X > 1 || point.Y < 0 || point.Y > 1 {
			t.Fatalf("point outside normalized space: %+v", point)
		}
		if len(point.Values) != projected.Dimension {
			t.Fatalf("point matrix dimension = %d, want %d", len(point.Values), projected.Dimension)
		}
	}
}
