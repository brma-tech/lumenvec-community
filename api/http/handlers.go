package http

import (
	"lumenvec/internal/api"
	"net/http"

	"github.com/gorilla/mux"
)

// Handlers struct to hold dependencies
type Handlers struct {
	API *api.Server
}

// NewHandlers initializes the HTTP handlers
func NewHandlers(api *api.Server) *Handlers {
	return &Handlers{API: api}
}

// RegisterRoutes registers the HTTP routes
func (h *Handlers) RegisterRoutes(r *mux.Router) {
	r.HandleFunc("/health", h.Health).Methods(http.MethodGet)
	r.HandleFunc("/vectors", h.ListVectors).Methods(http.MethodGet)
	r.HandleFunc("/vectors", h.AddVector).Methods(http.MethodPost)
	r.HandleFunc("/vectors/search", h.SearchVectors).Methods(http.MethodPost)
	r.HandleFunc("/vectors/{id}", h.GetVector).Methods(http.MethodGet)
	r.HandleFunc("/vectors/{id}", h.DeleteVector).Methods(http.MethodDelete)
}

// Health returns API health status.
func (h *Handlers) Health(w http.ResponseWriter, r *http.Request) {
	h.API.HealthHandler(w, r)
}

// ListVectors returns all stored vectors.
func (h *Handlers) ListVectors(w http.ResponseWriter, r *http.Request) {
	h.API.ListVectorsHandler(w, r)
}

// AddVector handles adding a new vector
func (h *Handlers) AddVector(w http.ResponseWriter, r *http.Request) {
	h.API.AddVectorHandler(w, r)
}

// SearchVectors handles similarity search on vectors.
func (h *Handlers) SearchVectors(w http.ResponseWriter, r *http.Request) {
	h.API.SearchVectorsHandler(w, r)
}

// GetVector handles retrieving a vector by ID
func (h *Handlers) GetVector(w http.ResponseWriter, r *http.Request) {
	h.API.GetVectorHandler(w, r)
}

// DeleteVector handles deleting a vector by ID
func (h *Handlers) DeleteVector(w http.ResponseWriter, r *http.Request) {
	h.API.DeleteVectorHandler(w, r)
}
