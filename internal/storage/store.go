package storage

// Store defines the interface for the LumenVec database storage.
type Store interface {
	// SaveVector saves a vector to the storage.
	SaveVector(id string, vector []float64) error

	// GetVector retrieves a vector from the storage by its ID.
	GetVector(id string) ([]float64, error)

	// DeleteVector removes a vector from the storage by its ID.
	DeleteVector(id string) error

	// ListVectors returns all vectors stored in the database.
	ListVectors() (map[string][]float64, error)

	// Close closes the storage connection.
	Close() error
}
