package core

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"

	"lumenvec/internal/index/ann"
)

const prebuiltANNTransactionVersion = 1

type prebuiltANNTransaction struct {
	Version      int                 `json:"version"`
	State        string              `json:"state"`
	Manifest     ann.SegmentManifest `json:"manifest"`
	SnapshotPath string              `json:"snapshot_path"`
	RecordsPath  string              `json:"records_path"`
}

func (s *Service) ApplyPrebuiltANNSnapshotSpoolFile(snapshotPath string, manifest ann.SegmentManifest, recordsPath string) error {
	if s.vectorPath != "" {
		if !pathWithinDirectory(s.vectorPath, snapshotPath) || !pathWithinDirectory(s.vectorPath, recordsPath) {
			return errors.New("prebuilt ANN transaction artifacts must remain inside the vector directory")
		}
	}
	replay, err := OpenPrebuiltANNRecordReplay(recordsPath, manifest)
	if err != nil {
		return err
	}
	transaction := prebuiltANNTransaction{
		Version: prebuiltANNTransactionVersion, State: "applying", Manifest: manifest,
		SnapshotPath: snapshotPath, RecordsPath: recordsPath,
	}
	path := s.prebuiltANNTransactionPath(manifest.SegmentID)
	if path != "" {
		if err := writePrebuiltANNTransaction(path, transaction); err != nil {
			return err
		}
	}
	if err := s.ApplyPrebuiltANNSnapshotFile(snapshotPath, manifest, replay); err != nil {
		if !errors.Is(err, errPrebuiltPublishedUncataloged) {
			if path != "" {
				_ = os.Remove(path)
			}
			_ = os.Remove(recordsPath)
		}
		return err
	}
	if path != "" {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if err := os.Remove(recordsPath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (s *Service) recoverPrebuiltANNTransactions() error {
	dir := s.prebuiltANNTransactionDir()
	if dir == "" {
		return nil
	}
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}
		path := filepath.Join(dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var transaction prebuiltANNTransaction
		if err := json.Unmarshal(data, &transaction); err != nil {
			return err
		}
		if err := transaction.Validate(); err != nil {
			return err
		}
		if !pathWithinDirectory(s.vectorPath, transaction.SnapshotPath) || !pathWithinDirectory(s.vectorPath, transaction.RecordsPath) {
			return errors.New("prebuilt ANN transaction contains an out-of-directory artifact")
		}
		replay, err := OpenPrebuiltANNRecordReplay(transaction.RecordsPath, transaction.Manifest)
		if err != nil {
			return err
		}
		committed, err := s.prebuiltANNCommitExists(transaction.Manifest)
		if err != nil {
			return err
		}
		if !committed {
			if err := replay(func(record PrebuiltANNRecord) error {
				_ = s.vectorStore.DeleteVector(record.ID)
				_ = s.index.DeleteVector(record.ID)
				return s.removeID(record.ID)
			}); err != nil {
				return err
			}
			// Crash recovery is intentionally allowed to rebuild the surviving
			// ANN once. Normal resharding remains file-backed and incremental.
			s.rebuildANNLocked()
			if err := s.ApplyPrebuiltANNSnapshotFile(transaction.SnapshotPath, transaction.Manifest, replay); err != nil {
				return err
			}
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
		if err := os.Remove(transaction.RecordsPath); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func pathWithinDirectory(base, candidate string) bool {
	if base == "" || candidate == "" {
		return false
	}
	absoluteBase, err := filepath.Abs(base)
	if err != nil {
		return false
	}
	absoluteCandidate, err := filepath.Abs(candidate)
	if err != nil {
		return false
	}
	relative, err := filepath.Rel(absoluteBase, absoluteCandidate)
	if err != nil {
		return false
	}
	return relative != "." && relative != ".." && !filepath.IsAbs(relative) && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func (s *Service) prebuiltANNCommitExists(manifest ann.SegmentManifest) (bool, error) {
	catalog, err := s.prebuiltANNCommitCatalog()
	if err != nil || catalog == nil {
		return false, err
	}
	for _, committed := range catalog.Snapshot() {
		if committed.SegmentID != manifest.SegmentID {
			continue
		}
		if committed.PayloadSHA256 != manifest.PayloadSHA256 || committed.PayloadBytes != manifest.PayloadBytes {
			return false, errors.New("prebuilt ANN transaction conflicts with committed segment")
		}
		return true, nil
	}
	return false, nil
}

func (s *Service) prebuiltANNTransactionDir() string {
	if s.vectorPath == "" {
		return ""
	}
	return filepath.Join(s.vectorPath, "prebuilt-ann-transactions")
}

func (s *Service) prebuiltANNTransactionPath(segmentID string) string {
	dir := s.prebuiltANNTransactionDir()
	if dir == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(segmentID))
	return filepath.Join(dir, hex.EncodeToString(sum[:])+".json")
}

func (t prebuiltANNTransaction) Validate() error {
	if t.Version != prebuiltANNTransactionVersion || t.State != "applying" || t.SnapshotPath == "" || t.RecordsPath == "" {
		return errors.New("invalid prebuilt ANN transaction")
	}
	return t.Manifest.Validate()
}

func writePrebuiltANNTransaction(path string, transaction prebuiltANNTransaction) error {
	if err := transaction.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(transaction, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".prebuilt-ann-transaction-*.tmp")
	if err != nil {
		return err
	}
	tmpPath := file.Name()
	defer os.Remove(tmpPath)
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpPath, path)
}
