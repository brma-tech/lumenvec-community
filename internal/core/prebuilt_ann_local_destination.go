package core

import (
	"errors"
	"io"
	"os"
	"path/filepath"

	"lumenvec/internal/index/ann"
)

func (s *Service) ReservePrebuiltIDs(ids []string) ([]IDMappingEntry, error) {
	return s.ReserveIDMappings(ids)
}

func (s *Service) UploadPrebuiltSnapshot(manifest ann.SegmentManifest, sourcePath string) (string, error) {
	if err := manifest.Validate(); err != nil {
		return "", err
	}
	if s.vectorPath == "" {
		return "", errors.New("prebuilt snapshot destination requires a vector path")
	}
	source, err := os.Open(sourcePath)
	if err != nil {
		return "", err
	}
	defer source.Close()
	path := filepath.Join(s.vectorPath, "incoming-ann", manifest.SegmentID+".snapshot")
	stager, err := ann.NewSnapshotStager(path, manifest)
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(stager, source); err != nil {
		stager.Abort()
		return "", err
	}
	if err := stager.Commit(); err != nil {
		return "", err
	}
	return path, nil
}

func (s *Service) ApplyPrebuiltSegment(manifest ann.SegmentManifest, replay PrebuiltANNReplay) error {
	if s.vectorPath == "" {
		return errors.New("prebuilt segment destination requires a vector path")
	}
	if replay == nil {
		return errors.New("prebuilt segment replay is required")
	}
	dir := filepath.Join(s.vectorPath, "incoming-ann")
	spool, err := NewPrebuiltANNRecordSpool(dir, manifest)
	if err != nil {
		return err
	}
	if err := replay(spool.Append); err != nil {
		spool.Abort()
		return err
	}
	if err := spool.Seal(); err != nil {
		return err
	}
	path := filepath.Join(dir, manifest.SegmentID+".snapshot")
	return s.ApplyPrebuiltANNSnapshotSpoolFile(path, manifest, spool.Path())
}
