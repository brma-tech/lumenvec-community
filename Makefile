APP_NAME=lumenvec

.PHONY: test vet build run bench tidy coverage proto docker-build compose-up compose-down compose-validate benchmark-gate core-performance-gate distributed-drill phase0-exit-gate phase1-gate community-product-gate community-exit-gate k8s-gate pages-gate tooling-gate edition-modules-gate edition-build-matrix content-curation-gate release-readiness mixed-benchmark backup-restore-benchmark loadgen loadgen-mixed release-assets

test:
	go test ./...

vet:
	go vet ./...

tidy:
	go mod tidy

build:
	go build -o $(APP_NAME) ./cmd/server

run:
	go run ./cmd/server

bench:
	go test ./internal/core -bench . -benchmem

coverage:
	go run ./tools/checkcoverage

proto:
	buf generate --template buf.gen.yaml --path api/proto/service.proto

docker-build:
	docker build -t $(APP_NAME):latest .

compose-up:
	docker compose up --build

compose-down:
	docker compose down

compose-validate:
	bash scripts/validate-observability.sh

benchmark-gate:
	bash scripts/benchmark-regression-gate.sh

core-performance-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-core-performance.ps1

distributed-drill:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-distributed-drill.ps1

phase0-exit-gate:
	python scripts/check-phase0-exit.py --output build/phase0/exit-status.json

phase1-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-phase1.ps1

community-product-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-community-product.ps1

community-exit-gate:
	python scripts/check-community-exit.py --output build/community-product/exit-status.json

k8s-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-kubernetes.ps1

pages-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-pages.ps1

tooling-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-release-tooling.ps1

edition-modules-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-edition-modules.ps1

edition-build-matrix:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check-edition-build-matrix.ps1

content-curation-gate:
	python scripts/check-content-curation.py

release-readiness:
	python scripts/check-release-readiness.py

security-gate:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-security-gate.ps1

mixed-benchmark:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-mixed-ingest-search-benchmark.ps1

backup-restore-benchmark:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run-backup-restore-benchmark.ps1

loadgen:
	go run ./tools/loadgen

loadgen-mixed:
	go run ./tools/loadgen --mixed --vectors 10000 --searches 5000 --dim 128 --batch-size 500 --concurrency 4 --k 10

release-assets:
	bash scripts/package-release.sh
