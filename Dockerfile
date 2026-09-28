FROM golang:1.26@sha256:3aff6657219a4d9c14e27fb1d8976c49c29fddb70ba835014f477e1c70636647 AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -mod=readonly \
    -trimpath \
    -o /out/lumenvec \
    ./cmd/server && \
    mkdir -p /out/data

FROM gcr.io/distroless/static-debian12:nonroot@sha256:f5b485ea962d9bd1186b2f6b3a061191539b905b82ec395de78cbfae51f20e35

LABEL org.opencontainers.image.title="LumenVec" \
      org.opencontainers.image.description="Vector database with HTTP and gRPC transports" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY --from=builder --chown=nonroot:nonroot /out/lumenvec /app/lumenvec
COPY --from=builder --chown=nonroot:nonroot /out/data /data

EXPOSE 19190
EXPOSE 19191

VOLUME ["/data"]

ENV VECTOR_DB_PROTOCOL=http \
    VECTOR_DB_PORT=19190 \
    VECTOR_DB_GRPC_PORT=19191 \
    VECTOR_DB_SNAPSHOT_PATH=/data/snapshot.json \
    VECTOR_DB_WAL_PATH=/data/wal.log \
    VECTOR_DB_VECTOR_PATH=/data/vectors

USER nonroot:nonroot

ENTRYPOINT ["/app/lumenvec"]
