# ==========================================
# STAGE 1: Builder
# ==========================================
FROM golang:1.25-alpine AS builder

# Necesitamos build-base (gcc, libc-dev, etc) para procesar el Assembly correctamente
RUN apk add --no-cache git build-base

WORKDIR /app

# Al estar dentro de /deployments, subimos un nivel para el contexto
COPY go.mod ./
# COPY go.sum ./

RUN go mod download
COPY . .

# IMPORTANTE: CGO_ENABLED=1 para asegurar que el linker maneje bien los objetos de Assembly
# GOARCH=amd64 es vital para los kernels AVX2
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /bin/blockvector-cli ./cmd/blockvector-cli/main.go
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /bin/blockvector-server ./cmd/blockvector-server/main.go

# ==========================================
# STAGE 2: Runtime
# ==========================================
FROM alpine:latest
# Al usar CGO_ENABLED=1, necesitamos las librerías de compatibilidad de libc
RUN apk --no-cache add ca-certificates libc6-compat

WORKDIR /app

COPY --from=builder /bin/blockvector-cli /app/blockvector-cli
COPY --from=builder /bin/blockvector-server /app/blockvector-server

EXPOSE 8080

CMD ["/app/blockvector-server"]