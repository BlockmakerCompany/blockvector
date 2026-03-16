# 🧊 BlockVector

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Language](https://img.shields.io/badge/Language-Go%20%2B%20Assembly-00ADD8?logo=go&logoColor=white)
![Platform](https://img.shields.io/badge/Architecture-x86__64%20%2F%20ARM64-red)
![Latency](https://img.shields.io/badge/Latency-Sub--7ms-green?logo=lightning)
![Company](https://img.shields.io/badge/Backed%20By-BlockMaker%20S.R.L.-black)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Images-blue?logo=docker&logoColor=white)](https://hub.docker.com/r/blockmakercompany/blockvector)

**BlockVector** is a high-performance, bare-metal vector search engine engineered for extreme low-latency environments. Built from the ground up by **Blockmaker Company** using Go and hand-optimized x86_64 Assembly (AVX2), it redefines how high-dimensional embeddings are processed at the edge.

While traditional vector databases rely on heavy abstractions and massive memory overhead, **BlockVector** talks directly to the silicon. It is designed for engineers who refuse to trade precision for performance.

> **Performance first:** Sub-7ms search latency for 100,000+ vectors of 1536 dimensions on standard consumer hardware. No fluff. Just raw compute.

---

## 📑 Table of Contents
* [🛰 Core Technologies](#-core-technologies)
* [⚡ Performance Metrics](#-performance-metrics)
* [🐳 Deployment & Quick Start](#-deployment--quick-start)
* [🔌 API Usage](#-api-usage)
* [⚙️ Configuration](#-configuration)
* [🧩 Integration & Dual-Use](#-integration--dual-use)
* [🔨 Building from Source](#-building-from-source)
* [🛠️ Internal Architecture](#-internal-architecture)
* [🚀 Roadmap](#-roadmap)

---

## 🛰 Core Technologies

* **Hand-Optimized AVX2 Engine:** Written in pure x86_64 Assembly. Leverages `VPMOVSXBW` and `VPMADDWD` to orchestrate 32-way parallel arithmetic per CPU cycle. This is not just fast; it is hardware-limited performance.
* **Int8 Scalar Quantization:** Ingests `float32` and crushes them into `int8` precision. This shatters the memory bandwidth bottleneck, reduces the footprint by 75%, and maximizes L3 cache residency without compromising semantic accuracy.
* **Instant-On Architecture (mmap):** Zero-copy dataset mapping via `syscall.Mmap`. Startup time is a constant **~70µs** regardless of the dataset size. The OS manages page faults, ensuring that memory is only utilized when accessed.
* **Lock-Free Multi-Core Scaling:** Distributes search workloads across the entire CPU topology. By avoiding mutexes and shared state during the scan phase, performance scales linearly with your physical core count.
* **Binary Heap Reduction:** Optimized Top-K retrieval using a Min-Heap data structure. Maintains $O(N \log K)$ complexity, ensuring the final filtering stage never bottlenecks the SIMD engine.
* **Minimalist API Surface:** A high-concurrency HTTP server built exclusively on the Go standard library. No external runtimes, no bloat, just a clean JSON interface for RAG and AI pipelines.

---

## ⚡ Performance Metrics

*Hardware: 11th Gen Intel® Core™ i5-1135G7 @ 2.40GHz*
*Dataset: 100,000 vectors × 1536 dimensions (OpenAI Standard)*

| Implementation | Data Type | Latency (Top-5) | Memory Efficiency |
| :--- | :--- | :--- | :--- |
| Standard Go (Single-threaded) | `float32` | ~169.0 ms | 614 MB (RSS) |
| Standard Go (Multi-threaded) | `int8` | ~107.0 ms | 153 MB (RSS) |
| **BlockVector (AVX2 + Parallel)** | **`int8`** | **~6.6 ms** | **153 MB (mmap)** |

### 🧠 Why is BlockVector so much faster?

* **SIMD Saturation (256-bit lanes):** While standard Go processes one vector element at a time (scalar), BlockVector uses **AVX2** to process **32 elements of 8-bit integers in a single CPU cycle**.
* **Fused Multiply-Add (FMA):** We reduce the number of CPU instructions by combining multiplication and addition into a single hardware step, minimizing pipeline stalls.
* **Zero-Copy Memory (mmap):** Standard databases copy data from the OS cache to the application's heap (causing GC pressure). BlockVector maps the binary dataset directly into the process's address space, achieving **zero-copy reads** and instant startup times.
* **Cache-Line Alignment:** Our dataset layout is optimized to fit perfectly into L1/L2 CPU caches, avoiding the "Memory Wall" that slows down standard implementations.

---

## 🐳 Deployment & Quick Start

BlockVector utilizes an optimized Docker deployment workflow. To ensure **Assembly** kernels are linked correctly and the **AVX2 engine** runs at 100% capacity, we use a multi-stage build process with native linking tools.

> [!IMPORTANT]
> All commands must be executed from the **project root**. The engine requires a processor with **AVX2** and **FMA** instruction set support.

### 1. Initialize the Vector Space
Before searching, you must ingest and quantize your data. The generator will create the memory-mapped binary artifact inside the `./data` folder on your host:

```bash
# The Makefile ensures the directory exists and runs the generator
make setup
```
> This will generate a `blockvector.bin` file of approximately 153MB (for 100k vectors).

### 2. Boot the Search Engine
Once the dataset is ready in `./data/blockvector.bin`, launch the high-concurrency search microservice:

```bash
# Starts the server in detached mode using Docker Compose
make run-server
```

### 3. Monitor & Performance Audit
Check the logs to verify that the architecture dispatcher has selected the **AVX2 engine** and that the `mmap` operation was successful:

```bash
# Direct access to the container logs
docker compose -f deployments/docker-compose.yaml logs -f server
```

---

### 🛠️ Deployment Troubleshooting

* **Architecture (Apple Silicon / ARM):** If you are on a Mac (M1/M2/M3), the build will force the `linux/amd64` platform. Ensure **Rosetta** emulation is enabled in Docker Desktop; otherwise, the x86_64 Assembly kernels will fail to execute.
* **Volume Permissions:** If the container fails to write to `./data`, ensure your host user has the necessary permissions:
  ```bash
  chmod -R 777 ./data
  ```
* **Hardware Fallback:** If the logs show `Architecture: generic`, your CPU does not support AVX2, and the engine is using the pure Go fallback (significantly slower).

---

### 🛠️ Troubleshooting de Despliegue

* **Arquitectura (Apple Silicon / ARM):** Si estás en un Mac (M1/M2/M3), el build forzará la plataforma `linux/amd64`. Asegúrate de tener activada la emulación de **Rosetta** en Docker Desktop; de lo contrario, los kernels de x86_64 Assembly no podrán ejecutarse.
* **Permisos de Volumen:** Si el contenedor falla al escribir en `./data`, verifica que el usuario de Docker tenga permisos en el host:
  ```bash
  chmod -R 777 ./data
  ```
* **Hardware:** Si los logs muestran `Architecture: generic`, significa que tu CPU no soporta AVX2 y el motor está usando el fallback de Go (mucho más lento).

---

## 🔌 API Usage

BlockVector exposes a minimalist REST API designed for high-throughput RAG pipelines. It accepts standard `float32` vectors, performing internal quantization and AVX2-accelerated scanning in a single step.

### Search Vectors (Top-K)
Perform a semantic search by sending a JSON query. The server automatically handles the conversion from float space to optimized integer space.

**Endpoint:** `POST /search`

**Example Request:**
```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.12, -0.05, 0.33, ..., 0.88],
    "top_k": 5
  }'
```

**Example Response:**
```json
{
  "results": [
    { "index": 420, "score": 12450 },
    { "index": 12, "score": 11890 },
    { "index": 777, "score": 10560 },
    { "index": 204, "score": 9840 },
    { "index": 55, "score": 8720 }
  ],
  "latency_ms": 6.62,
  "message": "Success"
}
```

> **Note:** The `score` represents the dot product of the quantized `int8` vectors. Since BlockVector preserves the angular relationship during quantization, higher scores indicate stronger semantic similarity.

---

## ⚙️ Configuration

BlockVector follows the **Twelve-Factor App** methodology, allowing you to tune the engine via environment variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BLOCKVECTOR_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARN`, `ERROR`) | `INFO` |
| `BLOCKVECTOR_PORT` | The port the HTTP server listens on | `:8080` |
| `BLOCKVECTOR_FILENAME` | Path to the memory-mapped binary dataset | `blockvector.bin` |
| `BLOCKVECTOR_DIMENSIONS` | Vector dimensionality (must match the dataset) | `1536` |

> **Note on Parallelism:** By default, BlockVector detects the number of physical CPU cores and spawns one worker per core to maximize SIMD throughput. You don't need to configure this manually.

---

## 🔨 Building from Source

## 🔨 Building & Testing

If you prefer to work outside of Docker or need to run raw performance benchmarks directly on your host machine, ensure you are in the project root and have **Go 1.25+** installed.

### Local Compilation
Builds both the Server and CLI binaries into the `./bin/` directory using the local Go toolchain:

```bash
# Compiles both artifacts for the local OS/Arch
make build-local
```

### Validation & Consistency
Run the test suite to ensure the **Assembly engine** is mathematically consistent with the Go fallback and correctly integrated:

```bash
# Verifies SIMD vs. Scalar results and core logic
make test
```

### Performance Benchmarking
Measure the raw nanosecond-level latency of the SIMD kernels and memory efficiency on your specific hardware:

```bash
# Runs micro-benchmarks for quantization and dot product
make bench
```

### Cleanup
Remove compiled binaries and generated data artifacts to reset the environment:

```bash
# Deletes ./bin and ./data/*.bin
make clean
```

---

## 🧩 Integration & Dual-Use

BlockVector is architected to be used either as a standalone microservice or as a high-performance library embedded directly into your application.

### 1. As a Go Library
Since BlockVector is native Go, you can import the core engine without any HTTP overhead:

```go
import "github.com/BlockmakerCompany/blockvector"

// Use the engine directly in your code
results := blockvector.LinearScanInt8TopK(query, dataset, 1536, 5)
```

### 2. As a C-Shared Library (C, Rust, Python)
You can compile BlockVector into a shared object (`.so` or `.dll`) to call it from other languages via FFI (Foreign Function Interface) with near-zero overhead.

**Build the shared library:**
```bash
go build -o libblockvector.so -buildmode=c-shared ./lib/export.go
```

#### 🦀 Rust Integration
```rust
#[link(name = "blockvector")]
extern "C" {
    fn LinearScanInt8TopK(query: *const i8, dataset: *const i8, dim: i32, k: i32) -> *mut Result;
}
```

#### 🐍 Python Integration (Direct FFI)
While the REST API is recommended for most Python use cases, you can use `ctypes` for maximum performance:
```python
import ctypes
lib = ctypes.CDLL("./libblockvector.so")
# Call the internal SIMD functions directly
```

### 3. As a Microservice (Standard)
The recommended way for most distributed systems. Use the Docker image and communicate via the JSON REST API (as shown in the [API Usage](#-api-usage) section).

---

## 🏗️ Internal Architecture

BlockVector is engineered to bypass traditional bottlenecks in high-dimensional search by optimizing every layer of the software-hardware interface:

### 1. Ingestion Pipeline
Standard `float32` arrays undergo **Max-Absolute Normalization**, scaling them into the `int8` range while preserving semantic directionality. This reduces the memory footprint by **4x** without losing significant search accuracy, allowing more data to reside in CPU cache.

### 2. Binary Storage & Persistence
Data is persisted in a flat, contiguous binary format (`blockvector.bin`). This layout ensures **sequential memory access**, which is highly friendly to CPU prefetchers and minimizes expensive cache misses during large-scale scans.

### 3. Zero-Copy Loading (mmap)
The engine leverages `syscall.Mmap` to map the dataset directly into the process's virtual address space:
* **Zero GC Pressure:** Since data lives outside the Go heap, the Garbage Collector never has to scan it, preventing "Stop the World" pauses.
* **Lazy Loading:** The OS manages memory lazily via page faults, allowing 100GB+ datasets to be "loaded" instantly without physical RAM allocation until accessed.

### 4. High-Performance Execution Engine
* **Hybrid Core:** A lightweight Go wrapper handles JSON orchestration, while heavy arithmetic is dispatched to the **AVX2 SIMD** core.
* **Assembly SIMD Kernels:** We process **32-way parallel 8-bit arithmetic** per cycle using hand-rolled assembly, saturating the CPU's execution ports.
* **Lock-Free Parallelism:** Workload is partitioned across physical cores. Each worker maintains a **local Min-Heap** to track Top-K matches independently, eliminating lock contention.
* **Final Reduction:** A global merge step consolidates local results, returning indices and scores with sub-millisecond tail latency.

---

## 🚀 Roadmap

BlockVector is an evolving engine. Our development path focuses on expanding hardware saturation and adding high-level indexing without sacrificing our "bare-metal" philosophy.

- [x] **ARM64 (NEON) Native Kernels:** Ported the SIMD logic to ARM64 for peak performance on Apple Silicon and AWS Graviton.
- [ ] **AVX-512 Support:** Developing 512-bit wide SIMD kernels to double the throughput on high-end Intel Xeon and AMD EPYC servers.
- [ ] **HNSW Graph Indexing:** Implementing a Hierarchical Navigable Small World (HNSW) layer over the mmap'ed raw vectors.
- [ ] **Official Language Wrappers:** Developing high-level, idiomatic packages for **Python (PyO3)** and **Rust**.
- [ ] **gRPC & Protobuf Support:** Adding a high-performance binary interface for distributed microservices.
- [ ] **NUMA-Aware Scaling:** Optimizing memory bandwidth distribution for multi-socket server hardware.
- [ ] **Live Hot-Reloading:** Real-time dataset swapping without service interruption.

---

---

## 🤝 Contact & Collaboration

This project is a testament to the power of low-level engineering and the "Zero-Dependency" philosophy. If you are interested in high-performance systems, operating system internals, or just want to discuss why Assembly is still relevant in the era of Cloud Native, let's connect!

**Fernando E. Mancuso** *Head of Engineering at Blockmaker S.R.L.*

* **LinkedIn**: [Fernando Ezequiel Mancuso](https://www.linkedin.com/in/fernando-ezequiel-mancuso-54a2737/)
* **Email**: [fernando.mancuso@blockmaker.net](mailto:fernando.mancuso@blockmaker.net)
* **GitHub**: [@fermancuso-blockmaker](https://github.com/fermancuso-blockmaker)

---

> "The best way to understand how a computer works is to stop asking the operating system for permission and start giving it orders."

---

## 🏢 Backed by BlockMaker S.R.L.

**BlockVector** was engineered from scratch by the engineering team at **BlockMaker S.R.L.**, led by **Fernando Ezequiel Mancuso** (Head of Engineering).

At BlockMaker, we believe in deep tech, zero-dependency architectures, and pushing the absolute limits of hardware efficiency. We are actively encouraging the global engineering community to fork, benchmark, and contribute to this project.

If you love low-level systems engineering and uncompromising performance, feel free to reach out at [fernando.mancuso@blockmaker.net](mailto:fernando.mancuso@blockmaker.net).