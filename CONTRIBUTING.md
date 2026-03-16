# Contributing to BlockVector 🧊

First off, thank you for considering contributing to **BlockVector**! We are building a bare-metal engine where every CPU cycle counts.

## 🏗️ Our Engineering Philosophy

1. **Performance First:** If a change introduces latency without a massive feature trade-off, it won't be merged.
2. **Mathematical Parity:** Any Assembly kernel (**AVX2**, **NEON**, **AVX-512**) must produce the exact same results as its Go-native reference.
3. **Zero-Dependency:** We avoid external libraries. If you need a utility, build it using the Go standard library or raw Assembly.

## 🛠️ Getting Started

### Prerequisites
* **Go 1.25+**
* **Docker & Docker Compose**
* Hardware with **AVX2** (Intel/AMD) or **NEON** (ARM64) support.

### Local Setup
1. Fork the repository and clone it.
2. Initialize the environment:
   ```bash
   make setup
   ```
3. Run the tests to ensure your baseline is stable:
   ```bash
   make test
   ```

## 🎯 Areas for Contribution

We are currently prioritizing the following areas from our [Roadmap](./README.md#🚀-roadmap):
* **AVX-512 Kernels:** Implementation of 512-bit wide SIMD logic.
* **Python/Rust Wrappers:** Improving the FFI layer for wider adoption.
* **HNSW Implementation:** Researching and implementing a zero-copy graph index.

## 📜 Coding Guidelines

### Go Code
* Follow standard `gofmt` and `go vet` rules.
* Keep the critical path (`pkg/blockvector/`) free of allocations to avoid GC pressure.

### Assembly Code
* **No Bloat:** Do not submit compiler-generated assembly. We only accept hand-optimized, human-readable `.s` files.
* **Comments:** Document register usage (e.g., `// V0: Query Vector`, `// V1: Data Chunk`).
* **Safety:** Always handle bounds checking at the logic entry point before entering the Assembly kernel.

## 🧪 Testing Requirements
All Pull Requests must pass the following check:
```bash
# Ensure mathematical consistency across all engines
make test

# Ensure no performance regressions
make bench
```
*If you add a new kernel, you must provide a corresponding benchmark in the test suite.*

## 📬 Submitting a PR
1. Create a feature branch (`git checkout -b feature/amazing-simd`).
2. Commit your changes with descriptive messages.
3. Push to the branch and open a Pull Request.
4. Ensure your PR description includes the performance impact (output of `make bench`).

---
**Backed by BlockMaker S.R.L.** *“In silicon we trust.”*