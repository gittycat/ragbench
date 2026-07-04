# RAG Desktop App — Architecture Recommendation (April 2026)

## Context

Cross-platform RAG application that:
- Indexes a local folder tree into a vector database
- Performs retrieval-augmented generation on user questions
- Supports local and online LLMs (configurable)
- Ships as a native desktop app on macOS and Windows

Current stack: ChromaDB + Python RAG backend + SvelteKit frontend (Docker-based).

---

## Core Principle

Replace the Python/Docker layer entirely. Both Apple and Microsoft now ship first-class on-device RAG primitives that make a Python runtime unnecessary and enable proper native app distribution.

---

## Option 1 — macOS Native (best macOS experience)

### Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | **SwiftUI** | Native macOS look and feel |
| RAG orchestration | **Swift + async/await** | Maps cleanly onto the chunk → embed → store / query → embed → search → prompt → stream pipeline |
| Embeddings | **`NLContextualEmbedding`** | Built-in since macOS 14, zero external dependencies, 512-dim BERT, sandboxed |
| Vector store | **VecturaKit** | Swift-native, pluggable backends (NL, MLX, BERT variants), most turnkey option |
| Local LLM inference | **MLX Swift v0.31.3** | Apple's own ML framework, Apple Silicon GPU/Neural Engine accelerated, supports Llama/Mistral/Phi/Qwen |
| Online LLMs | `URLSession` + `AsyncStream` | Direct API calls, streaming to UI |
| Document parsing | `PDFKit`, `NaturalLanguage`, `AttributedString` | In-box frameworks |
| Folder watching | `FSEvents` | Native macOS file system events |

### Distribution

Standard `.app` bundle. No Python runtime, no subprocess, no dynamic code execution.
- App Store compatible
- MLX models downloaded on first run to `~/Library/Application Support/`
- VecturaKit / sqlite-vec store is a single `.db` file

### What to rewrite vs keep

- **Keep**: SwiftUI frontend (extend it)
- **Port**: RAG orchestration logic (~200–400 lines Swift, algorithms unchanged)
- **Drop**: ChromaDB, Python runtime, SvelteKit frontend, inter-process communication

### Hard parts

- Document parsing beyond PDF (HTML, Word, Markdown) — requires third-party parsers
- MLX model management UX (download, cache, switch models)
- Embedding model parity if migrating an existing index

---

## Option 2 — Windows Native (best Windows experience)

Targets Windows 11 24H2+ on recent hardware (Copilot+ PCs, NVIDIA, AMD).

### Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | **WinUI 3 / Windows App SDK 1.8** | Fluent Design, materially more mature than in 2025, ships with Windows ML |
| Language | **C# / .NET 9** | |
| Inference | **Windows ML** (GA Sept 2025) | Unified on-device inference, replaces DirectML as primary layer |
| Local LLM | **Phi Silica** | In-box on Copilot+ PCs, NPU-optimized, zero download needed |
| Broader model support | **ONNX Runtime Generate() API** | For non-Phi models, full LLM loop control |
| Vector store + retrieval | **Windows Copilot Library Vector Embeddings API** | In-box on Windows 11 24H2+, handles RAG retrieval natively |
| Model management | **Foundry Local** | Microsoft's local model manager, 40+ model catalog |
| Online LLMs | `HttpClient` (.NET) | |
| Document parsing | **PdfPig**, `ML.NET` tokenizers | |

### Hardware acceleration

Windows ML targets all hardware automatically:
- Qualcomm Snapdragon X (NPU via QNN)
- NVIDIA (CUDA)
- AMD (DirectML)
- Intel (CPU/iGPU)

### Notes

Microsoft now ships a full in-box RAG stack on Windows 11 24H2+, comparable to Apple's NLContextualEmbedding approach but more complete. Windows App SDK 2.0 (targeting .NET 10) is in preview.

---

## Option 3 — Cross-Platform (one codebase, macOS + Windows)

Best choice for a small team that needs both platforms without maintaining two native codebases.

### Stack

| Layer | Technology | Version | Notes |
|---|---|---|---|
| Frontend | **SvelteKit** | current | Reuse existing frontend |
| App shell | **Tauri 2** | v2.10.3 | ~10–20 MB bundles vs ~120 MB Electron, stable since late 2024 |
| Language | **Rust** | | Backend logic |
| Local LLM inference | **llama.cpp** | b8763 | Metal (macOS), CUDA (NVIDIA), HIP (AMD), Vulkan (cross-platform) |
| Embeddings | **fastembed-rs** | v5.12.0 | ONNX-based, sync API, dense + sparse + quantized models |
| Vector store | **sqlite-vec** | v0.1.9 | SQLite extension, cross-platform, single `.db` file |
| Online LLMs | `reqwest` (Rust) | | |

### Distribution

- macOS: `.app` bundle, `.dmg` installer
- Windows: `.exe` installer, `.msi`
- Single CI pipeline for both

### Trade-offs vs native options

| | Cross-platform (Tauri) | Native (Swift / C#) |
|---|---|---|
| UI conventions | Web-in-shell feel | True platform native |
| Codebase | One | Two |
| Platform API access | Limited (Tauri plugins) | Full |
| Model management UX | Custom | Platform-integrated (MLX / Foundry) |
| App Store | No | Yes |
| Effort | Lower | Higher |

### Viability

Confirmed by multiple shipped production RAG desktop apps as of 2026, including ElectricSQL's reference architecture (Tauri + llama.cpp + fastembed). Main engineering overhead is model management UX, not framework maturity.

---

## Decision Matrix

| Priority | Recommended option |
|---|---|
| Best macOS experience, App Store viable | **Option 1 — macOS Native** |
| Best Windows experience on Copilot+ PCs | **Option 2 — Windows Native** |
| Small team, both platforms, ship faster | **Option 3 — Tauri cross-platform** |
| macOS primary, Windows secondary | Build Option 1 first, add Tauri wrapper for Windows later |
| Both platforms at equal quality | Two native codebases (significant investment) |

---

## What to Avoid

- **Bundled Python runtime**: complex code-signing, App Store rejection risk, ~80 MB overhead, slow startup
- **Docker/container-based distribution**: not a macOS/Windows app pattern, requires Docker Desktop installed
- **Electron**: viable but ~120 MB overhead; Tauri is the modern replacement
- **ChromaDB in production app**: designed for server/dev use, not embedded app distribution

---

*Last updated: April 2026. Sources: live web searches conducted April 12, 2026.*
