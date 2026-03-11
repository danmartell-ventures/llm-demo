# LLM Demo — How Language Models Work

An interactive web app that walks you through the internals of a large language model, step by step. Type a phrase and watch it flow through the same pipeline that powers ChatGPT, Claude, and Gemini.

![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go&logoColor=white)

## What You'll Learn

The demo breaks down the five core stages of how an LLM processes text:

1. **Tokenization** — How text gets split into subword pieces using BPE (Byte Pair Encoding)
2. **Embeddings** — How tokens become vectors in semantic space, where similar words cluster together
3. **Attention** — How each token decides which other tokens to focus on (the "magic" of transformers)
4. **Forward Pass** — The full layer-by-layer journey: embedding → positional encoding → attention → FFN → layer norm → output
5. **Next-Token Prediction** — How the model picks the next word using temperature and top-p sampling

Each step has a **simple mode** (plain English) and a **technical mode** (with the math).

## Quick Start

```bash
# Clone
git clone https://github.com/danmartell-ventures/llm-demo.git
cd llm-demo

# Build and run
go build -o llm-demo .
./llm-demo
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

## Project Structure

```
├── main.go                  # Entry point — server setup and routing
├── handlers/
│   ├── tokenizer.go         # BPE tokenization with vocabulary lookup
│   ├── embeddings.go        # Word embeddings with 2D projection and cosine similarity
│   ├── attention.go         # Multi-head attention matrix generation
│   ├── predict.go           # Next-token prediction with temperature/top-p
│   ├── forward.go           # Full forward pass visualization (8 stages)
│   └── math.go              # Shared math utilities (softmax)
├── static/
│   ├── index.html           # Single-page app shell
│   ├── style.css            # Dark theme UI
│   ├── app.js               # Core application logic and step navigation
│   └── visualizations.js    # D3-style canvas visualizations
├── tunnel-keepalive.sh      # Auto-restart script for apex-tunnel
├── go.mod
└── README.md
```

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /api/tokenize?text=...` | Tokenize text into BPE subwords |
| `GET /api/embeddings?word1=...&word2=...` | Get embedding map + cosine similarity |
| `GET /api/attention?text=...` | Generate multi-head attention matrices |
| `GET /api/predict?text=...&temperature=1.0&top_p=0.9` | Next-token predictions |
| `GET /api/forward-pass?text=...` | Full forward pass with intermediate representations |

## How It Works

This isn't running a real neural network — it uses simplified models that produce realistic outputs to teach the concepts:

- **Tokenizer**: Real BPE-style greedy longest-match against a curated vocabulary
- **Embeddings**: Precomputed 32-dimensional vectors with semantic clustering (animals near animals, emotions near emotions)
- **Attention**: Four distinct head patterns (local, global, positional, content-based) with proper softmax normalization
- **Prediction**: Bigram language model with temperature scaling and nucleus sampling
- **Forward Pass**: Correct positional encodings (sinusoidal), layer normalization, and FFN transformations

The goal is education, not production inference. Every number you see is computed the same way a real transformer does it — just at a smaller scale.

## Tech Stack

- **Backend**: Go with embedded static files (`embed.FS`)
- **Frontend**: Vanilla HTML/CSS/JS — no frameworks, no build step
- **Deployment**: Single binary, zero dependencies

## License

MIT
