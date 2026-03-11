package handlers

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
)

// ForwardPassStep describes one stage of the transformer forward pass.
type ForwardPassStep struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Technical   string      `json:"technical"`
	Data        interface{} `json:"data"`
}

// HandleForwardPass handles GET /api/forward-pass?text=...
// Walks through each layer of a transformer, showing intermediate representations.
func HandleForwardPass(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	if text == "" {
		text = "The cat sat"
	}

	tokens := Tokenize(text)
	tokenTexts := make([]string, len(tokens))
	tokenIDs := make([]int, len(tokens))
	for i, t := range tokens {
		tokenTexts[i] = t.Text
		tokenIDs[i] = t.ID
	}

	rng := rand.New(rand.NewSource(42))
	dim := 8

	embeddings := generateEmbeddings(rng, len(tokens), dim)
	posEncodings := generatePositionalEncodings(len(tokens), dim)
	combined := addVectors(embeddings, posEncodings)
	attentionScores := generateAttentionScores(len(tokens))
	ffnOutput := applyFFN(rng, combined, dim)
	layerNormOutput := applyLayerNorm(ffnOutput, dim)
	topTokens, logits := generateOutputLogits(rng)

	steps := []ForwardPassStep{
		{
			Name:        "Input Text",
			Description: "The raw text you typed — this is where everything starts. The model can't read text directly, so it needs to convert it into numbers first.",
			Technical:   fmt.Sprintf("Raw string input: %q. The model's vocabulary maps subword units to integer indices.", text),
			Data:        map[string]interface{}{"text": text},
		},
		{
			Name:        "Tokenization",
			Description: "The text is broken into smaller pieces called tokens. Think of it like breaking a sentence into puzzle pieces — some are whole words, some are parts of words.",
			Technical:   "BPE (Byte Pair Encoding) splits text into subword tokens from a learned vocabulary. Each token maps to an integer ID. Vocabulary size typically 32K–100K tokens.",
			Data:        map[string]interface{}{"tokens": tokenTexts, "ids": tokenIDs},
		},
		{
			Name:        "Embedding Lookup",
			Description: "Each token gets converted into a list of numbers (a vector) that captures its meaning. Similar words get similar vectors — like plotting words on a map where nearby words mean similar things.",
			Technical:   fmt.Sprintf("Each token ID indexes into an embedding matrix E ∈ ℝ^(V×d_model). Here d_model=%d (real models use 768–12288). The embedding captures semantic and syntactic properties.", dim),
			Data:        map[string]interface{}{"tokens": tokenTexts, "embeddings": embeddings, "dim": dim},
		},
		{
			Name:        "Positional Encoding",
			Description: "The model needs to know WHERE each word is in the sentence. We add a unique position signal to each embedding — like giving every word a seat number.",
			Technical:   "PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). Added element-wise to embeddings. Allows the model to learn relative positions.",
			Data:        map[string]interface{}{"tokens": tokenTexts, "positional": posEncodings, "combined": combined},
		},
		{
			Name:        "Multi-Head Attention",
			Description: "This is the magic. Each word looks at every other word and decides how much to pay attention to it. 'The cat sat on the mat' — when processing 'sat', the model pays extra attention to 'cat' (who sat?) and 'mat' (where?).",
			Technical:   "Attention(Q,K,V) = softmax(QK^T / √d_k)V. Q, K, V are linear projections of the input. Multiple heads (h=8 typically) attend to different aspects. Each head has dimension d_k = d_model / h.",
			Data:        map[string]interface{}{"tokens": tokenTexts, "attention": attentionScores},
		},
		{
			Name:        "Feed-Forward Network",
			Description: "After attention, each token passes through a small neural network independently. This is where the model 'thinks' about what it learned from the attention step — like processing information after a meeting.",
			Technical:   "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂. Two linear transformations with ReLU activation. Inner dimension is typically 4× the model dimension (d_ff = 4 × d_model).",
			Data:        map[string]interface{}{"tokens": tokenTexts, "output": ffnOutput},
		},
		{
			Name:        "Layer Normalization",
			Description: "The numbers are rescaled to keep them stable — preventing any value from getting too big or too small. Like a volume knob that keeps the signal clean.",
			Technical:   "LayerNorm(x) = γ · (x − μ) / √(σ² + ε) + β, where μ and σ² are computed per-token across the feature dimension. γ, β are learned parameters.",
			Data:        map[string]interface{}{"tokens": tokenTexts, "normalized": layerNormOutput},
		},
		{
			Name:        "Output Probabilities",
			Description: "Finally, the model produces a probability for every possible next token. The token with the highest probability is the prediction — but with some randomness (temperature) to keep things creative.",
			Technical:   "The final hidden state is projected to vocabulary size via W_out ∈ ℝ^(d_model×V), then softmax converts logits to probabilities: P(token_i) = exp(z_i) / Σ_j exp(z_j).",
			Data:        map[string]interface{}{"topTokens": topTokens, "probabilities": logits},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{"steps": steps})
}

func generateEmbeddings(rng *rand.Rand, n, dim int) [][]float64 {
	embeddings := make([][]float64, n)
	for i := range embeddings {
		embeddings[i] = make([]float64, dim)
		for j := range embeddings[i] {
			embeddings[i][j] = math.Round((rng.Float64()*2-1)*100) / 100
		}
	}
	return embeddings
}

func generatePositionalEncodings(n, dim int) [][]float64 {
	pe := make([][]float64, n)
	for i := range pe {
		pe[i] = make([]float64, dim)
		for j := range pe[i] {
			if j%2 == 0 {
				pe[i][j] = math.Round(math.Sin(float64(i)/math.Pow(10000, float64(j)/float64(dim)))*100) / 100
			} else {
				pe[i][j] = math.Round(math.Cos(float64(i)/math.Pow(10000, float64(j-1)/float64(dim)))*100) / 100
			}
		}
	}
	return pe
}

func addVectors(a, b [][]float64) [][]float64 {
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[i]))
		for j := range a[i] {
			result[i][j] = math.Round((a[i][j]+b[i][j])*100) / 100
		}
	}
	return result
}

func generateAttentionScores(n int) [][]float64 {
	scores := make([][]float64, n)
	for i := range scores {
		scores[i] = make([]float64, n)
		for j := range scores[i] {
			dist := math.Abs(float64(i - j))
			scores[i][j] = math.Exp(-dist * 0.3)
		}
		SoftmaxRow(scores[i])
		for j := range scores[i] {
			scores[i][j] = math.Round(scores[i][j]*100) / 100
		}
	}
	return scores
}

func applyFFN(rng *rand.Rand, input [][]float64, dim int) [][]float64 {
	output := make([][]float64, len(input))
	for i := range input {
		output[i] = make([]float64, dim)
		for j := range output[i] {
			output[i][j] = math.Round((input[i][j]*0.8+rng.Float64()*0.4-0.2)*100) / 100
		}
	}
	return output
}

func applyLayerNorm(input [][]float64, dim int) [][]float64 {
	output := make([][]float64, len(input))
	for i := range input {
		mean := 0.0
		for _, v := range input[i] {
			mean += v
		}
		mean /= float64(dim)

		variance := 0.0
		for _, v := range input[i] {
			variance += (v - mean) * (v - mean)
		}
		variance /= float64(dim)
		std := math.Sqrt(variance + 1e-5)

		output[i] = make([]float64, dim)
		for j := range output[i] {
			output[i][j] = math.Round((input[i][j]-mean)/std*100) / 100
		}
	}
	return output
}

func generateOutputLogits(rng *rand.Rand) ([]string, []float64) {
	topTokens := []string{"cat", "dog", "mat", "hat", "on", "the", "and", "was", "in", "is"}
	logits := make([]float64, len(topTokens))
	for i := range logits {
		logits[i] = rng.Float64()*4 - 1
	}
	logits[0] = 3.2
	logits[4] = 2.8
	SoftmaxRow(logits)
	for i := range logits {
		logits[i] = math.Round(logits[i]*1000) / 1000
	}
	return topTokens, logits
}
