package handlers

import (
	"encoding/json"
	"math"
	"math/rand"
	"net/http"
)

// WordEmbedding represents a word positioned in 2D semantic space with its full vector.
type WordEmbedding struct {
	Word     string    `json:"word"`
	X        float64   `json:"x"`
	Y        float64   `json:"y"`
	Category string    `json:"category"`
	Vector   []float64 `json:"vector,omitempty"`
}

// Semantic categories with representative words.
// Each category clusters together in embedding space.
var embeddingCategories = map[string][]string{
	"animals":  {"cat", "dog", "horse", "bird", "fish", "tiger", "lion", "elephant", "whale", "dolphin", "eagle", "owl", "snake", "bear", "wolf", "rabbit", "deer", "fox", "monkey", "penguin"},
	"colors":   {"red", "blue", "green", "yellow", "orange", "purple", "pink", "black", "white", "brown", "gray", "gold", "silver", "cyan", "magenta"},
	"emotions": {"happy", "sad", "angry", "fear", "love", "hate", "joy", "grief", "hope", "despair", "calm", "anxious", "proud", "shame", "surprise", "disgust", "trust", "envy"},
	"food":     {"apple", "bread", "cheese", "rice", "meat", "fish", "cake", "soup", "salad", "pizza", "pasta", "sushi", "curry", "steak", "burger", "taco"},
	"nature":   {"tree", "flower", "mountain", "river", "ocean", "forest", "desert", "island", "lake", "valley", "cloud", "rain", "snow", "wind", "sun", "moon", "star"},
	"body":     {"head", "hand", "eye", "heart", "brain", "arm", "leg", "foot", "face", "mouth", "ear", "nose", "finger", "bone", "blood", "skin"},
	"tech":     {"computer", "software", "algorithm", "data", "network", "code", "program", "server", "database", "internet", "robot", "model", "neural", "matrix", "vector", "tensor"},
	"actions":  {"run", "walk", "jump", "fly", "swim", "climb", "read", "write", "speak", "think", "learn", "teach", "build", "create", "destroy", "grow"},
	"size":     {"big", "small", "huge", "tiny", "large", "little", "giant", "miniature", "enormous", "microscopic"},
	"time":     {"today", "tomorrow", "yesterday", "morning", "evening", "night", "dawn", "dusk", "noon", "midnight", "hour", "minute", "second", "year", "month", "week"},
}

// precomputedEmbeddings holds all word embeddings, initialized at startup.
var precomputedEmbeddings []WordEmbedding

func init() {
	rng := rand.New(rand.NewSource(42))

	// Anchor positions for each category cluster in 2D space.
	categoryAnchors := map[string][2]float64{
		"animals":  {-60, -40},
		"colors":   {50, -50},
		"emotions": {-50, 50},
		"food":     {60, 30},
		"nature":   {-30, -70},
		"body":     {30, 60},
		"tech":     {70, -20},
		"actions":  {-70, 10},
		"size":     {0, 70},
		"time":     {0, -60},
	}

	for cat, words := range embeddingCategories {
		anchor := categoryAnchors[cat]
		for _, word := range words {
			x := anchor[0] + (rng.Float64()-0.5)*30
			y := anchor[1] + (rng.Float64()-0.5)*30

			// Generate a 32-dimensional embedding vector.
			vec := make([]float64, 32)
			for i := range vec {
				vec[i] = rng.NormFloat64() * 0.3
			}
			// Encode category signal into the first few dimensions.
			vec[0] = float64(len(cat)) * 0.1
			vec[1] = x * 0.01
			vec[2] = y * 0.01

			precomputedEmbeddings = append(precomputedEmbeddings, WordEmbedding{
				Word: word, X: x, Y: y, Category: cat, Vector: vec,
			})
		}
	}
}

// CosineSimilarity computes the cosine similarity between two vectors.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// HandleEmbeddings handles GET /api/embeddings?word1=...&word2=...
// Returns all embeddings (without vectors to stay under 32KB tunnel frame limit)
// and optionally computes similarity between two specified words.
func HandleEmbeddings(w http.ResponseWriter, r *http.Request) {
	word1 := r.URL.Query().Get("word1")
	word2 := r.URL.Query().Get("word2")

	type LightEmbedding struct {
		Word     string    `json:"word"`
		X        float64   `json:"x"`
		Y        float64   `json:"y"`
		Category string    `json:"category"`
		Vector   []float64 `json:"vector,omitempty"`
	}

	lightEmbeddings := make([]LightEmbedding, len(precomputedEmbeddings))
	for i, e := range precomputedEmbeddings {
		lightEmbeddings[i] = LightEmbedding{
			Word: e.Word, X: e.X, Y: e.Y, Category: e.Category,
		}
		// Only include the full vector for specifically requested words.
		if e.Word == word1 || e.Word == word2 {
			lightEmbeddings[i].Vector = e.Vector
		}
	}

	response := map[string]interface{}{
		"embeddings": lightEmbeddings,
	}

	if word1 != "" && word2 != "" {
		var vec1, vec2 []float64
		for _, e := range precomputedEmbeddings {
			if e.Word == word1 {
				vec1 = e.Vector
			}
			if e.Word == word2 {
				vec2 = e.Vector
			}
		}
		if vec1 != nil && vec2 != nil {
			response["similarity"] = CosineSimilarity(vec1, vec2)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
