package handlers

import (
	"encoding/json"
	"math"
	"math/rand"
	"net/http"
	"strings"
)

// HandleAttention handles GET /api/attention?text=...
// Returns multi-head attention matrices showing how tokens attend to each other.
func HandleAttention(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	if text == "" {
		text = "The cat sat on the mat because it was tired"
	}

	tokens := Tokenize(text)
	n := len(tokens)
	if n > 30 {
		tokens = tokens[:30]
		n = 30
	}

	tokenTexts := make([]string, n)
	for i, t := range tokens {
		tokenTexts[i] = t.Text
	}

	// Four attention heads, each with a different pattern.
	head1 := makeAttentionHead(n, "local", tokenTexts)
	head2 := makeAttentionHead(n, "global", tokenTexts)
	head3 := makeAttentionHead(n, "positional", tokenTexts)
	head4 := makeAttentionHead(n, "content", tokenTexts)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tokens": tokenTexts,
		"heads": map[string][][]float64{
			"local":      head1,
			"global":     head2,
			"positional": head3,
			"content":    head4,
		},
		"averaged": averageHeads(head1, head2, head3, head4),
	})
}

// makeAttentionHead generates a realistic attention matrix for a given pattern type.
func makeAttentionHead(n int, pattern string, tokens []string) [][]float64 {
	rng := rand.New(rand.NewSource(int64(n * len(pattern))))
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}

	switch pattern {
	case "local":
		// Nearby tokens attend strongly to each other.
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				dist := math.Abs(float64(i - j))
				matrix[i][j] = math.Exp(-dist * 0.5)
			}
		}
	case "global":
		// Function words ("the", "a", "is") receive broad attention.
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				matrix[i][j] = 0.05 + rng.Float64()*0.05
				t := strings.ToLower(strings.TrimPrefix(tokens[j], "Ġ"))
				if t == "the" || t == "a" || t == "is" || t == "was" || t == "." {
					matrix[i][j] = 0.3 + rng.Float64()*0.2
				}
			}
			matrix[i][i] += 0.2
		}
	case "positional":
		// Strong attention to the first few tokens (sentence start).
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if j <= 2 {
					matrix[i][j] = 0.3 + rng.Float64()*0.15
				} else {
					matrix[i][j] = 0.02 + rng.Float64()*0.05
				}
			}
			matrix[i][i] += 0.15
		}
	case "content":
		// Pronouns attend to their antecedents; identical tokens attend to each other.
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				ti := strings.ToLower(strings.TrimPrefix(tokens[i], "Ġ"))
				tj := strings.ToLower(strings.TrimPrefix(tokens[j], "Ġ"))
				matrix[i][j] = 0.03 + rng.Float64()*0.04

				if isPronoun(ti) && j < i {
					matrix[i][j] = 0.2 + rng.Float64()*0.15
				}
				if ti == tj && i != j {
					matrix[i][j] = 0.35 + rng.Float64()*0.15
				}
			}
			matrix[i][i] += 0.1
		}
	}

	for i := range matrix {
		SoftmaxRow(matrix[i])
	}
	return matrix
}

func isPronoun(word string) bool {
	pronouns := map[string]bool{
		"it": true, "he": true, "she": true, "they": true,
		"its": true, "his": true, "her": true, "their": true,
	}
	return pronouns[word]
}

// averageHeads computes the element-wise average of multiple attention matrices.
func averageHeads(heads ...[][]float64) [][]float64 {
	if len(heads) == 0 {
		return nil
	}
	n := len(heads[0])
	avg := make([][]float64, n)
	for i := range avg {
		avg[i] = make([]float64, n)
		for j := range avg[i] {
			for _, h := range heads {
				avg[i][j] += h[i][j]
			}
			avg[i][j] /= float64(len(heads))
		}
	}
	return avg
}
