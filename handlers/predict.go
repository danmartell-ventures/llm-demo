package handlers

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
)

// PredictionResult is a candidate next token with its probability.
type PredictionResult struct {
	Token string  `json:"token"`
	Prob  float64 `json:"prob"`
}

// ngramModel maps a context word to probable next words.
// This is a simplified bigram model — real LLMs use full transformer context.
var ngramModel = map[string]map[string]float64{
	"the":         {"cat": 0.12, "dog": 0.08, "model": 0.07, "world": 0.06, "data": 0.05, "best": 0.05, "first": 0.04, "new": 0.04, "old": 0.03, "big": 0.03, "end": 0.03, "same": 0.03, "most": 0.03, "next": 0.03, "other": 0.02},
	"a":           {"new": 0.1, "large": 0.08, "small": 0.07, "single": 0.06, "good": 0.05, "great": 0.05, "simple": 0.04, "deep": 0.04, "long": 0.03, "short": 0.03, "fast": 0.03, "big": 0.03, "few": 0.03, "very": 0.02, "bit": 0.02},
	"is":          {"a": 0.15, "the": 0.12, "not": 0.1, "an": 0.05, "very": 0.05, "used": 0.04, "one": 0.03, "called": 0.03, "known": 0.03, "based": 0.03, "also": 0.03, "that": 0.02, "how": 0.02, "what": 0.02, "it": 0.02},
	"i":           {"am": 0.12, "think": 0.1, "have": 0.09, "want": 0.07, "like": 0.06, "know": 0.06, "love": 0.05, "need": 0.05, "can": 0.04, "will": 0.04, "see": 0.03, "feel": 0.03, "believe": 0.03, "hope": 0.02, "wish": 0.02},
	"machine":     {"learning": 0.45, "translation": 0.1, "vision": 0.05, "intelligence": 0.04, "code": 0.03, "model": 0.03, "that": 0.02, "can": 0.02, "is": 0.02, "and": 0.02},
	"deep":        {"learning": 0.5, "neural": 0.08, "network": 0.06, "dive": 0.04, "understanding": 0.04, "analysis": 0.03, "inside": 0.02, "within": 0.02, "breath": 0.02, "thought": 0.02},
	"neural":      {"network": 0.45, "networks": 0.2, "net": 0.05, "architecture": 0.04, "model": 0.03, "machine": 0.02, "computation": 0.02, "processing": 0.02, "pathway": 0.02, "activity": 0.02},
	"language":    {"model": 0.3, "models": 0.15, "processing": 0.1, "understanding": 0.06, "generation": 0.05, "translation": 0.04, "learning": 0.03, "is": 0.02, "and": 0.02, "that": 0.02},
	"attention":   {"mechanism": 0.2, "is": 0.12, "head": 0.08, "heads": 0.07, "layer": 0.06, "weight": 0.05, "weights": 0.05, "matrix": 0.04, "score": 0.04, "pattern": 0.03},
	"transformer": {"model": 0.15, "architecture": 0.12, "is": 0.08, "based": 0.07, "network": 0.05, "layer": 0.05, "was": 0.04, "uses": 0.04, "has": 0.03, "can": 0.03},
	"cat":         {"sat": 0.15, "is": 0.1, "and": 0.08, "was": 0.07, "in": 0.06, "on": 0.05, "loves": 0.04, "jumped": 0.04, "ate": 0.03, "ran": 0.03},
	"to":          {"the": 0.12, "be": 0.1, "a": 0.07, "learn": 0.05, "make": 0.04, "do": 0.04, "get": 0.04, "have": 0.03, "use": 0.03, "find": 0.03, "see": 0.03, "go": 0.03, "know": 0.02, "say": 0.02, "take": 0.02},
	"we":          {"can": 0.12, "have": 0.1, "are": 0.09, "need": 0.07, "will": 0.06, "use": 0.05, "know": 0.04, "see": 0.04, "want": 0.03, "should": 0.03},
	"how":         {"to": 0.2, "the": 0.08, "does": 0.07, "do": 0.07, "it": 0.06, "can": 0.05, "much": 0.05, "many": 0.04, "about": 0.03, "well": 0.03},
	"what":        {"is": 0.2, "the": 0.1, "are": 0.08, "does": 0.06, "do": 0.05, "if": 0.04, "about": 0.04, "makes": 0.03, "happens": 0.03, "can": 0.03},
	"hello":       {"world": 0.25, "there": 0.15, "everyone": 0.08, "how": 0.07, "my": 0.05, "and": 0.04, "i": 0.04, "to": 0.03, "from": 0.02, "dear": 0.02},
	"artificial":  {"intelligence": 0.55, "neural": 0.06, "general": 0.04, "life": 0.03, "agent": 0.03, "system": 0.02, "network": 0.02, "mind": 0.02, "brain": 0.02, "world": 0.01},
}

// defaultDistribution is used when the last word isn't in our bigram model.
var defaultDistribution = map[string]float64{
	"the": 0.08, "a": 0.06, "is": 0.05, "and": 0.05, "to": 0.05,
	"of": 0.04, "in": 0.04, "that": 0.03, "it": 0.03, "for": 0.03,
	"was": 0.03, "on": 0.02, "with": 0.02, "as": 0.02, "are": 0.02,
}

// HandlePredict handles GET /api/predict?text=...&temperature=...&top_p=...
// Returns ranked next-token predictions with probabilities.
func HandlePredict(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	temperature := 1.0
	topP := 0.9

	if t := r.URL.Query().Get("temperature"); t != "" {
		fmt.Sscanf(t, "%f", &temperature)
	}
	if p := r.URL.Query().Get("top_p"); p != "" {
		fmt.Sscanf(p, "%f", &topP)
	}

	temperature = math.Max(0.1, math.Min(2.0, temperature))

	words := strings.Fields(strings.ToLower(text))
	var lastWord string
	if len(words) > 0 {
		lastWord = words[len(words)-1]
	}

	baseDist, ok := ngramModel[lastWord]
	if !ok {
		baseDist = defaultDistribution
	}

	// Apply temperature scaling to logits.
	type kv struct {
		k string
		v float64
	}
	var pairs []kv
	for k, v := range baseDist {
		pairs = append(pairs, kv{k, math.Log(v+1e-10) / temperature})
	}

	// Softmax over temperature-scaled logits.
	maxLogit := pairs[0].v
	for _, p := range pairs {
		if p.v > maxLogit {
			maxLogit = p.v
		}
	}
	sum := 0.0
	for i := range pairs {
		pairs[i].v = math.Exp(pairs[i].v - maxLogit)
		sum += pairs[i].v
	}
	for i := range pairs {
		pairs[i].v /= sum
	}

	sort.Slice(pairs, func(i, j int) bool { return pairs[i].v > pairs[j].v })

	// Apply nucleus (top-p) sampling.
	var results []PredictionResult
	cumProb := 0.0
	for _, p := range pairs {
		if cumProb >= topP && len(results) > 0 {
			break
		}
		results = append(results, PredictionResult{Token: p.k, Prob: p.v})
		cumProb += p.v
	}

	// Renormalize.
	totalProb := 0.0
	for _, r := range results {
		totalProb += r.Prob
	}
	for i := range results {
		results[i].Prob /= totalProb
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"predictions": results,
		"lastWord":    lastWord,
		"temperature": temperature,
		"topP":        topP,
	})
}
