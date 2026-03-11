package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
)

//go:embed static/*
var staticFiles embed.FS

func main() {
	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}

	http.Handle("/", http.FileServer(http.FS(staticFS)))
	http.HandleFunc("/api/tokenize", handleTokenize)
	http.HandleFunc("/api/embeddings", handleEmbeddings)
	http.HandleFunc("/api/attention", handleAttention)
	http.HandleFunc("/api/predict", handlePredict)
	http.HandleFunc("/api/forward-pass", handleForwardPass)

	fmt.Println("🧠 LLM Demo running at http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

// ─── TOKENIZER ──────────────────────────────────────────────

var bpeVocab = map[string]int{
	"Ġ": 256,
	"The": 500, "A": 501, "I": 502, "It": 503, "In": 504, "This": 505, "We": 506,
	"He": 507, "She": 508, "They": 509, "How": 510, "What": 511, "When": 512,
	"Hello": 513, "My": 514, "You": 515, "Is": 516, "Are": 517, "Do": 518,
	"the": 1000, "Ġthe": 1001, "Ġa": 1002, "Ġis": 1003,
	"Ġof": 1004, "Ġand": 1005, "Ġto": 1006, "Ġin": 1007, "Ġthat": 1008,
	"Ġit": 1009, "Ġfor": 1010, "Ġwas": 1011, "Ġon": 1012, "Ġare": 1013,
	"Ġwith": 1014, "Ġas": 1015, "Ġhis": 1016, "Ġthey": 1017, "Ġbe": 1018,
	"Ġat": 1019, "Ġone": 1020, "Ġhave": 1021, "Ġthis": 1022, "Ġfrom": 1023,
	"Ġhad": 1024, "Ġnot": 1025, "Ġbut": 1026, "Ġwhat": 1027, "Ġall": 1028,
	"Ġwere": 1029, "Ġwe": 1030, "Ġwhen": 1031, "Ġyour": 1032, "Ġcan": 1033,
	"Ġthere": 1034, "Ġuse": 1035, "Ġan": 1036, "Ġeach": 1037,
	"Ġshe": 1038, "Ġwhich": 1039, "Ġdo": 1040, "Ġhow": 1041,
	"Ġwill": 1042, "Ġup": 1043, "Ġother": 1044, "Ġabout": 1045,
	"Ġout": 1046, "Ġmany": 1047, "Ġthen": 1048, "Ġthem": 1049,
	"Ġthese": 1050, "Ġso": 1051, "Ġsome": 1052, "Ġher": 1053,
	"Ġwould": 1054, "Ġmake": 1055, "Ġlike": 1056, "Ġhas": 1057,
	"Ġhim": 1058, "Ġinto": 1059, "Ġtime": 1060, "Ġlook": 1061,
	"Ġmore": 1062, "Ġgo": 1063, "Ġcome": 1064, "Ġcould": 1065,
	"Ġno": 1066, "Ġmy": 1067, "Ġthan": 1068, "Ġbeen": 1069,
	"Ġcall": 1070, "Ġwho": 1071, "Ġits": 1072, "Ġnow": 1073,
	"Ġfind": 1074, "Ġlong": 1075, "Ġdown": 1076, "Ġday": 1077,
	"Ġdid": 1078, "Ġget": 1079, "Ġmade": 1080, "Ġmay": 1081,
	"Ġpart": 1082, "Ġover": 1083, "Ġnew": 1084, "Ġafter": 1085,
	"ing": 2000, "tion": 2001, "ed": 2002, "er": 2003, "es": 2004,
	"en": 2005, "al": 2006, "re": 2007, "on": 2008, "ly": 2009,
	"an": 2010, "or": 2011, "le": 2012, "se": 2013, "ent": 2014,
	"ar": 2015, "ment": 2016, "at": 2017, "ous": 2018, "ness": 2019,
	"able": 2020, "ful": 2021, "ive": 2022, "ight": 2023, "ure": 2024,
	"Ġlang": 3000, "uage": 3001, "Ġmodel": 3002, "Ġlearn": 3003,
	"Ġneural": 3004, "Ġnetwork": 3005, "Ġtrans": 3006, "former": 3007,
	"Ġattention": 3008, "Ġtoken": 3009, "Ġembed": 3010, "ding": 3011,
	"Ġartificial": 3012, "Ġintelligence": 3013, "Ġmachine": 3014,
	"Ġdeep": 3015, "Ġdata": 3016, "Ġtrain": 3017,
	"Ġhello": 4000, "Ġworld": 4001, "Ġhappy": 4002, "Ġsad": 4003,
	"Ġlove": 4004, "Ġhate": 4005, "Ġgood": 4006, "Ġbad": 4007,
	"Ġbeautiful": 4008, "Ġugly": 4009, "Ġfast": 4010, "Ġslow": 4011,
	"Ġbig": 4012, "Ġsmall": 4013, "Ġcat": 4014, "Ġdog": 4015,
	"Ġhorse": 4016, "Ġbird": 4017, "Ġfish": 4018, "Ġtree": 4019,
	"Ġflower": 4020, "Ġwater": 4021, "Ġfire": 4022, "Ġearth": 4023,
	"Ġsky": 4024, "Ġsun": 4025, "Ġmoon": 4026, "Ġstar": 4027,
	"Ġhuman": 4028, "Ġlife": 4029, "Ġwork": 4030, "Ġplay": 4031,
	"Ġread": 4032, "Ġwrite": 4033, "Ġspeak": 4034, "Ġthink": 4035,
	"Ġknow": 4036, "Ġfeel": 4037, "Ġsee": 4038, "Ġhear": 4039,
	"Ġrun": 4040, "Ġwalk": 4041, "Ġeat": 4042, "Ġsleep": 4043,
}

type Token struct {
	Text string `json:"text"`
	ID   int    `json:"id"`
}

func tokenize(text string) []Token {
	var tokens []Token
	words := splitIntoWords(text)

	for _, word := range words {
		wordTokens := tokenizeWord(word)
		tokens = append(tokens, wordTokens...)
	}
	return tokens
}

func splitIntoWords(text string) []string {
	var words []string
	fields := strings.Fields(text)
	for i, f := range fields {
		if i == 0 {
			words = append(words, f)
		} else {
			words = append(words, "Ġ"+f)
		}
	}
	return words
}

func tokenizeWord(word string) []Token {
	// Try whole word first
	if id, ok := bpeVocab[word]; ok {
		return []Token{{Text: displayToken(word), ID: id}}
	}

	// Greedy longest-match from left
	var tokens []Token
	remaining := word
	for len(remaining) > 0 {
		bestLen := 0
		bestID := -1
		for l := len(remaining); l > 0; l-- {
			sub := remaining[:l]
			if id, ok := bpeVocab[sub]; ok {
				bestLen = l
				bestID = id
				break
			}
		}
		if bestLen == 0 {
			// Single character fallback
			ch := remaining[0]
			tokens = append(tokens, Token{Text: string(ch), ID: int(ch)})
			remaining = remaining[1:]
		} else {
			tokens = append(tokens, Token{Text: displayToken(remaining[:bestLen]), ID: bestID})
			remaining = remaining[bestLen:]
		}
	}
	return tokens
}

func displayToken(t string) string {
	return strings.ReplaceAll(t, "Ġ", "Ġ")
}

func handleTokenize(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	if text == "" {
		text = "The transformer model uses attention mechanisms"
	}
	tokens := tokenize(text)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tokens": tokens,
		"count":  len(tokens),
	})
}

// ─── EMBEDDINGS ─────────────────────────────────────────────

type WordEmbedding struct {
	Word     string    `json:"word"`
	X        float64   `json:"x"`
	Y        float64   `json:"y"`
	Category string    `json:"category"`
	Vector   []float64 `json:"vector,omitempty"`
}

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

var precomputedEmbeddings []WordEmbedding

func init() {
	rng := rand.New(rand.NewSource(42))
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
			vec := make([]float64, 32)
			for i := range vec {
				vec[i] = rng.NormFloat64() * 0.3
			}
			// Encode category info into first few dims
			catHash := float64(len(cat)) * 0.1
			vec[0] = catHash
			vec[1] = x * 0.01
			vec[2] = y * 0.01

			precomputedEmbeddings = append(precomputedEmbeddings, WordEmbedding{
				Word:     word,
				X:        x,
				Y:        y,
				Category: cat,
				Vector:   vec,
			})
		}
	}
}

func cosineSimilarity(a, b []float64) float64 {
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

func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	word1 := r.URL.Query().Get("word1")
	word2 := r.URL.Query().Get("word2")

	// Strip vectors from bulk response to keep payload under 32KB (tunnel frame limit)
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
			Word:     e.Word,
			X:        e.X,
			Y:        e.Y,
			Category: e.Category,
		}
		// Only include vector if this word was specifically requested
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
			response["similarity"] = cosineSimilarity(vec1, vec2)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ─── ATTENTION ──────────────────────────────────────────────

func handleAttention(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	if text == "" {
		text = "The cat sat on the mat because it was tired"
	}

	tokens := tokenize(text)
	n := len(tokens)
	if n > 30 {
		tokens = tokens[:30]
		n = 30
	}

	tokenTexts := make([]string, n)
	for i, t := range tokens {
		tokenTexts[i] = t.Text
	}

	// Generate realistic attention patterns
	// Head 1: Local attention (nearby tokens)
	head1 := makeAttentionHead(n, "local", tokenTexts)
	// Head 2: Global attention (special tokens like "the", punctuation)
	head2 := makeAttentionHead(n, "global", tokenTexts)
	// Head 3: Positional (attend to beginning)
	head3 := makeAttentionHead(n, "positional", tokenTexts)
	// Head 4: Content-based (semantic similarity)
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

func makeAttentionHead(n int, pattern string, tokens []string) [][]float64 {
	rng := rand.New(rand.NewSource(int64(n * len(pattern))))
	matrix := make([][]float64, n)
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}

	switch pattern {
	case "local":
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				dist := math.Abs(float64(i - j))
				matrix[i][j] = math.Exp(-dist * 0.5)
			}
		}
	case "global":
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
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				ti := strings.ToLower(strings.TrimPrefix(tokens[i], "Ġ"))
				tj := strings.ToLower(strings.TrimPrefix(tokens[j], "Ġ"))
				matrix[i][j] = 0.03 + rng.Float64()*0.04

				// Pronouns attend to nouns
				if (ti == "it" || ti == "he" || ti == "she" || ti == "they" || ti == "its" || ti == "his" || ti == "her" || ti == "their") && j < i {
					matrix[i][j] = 0.2 + rng.Float64()*0.15
				}
				// Same or similar tokens attend to each other
				if ti == tj && i != j {
					matrix[i][j] = 0.35 + rng.Float64()*0.15
				}
			}
			matrix[i][i] += 0.1
		}
	}

	// Softmax each row
	for i := range matrix {
		softmaxRow(matrix[i])
	}
	return matrix
}

func softmaxRow(row []float64) {
	maxVal := row[0]
	for _, v := range row {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := 0.0
	for i := range row {
		row[i] = math.Exp(row[i] - maxVal)
		sum += row[i]
	}
	for i := range row {
		row[i] /= sum
	}
}

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

// ─── NEXT-TOKEN PREDICTION ─────────────────────────────────

type PredictionResult struct {
	Token string  `json:"token"`
	Prob  float64 `json:"prob"`
}

var ngramModel = map[string]map[string]float64{
	"the": {"cat": 0.12, "dog": 0.08, "model": 0.07, "world": 0.06, "data": 0.05, "best": 0.05, "first": 0.04, "new": 0.04, "old": 0.03, "big": 0.03, "end": 0.03, "same": 0.03, "most": 0.03, "next": 0.03, "other": 0.02},
	"a":   {"new": 0.1, "large": 0.08, "small": 0.07, "single": 0.06, "good": 0.05, "great": 0.05, "simple": 0.04, "deep": 0.04, "long": 0.03, "short": 0.03, "fast": 0.03, "big": 0.03, "few": 0.03, "very": 0.02, "bit": 0.02},
	"is":  {"a": 0.15, "the": 0.12, "not": 0.1, "an": 0.05, "very": 0.05, "used": 0.04, "one": 0.03, "called": 0.03, "known": 0.03, "based": 0.03, "also": 0.03, "that": 0.02, "how": 0.02, "what": 0.02, "it": 0.02},
	"i":   {"am": 0.12, "think": 0.1, "have": 0.09, "want": 0.07, "like": 0.06, "know": 0.06, "love": 0.05, "need": 0.05, "can": 0.04, "will": 0.04, "see": 0.03, "feel": 0.03, "believe": 0.03, "hope": 0.02, "wish": 0.02},
	"machine":  {"learning": 0.45, "translation": 0.1, "vision": 0.05, "intelligence": 0.04, "code": 0.03, "model": 0.03, "that": 0.02, "can": 0.02, "is": 0.02, "and": 0.02},
	"deep":     {"learning": 0.5, "neural": 0.08, "network": 0.06, "dive": 0.04, "understanding": 0.04, "analysis": 0.03, "inside": 0.02, "within": 0.02, "breath": 0.02, "thought": 0.02},
	"neural":   {"network": 0.45, "networks": 0.2, "net": 0.05, "architecture": 0.04, "model": 0.03, "machine": 0.02, "computation": 0.02, "processing": 0.02, "pathway": 0.02, "activity": 0.02},
	"language":  {"model": 0.3, "models": 0.15, "processing": 0.1, "understanding": 0.06, "generation": 0.05, "translation": 0.04, "learning": 0.03, "is": 0.02, "and": 0.02, "that": 0.02},
	"attention": {"mechanism": 0.2, "is": 0.12, "head": 0.08, "heads": 0.07, "layer": 0.06, "weight": 0.05, "weights": 0.05, "matrix": 0.04, "score": 0.04, "pattern": 0.03},
	"transformer": {"model": 0.15, "architecture": 0.12, "is": 0.08, "based": 0.07, "network": 0.05, "layer": 0.05, "was": 0.04, "uses": 0.04, "has": 0.03, "can": 0.03},
	"cat":  {"sat": 0.15, "is": 0.1, "and": 0.08, "was": 0.07, "in": 0.06, "on": 0.05, "loves": 0.04, "jumped": 0.04, "ate": 0.03, "ran": 0.03},
	"to":   {"the": 0.12, "be": 0.1, "a": 0.07, "learn": 0.05, "make": 0.04, "do": 0.04, "get": 0.04, "have": 0.03, "use": 0.03, "find": 0.03, "see": 0.03, "go": 0.03, "know": 0.02, "say": 0.02, "take": 0.02},
	"we":   {"can": 0.12, "have": 0.1, "are": 0.09, "need": 0.07, "will": 0.06, "use": 0.05, "know": 0.04, "see": 0.04, "want": 0.03, "should": 0.03},
	"how":  {"to": 0.2, "the": 0.08, "does": 0.07, "do": 0.07, "it": 0.06, "can": 0.05, "much": 0.05, "many": 0.04, "about": 0.03, "well": 0.03},
	"what": {"is": 0.2, "the": 0.1, "are": 0.08, "does": 0.06, "do": 0.05, "if": 0.04, "about": 0.04, "makes": 0.03, "happens": 0.03, "can": 0.03},
	"hello": {"world": 0.25, "there": 0.15, "everyone": 0.08, "how": 0.07, "my": 0.05, "and": 0.04, "i": 0.04, "to": 0.03, "from": 0.02, "dear": 0.02},
	"artificial": {"intelligence": 0.55, "neural": 0.06, "general": 0.04, "life": 0.03, "agent": 0.03, "system": 0.02, "network": 0.02, "mind": 0.02, "brain": 0.02, "world": 0.01},
}

func handlePredict(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	temperature := 1.0
	topP := 0.9

	if t := r.URL.Query().Get("temperature"); t != "" {
		fmt.Sscanf(t, "%f", &temperature)
	}
	if p := r.URL.Query().Get("top_p"); p != "" {
		fmt.Sscanf(p, "%f", &topP)
	}

	if temperature < 0.1 {
		temperature = 0.1
	}
	if temperature > 2.0 {
		temperature = 2.0
	}

	words := strings.Fields(strings.ToLower(text))
	var lastWord string
	if len(words) > 0 {
		lastWord = words[len(words)-1]
	}

	// Get base distribution
	baseDist, ok := ngramModel[lastWord]
	if !ok {
		// Default distribution
		baseDist = map[string]float64{
			"the": 0.08, "a": 0.06, "is": 0.05, "and": 0.05, "to": 0.05,
			"of": 0.04, "in": 0.04, "that": 0.03, "it": 0.03, "for": 0.03,
			"was": 0.03, "on": 0.02, "with": 0.02, "as": 0.02, "are": 0.02,
		}
	}

	// Apply temperature
	type kv struct {
		k string
		v float64
	}
	var pairs []kv
	for k, v := range baseDist {
		logit := math.Log(v + 1e-10)
		pairs = append(pairs, kv{k, logit / temperature})
	}

	// Softmax
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

	// Sort by probability desc
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].v > pairs[j].v
	})

	// Apply top-p
	var results []PredictionResult
	cumProb := 0.0
	for _, p := range pairs {
		if cumProb >= topP && len(results) > 0 {
			break
		}
		results = append(results, PredictionResult{Token: p.k, Prob: p.v})
		cumProb += p.v
	}

	// Renormalize
	totalProb := 0.0
	for _, r := range results {
		totalProb += r.Prob
	}
	for i := range results {
		results[i].Prob = results[i].Prob / totalProb
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"predictions": results,
		"lastWord":    lastWord,
		"temperature": temperature,
		"topP":        topP,
	})
}

// ─── FORWARD PASS ───────────────────────────────────────────

type ForwardPassStep struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Technical   string      `json:"technical"`
	Data        interface{} `json:"data"`
}

func handleForwardPass(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	if text == "" {
		text = "The cat sat"
	}
	tokens := tokenize(text)
	tokenTexts := make([]string, len(tokens))
	tokenIDs := make([]int, len(tokens))
	for i, t := range tokens {
		tokenTexts[i] = t.Text
		tokenIDs[i] = t.ID
	}

	rng := rand.New(rand.NewSource(42))
	dim := 8

	// Generate embedding vectors
	embeddings := make([][]float64, len(tokens))
	for i := range embeddings {
		embeddings[i] = make([]float64, dim)
		for j := range embeddings[i] {
			embeddings[i][j] = math.Round((rng.Float64()*2-1)*100) / 100
		}
	}

	// Positional encodings
	posEncodings := make([][]float64, len(tokens))
	for i := range posEncodings {
		posEncodings[i] = make([]float64, dim)
		for j := range posEncodings[i] {
			if j%2 == 0 {
				posEncodings[i][j] = math.Round(math.Sin(float64(i)/math.Pow(10000, float64(j)/float64(dim)))*100) / 100
			} else {
				posEncodings[i][j] = math.Round(math.Cos(float64(i)/math.Pow(10000, float64(j-1)/float64(dim)))*100) / 100
			}
		}
	}

	// Combined
	combined := make([][]float64, len(tokens))
	for i := range combined {
		combined[i] = make([]float64, dim)
		for j := range combined[i] {
			combined[i][j] = math.Round((embeddings[i][j]+posEncodings[i][j])*100) / 100
		}
	}

	// Attention scores (simplified)
	n := len(tokens)
	attentionScores := make([][]float64, n)
	for i := range attentionScores {
		attentionScores[i] = make([]float64, n)
		for j := range attentionScores[i] {
			dist := math.Abs(float64(i - j))
			attentionScores[i][j] = math.Exp(-dist * 0.3)
		}
		softmaxRow(attentionScores[i])
		for j := range attentionScores[i] {
			attentionScores[i][j] = math.Round(attentionScores[i][j]*100) / 100
		}
	}

	// FFN output
	ffnOutput := make([][]float64, len(tokens))
	for i := range ffnOutput {
		ffnOutput[i] = make([]float64, dim)
		for j := range ffnOutput[i] {
			val := combined[i][j]*0.8 + rng.Float64()*0.4 - 0.2
			ffnOutput[i][j] = math.Round(val*100) / 100
		}
	}

	// Layer norm
	layerNormOutput := make([][]float64, len(tokens))
	for i := range layerNormOutput {
		mean := 0.0
		for _, v := range ffnOutput[i] {
			mean += v
		}
		mean /= float64(dim)
		variance := 0.0
		for _, v := range ffnOutput[i] {
			variance += (v - mean) * (v - mean)
		}
		variance /= float64(dim)
		std := math.Sqrt(variance + 1e-5)

		layerNormOutput[i] = make([]float64, dim)
		for j := range layerNormOutput[i] {
			layerNormOutput[i][j] = math.Round((ffnOutput[i][j]-mean)/std*100) / 100
		}
	}

	// Final logits (top tokens)
	topTokens := []string{"cat", "dog", "mat", "hat", "on", "the", "and", "was", "in", "is"}
	logits := make([]float64, len(topTokens))
	for i := range logits {
		logits[i] = rng.Float64()*4 - 1
	}
	logits[0] = 3.2 // Make "on" or similar likely
	logits[4] = 2.8
	softmaxRow(logits)
	for i := range logits {
		logits[i] = math.Round(logits[i]*1000) / 1000
	}

	steps := []ForwardPassStep{
		{
			Name:        "Input Text",
			Description: "The raw text you typed — this is where everything starts. The model can't read text directly, so it needs to convert it into numbers first.",
			Technical:   "Raw string input: \"" + text + "\". The model's vocabulary maps subword units to integer indices.",
			Data:        map[string]interface{}{"text": text},
		},
		{
			Name:        "Tokenization",
			Description: "The text is broken into smaller pieces called tokens. Think of it like breaking a sentence into puzzle pieces — some are whole words, some are parts of words.",
			Technical:   "BPE (Byte Pair Encoding) splits text into subword tokens from a learned vocabulary. Each token maps to an integer ID. Vocabulary size typically 32K-100K tokens.",
			Data:        map[string]interface{}{"tokens": tokenTexts, "ids": tokenIDs},
		},
		{
			Name:        "Embedding Lookup",
			Description: "Each token gets converted into a list of numbers (a vector) that captures its meaning. Similar words get similar vectors — like plotting words on a map where nearby words mean similar things.",
			Technical:   fmt.Sprintf("Each token ID indexes into an embedding matrix E ∈ ℝ^(V×d_model). Here d_model=%d (real models use 768-12288). The embedding captures semantic and syntactic properties.", dim),
			Data:        map[string]interface{}{"tokens": tokenTexts, "embeddings": embeddings, "dim": dim},
		},
		{
			Name:        "Positional Encoding",
			Description: "The model needs to know WHERE each word is in the sentence. We add a unique position signal to each embedding — like giving every word a seat number.",
			Technical:   "PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos/10000^(2i/d_model)). Added element-wise to embeddings. Allows the model to learn relative positions.",
			Data:        map[string]interface{}{"tokens": tokenTexts, "positional": posEncodings, "combined": combined},
		},
		{
			Name:        "Multi-Head Attention",
			Description: "This is the magic. Each word looks at every other word and decides how much to pay attention to it. 'The cat sat on the mat' — when processing 'sat', the model pays extra attention to 'cat' (who sat?) and 'mat' (where?).",
			Technical:   "Attention(Q,K,V) = softmax(QK^T/√d_k)V. Q,K,V are linear projections of the input. Multiple heads (h=8 typically) allow attending to different aspects. Each head has dimension d_k = d_model/h.",
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
			Technical:   "LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β, where μ and σ² are computed per-token across the feature dimension. γ, β are learned parameters.",
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
	json.NewEncoder(w).Encode(map[string]interface{}{
		"steps": steps,
	})
}
