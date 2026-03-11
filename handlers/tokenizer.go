package handlers

import (
	"encoding/json"
	"net/http"
	"strings"
)

// Token represents a single BPE token with its text and vocabulary ID.
type Token struct {
	Text string `json:"text"`
	ID   int    `json:"id"`
}

// bpeVocab is a simplified BPE vocabulary mapping subword units to token IDs.
// Real models like GPT-2 use 50,257 tokens; this is a curated subset for demonstration.
var bpeVocab = map[string]int{
	"Ġ": 256,
	// Sentence starters
	"The": 500, "A": 501, "I": 502, "It": 503, "In": 504, "This": 505, "We": 506,
	"He": 507, "She": 508, "They": 509, "How": 510, "What": 511, "When": 512,
	"Hello": 513, "My": 514, "You": 515, "Is": 516, "Are": 517, "Do": 518,
	// Common words (with leading space marker "Ġ")
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
	// Subword suffixes
	"ing": 2000, "tion": 2001, "ed": 2002, "er": 2003, "es": 2004,
	"en": 2005, "al": 2006, "re": 2007, "on": 2008, "ly": 2009,
	"an": 2010, "or": 2011, "le": 2012, "se": 2013, "ent": 2014,
	"ar": 2015, "ment": 2016, "at": 2017, "ous": 2018, "ness": 2019,
	"able": 2020, "ful": 2021, "ive": 2022, "ight": 2023, "ure": 2024,
	// Domain-specific (AI/ML)
	"Ġlang": 3000, "uage": 3001, "Ġmodel": 3002, "Ġlearn": 3003,
	"Ġneural": 3004, "Ġnetwork": 3005, "Ġtrans": 3006, "former": 3007,
	"Ġattention": 3008, "Ġtoken": 3009, "Ġembed": 3010, "ding": 3011,
	"Ġartificial": 3012, "Ġintelligence": 3013, "Ġmachine": 3014,
	"Ġdeep": 3015, "Ġdata": 3016, "Ġtrain": 3017,
	// Common nouns/adjectives/verbs
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

// Tokenize splits text into BPE tokens using greedy longest-match.
func Tokenize(text string) []Token {
	var tokens []Token
	words := splitIntoWords(text)
	for _, word := range words {
		tokens = append(tokens, tokenizeWord(word)...)
	}
	return tokens
}

// splitIntoWords splits text on whitespace, prepending "Ġ" to non-initial words
// to match the GPT-2 BPE convention where spaces are encoded as part of the token.
func splitIntoWords(text string) []string {
	var words []string
	for i, f := range strings.Fields(text) {
		if i == 0 {
			words = append(words, f)
		} else {
			words = append(words, "Ġ"+f)
		}
	}
	return words
}

// tokenizeWord applies greedy longest-match BPE to a single word.
func tokenizeWord(word string) []Token {
	if id, ok := bpeVocab[word]; ok {
		return []Token{{Text: displayToken(word), ID: id}}
	}

	var tokens []Token
	remaining := word
	for len(remaining) > 0 {
		bestLen := 0
		bestID := -1
		for l := len(remaining); l > 0; l-- {
			if id, ok := bpeVocab[remaining[:l]]; ok {
				bestLen = l
				bestID = id
				break
			}
		}
		if bestLen == 0 {
			tokens = append(tokens, Token{Text: string(remaining[0]), ID: int(remaining[0])})
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

// HandleTokenize handles GET /api/tokenize?text=...
func HandleTokenize(w http.ResponseWriter, r *http.Request) {
	text := r.URL.Query().Get("text")
	if text == "" {
		text = "The transformer model uses attention mechanisms"
	}
	tokens := Tokenize(text)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"tokens": tokens,
		"count":  len(tokens),
	})
}
