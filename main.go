package main

import (
	"embed"
	"fmt"
	"io/fs"
	"log"
	"net/http"

	"llm-demo/handlers"
)

//go:embed static/*
var staticFiles embed.FS

func main() {
	staticFS, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}

	http.Handle("/", http.FileServer(http.FS(staticFS)))
	http.HandleFunc("/api/tokenize", handlers.HandleTokenize)
	http.HandleFunc("/api/embeddings", handlers.HandleEmbeddings)
	http.HandleFunc("/api/attention", handlers.HandleAttention)
	http.HandleFunc("/api/predict", handlers.HandlePredict)
	http.HandleFunc("/api/forward-pass", handlers.HandleForwardPass)

	fmt.Println("🧠 LLM Demo running at http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
