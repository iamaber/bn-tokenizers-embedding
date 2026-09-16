package tokenizer

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"
)

// BenchmarkValidation uses the same normalized corpus as the Python comparison.
func BenchmarkValidation(b *testing.B) {
	model, corpus := os.Getenv("BNTOK_MODEL"), os.Getenv("BNTOK_CORPUS")
	if model == "" || corpus == "" {
		b.Skip("set BNTOK_MODEL and BNTOK_CORPUS")
	}
	tok, err := Load(model)
	if err != nil {
		b.Fatal(err)
	}
	f, err := os.Open(corpus)
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	var texts []string
	var bytes int64
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 65536), 16*1024*1024)
	for scanner.Scan() {
		var record struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			b.Fatal(err)
		}
		text := Normalize(record.Text)
		texts = append(texts, text)
		bytes += int64(len(text))
	}
	if err := scanner.Err(); err != nil {
		b.Fatal(err)
	}
	if len(texts) == 0 {
		b.Fatal("empty corpus")
	}
	b.ReportAllocs()
	b.SetBytes(bytes)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, text := range texts {
			tok.Encode(text)
		}
	}
}
