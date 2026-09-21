package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

func TestEvaluateCountsFusedByteFallback(t *testing.T) {
	dir := t.TempDir()
	model := tokenizer.Model{Version: 2, Algorithm: "bpe", Normalization: tokenizer.Normalization,
		Pieces: []tokenizer.Piece{{Text: "a"}}, SpaceFusion: []int{tokenizer.ByteOffset + 0xf0}}
	path := filepath.Join(dir, "model.json")
	if err := model.Save(path); err != nil {
		t.Fatal(err)
	}
	input := filepath.Join(dir, "validation.jsonl")
	if err := os.WriteFile(input, []byte("{\"text\":\"a 🙂\",\"bucket\":\"mixed\",\"split\":\"validation\"}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	output, err := os.Create(filepath.Join(dir, "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	previous := os.Stdout
	os.Stdout = output
	defer func() { os.Stdout = previous }()
	if err := run([]string{"evaluate", "-model", path, "-input", input}); err != nil {
		t.Fatal(err)
	}
	if _, err := output.Seek(0, 0); err != nil {
		t.Fatal(err)
	}
	var result struct {
		Buckets map[string]metrics `json:"buckets"`
	}
	if err := json.NewDecoder(output).Decode(&result); err != nil {
		t.Fatal(err)
	}
	m := result.Buckets["mixed"]
	if m.Tokens != 5 || m.SpaceTokens != 1 || m.FallbackTokens != 4 || m.Fertility != 2.5 || m.FallbackRate != .8 || m.UsedIDs != 5 {
		t.Fatalf("incorrect fused metrics: %+v", m)
	}
}
