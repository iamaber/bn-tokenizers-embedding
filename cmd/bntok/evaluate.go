package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/iamaber/bn-tokenizers-embedding/corpus"
	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

type metrics struct {
	Examples              int     `json:"examples"`
	Tokens                int     `json:"tokens"`
	Words                 int     `json:"whitespace_words"`
	Bytes                 int     `json:"normalized_bytes"`
	Characters            int     `json:"normalized_characters"`
	ByteBaselineTokens    int     `json:"byte_baseline_tokens"`
	FallbackTokens        int     `json:"fallback_tokens_excluding_spaces"`
	Truncated             int     `json:"truncated_examples"`
	TruncationRate        float64 `json:"truncation_rate"`
	FallbackRate          float64 `json:"fallback_token_rate_excluding_spaces"`
	UsedIDs               int     `json:"used_vocabulary_ids"`
	VocabUtilization      float64 `json:"vocabulary_utilization"`
	Fertility             float64 `json:"tokens_per_word"`
	SequenceTokensPerWord float64 `json:"sequence_tokens_per_word_including_spaces"`
	SpaceTokens           int     `json:"space_tokens"`
	P50                   int     `json:"p50_tokens"`
	P95                   int     `json:"p95_tokens"`
	P99                   int     `json:"p99_tokens"`
	EncodeSeconds         float64 `json:"encode_seconds"`
	MBPerSecond           float64 `json:"encode_MB_per_second"`
	lengths               []int
	used                  map[int]bool
}

func evaluate(f *flag.FlagSet, args []string) error {
	path := f.String("model", "models/unigram-32000.json", "model file")
	input := f.String("input", "data/processed/validation.jsonl", "held-out JSONL")
	limit := f.Int("max-tokens", 512, "encoder sequence limit")
	special := f.Int("special-tokens", 2, "reserved encoder positions (e.g. BOS/EOS)")
	if err := f.Parse(args); err != nil {
		return err
	}
	if *limit <= 0 || *special < 0 || *special >= *limit {
		return fmt.Errorf("invalid sequence budget")
	}
	t, err := tokenizer.Load(*path)
	if err != nil {
		return err
	}
	results := map[string]*metrics{}
	err = corpus.Read(*input, func(r corpus.Record) error {
		if r.Split != "validation" && r.Split != "test" {
			return fmt.Errorf("evaluation requires held-out records")
		}
		m := results[r.Bucket]
		if m == nil {
			m = &metrics{used: map[int]bool{}}
			results[r.Bucket] = m
		}
		start := time.Now()
		ids := t.Encode(r.Text)
		m.EncodeSeconds += time.Since(start).Seconds()
		decoded, err := t.Decode(ids)
		if err != nil {
			return err
		}
		normalized := tokenizer.Normalize(r.Text)
		if decoded != normalized {
			return fmt.Errorf("round-trip failure in %s", r.Group)
		}
		m.Examples++
		m.Tokens += len(ids)
		m.Words += len(strings.Fields(normalized))
		m.Bytes += len(normalized)
		m.Characters += utf8.RuneCountInString(normalized)
		m.ByteBaselineTokens += len(normalized)
		for _, id := range ids {
			if id == tokenizer.ByteOffset+32 {
				m.SpaceTokens++
			}
			m.used[id] = true
			if id >= tokenizer.ByteOffset && id < tokenizer.PieceOffset && id != tokenizer.ByteOffset+32 {
				m.FallbackTokens++
			}
		}
		if len(ids)+*special > *limit {
			m.Truncated++
		}
		m.lengths = append(m.lengths, len(ids))
		return nil
	})
	if err != nil {
		return err
	}
	if len(results) == 0 {
		return fmt.Errorf("empty evaluation set")
	}
	for _, m := range results {
		sort.Ints(m.lengths)
		m.P50 = m.lengths[(len(m.lengths)-1)*50/100]
		m.P95 = m.lengths[(len(m.lengths)-1)*95/100]
		m.P99 = m.lengths[(len(m.lengths)-1)*99/100]
		m.Fertility = float64(m.Tokens-m.SpaceTokens) / float64(m.Words)
		m.SequenceTokensPerWord = float64(m.Tokens) / float64(m.Words)
		m.TruncationRate = float64(m.Truncated) / float64(m.Examples)
		m.FallbackRate = float64(m.FallbackTokens) / float64(max(1, m.Tokens-m.SpaceTokens))
		m.UsedIDs = len(m.used)
		m.VocabUtilization = float64(m.UsedIDs) / float64(t.VocabSize())
		m.MBPerSecond = float64(m.Bytes) / 1e6 / m.EncodeSeconds
	}
	return corpus.CopyJSON(os.Stdout, map[string]any{"vocabulary": t.VocabSize(), "buckets": results, "round_trip_failures": 0, "max_tokens": *limit, "special_tokens": *special, "unknown_tokens": 0})
}
