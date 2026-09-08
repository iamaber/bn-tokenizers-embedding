package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/iamaber/bn-tokenizers-embedding/corpus"
	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: bntok prepare|train|encode|evaluate [flags]")
	}
	f := flag.NewFlagSet(args[0], flag.ContinueOnError)
	switch args[0] {
	case "audit":
		dir := f.String("input", "data/processed", "prepared directory")
		raw := f.String("raw", "data/raw", "raw official held-out sources")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		report, err := corpus.Audit(*dir)
		if err != nil {
			return err
		}
		n, err := corpus.AuditOfficial(*dir, *raw)
		if err != nil {
			return err
		}
		report["official_heldout_keys_checked"] = n
		report["official_heldout_train_overlap"] = 0
		return corpus.CopyJSON(os.Stdout, report)
	case "fetch":
		manifest := f.String("manifest", "data/sources.lock.json", "frozen download manifest")
		root := f.String("raw", "data/raw", "download directory")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		return corpus.Fetch(*manifest, *root)
	case "prepare":
		manifest := f.String("manifest", "data/sources.lock.json", "frozen download manifest")
		raw := f.String("raw", "data/raw", "download directory")
		out := f.String("out", "data/processed", "split directory")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		if err := corpus.VerifyManifest(*manifest, *raw); err != nil {
			return err
		}
		r, err := corpus.LoadSources(*raw)
		if err != nil {
			return err
		}
		s, err := corpus.Prepare(r, *out)
		if err != nil {
			return err
		}
		return corpus.CopyJSON(os.Stdout, s)
	case "train":
		algorithm := f.String("algorithm", "unigram", "unigram or bpe")
		input := f.String("input", "data/processed/train.jsonl", "training records")
		out := f.String("out", "models/unigram-32000.json", "output model")
		vocab := f.Int("vocab", 32000, "total vocabulary including 260 reserved/byte IDs")
		seed := f.Int("seed-size", 64000, "initial vocabulary size")
		em := f.Int("em", 2, "EM iterations per pruning round")
		length := f.Int("max-piece-runes", 12, "maximum learned piece length")
		balanced := f.Bool("balanced", false, "use 25/25/25/25 rather than 40/25/25/10 sentence mixture")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		mixture := map[string]float64{"bangla": .4, "english": .25, "banglish": .25, "code-mixed": .1}
		if *balanced {
			for k := range mixture {
				mixture[k] = .25
			}
		}
		counts, err := corpus.WordCounts(*input, mixture)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "training on %d distinct words\n", len(counts))
		started := time.Now()
		var m tokenizer.Model
		switch *algorithm {
		case "unigram":
			m, err = tokenizer.Train(counts, tokenizer.TrainConfig{VocabSize: *vocab, SeedSize: *seed, MaxPieceRunes: *length, EMIterations: *em, Log: os.Stderr})
		case "bpe":
			m, err = tokenizer.TrainBPE(counts, *vocab, *length, os.Stderr)
		default:
			return fmt.Errorf("unknown algorithm %q", *algorithm)
		}
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(*out), 0755); err != nil {
			return err
		}
		if err := m.Save(*out); err != nil {
			return err
		}
		hash, err := corpus.FileHash(*input)
		if err != nil {
			return err
		}
		modelHash, err := corpus.FileHash(*out)
		if err != nil {
			return err
		}
		meta, err := os.Create(*out + ".meta.json")
		if err != nil {
			return err
		}
		defer meta.Close()
		return corpus.CopyJSON(meta, map[string]any{"algorithm": *algorithm, "vocabulary": len(m.Pieces) + tokenizer.PieceOffset, "seed_size": *seed, "em_iterations": *em, "max_piece_runes": *length, "mixture": mixture, "weight_unit": "equal sentence mass; inverse word-count within each sentence", "train_sha256": hash, "model_sha256": modelHash, "go": runtime.Version(), "os": runtime.GOOS, "arch": runtime.GOARCH, "training_seconds": time.Since(started).Seconds()})
	case "encode":
		path := f.String("model", "models/unigram-32000.json", "model file")
		text := f.String("text", "আমি আজ ভালো আছি। ami aj bhalo achi", "text to encode")
		if err := f.Parse(args[1:]); err != nil {
			return err
		}
		t, err := tokenizer.Load(*path)
		if err != nil {
			return err
		}
		ids := t.Encode(*text)
		decoded, err := t.Decode(ids)
		if err != nil {
			return err
		}
		return corpus.CopyJSON(os.Stdout, map[string]any{"ids": ids, "decoded": decoded, "tokens": len(ids)})
	case "evaluate":
		return evaluate(f, args[1:])
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}
