package tokenizer

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

type TrainConfig struct {
	VocabSize     int
	SeedSize      int
	MaxPieceRunes int
	EMIterations  int
	Log           io.Writer
}

type word struct {
	text  string
	count float64
}

// Train fits a Unigram language model with forward-backward EM and iterative
// expected-count pruning. This is not SentencePiece's loss-based pruning recipe
// and its JSON models are deliberately not advertised as SentencePiece compatible.
// Counts may be fractional, allowing a documented sentence-weighted mixture.
func Train(counts map[string]float64, c TrainConfig) (Model, error) {
	if c.VocabSize <= PieceOffset || c.SeedSize < c.VocabSize || c.MaxPieceRunes < 1 || c.EMIterations < 1 {
		return Model{}, fmt.Errorf("invalid training configuration")
	}
	words := make([]word, 0, len(counts))
	seeds := map[string]float64{}
	chars := map[string]bool{}
	keys := make([]string, 0, len(counts))
	for s := range counts {
		keys = append(keys, s)
	}
	sort.Strings(keys)
	for _, s := range keys {
		f := counts[s]
		if f <= 0 || math.IsNaN(f) || math.IsInf(f, 0) || s == "" || s != Normalize(s) || strings.Contains(s, " ") {
			return Model{}, fmt.Errorf("invalid training word")
		}
		words = append(words, word{s, f})
		r := []rune(s)
		for i := range r {
			chars[string(r[i])] = true
			for n := 1; n <= c.MaxPieceRunes && i+n <= len(r); n++ {
				seeds[string(r[i:i+n])] += f
			}
		}
	}
	if len(words) == 0 {
		return Model{}, fmt.Errorf("empty training corpus")
	}
	if len(seeds) < c.VocabSize-PieceOffset {
		return Model{}, fmt.Errorf("only %d candidate pieces: insufficient corpus for vocabulary %d", len(seeds), c.VocabSize)
	}
	target := c.VocabSize - PieceOffset
	if len(chars) > target {
		return Model{}, fmt.Errorf("vocabulary too small for %d characters and %d reserved IDs", len(chars), PieceOffset)
	}
	pieces := make([]Piece, 0, len(seeds))
	for s, f := range seeds {
		pieces = append(pieces, Piece{s, f})
	}
	prune := func(n int) {
		sort.Slice(pieces, func(i, j int) bool {
			a, b := pieces[i], pieces[j]
			if chars[a.Text] != chars[b.Text] {
				return chars[a.Text]
			}
			if a.Score != b.Score {
				return a.Score > b.Score
			}
			return a.Text < b.Text
		})
		if n < len(pieces) {
			pieces = pieces[:n]
		}
		sort.Slice(pieces, func(i, j int) bool { return pieces[i].Text < pieces[j].Text })
	}
	prune(c.SeedSize - PieceOffset)
	normalize := func() {
		total := 0.0
		for _, p := range pieces {
			total += math.Max(p.Score, 1e-12)
		}
		for i := range pieces {
			pieces[i].Score = math.Log(math.Max(pieces[i].Score, 1e-12) / total)
		}
	}
	normalize()
	for round := 0; ; round++ {
		var expected []float64
		for iteration := 0; iteration < c.EMIterations; iteration++ {
			m := Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: pieces}
			t, err := New(m)
			if err != nil {
				return Model{}, err
			}
			expected = make([]float64, len(pieces))
			loss := 0.0
			for _, w := range words {
				alpha := make([]float64, len(w.text)+1)
				beta := make([]float64, len(alpha))
				for i := range alpha {
					alpha[i] = math.Inf(-1)
					beta[i] = math.Inf(-1)
				}
				alpha[0] = 0
				beta[len(w.text)] = 0
				for i := range w.text {
					t.matches(w.text, i, func(end, id int) { alpha[end] = logAdd(alpha[end], alpha[i]+pieces[id].Score) })
				}
				for i := len(w.text) - 1; i >= 0; i-- {
					if !utf8.RuneStart(w.text[i]) {
						continue
					}
					t.matches(w.text, i, func(end, id int) { beta[i] = logAdd(beta[i], pieces[id].Score+beta[end]) })
				}
				z := alpha[len(w.text)]
				loss -= w.count * z
				for i := range w.text {
					t.matches(w.text, i, func(end, id int) { expected[id] += w.count * math.Exp(alpha[i]+pieces[id].Score+beta[end]-z) })
				}
			}
			for i := range pieces {
				pieces[i].Score = expected[i]
			}
			normalize()
			if c.Log != nil {
				fmt.Fprintf(c.Log, "round=%d em=%d pieces=%d negative_log_likelihood=%.3f\n", round, iteration, len(pieces), loss)
			}
		}
		if len(pieces) <= target {
			break
		}
		for i := range pieces {
			pieces[i].Score = expected[i]
		}
		n := max(target, int(float64(len(pieces))*0.8))
		prune(n)
		normalize()
	}
	return Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: pieces}, nil
}

func logAdd(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	if a < b {
		a, b = b, a
	}
	return a + math.Log1p(math.Exp(b-a))
}
