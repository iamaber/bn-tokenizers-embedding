package tokenizer

import (
	"fmt"
	"io"
	"math"
	"sort"
)

type TrainConfig struct {
	VocabSize     int
	SeedSize      int
	MaxPieceRunes int
	EMIterations  int
	Log           io.Writer
}

// Train fits a Unigram language model with forward-backward EM and iterative
// expected-count pruning. This is not SentencePiece's loss-based pruning recipe
// and its JSON models are deliberately not advertised as SentencePiece compatible.
// Counts may be fractional, allowing a documented sentence-weighted mixture.
func Train(counts map[string]float64, c TrainConfig) (Model, error) {
	if c.VocabSize <= PieceOffset || c.SeedSize < c.VocabSize || c.MaxPieceRunes < 1 || c.EMIterations < 1 {
		return Model{}, fmt.Errorf("invalid training configuration")
	}
	input, valid := prepareTrainingInput(counts)
	if !valid {
		return Model{}, fmt.Errorf("invalid training word")
	}
	words, chars := input.words, input.characters
	seeds := map[string]float64{}
	for _, w := range words {
		r := []rune(w.text)
		for i := range r {
			for n := 1; n <= c.MaxPieceRunes && i+n <= len(r); n++ {
				seeds[string(r[i:i+n])] += w.count
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
			var loss float64
			expected, loss = t.unigramExpectation(words)
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
