package tokenizer

import (
	"math"
	"sort"
	"strings"
)

type word struct {
	text  string
	count float64
}

// trainingInput owns the shared corpus contract and accumulation order. Empty
// corpora and vocabulary budgets are interpreted by each training algorithm.
type trainingInput struct {
	words      []word
	characters map[string]bool
}

func prepareTrainingInput(counts map[string]float64) (trainingInput, bool) {
	input := trainingInput{words: make([]word, 0, len(counts)), characters: map[string]bool{}}
	keys := make([]string, 0, len(counts))
	for s := range counts {
		keys = append(keys, s)
	}
	sort.Strings(keys)
	for _, s := range keys {
		f := counts[s]
		if f <= 0 || math.IsNaN(f) || math.IsInf(f, 0) || s == "" || s != Normalize(s) || strings.Contains(s, " ") {
			return trainingInput{}, false
		}
		input.words = append(input.words, word{s, f})
		for _, r := range s {
			input.characters[string(r)] = true
		}
	}
	return input, true
}

func (input trainingInput) alphabet() []string {
	alphabet := make([]string, 0, len(input.characters))
	for s := range input.characters {
		alphabet = append(alphabet, s)
	}
	sort.Strings(alphabet)
	return alphabet
}
