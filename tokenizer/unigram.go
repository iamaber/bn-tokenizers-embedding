package tokenizer

import (
	"math"
	"math/rand"
	"unicode/utf8"
)

const unigramByteFallbackScore = -30

// Unigram's three recurrences share piece lookup, but deliberately keep their
// traversal and arithmetic separate: training excludes byte fallback, whereas
// inference includes it. Match order also determines Viterbi ties and RNG draws.

// matches walks byte trie edges; learned pieces always end on rune boundaries.
func (t *Tokenizer) matches(s string, start int, visit func(end, id int)) {
	n := 0
	for j := start; j < len(s); j++ {
		v, ok := t.trie[n].next[s[j]]
		if !ok {
			break
		}
		n = v
		if t.trie[n].id >= 0 {
			visit(j+1, t.trie[n].id)
		}
	}
}

type viterbiState struct {
	score       float64
	prev, token int
}

func (t *Tokenizer) appendUnigram(ids []int, s string, states []viterbiState) []int {
	states[0] = viterbiState{}
	for i := 1; i <= len(s); i++ {
		states[i].score = math.Inf(-1)
	}
	for i := 0; i < len(s); i++ {
		// Byte fallback competes only at a low fixed score. It guarantees coverage.
		if v := states[i].score + unigramByteFallbackScore; v > states[i+1].score {
			states[i+1] = viterbiState{v, i, ByteOffset + int(s[i])}
		}
		t.matches(s, i, func(end, id int) {
			if v := states[i].score + t.model.Pieces[id].Score; v > states[end].score {
				states[end] = viterbiState{v, i, PieceOffset + id}
			}
		})
	}
	start := len(ids)
	for end := len(s); end > 0; end = states[end].prev {
		ids = append(ids, states[end].token)
	}
	for i, j := start, len(ids)-1; i < j; i, j = i+1, j-1 {
		ids[i], ids[j] = ids[j], ids[i]
	}
	return ids
}

func (t *Tokenizer) appendUnigramSample(ids []int, s string, alpha float64, rng *rand.Rand) []int {
	beta := make([]float64, len(s)+1)
	for i := len(s) - 1; i >= 0; i-- {
		beta[i] = unigramByteFallbackScore*alpha + beta[i+1]
		t.matches(s, i, func(end, id int) { beta[i] = logAdd(beta[i], t.model.Pieces[id].Score*alpha+beta[end]) })
	}
	for i := 0; i < len(s); {
		type choice struct {
			end, id     int
			probability float64
		}
		choices := []choice{{i + 1, ByteOffset + int(s[i]), math.Exp(unigramByteFallbackScore*alpha + beta[i+1] - beta[i])}}
		t.matches(s, i, func(end, id int) {
			choices = append(choices, choice{end, PieceOffset + id, math.Exp(t.model.Pieces[id].Score*alpha + beta[end] - beta[i])})
		})
		u := rng.Float64()
		selected := choices[len(choices)-1]
		for _, c := range choices {
			u -= c.probability
			if u <= 0 {
				selected = c
				break
			}
		}
		ids = append(ids, selected.id)
		i = selected.end
	}
	return ids
}

// unigramExpectation computes one EM E-step against the immutable model scores.
// Corpus order and floating-point addition order are part of reproducible training.
func (t *Tokenizer) unigramExpectation(words []word) ([]float64, float64) {
	pieces := t.model.Pieces
	expected := make([]float64, len(pieces))
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
	return expected, loss
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
