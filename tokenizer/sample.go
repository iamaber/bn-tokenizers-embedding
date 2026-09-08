package tokenizer

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// Sample draws a complete Unigram segmentation from P(segmentation)^alpha.
// The caller owns and seeds rng; inference should use deterministic Encode.
// Unknown characters retain the same fixed-score byte fallback as Encode.
func (t *Tokenizer) Sample(text string, alpha float64, rng *rand.Rand) ([]int, error) {
	if t.model.Algorithm != "unigram" || rng == nil || alpha <= 0 || math.IsNaN(alpha) || math.IsInf(alpha, 0) {
		return nil, fmt.Errorf("sampling requires Unigram, positive finite alpha and an RNG")
	}
	var ids []int
	for n, s := range strings.Split(Normalize(text), " ") {
		if n > 0 {
			ids = append(ids, ByteOffset+32)
		}
		beta := make([]float64, len(s)+1)
		for i := len(s) - 1; i >= 0; i-- {
			beta[i] = -30*alpha + beta[i+1]
			t.matches(s, i, func(end, id int) { beta[i] = logAdd(beta[i], t.model.Pieces[id].Score*alpha+beta[end]) })
		}
		for i := 0; i < len(s); {
			type choice struct {
				end, id     int
				probability float64
			}
			choices := []choice{{i + 1, ByteOffset + int(s[i]), math.Exp(-30*alpha + beta[i+1] - beta[i])}}
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
	}
	return ids, nil
}
