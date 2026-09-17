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
		ids = t.appendUnigramSample(ids, s, alpha, rng)
	}
	return ids, nil
}
