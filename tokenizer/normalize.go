// Package tokenizer implements a joint Unicode Unigram tokenizer in pure Go.
package tokenizer

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const Normalization = "nfc-whitespace-v1"

// Normalize applies NFC, trims whitespace and collapses Unicode whitespace.
// Case, joiners, punctuation, accents and spelling variants are preserved.
// Invalid UTF-8 is replaced with U+FFFD before normalization.
func Normalize(s string) string {
	s = norm.NFC.String(strings.ToValidUTF8(s, "\uFFFD"))
	previousSpace := true
	for _, r := range s {
		if unicode.IsSpace(r) {
			if r != ' ' || previousSpace {
				return strings.Join(strings.Fields(s), " ")
			}
			previousSpace = true
		} else {
			previousSpace = false
		}
	}
	return strings.TrimSuffix(s, " ")
}
