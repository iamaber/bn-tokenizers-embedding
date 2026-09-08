// Package tokenizer implements a joint Unicode Unigram tokenizer in pure Go.
package tokenizer

import (
	"golang.org/x/text/unicode/norm"
	"strings"
)

const Normalization = "nfc-whitespace-v1"

// Normalize applies NFC, trims whitespace and collapses Unicode whitespace.
// Case, joiners, punctuation, accents and spelling variants are preserved.
// Invalid UTF-8 is replaced with U+FFFD before normalization.
func Normalize(s string) string {
	return strings.Join(strings.Fields(norm.NFC.String(strings.ToValidUTF8(s, "\uFFFD"))), " ")
}
