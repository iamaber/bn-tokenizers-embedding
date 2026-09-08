package tokenizer

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestBPERankAndFallback(t *testing.T) {
	m := Model{Version: 1, Algorithm: "bpe", Normalization: Normalization, Pieces: []Piece{{"a", 0}, {"b", 0}, {"c", 0}, {"bc", 0}, {"ab", 0}}, Merges: []Merge{{1, 2, 3}, {0, 1, 4}}}
	tok, err := New(m)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.Encode("abc"); !reflect.DeepEqual(got, []int{PieceOffset, PieceOffset + 3}) {
		t.Fatal(got)
	}
	for _, s := range []string{"আমি 🦊 abc", "<unk> ▁", "ক্\u200dষ", "e\u0301"} {
		got, err := tok.Decode(tok.Encode(s))
		if err != nil || got != Normalize(s) {
			t.Fatal(got, err)
		}
	}
}

func TestRejectUnreachableBPEPiece(t *testing.T) {
	m := Model{Version: 1, Algorithm: "bpe", Normalization: Normalization, Pieces: []Piece{{"a", 0}, {"b", 0}, {"ab", 0}, {"unreachable", 0}}, Merges: []Merge{{0, 1, 2}}}
	if _, err := New(m); err == nil {
		t.Fatal("accepted unreachable vocabulary")
	}
}

func BenchmarkBPERepeatedWord(b *testing.B) {
	tok, err := New(Model{Version: 1, Algorithm: "bpe", Normalization: Normalization, Pieces: []Piece{{"a", 0}, {"aa", 0}, {"aaaa", 0}}, Merges: []Merge{{0, 0, 1}, {1, 1, 2}}})
	if err != nil {
		b.Fatal(err)
	}
	for _, n := range []int{1000, 10000, 100000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			s := strings.Repeat("a", n)
			b.SetBytes(int64(n))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tok.Encode(s)
			}
		})
	}
}

func TestBPETrainDeterministic(t *testing.T) {
	counts := map[string]float64{"low": 5, "lower": 2, "newest": 6, "widest": 3}
	a, err := TrainBPE(counts, 280, 12, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := TrainBPE(counts, 280, 12, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("BPE training not deterministic")
	}
	tok, err := New(a)
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() != 280 {
		t.Fatal("wrong vocabulary")
	}
	for s := range counts {
		got, err := tok.Decode(tok.Encode(s))
		if err != nil || got != s {
			t.Fatal(got, err)
		}
	}
}
