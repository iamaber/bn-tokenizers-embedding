package tokenizer

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"golang.org/x/text/unicode/norm"
)

func TestLoadedPieceCachePreservesSegmentation(t *testing.T) {
	models := []Model{
		{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: []Piece{{"a", -1}, {"b", -1}, {"ab", -10}}},
		{Version: 1, Algorithm: "bpe", Normalization: Normalization, Pieces: []Piece{{"a", 0}, {"b", 0}, {"c", 0}, {"bc", 0}, {"ab", 0}, {"abc", 0}}, Merges: []Merge{{1, 2, 3}, {0, 1, 4}, {4, 2, 5}}},
	}
	for _, m := range models {
		path := filepath.Join(t.TempDir(), "model.json")
		if err := m.Save(path); err != nil {
			t.Fatal(err)
		}
		cached, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		plain, err := New(m)
		if err != nil {
			t.Fatal(err)
		}
		for _, text := range []string{"ab abc a", strings.Repeat("abc", 200) + " ab abc", "a 🙂 ab\x00abc", ""} {
			want := plain.Encode(text)
			got := cached.Encode(text)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s %q: %v != %v", m.Algorithm, text, got, want)
			}
			if len(got) > 0 {
				got[0] = -1
			}
			if !reflect.DeepEqual(cached.Encode(text), want) {
				t.Fatal("returned IDs alias cache")
			}
		}
	}
}

func FuzzNormalizeEquivalent(f *testing.F) {
	for _, text := range []string{"", "  a b  ", "a\u00a0b", "বাংলা", "a\x00b", "e\u0301", "a\xffb", "a\u2028b"} {
		f.Add(text)
	}
	f.Fuzz(func(t *testing.T, text string) {
		want := strings.Join(strings.Fields(norm.NFC.String(strings.ToValidUTF8(text, "\uFFFD"))), " ")
		if got := Normalize(text); got != want {
			t.Fatalf("%q != %q", got, want)
		}
	})
}
