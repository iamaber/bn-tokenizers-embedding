package tokenizer

import (
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestFusionAndWordCache(t *testing.T) {
	base := Model{Version: 1, Algorithm: "bpe", Normalization: Normalization,
		Pieces: []Piece{{"a", 0}, {"b", 0}, {"ab", 0}}, Merges: []Merge{{0, 1, 2}}}
	plain, err := New(base)
	if err != nil {
		t.Fatal(err)
	}
	fused := base
	fused.Version = 2
	fused.SpaceFusion = []int{262, ByteOffset + 0xf0}
	fused.CacheWords = []string{"abab", "🙂", "a\x00b"}
	path := filepath.Join(t.TempDir(), "model.json")
	if err := fused.Save(path); err != nil {
		t.Fatal(err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if tok.VocabSize() != 265 {
		t.Fatal(tok.VocabSize())
	}
	for _, text := range []string{"", "abab", " abab\tabab 🙂 a\x00b ", "a 🙂🙂 unseen বাংলা", "<unk> ▁"} {
		ids := tok.Encode(text)
		var expanded []int
		for _, id := range ids {
			if id >= 263 {
				expanded = append(expanded, 36, fused.SpaceFusion[id-263])
			} else {
				expanded = append(expanded, id)
			}
		}
		want := plain.Encode(text)
		expandedAPI, err := tok.Expand(ids)
		if err != nil || !reflect.DeepEqual(expandedAPI, want) {
			t.Fatalf("Expand: %v %v", expandedAPI, err)
		}
		if len(want) != len(expanded) || (len(want) > 0 && !reflect.DeepEqual(want, expanded)) {
			t.Fatalf("%q: %v != %v", text, expanded, want)
		}
		decoded, err := tok.Decode(ids)
		if err != nil || decoded != Normalize(text) {
			t.Fatalf("%q: %q %v", text, decoded, err)
		}
		if len(ids) > 0 {
			ids[0] = -1
		}
		if decoded, err := tok.Decode(tok.Encode(text)); err != nil || decoded != Normalize(text) {
			t.Fatal("cache alias")
		}
	}
	if _, err := tok.Decode([]int{264}); err == nil {
		t.Fatal("accepted incomplete UTF-8 byte")
	}
	if _, err := tok.Decode([]int{265}); err == nil {
		t.Fatal("accepted out of range ID")
	}
	for _, id := range []int{-1, 0, 265} {
		if _, err := tok.Expand([]int{id}); err == nil {
			t.Fatal("Expand accepted invalid ID", id)
		}
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				if got := tok.Encode("abab abab"); !reflect.DeepEqual(got, []int{262, 262, 263, 262}) {
					t.Error(got)
				}
			}
		}()
	}
	wg.Wait()
	base.CacheWords = fused.CacheWords
	if err := base.Save(path); err != nil {
		t.Fatal(err)
	}
	cached, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cached.Encode("abab 🙂 a\x00b"), plain.Encode("abab 🙂 a\x00b")) {
		t.Fatal("word cache changed IDs")
	}
}

func TestRejectInvalidFusionAndCache(t *testing.T) {
	for _, model := range []Model{
		{Version: 1, Algorithm: "bpe", SpaceFusion: []int{4}},
		{Version: 2, Algorithm: "unigram", SpaceFusion: []int{4}},
		{Version: 2, Algorithm: "bpe", SpaceFusion: []int{0}},
		{Version: 2, Algorithm: "bpe", SpaceFusion: []int{36}},
		{Version: 2, Algorithm: "bpe", SpaceFusion: []int{260}},
		{Version: 2, Algorithm: "bpe", SpaceFusion: []int{4, 4}},
		{Version: 1, Algorithm: "bpe", CacheWords: []string{"a b"}},
		{Version: 1, Algorithm: "bpe", CacheWords: []string{""}},
		{Version: 1, Algorithm: "bpe", CacheWords: []string{"a\xff"}},
	} {
		model.Normalization = Normalization
		if _, err := New(model); err == nil {
			t.Fatalf("accepted %+v", model)
		}
	}
}
