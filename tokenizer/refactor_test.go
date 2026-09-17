package tokenizer

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func refactorCounts() map[string]float64 {
	return map[string]float64{"ami": 30.25, "amra": 20.5, "bhalo": 30, "valo": 10.125, "hello": 10, "আমি": 30, "ভালো": 20, "aaaa": 7.75, "abab": 7.75}
}

func refactorConfig() TrainConfig {
	return TrainConfig{VocabSize: 284, SeedSize: 310, MaxPieceRunes: 6, EMIterations: 3}
}

func TestTrainingGolden(t *testing.T) {
	for _, algorithm := range []string{"unigram", "bpe"} {
		t.Run(algorithm, func(t *testing.T) {
			var m Model
			var err error
			if algorithm == "unigram" {
				m, err = Train(refactorCounts(), refactorConfig())
			} else {
				m, err = TrainBPE(refactorCounts(), 284, 6, nil)
			}
			if err != nil {
				t.Fatal(err)
			}
			data, err := json.Marshal(m)
			if err != nil {
				t.Fatal(err)
			}
			want := map[string]string{
				"unigram": "960b9f7a05b457231dfbbb3c2ef82d7a0712433b66d1c5abc715ddc60a6301a7",
				"bpe":     "e8ed574fd550594a8b7a77c5eec5362d7ca4d1fa761b85fae7dcbaf2e438a7ae",
			}[algorithm]
			if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
				t.Fatalf("serialized model hash = %s, want %s", got, want)
			}
		})
	}
}

func TestSamplingGolden(t *testing.T) {
	tok := testModel(t)
	rng := rand.New(rand.NewSource(947))
	var outputs [][]int
	for _, alpha := range []float64{1, .01, .5, 2} {
		for _, text := range []string{"abc ab ami আমি", "\t🙂 a\xffb e\u0301\n", "", "aaaa abc"} {
			ids, err := tok.Sample(text, alpha, rng)
			if err != nil {
				t.Fatal(err)
			}
			outputs = append(outputs, ids)
		}
	}
	data, err := json.Marshal(outputs)
	if err != nil {
		t.Fatal(err)
	}
	const want = "b419266e1e824b19ff403bbf00df1e2fa7b8d13e0012647b8548bb5fce9dbc3a"
	if got := fmt.Sprintf("%x", sha256.Sum256(data)); got != want {
		t.Fatalf("sample hash = %s, want %s; IDs: %s", got, want, data)
	}
	if got := rng.Int63(); got != 8802221791825287930 {
		t.Fatalf("RNG stream changed: %d", got)
	}
}

func TestTrainingValidationContract(t *testing.T) {
	for _, tc := range []struct {
		name         string
		counts       map[string]float64
		vocab        int
		unigram, bpe string
	}{
		{"config first", map[string]float64{"": -1}, 260, "invalid training configuration", "invalid BPE configuration"},
		{"empty", nil, 261, "empty training corpus", "insufficient merge candidates for vocabulary 261"},
		{"candidate first", map[string]float64{"a": 1}, 263, "only 1 candidate pieces: insufficient corpus for vocabulary 263", "insufficient merge candidates for vocabulary 263"},
		{"alphabet", map[string]float64{"abc": 1}, 262, "vocabulary too small for 3 characters and 260 reserved IDs", "vocabulary too small"},
		{"invalid first", map[string]float64{"abc": 1, "z z": 1}, 261, "invalid training word", "invalid word"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := refactorConfig()
			c.VocabSize = tc.vocab
			_, err := Train(tc.counts, c)
			if err == nil || err.Error() != tc.unigram {
				t.Fatalf("Unigram: %v", err)
			}
			_, err = TrainBPE(tc.counts, tc.vocab, 6, nil)
			if err == nil || err.Error() != tc.bpe {
				t.Fatalf("BPE: %v", err)
			}
		})
	}
	for _, counts := range []map[string]float64{
		{"": 1}, {"a b": 1}, {" a": 1}, {"a\tb": 1}, {"e\u0301": 1}, {"\xff": 1},
		{"a": 0}, {"a": -1}, {"a": math.NaN()}, {"a": math.Inf(1)}, {"a": math.Inf(-1)},
	} {
		if _, err := Train(counts, refactorConfig()); err == nil || err.Error() != "invalid training word" {
			t.Fatal(err)
		}
		if _, err := TrainBPE(counts, 284, 6, nil); err == nil || err.Error() != "invalid word" {
			t.Fatal(err)
		}
	}
}

func TestUnigramTieAndFallbackContract(t *testing.T) {
	tok, err := New(Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: []Piece{{"a", -30}, {"b", -1}, {"ab", -31}}})
	if err != nil {
		t.Fatal(err)
	}
	// Equal scores retain the first path; byte fallback precedes piece matches.
	if got, want := tok.Encode("a ab 🙂"), []int{101, 36, 262, 36, 244, 163, 157, 134}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestUnigramTrainingExcludesFallback(t *testing.T) {
	tok, err := New(Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: []Piece{{"আমি", -1000}}})
	if err != nil {
		t.Fatal(err)
	}
	// This learned path loses to bytes during inference but must receive the
	// entire weighted expected count during training, even at this low score.
	expected, loss := tok.unigramExpectation([]word{{"আমি", 2.5}})
	if !reflect.DeepEqual(expected, []float64{2.5}) || loss != 2500 {
		t.Fatalf("expected counts %v, loss %g", expected, loss)
	}
	want := make([]int, 0, len("আমি"))
	for _, b := range []byte("আমি") {
		want = append(want, ByteOffset+int(b))
	}
	if got := tok.Encode("আমি"); !reflect.DeepEqual(got, want) {
		t.Fatalf("Encode: %v, want %v", got, want)
	}
	got, err := tok.Sample("আমি", 1, rand.New(rand.NewSource(42)))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("Sample: %v, %v; want %v", got, err, want)
	}
}

func BenchmarkRefactor(b *testing.B) {
	b.Run("TrainUnigram", func(b *testing.B) {
		counts, config := refactorCounts(), refactorConfig()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := Train(counts, config); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("TrainBPE", func(b *testing.B) {
		counts := refactorCounts()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := TrainBPE(counts, 284, 6, nil); err != nil {
				b.Fatal(err)
			}
		}
	})
	for _, alpha := range []float64{1, .01} {
		b.Run(fmt.Sprintf("Sample/%g", alpha), func(b *testing.B) {
			tok, rng := testModel(b), rand.New(rand.NewSource(947))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := tok.Sample("abc আমি 🙂 ami", alpha, rng); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
