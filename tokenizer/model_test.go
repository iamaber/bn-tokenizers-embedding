package tokenizer

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func testModel(t testing.TB) *Tokenizer {
	t.Helper()
	tok, err := New(Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: []Piece{{"আমি", -1}, {"ভালো", -2}, {"ami", -1}, {"bhalo", -2}, {"a", -1}, {"ab", -2}, {"bc", -.1}, {"c", -10}}})
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func TestRoundTrip(t *testing.T) {
	tok := testModel(t)
	for _, s := range []string{"", "  আমি\tভালো\nআছি। ", "ami aaj valo asi", "meetingটা filesগুলো updateটা", "ক্\u200Dষ ক্\u200Cষ", "e\u0301 café", "１２৩ ১২৩ 123", "<unk> ▁ <0x20>", "🙂🚀 https://example.org/?q=ami", "Hello\u00a0WORLD", "a\xffb"} {
		got, err := tok.Decode(tok.Encode(s))
		if err != nil || got != Normalize(s) {
			t.Fatalf("round trip %q: %q %v", s, got, err)
		}
		if Normalize(Normalize(s)) != Normalize(s) {
			t.Fatal("normalization not idempotent")
		}
	}
}

func TestViterbiNotGreedy(t *testing.T) {
	tok := testModel(t)
	got := tok.Encode("abc")
	want := []int{PieceOffset + 4, PieceOffset + 6}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestInvalidModelAndIDs(t *testing.T) {
	for _, pieces := range [][]Piece{{{"", -1}}, {{"a", math.NaN()}}, {{"a", 1}}, {{"a", -1}, {"a", -2}}} {
		if _, err := New(Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: pieces}); err == nil {
			t.Fatal("accepted invalid model")
		}
	}
	tok := testModel(t)
	for _, ids := range [][]int{{0}, {-1}, {99999}, {ByteOffset + 255}} {
		if _, err := tok.Decode(ids); err == nil {
			t.Fatalf("accepted %v", ids)
		}
	}
}

func TestTrainAndPersist(t *testing.T) {
	c := TrainConfig{VocabSize: 280, SeedSize: 320, MaxPieceRunes: 6, EMIterations: 2}
	counts := map[string]float64{"ami": 30, "amra": 20, "bhalo": 30, "valo": 10, "hello": 10, "আমি": 30, "ভালো": 20}
	a, err := Train(counts, c)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Train(counts, c)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("training is nondeterministic")
	}
	path := filepath.Join(t.TempDir(), "model.json")
	if err := a.Save(path); err != nil {
		t.Fatal(err)
	}
	tok, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	s := "আমি bhalo unseen 🦊"
	decoded, err := tok.Decode(tok.Encode(s))
	if err != nil || decoded != s {
		t.Fatal(decoded, err)
	}
	if tok.VocabSize() > c.VocabSize {
		t.Fatal("vocabulary budget exceeded")
	}
}

func FuzzRoundTrip(f *testing.F) {
	f.Add("আমি ভালো আছি। ami valo")
	f.Add("\xff\x00🦊")
	tok := testModel(f)
	f.Fuzz(func(t *testing.T, s string) {
		if len(s) > 10000 {
			t.Skip()
		}
		got, err := tok.Decode(tok.Encode(s))
		if err != nil || got != Normalize(s) {
			t.Fatalf("%q %q %v", s, got, err)
		}
	})
}

func BenchmarkEncode(b *testing.B) {
	tok := testModel(b)
	s := strings.Repeat("আমি ভালো আছি। ami aj bhalo achi meetingটা কখন? ", 8)
	b.SetBytes(int64(len(s)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Encode(s)
	}
}

func TestSampling(t *testing.T) {
	tok, err := New(Model{Version: 1, Algorithm: "unigram", Normalization: Normalization, Pieces: []Piece{{"a", math.Log(.5)}, {"aa", math.Log(.5)}}})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(42))
	whole := 0
	for range 3000 {
		ids, err := tok.Sample("aa", 1, rng)
		if err != nil {
			t.Fatal(err)
		}
		s, err := tok.Decode(ids)
		if err != nil || s != "aa" {
			t.Fatal(s, err)
		}
		if len(ids) == 1 {
			whole++
		}
	}
	if math.Abs(float64(whole)/3000-2.0/3) > .04 {
		t.Fatalf("incorrect posterior sampling: %d", whole)
	}
}

func BenchmarkTrained(b *testing.B) {
	path := os.Getenv("BNTOK_MODEL")
	if path == "" {
		b.Skip("set BNTOK_MODEL to benchmark trained artifact")
	}
	tok, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	s := strings.Repeat("আমি আজ ভালো আছি। ami aaj valo asi. meetingটা কখন? The passport renewal fee is listed online. 🙂 ", 8)
	b.ReportAllocs()
	b.SetBytes(int64(len(s)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Encode(s)
	}
}
