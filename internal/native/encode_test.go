package native

import (
	"encoding/binary"
	"reflect"
	"testing"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

func TestBinaryEncode(t *testing.T) {
	model, err := tokenizer.New(tokenizer.Model{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}})
	if err != nil {
		t.Fatal(err)
	}
	r := Registry{models: map[uint64]*tokenizer.Tokenizer{1: model}}
	texts := []string{"আমি", "", "a\x00b 🙂", "\t বাংলা\n"}
	var input []byte
	for _, text := range texts {
		input = binary.LittleEndian.AppendUint32(input, uint32(len(text)))
		input = append(input, text...)
	}
	output := r.EncodeBinary(1, input)
	if output[0] != 0 {
		t.Fatalf("encode: %s", output[1:])
	}
	output = output[1:]
	for _, text := range texts {
		n := int(binary.LittleEndian.Uint32(output))
		output = output[4:]
		ids := make([]int, n)
		for i := range ids {
			ids[i] = int(binary.LittleEndian.Uint32(output))
			output = output[4:]
		}
		if !reflect.DeepEqual(ids, model.Encode(text)) {
			t.Fatalf("different IDs for %q", text)
		}
	}
	if len(output) != 0 {
		t.Fatal("trailing output")
	}
	if got := r.EncodeBinary(1, nil); !reflect.DeepEqual(got, []byte{0}) {
		t.Fatal(got)
	}
	for _, input := range [][]byte{{0}, {255, 255, 255, 255}, {2, 0, 0, 0, 'a'}} {
		if r.EncodeBinary(1, input)[0] != 1 {
			t.Fatal("accepted malformed input")
		}
	}
	if r.EncodeBinary(99, nil)[0] != 1 {
		t.Fatal("accepted closed handle")
	}
}

func FuzzBinaryEncode(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0})
	f.Add([]byte{1, 0, 0, 0, 'a'})
	var r Registry
	f.Fuzz(func(t *testing.T, input []byte) {
		if result := r.EncodeBinary(1, input); len(result) == 0 || result[0] != 1 {
			t.Fatal("invalid handle must return an error")
		}
	})
}
