package native

import (
	"encoding/binary"
	"encoding/json"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

// Both wire formats operate on handles created and released through the C control path.
func TestWireContracts(t *testing.T) {
	models := []tokenizer.Model{
		{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}},
		{Version: 1, Algorithm: "bpe", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "a"}, {Text: "b"}, {Text: "ab"}}, Merges: []tokenizer.Merge{{Left: 0, Right: 1, Result: 2}}},
	}
	for _, model := range models {
		t.Run(model.Algorithm, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "model.json")
			if err := model.Save(path); err != nil {
				t.Fatal(err)
			}
			var registry Registry
			control := func(req request) response {
				t.Helper()
				data, err := json.Marshal(req)
				if err != nil {
					t.Fatal(err)
				}
				var res response
				if err := json.Unmarshal(registry.Call(data), &res); err != nil {
					t.Fatal(err)
				}
				return res
			}
			loaded := control(request{Operation: "load", Path: path})
			if loaded.Error != "" || loaded.Handle == 0 || loaded.Vocabulary != tokenizer.PieceOffset+len(model.Pieces) {
				t.Fatal(loaded)
			}
			for _, texts := range [][]string{nil, {"", "ab", "আমি\tভালো 🙂", "a\x00b"}} {
				var input []byte
				for _, text := range texts {
					input = binary.LittleEndian.AppendUint32(input, uint32(len(text)))
					input = append(input, text...)
				}
				batch := control(request{Operation: "encode_batch", Handle: loaded.Handle, Texts: texts})
				if batch.Error != "" {
					t.Fatal(batch.Error)
				}
				want := []byte{0}
				for i, ids := range batch.Batch {
					single := control(request{Operation: "encode", Handle: loaded.Handle, Text: texts[i]})
					if single.Error != "" || !slices.Equal(single.IDs, ids) {
						t.Fatalf("single/batch mismatch: %+v, %v", single, ids)
					}
					decoded := control(request{Operation: "decode", Handle: loaded.Handle, IDs: ids})
					if decoded.Error != "" || decoded.Text != tokenizer.Normalize(texts[i]) {
						t.Fatal(decoded)
					}
					want = binary.LittleEndian.AppendUint32(want, uint32(len(ids)))
					for _, id := range ids {
						want = binary.LittleEndian.AppendUint32(want, uint32(id))
					}
				}
				if got := registry.EncodeBinary(loaded.Handle, input); !reflect.DeepEqual(got, want) {
					t.Fatalf("binary/JSON mismatch: %v != %v", got, want)
				}
			}
			for range 2 {
				if closed := control(request{Operation: "close", Handle: loaded.Handle}); closed.Error != "" {
					t.Fatal(closed.Error)
				}
			}
			closed := control(request{Operation: "encode_batch", Handle: loaded.Handle})
			if closed.Error != "unknown or closed tokenizer handle" {
				t.Fatal(closed)
			}
			if got := registry.EncodeBinary(loaded.Handle, nil); string(got) != "\x01"+closed.Error {
				t.Fatalf("binary close error: %q", got)
			}
		})
	}
}
