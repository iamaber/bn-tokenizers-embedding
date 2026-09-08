package native

import (
	"encoding/json"
	"path/filepath"
	"sync"
	"testing"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

func TestRegistryLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.json")
	m := tokenizer.Model{Version: 1, Algorithm: "unigram", Normalization: tokenizer.Normalization, Pieces: []tokenizer.Piece{{Text: "আমি", Score: -1}}}
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	var registry Registry
	loaded, err := registry.call(request{Operation: "load", Path: path})
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	for range 16 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for range 20 {
				encoded, err := registry.call(request{Operation: "encode", Handle: loaded.Handle, Text: "আমি hello 🙂"})
				if err != nil {
					t.Error(err)
					return
				}
				decoded, err := registry.call(request{Operation: "decode", Handle: loaded.Handle, IDs: encoded.IDs})
				if err != nil || decoded.Text != "আমি hello 🙂" {
					t.Error(decoded, err)
					return
				}
			}
		}()
	}
	workers.Wait()
	if _, err := registry.call(request{Operation: "close", Handle: loaded.Handle}); err != nil {
		t.Fatal(err)
	}
	if len(registry.models) != 0 {
		t.Fatal("model handle retained after close")
	}
	if _, err := registry.call(request{Operation: "encode", Handle: loaded.Handle}); err == nil {
		t.Fatal("accepted closed handle")
	}
}

func TestMalformedRequest(t *testing.T) {
	var r Registry
	for _, s := range []string{"{", `{"op":"decode","ids":[true]}`, `{"op":"encode","handle":99}`} {
		var result response
		if err := json.Unmarshal(r.Call([]byte(s)), &result); err != nil {
			t.Fatal(err)
		}
		if result.Error == "" {
			t.Fatal("missing error")
		}
	}
}
