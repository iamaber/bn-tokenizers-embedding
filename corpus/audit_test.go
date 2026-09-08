package corpus

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestWordWeightsDeterministic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "train.jsonl")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, bucket := range []string{"bangla", "english", "banglish", "code-mixed"} {
		if err := json.NewEncoder(f).Encode(Record{Text: "shared one two", Bucket: bucket, Split: "train"}); err != nil {
			t.Fatal(err)
		}
	}
	f.Close()
	mix := map[string]float64{"bangla": .4, "english": .25, "banglish": .25, "code-mixed": .1}
	a, err := WordCounts(path, mix)
	if err != nil {
		t.Fatal(err)
	}
	for range 20 {
		b, err := WordCounts(path, mix)
		if err != nil || !reflect.DeepEqual(a, b) {
			t.Fatal("nondeterministic weights", err)
		}
	}
}

func TestRejectUngrouped(t *testing.T) {
	if _, err := Prepare([]Record{{Text: "this has no group", Bucket: "english", Source: "fixture"}}, t.TempDir()); err == nil {
		t.Fatal("accepted blank group")
	}
}

func TestManifestTampering(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "sample.txt")
	if err := os.WriteFile(p, []byte("hello"), 0644); err != nil {
		t.Fatal(err)
	}
	h, err := FileHash(p)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal([]Download{{Path: "sample.txt", Bytes: 5, SHA256: h, URL: "https://example.org"}})
	manifest := filepath.Join(dir, "lock.json")
	os.WriteFile(manifest, b, 0644)
	if err := VerifyManifest(manifest, dir); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(p, []byte("other"), 0644)
	if err := VerifyManifest(manifest, dir); err == nil {
		t.Fatal("accepted changed raw data")
	}
}
