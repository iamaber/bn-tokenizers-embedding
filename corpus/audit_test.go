package corpus

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestAuditRereadsCorruptedSplits(t *testing.T) {
	for _, tc := range []struct {
		name   string
		record Record
		want   string
	}{
		{"canonical duplicate", Record{Text: "CAFÉ HAS GOOD COFFEE!!!", Group: "new", Split: "validation"}, "duplicate text"},
		{"group overlap", Record{Text: "A newly injected sentence", Group: "solo", Split: "validation"}, "group overlaps"},
		{"mislabeled split", Record{Text: "A newly injected sentence", Group: "new", Split: "train"}, "mislabeled split"},
		{"empty group", Record{Text: "A newly injected sentence", Split: "validation"}, "empty group"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := t.TempDir()
			if _, err := Prepare(preparationFixture(), out); err != nil {
				t.Fatal(err)
			}
			if _, err := Audit(out); err != nil {
				t.Fatalf("valid corpus failed audit: %v", err)
			}
			writeAuditRecords(t, filepath.Join(out, "validation.jsonl"), tc.record)
			if _, err := Audit(out); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("corruption not detected (%s): %v", tc.want, err)
			}
		})
	}
}

// Write saved rows directly: corruption checks must not route through Prepare's
// production decisions, which would remove duplicates and propagate holdouts.
func writeAuditRecords(t *testing.T, path string, records ...Record) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, r := range records {
		if err := json.NewEncoder(f).Encode(r); err != nil {
			t.Fatal(err)
		}
	}
}

func TestAuditOfficialRereadsSavedAndRawHoldouts(t *testing.T) {
	for _, split := range []string{"validation", "test"} {
		for _, form := range []string{"AMI AJ ONEK BHALO ACHI!!!", "আমি আজ অনেক ভালো আছি!"} {
			for _, corrupt := range []string{"saved training", "raw holdout"} {
				t.Run(split+"/"+form+"/"+corrupt, func(t *testing.T) {
					prepared, raw := t.TempDir(), t.TempDir()
					dir := filepath.Join(raw, "banglatlit")
					if err := os.MkdirAll(dir, 0755); err != nil {
						t.Fatal(err)
					}
					for _, name := range []string{"validation", "test"} {
						if err := os.WriteFile(filepath.Join(dir, name+".csv"), []byte("id,banglish,bangla\n"), 0644); err != nil {
							t.Fatal(err)
						}
					}
					writeHoldout := func() {
						t.Helper()
						if err := os.WriteFile(filepath.Join(dir, split+".csv"), []byte("id,banglish,bangla\n1,ami aj onek bhalo achi,আমি আজ অনেক ভালো আছি\n"), 0644); err != nil {
							t.Fatal(err)
						}
					}
					train := filepath.Join(prepared, "train.jsonl")
					leak := Record{Text: form, Group: "imported-pt", Source: "pt", Split: "train"}
					if corrupt == "saved training" {
						writeHoldout()
						writeAuditRecords(t, train, Record{Text: "An unrelated training sentence", Group: "safe", Split: "train"})
					} else {
						writeAuditRecords(t, train, leak)
					}
					if _, err := AuditOfficial(prepared, raw); err != nil {
						t.Fatalf("initial independent audit failed: %v", err)
					}
					if corrupt == "saved training" {
						writeAuditRecords(t, train, leak)
					} else {
						writeHoldout()
					}
					count, err := AuditOfficial(prepared, raw)
					if count != 2 || err == nil || !strings.Contains(err.Error(), "official holdout leaked") {
						t.Fatalf("changed %s was not detected: count=%d, err=%v", corrupt, count, err)
					}
				})
			}
		}
	}
}

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
