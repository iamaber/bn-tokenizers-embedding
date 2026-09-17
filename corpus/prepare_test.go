package corpus

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func preparationFixture() []Record {
	return []Record{
		{Text: "  Cafe\u0301   has good coffee  ", Bucket: "english", Source: "fixture", Group: "solo", Attribution: "A&B"},
		{Text: "one two three four five six seven eight nine ten eleven twelve", Bucket: "english", Source: "fixture", Group: "z-near", Split: "train"},
		{Text: "twelve eleven ten nine eight seven six five four three two one", Bucket: "english", Source: "fixture", Group: "y-near", Split: "validation"},
		{Text: "tiny bridge", Bucket: "english", Source: "fixture", Group: "y-near"},
		{Text: "TINY BRIDGE!", Bucket: "english", Source: "fixture", Group: "a-heldout", Split: "test"},
		{Text: "আমি আজ অনেক ভালো আছি", Bucket: "bangla", Source: "fixture", Group: "a-heldout"},
		{Text: "ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE!!!", Bucket: "english", Source: "duplicate", Group: "copy"},
		{Text: "A separate validation sentence", Bucket: "english", Source: "fixture", Group: "validation-only", Split: "validation"},
		{Text: "Please email person@example.com", Bucket: "english", Source: "fixture", Group: "private"},
	}
}

func TestPrepareFilteredBridge(t *testing.T) {
	records := []Record{
		{Text: "A sentence from the training side", Bucket: "english", Source: "fixture", Group: "z-train", Split: "train"},
		{Text: "tiny", Bucket: "english", Source: "fixture", Group: "z-train"},
		{Text: "TINY!", Bucket: "english", Source: "fixture", Group: "a-test", Split: "test"},
	}
	out := t.TempDir()
	st, err := Prepare(records, out)
	if err != nil {
		t.Fatal(err)
	}
	if st.Rejected != 2 || st.Duplicates != 0 || st.Counts["test/english"] != 1 {
		t.Fatalf("rejected bridge must carry test priority: %+v", st)
	}
	if err := Read(filepath.Join(out, "test.jsonl"), func(r Record) error {
		if r.Group != "a-test" || r.Split != "test" {
			return fmt.Errorf("wrong connected group: %+v", r)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareFiltersBeforeDedupe(t *testing.T) {
	records := []Record{
		{Text: "password = value here", Bucket: "english", Source: "private", Group: "heldout", Split: "test"},
		{Text: "password value here", Bucket: "english", Source: "first-acceptable", Group: "copy"},
		{Text: "PASSWORD VALUE HERE!", Bucket: "english", Source: "duplicate", Group: "copy"},
	}
	out := t.TempDir()
	st, err := Prepare(records, out)
	if err != nil {
		t.Fatal(err)
	}
	if st.Rejected != 1 || st.Duplicates != 1 || st.Counts["test/english"] != 1 {
		t.Fatalf("unexpected accounting: %+v", st)
	}
	if err := Read(filepath.Join(out, "test.jsonl"), func(r Record) error {
		if r.Source != "first-acceptable" {
			return fmt.Errorf("lost first acceptable provenance: %+v", r)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareErrorTiming(t *testing.T) {
	for _, invalid := range []Record{
		{Text: "  leave   untouched ", Bucket: "english", Source: "fixture"},
		{Text: "  leave   untouched ", Bucket: "english", Source: "fixture", Group: "bad", Split: "invalid"},
	} {
		t.Run(invalid.Split+invalid.Group, func(t *testing.T) {
			records := preparationFixture()[:3]
			records = append(records, invalid, Record{Text: "  also   untouched "})
			out := filepath.Join(t.TempDir(), "not-created")
			st, err := Prepare(records, out)
			if err == nil || st.Input != 5 || st.NearLinks != 1 || st.Rejected != 0 || len(st.Counts) != 0 {
				t.Fatalf("unexpected validation result: %+v, %v", st, err)
			}
			if records[0].Text != "Café has good coffee" || records[3] != invalid || records[4].Text != "  also   untouched " {
				t.Fatalf("wrong normalization boundary: %+v", records)
			}
			if _, err := os.Stat(out); !os.IsNotExist(err) {
				t.Fatalf("validation must precede output creation: %v", err)
			}
		})
	}
	t.Run("output creation", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "file")
		if err := os.WriteFile(out, nil, 0644); err != nil {
			t.Fatal(err)
		}
		records := preparationFixture()
		st, err := Prepare(records, out)
		if err == nil || st.NearLinks != 1 || st.Rejected != 0 || st.Duplicates != 0 || len(st.Counts) != 0 || len(st.Characters) != 0 {
			t.Fatalf("filtering ran before output creation: %+v, %v", st, err)
		}
		if records[0].Text != "Café has good coffee" {
			t.Fatal("normalization must precede output creation")
		}
	})
}

// Literal bytes captured from Prepare before extracting its decision pipeline.
func TestPrepareOutputBaseline(t *testing.T) {
	records := preparationFixture()
	before := append([]Record(nil), records...)
	out := t.TempDir()
	st, err := Prepare(records, out)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"train.jsonl":      `{"text":"Café has good coffee","bucket":"english","source":"fixture","group":"solo","attribution":"A\u0026B","split":"train"}` + "\n",
		"validation.jsonl": `{"text":"A separate validation sentence","bucket":"english","source":"fixture","group":"validation-only","attribution":"","split":"validation"}` + "\n",
		"test.jsonl": `{"text":"one two three four five six seven eight nine ten eleven twelve","bucket":"english","source":"fixture","group":"a-heldout","attribution":"","split":"test"}
{"text":"twelve eleven ten nine eight seven six five four three two one","bucket":"english","source":"fixture","group":"a-heldout","attribution":"","split":"test"}
{"text":"TINY BRIDGE!","bucket":"english","source":"fixture","group":"a-heldout","attribution":"","split":"test"}
{"text":"আমি আজ অনেক ভালো আছি","bucket":"bangla","source":"fixture","group":"a-heldout","attribution":"","split":"test"}
`,
		"stats.json": `{
  "input_records": 9,
  "quality_rejected": 2,
  "duplicates_removed": 1,
  "near_duplicate_links": 1,
  "split_bucket_records": {
    "test/bangla": 1,
    "test/english": 3,
    "train/english": 1,
    "validation/english": 1
  },
  "split_bucket_characters": {
    "test/bangla": 20,
    "test/english": 136,
    "train/english": 20,
    "validation/english": 30
  }
}`,
	}
	for name, expected := range want {
		got, err := os.ReadFile(filepath.Join(out, name))
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != expected {
			t.Errorf("%s bytes differ:\n%s", name, got)
		}
	}
	before[0].Text = "Café has good coffee"
	if !reflect.DeepEqual(records, before) {
		t.Errorf("Prepare must mutate only normalized input text: %+v", records)
	}
	expectedStats := Stats{Input: 9, Rejected: 2, Duplicates: 1, NearLinks: 1,
		Counts:     map[string]int{"test/bangla": 1, "test/english": 3, "train/english": 1, "validation/english": 1},
		Characters: map[string]int{"test/bangla": 20, "test/english": 136, "train/english": 20, "validation/english": 30}}
	if !reflect.DeepEqual(st, expectedStats) {
		t.Fatalf("stats differ: %+v", st)
	}
}
