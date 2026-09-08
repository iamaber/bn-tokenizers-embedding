package corpus

import (
	"path/filepath"
	"testing"
)

func TestHeldOutPropagation(t *testing.T) {
	r := []Record{
		{Text: "ami aj onek bhalo achi", Bucket: "banglish", Group: "pair1", Split: "test"},
		{Text: "আমি আজ অনেক ভালো আছি", Bucket: "bangla", Group: "pair1", Split: "train"},
		{Text: "AMI AJ ONEK BHALO ACHI!!!", Bucket: "banglish", Group: "pair2", Split: "train"},
		{Text: "এই বাক্যটি একই দলে থাকবে", Bucket: "bangla", Group: "pair2", Split: "train"},
		{Text: "Please email person@example.com", Bucket: "english", Group: "private"},
	}
	for i := range r {
		r[i].Source = "fixture"
	}
	out := t.TempDir()
	st, err := Prepare(r, out)
	if err != nil {
		t.Fatal(err)
	}
	if st.Duplicates != 1 || st.Rejected != 1 {
		t.Fatalf("bad accounting %+v", st)
	}
	count := 0
	for _, split := range []string{"train", "validation", "test"} {
		err := Read(filepath.Join(out, split+".jsonl"), func(r Record) error {
			count++
			if split != "test" {
				t.Errorf("paired record leaked into %s", split)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if count != 3 {
		t.Fatalf("got %d records", count)
	}
}

func TestNormalizationAndPrivacy(t *testing.T) {
	if Key("Café!") != Key("cafe\u0301") {
		t.Fatal("canonical key mismatch")
	}
	if acceptable("Call 01712345678 now") || acceptable("password = notasecret") {
		t.Fatal("privacy filter failure")
	}
	if !acceptable("আমি ভালো আছি আজকে") {
		t.Fatal("rejected Bangla")
	}
}
