package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/iamaber/bn-tokenizers-embedding/corpus"
)

func TestPrepareRecordsPreservesHoldoutsAndRefusesOverwrite(t *testing.T) {
	root := t.TempDir()
	input := filepath.Join(root, "records.jsonl")
	text := `{"text":"এটি একটি পরীক্ষার বাক্য","bucket":"bangla","source":"fixture","group":"one","split":"train"}
{"text":"This is the aligned sentence","bucket":"english","source":"fixture","group":"one","split":"test"}
`
	if err := os.WriteFile(input, []byte(text), 0600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(root, "prepared")
	if err := run([]string{"prepare-records", "-input", input, "-out", out}); err != nil {
		t.Fatal(err)
	}
	count := 0
	if err := corpus.Read(filepath.Join(out, "test.jsonl"), func(r corpus.Record) error {
		count++
		if r.Split != "test" {
			t.Fatal("heldout priority lost")
		}
		return nil
	}); err != nil || count != 2 {
		t.Fatalf("read test: count=%d err=%v", count, err)
	}
	if err := run([]string{"prepare-records", "-input", input, "-out", out}); err == nil || !strings.Contains(err.Error(), "exists") {
		t.Fatalf("overwrite must fail: %v", err)
	}
	if err := run([]string{"prepare-records"}); err == nil {
		t.Fatal("missing input/output must fail")
	}
}
