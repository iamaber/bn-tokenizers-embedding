// Package corpus prepares provenance-preserving, grouped tokenizer splits.
package corpus

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

type Record struct {
	Text        string `json:"text"`
	Bucket      string `json:"bucket"`
	Source      string `json:"source"`
	Group       string `json:"group"`
	Attribution string `json:"attribution"`
	Split       string `json:"split,omitempty"`
}

type Stats struct {
	Input      int            `json:"input_records"`
	Rejected   int            `json:"quality_rejected"`
	Duplicates int            `json:"duplicates_removed"`
	NearLinks  int            `json:"near_duplicate_links"`
	Counts     map[string]int `json:"split_bucket_records"`
	Characters map[string]int `json:"split_bucket_characters"`
}

// Key catches punctuation, case and whitespace variants without changing the
// text used for fitting. It is a leakage/dedup key, not inference normalization.
func Key(s string) string {
	var b strings.Builder
	for _, r := range tokenizer.Normalize(s) {
		if unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r) {
			b.WriteRune(unicode.ToLower(r))
		}
	}
	return b.String()
}

// Prepare groups paired forms, article paragraphs, exact canonical duplicates,
// and long-text SimHash neighbors before assigning splits. Official test/dev
// membership wins over hash assignment and propagates to connected forms.
func Prepare(records []Record, out string) (Stats, error) {
	st := Stats{Input: len(records), Counts: map[string]int{}, Characters: map[string]int{}}
	prepared, err := prepareRecords(records, &st)
	if err != nil {
		return st, err
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		return st, err
	}
	files := map[string]*os.File{}
	writers := map[string]*bufio.Writer{}
	for _, s := range []string{"train", "validation", "test"} {
		f, err := os.Create(filepath.Join(out, s+".jsonl"))
		if err != nil {
			return st, err
		}
		files[s] = f
		defer f.Close()
		writers[s] = bufio.NewWriter(f)
	}
	for r := range prepared {
		if err := json.NewEncoder(writers[r.Split]).Encode(r); err != nil {
			return st, err
		}
		k := r.Split + "/" + r.Bucket
		st.Counts[k]++
		st.Characters[k] += utf8.RuneCountInString(r.Text)
	}
	for s, w := range writers {
		if err := w.Flush(); err != nil {
			return st, err
		}
		if err := files[s].Sync(); err != nil {
			return st, err
		}
	}
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return st, err
	}
	err = os.WriteFile(filepath.Join(out, "stats.json"), b, 0644)
	return st, err
}

func Read(path string, fn func(Record) error) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 65536), 4<<20)
	for s.Scan() {
		var r Record
		if err := json.Unmarshal(s.Bytes(), &r); err != nil {
			return err
		}
		if err := fn(r); err != nil {
			return err
		}
	}
	return s.Err()
}

// WordCounts gives each sentence equal mass within its bucket, then applies
// explicit bucket weights. It never infers Romanized Bangla from Latin script.
func WordCounts(path string, mixture map[string]float64) (map[string]float64, error) {
	perBucket := map[string]map[string]float64{}
	sentences := map[string]int{}
	err := Read(path, func(r Record) error {
		if r.Split != "train" {
			return fmt.Errorf("refusing to fit on split %q", r.Split)
		}
		if perBucket[r.Bucket] == nil {
			perBucket[r.Bucket] = map[string]float64{}
		}
		words := strings.Fields(tokenizer.Normalize(r.Text))
		if len(words) == 0 {
			return nil
		}
		sentences[r.Bucket]++
		for _, w := range words {
			if utf8.RuneCountInString(w) <= 64 {
				perBucket[r.Bucket][w] += 1 / float64(len(words))
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	counts := map[string]float64{}
	buckets := make([]string, 0, len(mixture))
	for bucket := range mixture {
		buckets = append(buckets, bucket)
	}
	sort.Strings(buckets)
	for _, bucket := range buckets {
		weight := mixture[bucket]
		if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("nonpositive mixture weight")
		}
		if sentences[bucket] == 0 {
			return nil, fmt.Errorf("missing required bucket %s", bucket)
		}
		for w, n := range perBucket[bucket] {
			counts[w] += n * weight * 1e6 / float64(sentences[bucket])
		}
	}
	return counts, nil
}

// CopyJSON writes a small reproducibility report.
func CopyJSON(w io.Writer, v any) error {
	e := json.NewEncoder(w)
	e.SetIndent("", "  ")
	return e.Encode(v)
}
