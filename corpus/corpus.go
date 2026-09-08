// Package corpus prepares provenance-preserving, grouped tokenizer splits.
package corpus

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"regexp"
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

var privateText = regexp.MustCompile(`(?i)([a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|(?:\+?880|\b0)1[3-9][0-9]{8}\b|(?:api[_-]?key|password|secret)\s*[:=]\s*\S+)`)

func acceptable(s string) bool {
	n := utf8.RuneCountInString(s)
	return n >= 12 && n <= 2000 && !privateText.MatchString(s) && !strings.ContainsRune(s, '\uFFFD') && len(Key(s)) > 0
}

type groups struct{ parent []int }

func (g *groups) root(i int) int {
	for g.parent[i] != i {
		g.parent[i] = g.parent[g.parent[i]]
		i = g.parent[i]
	}
	return i
}
func (g *groups) join(a, b int) {
	a = g.root(a)
	b = g.root(b)
	if a > b {
		a, b = b, a
	}
	g.parent[b] = a
}

// Prepare groups paired forms, article paragraphs, exact canonical duplicates,
// and long-text SimHash neighbors before assigning splits. Official test/dev
// membership wins over hash assignment and propagates to connected forms.
func Prepare(records []Record, out string) (Stats, error) {
	st := Stats{Input: len(records), Counts: map[string]int{}, Characters: map[string]int{}}
	g := groups{make([]int, len(records))}
	for i := range g.parent {
		g.parent[i] = i
	}
	byGroup := map[string]int{}
	exact := map[string]int{}
	type signature struct {
		index int
		hash  uint64
	}
	bands := map[uint64][]signature{}
	for i := range records {
		r := &records[i]
		if r.Group == "" || r.Source == "" || r.Bucket == "" {
			return st, fmt.Errorf("record %d requires group, source and bucket", i)
		}
		if r.Split != "" && r.Split != "train" && r.Split != "validation" && r.Split != "test" {
			return st, fmt.Errorf("invalid split %q", r.Split)
		}
		r.Text = tokenizer.Normalize(r.Text)
		if j, ok := byGroup[r.Group]; ok {
			g.join(i, j)
		} else {
			byGroup[r.Group] = i
		}
		key := Key(r.Text)
		if key == "" {
			continue
		}
		if j, ok := exact[key]; ok {
			g.join(i, j)
		} else {
			exact[key] = i
		}
		h, ok := simhash(r.Text)
		if !ok {
			continue
		}
		for band := 0; band < 4; band++ {
			k := uint64(band)<<16 | ((h >> uint(16*band)) & 65535)
			for _, v := range bands[k] {
				if bits.OnesCount64(h^v.hash) <= 3 {
					if g.root(i) != g.root(v.index) {
						st.NearLinks++
						g.join(i, v.index)
					}
				}
			}
			bands[k] = append(bands[k], signature{i, h})
		}
	}
	priority := map[int]int{}
	groupName := map[int]string{}
	for i, r := range records {
		root := g.root(i)
		p := 0
		if r.Split == "validation" {
			p = 1
		}
		if r.Split == "test" {
			p = 2
		}
		priority[root] = max(priority[root], p)
		if name, ok := groupName[root]; !ok || r.Group < name {
			groupName[root] = r.Group
		}
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
	seen := map[string]bool{}
	for i, r := range records {
		if !acceptable(r.Text) {
			st.Rejected++
			continue
		}
		key := Key(r.Text)
		if seen[key] {
			st.Duplicates++
			continue
		}
		seen[key] = true
		root := g.root(i)
		r.Group = groupName[root]
		r.Split = splitFor(r.Group)
		if priority[root] == 1 {
			r.Split = "validation"
		}
		if priority[root] == 2 {
			r.Split = "test"
		}
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

func splitFor(group string) string {
	h := sha256.Sum256([]byte("bn-corpus-v1:" + group))
	n := binary.BigEndian.Uint64(h[:8]) % 100
	if n < 2 {
		return "test"
	}
	if n < 4 {
		return "validation"
	}
	return "train"
}

func simhash(s string) (uint64, bool) {
	words := strings.Fields(strings.ToLower(s))
	if len(words) < 12 {
		return 0, false
	}
	var weights [64]int
	for _, w := range words {
		h := sha256.Sum256([]byte(w))
		v := binary.LittleEndian.Uint64(h[:8])
		for b := range 64 {
			if v&(uint64(1)<<b) != 0 {
				weights[b]++
			} else {
				weights[b]--
			}
		}
	}
	var h uint64
	for b, w := range weights {
		if w >= 0 {
			h |= uint64(1) << b
		}
	}
	return h, true
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
