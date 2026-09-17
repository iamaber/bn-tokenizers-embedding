package corpus

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"iter"
	"math/bits"
	"regexp"
	"strings"
	"unicode/utf8"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

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

// prepareRecords normalizes the input in place and connects all records before
// choosing splits: even rejected text can bridge groups or carry heldout priority.
// The returned sequence is consumed once, in input order. Filtering is deferred
// until consumption so output errors preserve Prepare's partial statistics;
// the writer accounts for records only after successfully encoding them.
// Audits intentionally reconstruct their checks from saved files independently.
func prepareRecords(records []Record, st *Stats) (iter.Seq[Record], error) {
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
			return nil, fmt.Errorf("record %d requires group, source and bucket", i)
		}
		if r.Split != "" && r.Split != "train" && r.Split != "validation" && r.Split != "test" {
			return nil, fmt.Errorf("invalid split %q", r.Split)
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
	return func(yield func(Record) bool) {
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
			if !yield(r) {
				return
			}
		}
	}, nil
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
