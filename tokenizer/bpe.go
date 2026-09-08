package tokenizer

import (
	"container/heap"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"unicode/utf8"
)

// Merge contains piece indexes (not external IDs) and is applied in list order.
type Merge struct {
	Left   int `json:"left"`
	Right  int `json:"right"`
	Result int `json:"result"`
}

func (t *Tokenizer) initBPE() error {
	t.model.Merges = append([]Merge(nil), t.model.Merges...)
	t.mergeRanks = map[[2]int]int{}
	t.characters = map[rune]int{}
	available := map[int]bool{}
	for i, p := range t.model.Pieces {
		if utf8.RuneCountInString(p.Text) == 1 {
			r, _ := utf8.DecodeRuneInString(p.Text)
			t.characters[r] = i
			available[i] = true
		}
	}
	for rank, m := range t.model.Merges {
		if !available[m.Left] || !available[m.Right] || m.Result < 0 || m.Result >= len(t.model.Pieces) {
			return fmt.Errorf("invalid BPE merge")
		}
		if t.model.Pieces[m.Result].Text != t.model.Pieces[m.Left].Text+t.model.Pieces[m.Right].Text {
			return fmt.Errorf("BPE merge text mismatch")
		}
		pair := [2]int{m.Left, m.Right}
		if _, ok := t.mergeRanks[pair]; ok {
			return fmt.Errorf("duplicate BPE merge")
		}
		t.mergeRanks[pair] = rank
		available[m.Result] = true
	}
	if len(available) != len(t.model.Pieces) {
		return fmt.Errorf("BPE vocabulary contains unreachable pieces")
	}
	return nil
}

func (t *Tokenizer) encodeBPE(s string) []int {
	ids := make([]int, 0, len(s))
	for _, r := range s {
		if id, ok := t.characters[r]; ok {
			ids = append(ids, PieceOffset+id)
		} else {
			for _, b := range []byte(string(r)) {
				ids = append(ids, ByteOffset+int(b))
			}
		}
	}
	for {
		best, pos := len(t.model.Merges), -1
		for i := 0; i+1 < len(ids); i++ {
			rank, ok := t.mergeRanks[[2]int{ids[i] - PieceOffset, ids[i+1] - PieceOffset}]
			if ok && rank < best {
				best, pos = rank, i
			}
		}
		if pos < 0 {
			break
		}
		merge := t.model.Merges[best]
		// Replace every non-overlapping occurrence in one pass, matching training.
		// Repeated long words must not rescan/shift the entire slice per occurrence.
		out := ids[:0]
		for i := 0; i < len(ids); i++ {
			if i+1 < len(ids) && ids[i] == PieceOffset+merge.Left && ids[i+1] == PieceOffset+merge.Right {
				out = append(out, PieceOffset+merge.Result)
				i++
			} else {
				out = append(out, ids[i])
			}
		}
		ids = out
	}
	return ids
}

type pairItem struct {
	pair     [2]int
	count    float64
	revision int
}
type pairHeap []pairItem

func (h pairHeap) Len() int { return len(h) }
func (h pairHeap) Less(i, j int) bool {
	a, b := h[i], h[j]
	if a.count != b.count {
		return a.count > b.count
	}
	if a.pair[0] != b.pair[0] {
		return a.pair[0] < b.pair[0]
	}
	return a.pair[1] < b.pair[1]
}
func (h pairHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *pairHeap) Push(v any)   { *h = append(*h, v.(pairItem)) }
func (h *pairHeap) Pop() any     { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }

// TrainBPE updates only words affected by a merge. Corpus weights, normalization,
// word boundaries, all observed characters and fallback match the Unigram run.
func TrainBPE(counts map[string]float64, vocab, maxPieceRunes int, log io.Writer) (Model, error) {
	m := Model{Version: 1, Algorithm: "bpe", Normalization: Normalization}
	if vocab <= PieceOffset || maxPieceRunes < 1 {
		return m, fmt.Errorf("invalid BPE configuration")
	}
	keys := make([]string, 0, len(counts))
	chars := map[string]bool{}
	for s, n := range counts {
		if s == "" || s != Normalize(s) || strings.Contains(s, " ") || n <= 0 || math.IsNaN(n) || math.IsInf(n, 0) {
			return m, fmt.Errorf("invalid word")
		}
		keys = append(keys, s)
		for _, r := range s {
			chars[string(r)] = true
		}
	}
	sort.Strings(keys)
	alphabet := make([]string, 0, len(chars))
	for s := range chars {
		alphabet = append(alphabet, s)
	}
	sort.Strings(alphabet)
	if len(alphabet)+PieceOffset > vocab {
		return m, fmt.Errorf("vocabulary too small")
	}
	index := map[string]int{}
	for _, s := range alphabet {
		index[s] = len(m.Pieces)
		m.Pieces = append(m.Pieces, Piece{s, 0})
	}
	words := make([][]int, len(keys))
	freq := make([]float64, len(keys))
	for i, s := range keys {
		freq[i] = counts[s]
		for _, r := range s {
			words[i] = append(words[i], index[string(r)])
		}
	}
	totals := map[[2]int]float64{}
	members := map[[2]int]map[int]bool{}
	revisions := map[[2]int]int{}
	done := map[[2]int]bool{}
	local := func(ids []int) map[[2]int]int {
		out := map[[2]int]int{}
		for i := 0; i+1 < len(ids); i++ {
			out[[2]int{ids[i], ids[i+1]}]++
		}
		return out
	}
	for i, ids := range words {
		for p, n := range local(ids) {
			totals[p] += float64(n) * freq[i]
			if members[p] == nil {
				members[p] = map[int]bool{}
			}
			members[p][i] = true
		}
	}
	h := &pairHeap{}
	for p, n := range totals {
		heap.Push(h, pairItem{p, n, 0})
	}
	for len(m.Pieces)+PieceOffset < vocab {
		if h.Len() == 0 {
			return m, fmt.Errorf("insufficient merge candidates for vocabulary %d", vocab)
		}
		item := heap.Pop(h).(pairItem)
		p := item.pair
		if item.revision != revisions[p] || done[p] || item.count <= 1e-12 {
			continue
		}
		text := m.Pieces[p[0]].Text + m.Pieces[p[1]].Text
		if utf8.RuneCountInString(text) > maxPieceRunes {
			done[p] = true
			continue
		}
		result, ok := index[text]
		if !ok {
			result = len(m.Pieces)
			index[text] = result
			m.Pieces = append(m.Pieces, Piece{text, 0})
		}
		m.Merges = append(m.Merges, Merge{p[0], p[1], result})
		done[p] = true
		affected := make([]int, 0, len(members[p]))
		for i := range members[p] {
			affected = append(affected, i)
		}
		sort.Ints(affected)
		changed := map[[2]int]bool{}
		for _, i := range affected {
			old := words[i]
			before := local(old)
			afterIDs := make([]int, 0, len(old))
			for j := 0; j < len(old); j++ {
				if j+1 < len(old) && old[j] == p[0] && old[j+1] == p[1] {
					afterIDs = append(afterIDs, result)
					j++
				} else {
					afterIDs = append(afterIDs, old[j])
				}
			}
			words[i] = afterIDs
			after := local(afterIDs)
			for q, n := range before {
				totals[q] -= float64(n) * freq[i]
				delete(members[q], i)
				changed[q] = true
			}
			for q, n := range after {
				totals[q] += float64(n) * freq[i]
				if members[q] == nil {
					members[q] = map[int]bool{}
				}
				members[q][i] = true
				changed[q] = true
			}
		}
		for q := range changed {
			revisions[q]++
			if !done[q] && totals[q] > 1e-12 {
				heap.Push(h, pairItem{q, totals[q], revisions[q]})
			}
		}
		if log != nil && len(m.Merges)%1000 == 0 {
			fmt.Fprintf(log, "merges=%d vocabulary=%d\n", len(m.Merges), len(m.Pieces)+PieceOffset)
		}
	}
	return m, nil
}
