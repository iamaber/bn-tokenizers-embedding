package tokenizer

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// IDs 0..3 are reserved; 4..259 are literal UTF-8 bytes, never text aliases.
const ByteOffset = 4
const PieceOffset = 260

type Piece struct {
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

type Model struct {
	Version       int      `json:"version"`
	Algorithm     string   `json:"algorithm"`
	Normalization string   `json:"normalization"`
	Pieces        []Piece  `json:"pieces"`
	Merges        []Merge  `json:"merges,omitempty"`
	CacheWords    []string `json:"cache_words,omitempty"`
	SpaceFusion   []int    `json:"space_fusion,omitempty"`
}

type node struct {
	next map[byte]int
	id   int
}

// Tokenizer is immutable after construction and safe for concurrent Encode calls.
type Tokenizer struct {
	model         Model
	trie          []node
	mergeRanks    map[uint64]int
	characters    map[rune]int
	encodedPieces map[string][]int
	spaceFusion   []int
}

func New(m Model) (*Tokenizer, error) {
	if uint64(len(m.Pieces))+uint64(len(m.SpaceFusion)) > math.MaxUint32-PieceOffset {
		return nil, fmt.Errorf("vocabulary exceeds 32-bit token IDs")
	}
	if (m.Version != 1 && m.Version != 2) || (m.Algorithm != "unigram" && m.Algorithm != "bpe") || m.Normalization != Normalization {
		return nil, fmt.Errorf("unsupported model contract")
	}
	if len(m.SpaceFusion) > 0 && (m.Version != 2 || m.Algorithm != "bpe") {
		return nil, fmt.Errorf("space fusion requires version 2 BPE")
	}
	for _, word := range m.CacheWords {
		if word == "" || len(word) > 1024 || strings.Contains(word, " ") || word != Normalize(word) {
			return nil, fmt.Errorf("invalid cache word")
		}
	}
	t := &Tokenizer{model: m, trie: []node{{id: -1}}}
	t.model.Pieces = append([]Piece(nil), m.Pieces...)
	t.model.CacheWords = append([]string(nil), m.CacheWords...)
	t.model.SpaceFusion = append([]int(nil), m.SpaceFusion...)
	if len(m.SpaceFusion) > 0 {
		base := PieceOffset + len(m.Pieces)
		t.spaceFusion = make([]int, base)
		for i, id := range m.SpaceFusion {
			if id < ByteOffset || id >= base || id == ByteOffset+32 || t.spaceFusion[id] != 0 {
				return nil, fmt.Errorf("invalid or duplicate space fusion ID %d", id)
			}
			t.spaceFusion[id] = base + i
		}
	}
	for i, p := range m.Pieces {
		if p.Text == "" || !utf8.ValidString(p.Text) || math.IsNaN(p.Score) || math.IsInf(p.Score, 0) || p.Score > 0 {
			return nil, fmt.Errorf("invalid piece %d", i)
		}
		n := 0
		for j := range len(p.Text) {
			if t.trie[n].next == nil {
				t.trie[n].next = make(map[byte]int)
			}
			v, ok := t.trie[n].next[p.Text[j]]
			if !ok {
				v = len(t.trie)
				t.trie[n].next[p.Text[j]] = v
				t.trie = append(t.trie, node{id: -1})
			}
			n = v
		}
		if t.trie[n].id >= 0 {
			return nil, fmt.Errorf("duplicate piece %q", p.Text)
		}
		t.trie[n].id = i
	}
	if m.Algorithm == "bpe" {
		if err := t.initBPE(); err != nil {
			return nil, err
		}
	}
	return t, nil
}

func Load(path string) (*Tokenizer, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Model
	if err = json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	t, err := New(m)
	if err != nil {
		return nil, err
	}
	// Loaded models cache vocabulary strings and optional training words, never
	// caller text. Compute their actual segmentation rather than assuming that
	// every learned piece wins against competing Unigram paths or BPE merge ranks.
	t.encodedPieces = make(map[string][]int, len(m.Pieces)+len(m.CacheWords))
	for _, piece := range m.Pieces {
		if !strings.Contains(piece.Text, " ") && piece.Text == Normalize(piece.Text) {
			t.encodedPieces[piece.Text] = t.Encode(piece.Text)
		}
	}
	for _, word := range m.CacheWords {
		t.encodedPieces[word] = t.Encode(word)
	}
	return t, nil
}

func (m Model) Save(path string) error {
	if _, err := New(m); err != nil {
		return err
	}
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0644)
}

// Encode returns deterministic IDs. Version 2 can fuse a space with the first
// token of the following word; literal markers and special-token strings round-trip.
func (t *Tokenizer) Encode(text string) []int {
	s := Normalize(text)
	ids := make([]int, 0, len(s)/2)
	var local [128]viterbiState
	scratch := local[:]
	for len(s) > 0 {
		w, rest, _ := strings.Cut(s, " ")
		if len(ids) > 0 {
			ids = append(ids, ByteOffset+32)
		}
		start := len(ids)
		if cached, ok := t.encodedPieces[w]; ok {
			ids = append(ids, cached...)
		} else if t.model.Algorithm == "bpe" {
			ids = t.appendBPE(ids, w)
		} else {
			if len(scratch) < len(w)+1 {
				scratch = make([]viterbiState, len(w)+1)
			}
			ids = t.appendUnigram(ids, w, scratch[:len(w)+1])
		}
		if start > 0 && len(t.spaceFusion) > 0 {
			if fused := t.spaceFusion[ids[start]]; fused != 0 {
				ids[start-1] = fused
				copy(ids[start:], ids[start+1:])
				ids = ids[:len(ids)-1]
			}
		}
		s = rest
	}
	return ids
}

// Decode rejects reserved/invalid IDs rather than silently losing text.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
		base := PieceOffset + len(t.model.Pieces)
		if id >= base && id < t.VocabSize() {
			b.WriteByte(' ')
			id = t.model.SpaceFusion[id-base]
		}
		switch {
		case id >= ByteOffset && id < PieceOffset:
			b.WriteByte(byte(id - ByteOffset))
		case id >= PieceOffset && id < PieceOffset+len(t.model.Pieces):
			b.WriteString(t.model.Pieces[id-PieceOffset].Text)
		default:
			return "", fmt.Errorf("invalid or reserved token ID %d", id)
		}
	}
	if !utf8.ValidString(b.String()) {
		return "", fmt.Errorf("token sequence is not valid UTF-8")
	}
	return b.String(), nil
}

func (t *Tokenizer) VocabSize() int {
	return PieceOffset + len(t.model.Pieces) + len(t.model.SpaceFusion)
}

// Expand returns base token IDs, undoing space fusion without decoding text.
// It rejects reserved/out-of-range IDs; UTF-8 validity is checked by Decode.
func (t *Tokenizer) Expand(ids []int) ([]int, error) {
	base := PieceOffset + len(t.model.Pieces)
	result := make([]int, 0, len(ids))
	for _, id := range ids {
		if id < ByteOffset || id >= t.VocabSize() {
			return nil, fmt.Errorf("invalid or reserved token ID %d", id)
		}
		if id >= base {
			result = append(result, ByteOffset+32, t.model.SpaceFusion[id-base])
		} else {
			result = append(result, id)
		}
	}
	return result, nil
}
