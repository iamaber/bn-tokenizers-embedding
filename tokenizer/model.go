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
	Version       int     `json:"version"`
	Algorithm     string  `json:"algorithm"`
	Normalization string  `json:"normalization"`
	Pieces        []Piece `json:"pieces"`
	Merges        []Merge `json:"merges,omitempty"`
}

type node struct {
	next map[byte]int
	id   int
}

// Tokenizer is immutable after construction and safe for concurrent Encode calls.
type Tokenizer struct {
	model      Model
	trie       []node
	mergeRanks map[[2]int]int
	characters map[rune]int
}

func New(m Model) (*Tokenizer, error) {
	if m.Version != 1 || (m.Algorithm != "unigram" && m.Algorithm != "bpe") || m.Normalization != Normalization {
		return nil, fmt.Errorf("unsupported model contract")
	}
	t := &Tokenizer{model: m, trie: []node{{id: -1}}}
	t.model.Pieces = append([]Piece(nil), m.Pieces...)
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
	return New(m)
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

// matches walks byte trie edges; learned pieces always end on rune boundaries.
func (t *Tokenizer) matches(s string, start int, visit func(end, id int)) {
	n := 0
	for j := start; j < len(s); j++ {
		v, ok := t.trie[n].next[s[j]]
		if !ok {
			break
		}
		n = v
		if t.trie[n].id >= 0 {
			visit(j+1, t.trie[n].id)
		}
	}
}

// Encode returns the maximum-likelihood segmentation. Whitespace is an explicit
// byte token, so literal SentencePiece markers and special-token strings round-trip.
func (t *Tokenizer) Encode(text string) []int {
	s := Normalize(text)
	ids := make([]int, 0, len(s)/2)
	for i, w := range strings.Split(s, " ") {
		if i > 0 {
			ids = append(ids, ByteOffset+32)
		}
		ids = append(ids, t.encodeWord(w)...)
	}
	return ids
}

func (t *Tokenizer) encodeWord(s string) []int {
	if t.model.Algorithm == "bpe" {
		return t.encodeBPE(s)
	}
	dp := make([]float64, len(s)+1)
	prev := make([]int, len(s)+1)
	token := make([]int, len(s)+1)
	for i := 1; i <= len(s); i++ {
		dp[i] = math.Inf(-1)
	}
	for i := 0; i < len(s); i++ {
		// Byte fallback competes only at a low fixed score. It guarantees coverage.
		if v := dp[i] - 30; v > dp[i+1] {
			dp[i+1] = v
			prev[i+1] = i
			token[i+1] = ByteOffset + int(s[i])
		}
		t.matches(s, i, func(end, id int) {
			if v := dp[i] + t.model.Pieces[id].Score; v > dp[end] {
				dp[end] = v
				prev[end] = i
				token[end] = PieceOffset + id
			}
		})
	}
	ids := make([]int, 0, len(s))
	for end := len(s); end > 0; end = prev[end] {
		ids = append(ids, token[end])
	}
	for i, j := 0, len(ids)-1; i < j; i, j = i+1, j-1 {
		ids[i], ids[j] = ids[j], ids[i]
	}
	return ids
}

// Decode rejects reserved/invalid IDs rather than silently losing text.
func (t *Tokenizer) Decode(ids []int) (string, error) {
	var b strings.Builder
	for _, id := range ids {
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

func (t *Tokenizer) VocabSize() int { return PieceOffset + len(t.model.Pieces) }
