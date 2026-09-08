// Package native owns the small JSON ABI used by the Python binding.
// No Go pointers cross the C boundary; callers receive numeric model handles.
package native

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/iamaber/bn-tokenizers-embedding/tokenizer"
)

type request struct {
	Operation string   `json:"op"`
	Handle    uint64   `json:"handle"`
	Path      string   `json:"path"`
	Text      string   `json:"text"`
	Texts     []string `json:"texts"`
	IDs       []int    `json:"ids"`
}

type response struct {
	Handle     uint64  `json:"handle,omitempty"`
	Vocabulary int     `json:"vocabulary,omitempty"`
	IDs        []int   `json:"ids,omitempty"`
	Batch      [][]int `json:"batch,omitempty"`
	Text       string  `json:"text"`
	Error      string  `json:"error,omitempty"`
}

// Registry owns models until Python closes or finalizes their handles.
type Registry struct {
	mu     sync.RWMutex
	next   uint64
	models map[uint64]*tokenizer.Tokenizer
}

func (r *Registry) Call(data []byte) []byte {
	var req request
	var result response
	if err := json.Unmarshal(data, &req); err != nil {
		result.Error = err.Error()
	} else {
		var err error
		result, err = r.call(req)
		if err != nil {
			result.Error = err.Error()
		}
	}
	// response contains only JSON-safe primitive values.
	encoded, _ := json.Marshal(result)
	return encoded
}

func (r *Registry) call(req request) (response, error) {
	if req.Operation == "load" {
		model, err := tokenizer.Load(req.Path)
		if err != nil {
			return response{}, err
		}
		r.mu.Lock()
		defer r.mu.Unlock()
		if r.models == nil {
			r.models = map[uint64]*tokenizer.Tokenizer{}
		}
		r.next++
		r.models[r.next] = model
		return response{Handle: r.next, Vocabulary: model.VocabSize()}, nil
	}
	if req.Operation == "normalize" {
		return response{Text: tokenizer.Normalize(req.Text)}, nil
	}
	if req.Operation == "close" {
		r.mu.Lock()
		delete(r.models, req.Handle)
		r.mu.Unlock()
		return response{}, nil
	}
	r.mu.RLock()
	model, ok := r.models[req.Handle]
	r.mu.RUnlock()
	if !ok {
		return response{}, fmt.Errorf("unknown or closed tokenizer handle")
	}
	switch req.Operation {
	case "encode":
		return response{IDs: model.Encode(req.Text)}, nil
	case "encode_batch":
		batch := make([][]int, len(req.Texts))
		for i, text := range req.Texts {
			batch[i] = model.Encode(text)
		}
		return response{Batch: batch}, nil
	case "decode":
		text, err := model.Decode(req.IDs)
		return response{Text: text}, err
	default:
		return response{}, fmt.Errorf("unknown operation %q", req.Operation)
	}
}
