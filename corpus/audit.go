package corpus

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func FileHash(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// Audit checks canonical text and connected-group disjointness in saved splits.
func Audit(dir string) (map[string]any, error) {
	keys := map[string]string{}
	groups := map[string]string{}
	counts := map[string]int{}
	sources := map[string]int{}
	hashes := map[string]string{}
	for _, split := range []string{"train", "validation", "test"} {
		path := filepath.Join(dir, split+".jsonl")
		hash, err := FileHash(path)
		if err != nil {
			return nil, err
		}
		hashes[split] = hash
		err = Read(path, func(r Record) error {
			if r.Split != split {
				return fmt.Errorf("mislabeled split")
			}
			if r.Group == "" {
				return fmt.Errorf("empty group")
			}
			key := Key(r.Text)
			if old, ok := keys[key]; ok {
				return fmt.Errorf("duplicate text in %s and %s", old, split)
			}
			keys[key] = split
			if old, ok := groups[r.Group]; ok && old != split {
				return fmt.Errorf("group overlaps %s and %s", old, split)
			}
			groups[r.Group] = split
			counts[split]++
			sources[split+"/"+r.Source]++
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return map[string]any{"canonical_text_overlap": 0, "group_overlap": 0, "counts": counts, "sources": sources, "sha256": hashes, "groups": len(groups)}, nil
}
