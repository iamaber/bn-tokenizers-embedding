package corpus

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// AuditOfficial independently checks both forms of official BanglaTLit holdouts
// against the saved training corpus, including copies imported through PT rows.
func AuditOfficial(prepared, raw string) (int, error) {
	heldout := map[string]bool{}
	for _, split := range []string{"validation", "test"} {
		f, err := os.Open(filepath.Join(raw, "banglatlit", split+".csv"))
		if err != nil {
			return 0, err
		}
		r := csv.NewReader(f)
		if _, err := r.Read(); err != nil {
			f.Close()
			return 0, err
		}
		for {
			row, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				f.Close()
				return 0, err
			}
			if len(row) != 3 {
				f.Close()
				return 0, fmt.Errorf("unexpected holdout schema")
			}
			for _, s := range row[1:] {
				if k := Key(s); k != "" {
					heldout[k] = true
				}
			}
		}
		f.Close()
	}
	err := Read(filepath.Join(prepared, "train.jsonl"), func(r Record) error {
		if heldout[Key(r.Text)] {
			return fmt.Errorf("official holdout leaked into training")
		}
		return nil
	})
	return len(heldout), err
}
