package corpus

import (
	"archive/zip"
	"bufio"
	"crypto/sha256"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/parquet-go/parquet-go"
)

// LoadSources reads the frozen local downloads; it never fetches a moving URL.
func LoadSources(raw string) ([]Record, error) {
	var records []Record
	for _, split := range []string{"train", "validation", "test"} {
		f, err := os.Open(filepath.Join(raw, "banglatlit", split+".csv"))
		if err != nil {
			return nil, err
		}
		r := csv.NewReader(f)
		header, err := r.Read()
		if err != nil {
			f.Close()
			return nil, err
		}
		if strings.Join(header, ",") != "id,text_transliterated,text_bengali" {
			f.Close()
			return nil, fmt.Errorf("unexpected BanglaTLit schema")
		}
		for {
			row, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				f.Close()
				return nil, err
			}
			group := "banglatlit:" + row[0]
			for j, bucket := range []string{"banglish", "bangla"} {
				if row[j+1] == "" {
					continue
				}
				b := bucket
				// Script-mixed is an observable proxy, not a language classifier.
				if mixed(row[j+1]) {
					b = "code-mixed"
				}
				records = append(records, Record{row[j+1], b, "banglatlit", group, "https://github.com/farhanishmam/BanglaTLit; upstream MIT", split})
			}
		}
		f.Close()
	}
	for _, lang := range []string{"ben", "fra"} {
		z, err := zip.OpenReader(filepath.Join(raw, "tatoeba", lang+"-eng.zip"))
		if err != nil {
			return nil, err
		}
		found := false
		for _, entry := range z.File {
			if entry.Name != lang+".txt" {
				continue
			}
			found = true
			f, err := entry.Open()
			if err != nil {
				z.Close()
				return nil, err
			}
			s := bufio.NewScanner(f)
			s.Buffer(make([]byte, 65536), 4<<20)
			for s.Scan() {
				row := strings.SplitN(s.Text(), "\t", 3)
				if len(row) != 3 {
					f.Close()
					z.Close()
					return nil, fmt.Errorf("bad Tatoeba row")
				}
				h := sha256.Sum256([]byte(row[0]))
				group := fmt.Sprintf("tatoeba:%x", h[:16])
				records = append(records, Record{row[0], "english", "tatoeba-" + lang, group, row[2], ""})
				if lang == "ben" {
					records = append(records, Record{row[1], "bangla", "tatoeba-ben", group, row[2], ""})
				}
			}
			err = s.Err()
			f.Close()
			if err != nil {
				z.Close()
				return nil, err
			}
		}
		z.Close()
		if !found {
			return nil, fmt.Errorf("missing %s.txt", lang)
		}
	}
	for shard := 0; shard < 2; shard++ {
		f, err := os.Open(filepath.Join(raw, "wikipedia", fmt.Sprintf("bn-%d.parquet", shard)))
		if err != nil {
			return nil, err
		}
		r := parquet.NewGenericReader[wikiRow](f)
		batch := make([]wikiRow, 128)
		for {
			n, err := r.Read(batch)
			for _, row := range batch[:n] {
				// Deterministic 20% article sample, first 10 substantial paragraphs.
				h := sha256.Sum256([]byte(row.ID))
				if h[0]%5 != 0 {
					continue
				}
				kept := 0
				for _, p := range strings.Split(row.Text, "\n") {
					if len(strings.Fields(p)) < 8 {
						continue
					}
					records = append(records, Record{p, "bangla", "wikipedia-bn-20231101", "wiki-bn:" + row.ID, row.URL + "; Wikipedia contributors; CC BY-SA 3.0 / GFDL", ""})
					kept++
					if kept >= 10 {
						break
					}
				}
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				r.Close()
				f.Close()
				return nil, err
			}
		}
		r.Close()
		f.Close()
	}
	return records, nil
}

type wikiRow struct {
	ID    string `parquet:"id"`
	URL   string `parquet:"url"`
	Title string `parquet:"title"`
	Text  string `parquet:"text"`
}

func mixed(s string) bool {
	bn, latin := false, false
	for _, r := range s {
		if unicode.In(r, unicode.Bengali) {
			bn = true
		}
		if unicode.In(r, unicode.Latin) {
			latin = true
		}
	}
	return bn && latin
}
