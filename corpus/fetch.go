package corpus

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Download struct {
	Path   string `json:"path"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
	Bytes  int64  `json:"bytes"`
}

// VerifyManifest checks every required raw file without network access.
func VerifyManifest(manifest, root string) error {
	b, err := os.ReadFile(manifest)
	if err != nil {
		return err
	}
	var entries []Download
	if err := json.Unmarshal(b, &entries); err != nil {
		return err
	}
	if len(entries) == 0 {
		return fmt.Errorf("empty source manifest")
	}
	for _, e := range entries {
		if !filepath.IsLocal(e.Path) {
			return fmt.Errorf("invalid manifest path")
		}
		if err := verify(filepath.Join(root, e.Path), e); err != nil {
			return err
		}
	}
	return nil
}

// Fetch validates cached and newly downloaded files against a frozen manifest.
func Fetch(manifest, root string) error {
	b, err := os.ReadFile(manifest)
	if err != nil {
		return err
	}
	var entries []Download
	if err := json.Unmarshal(b, &entries); err != nil {
		return err
	}
	client := http.Client{Timeout: 10 * time.Minute}
	if len(entries) == 0 {
		return fmt.Errorf("empty source manifest")
	}
	for _, entry := range entries {
		if !filepath.IsLocal(entry.Path) || !strings.HasPrefix(entry.URL, "https://") || entry.Bytes <= 0 {
			return fmt.Errorf("invalid manifest entry %q", entry.Path)
		}
		want, err := hex.DecodeString(entry.SHA256)
		if err != nil || len(want) != 32 {
			return fmt.Errorf("invalid SHA256")
		}
		path := filepath.Join(root, entry.Path)
		if err := verify(path, entry); err == nil {
			continue
		}
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return err
		}
		fmt.Fprintln(os.Stderr, "fetch", entry.Path)
		if err := download(&client, path, entry); err != nil {
			return err
		}
	}
	return nil
}

func verify(path string, e Download) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return err
	}
	if n != e.Bytes || hex.EncodeToString(h.Sum(nil)) != e.SHA256 {
		return fmt.Errorf("checksum mismatch: %s", path)
	}
	return nil
}

func download(client *http.Client, path string, e Download) error {
	resp, err := client.Get(e.URL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download %s: HTTP %d", e.Path, resp.StatusCode)
	}
	f, err := os.CreateTemp(filepath.Dir(path), "download-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	_, err = io.Copy(f, io.LimitReader(resp.Body, e.Bytes+1))
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if err := verify(f.Name(), e); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
