package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"path/filepath"
	"strings"

	"github.com/gen2brain/webp"
	"golang.org/x/image/bmp"
	"golang.org/x/image/tiff"
)

// Output formats, keyed by canonical name.
var formatMIME = map[string]string{
	"png":  "image/png",
	"jpg":  "image/jpeg",
	"webp": "image/webp",
	"bmp":  "image/bmp",
	"tiff": "image/tiff",
}

// normalizeFormat maps a format name, file extension or image.Decode format
// name to a canonical output format. GIF maps to PNG: re-quantising an
// enhanced image to 256 colours would throw the enhancement away.
func normalizeFormat(s string) (string, bool) {
	switch strings.TrimPrefix(strings.ToLower(s), ".") {
	case "png", "gif":
		return "png", true
	case "jpg", "jpeg":
		return "jpg", true
	case "webp":
		return "webp", true
	case "bmp":
		return "bmp", true
	case "tif", "tiff":
		return "tiff", true
	}
	return "", false
}

// formatFromPath returns the output format implied by a file's extension.
func formatFromPath(path string) (string, error) {
	f, ok := normalizeFormat(filepath.Ext(path))
	if !ok {
		return "", fmt.Errorf("unsupported output extension %q (use .png, .jpg, .webp, .bmp or .tiff)", filepath.Ext(path))
	}
	return f, nil
}

// encodeImage writes img in the given canonical format. quality applies to
// lossy formats (jpg, webp).
func encodeImage(w io.Writer, img image.Image, format string, quality int) error {
	switch format {
	case "png":
		return (&png.Encoder{CompressionLevel: png.BestSpeed}).Encode(w, img)
	case "jpg":
		return jpeg.Encode(w, img, &jpeg.Options{Quality: quality})
	case "webp":
		// Method 2 is ~2.5x faster than the default 4 for only ~5% larger files.
		return webp.Encode(w, img, webp.Options{Quality: quality, Method: 2})
	case "bmp":
		return bmp.Encode(w, img)
	case "tiff":
		return tiff.Encode(w, img, &tiff.Options{Compression: tiff.Deflate})
	}
	return fmt.Errorf("unsupported output format %q", format)
}
