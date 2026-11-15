// Command enhance upscales and restores low-resolution / blurry images.
//
// Two engines are available:
//
//   - classic: pure Go pipeline (deblock, chroma fix, Lanczos upscale,
//     halo-free sharpening). No dependencies, works everywhere.
//   - ai: shells out to Real-ESRGAN (realesrgan-ncnn-vulkan), a neural
//     super-resolution model that can reconstruct detail classical filters
//     cannot. Its output is then lightly polished by the classic sharpener.
//
// With -engine auto (default) the AI engine is used when the binary is found.
package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	_ "image/gif"

	"Image-enhancement/enhance"

	_ "golang.org/x/image/webp"
)

func main() {
	def := enhance.DefaultOptions()
	var (
		in       = flag.String("i", "", "input image (png, jpg, webp, gif, bmp, tiff)")
		out      = flag.String("o", "", "output image (.png, .jpg, .webp, .bmp, .tiff); default <input>_enhanced.<same format as input>")
		engine   = flag.String("engine", "auto", "auto | classic | ai")
		aiBin    = flag.String("ai-bin", "realesrgan-ncnn-vulkan", "path to Real-ESRGAN ncnn binary")
		aiModel  = flag.String("ai-model", "fast", "Real-ESRGAN model: fast, quality, or a model name in tools/realesrgan/models")
		scale    = flag.Float64("scale", def.Scale, "upscale factor")
		denoise  = flag.Float64("denoise", def.Denoise, "artifact removal strength (0 = off, ~0.01-0.08)")
		sharpen  = flag.Float64("sharpen", def.Sharpen, "sharpening amount (0 = off, ~0.5-3)")
		clarity  = flag.Float64("clarity", def.Clarity, "local contrast boost (0 = off, ~0.1-0.6)")
		halo     = flag.Float64("halo", def.HaloLimit, "allowed edge overshoot (lower = fewer halos)")
		noChroma = flag.Bool("no-chroma-fix", false, "disable luma-guided chroma cleanup")
		quality  = flag.Int("q", 95, "quality for lossy output (jpg, webp), 1-100")
		serve    = flag.String("serve", "", "run HTTP API on this address (e.g. :8080) instead of the CLI")
		maxMB    = flag.Int64("max-upload-mb", 20, "API: maximum upload size in MB")
		maxSide  = flag.Int("max-side", 2000, "API: maximum input width/height in pixels")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage:\n  %[1]s -i input.webp [-o out.png] [flags]\n  %[1]s -serve :8080 [flags]\n\n", filepath.Base(os.Args[0]))
		flag.PrintDefaults()
	}
	flag.Parse()
	*aiBin = resolveAIBin(*aiBin)
	*aiModel = resolveModel(*aiModel)

	if *serve != "" {
		cfg := serverConfig{
			Addr:      *serve,
			AIBin:     *aiBin,
			AIModel:   *aiModel,
			Defaults:  enhance.Options{Scale: 4, ChromaFix: !*noChroma, Sharpen: *sharpen, Clarity: *clarity, HaloLimit: *halo},
			MaxUpload: *maxMB << 20,
			MaxSide:   *maxSide,
			Quality:   *quality,
		}
		if err := runServer(cfg); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}

	if *in == "" && flag.NArg() > 0 {
		*in = flag.Arg(0)
	}
	if *in == "" {
		flag.Usage()
		os.Exit(2)
	}
	opts := enhance.Options{
		Scale:     *scale,
		Denoise:   *denoise,
		ChromaFix: !*noChroma,
		Sharpen:   *sharpen,
		Clarity:   *clarity,
		HaloLimit: *halo,
		Logf:      func(f string, a ...any) { fmt.Printf("  • "+f+"\n", a...) },
	}

	start := time.Now()
	outPath, err := run(*in, *out, *engine, *aiBin, *aiModel, opts, *quality)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	fmt.Printf("done in %s -> %s\n", time.Since(start).Round(time.Millisecond), outPath)
}

// run enhances in and writes the result. If out is empty, the output keeps
// the input's format: <input>_enhanced.<ext>. It returns the path written.
func run(in, out, engine, aiBin, aiModel string, opts enhance.Options, quality int) (string, error) {
	src, inFormat, err := loadImage(in)
	if err != nil {
		return "", err
	}
	b := src.Bounds()
	fmt.Printf("input: %s (%s, %dx%d)\n", in, inFormat, b.Dx(), b.Dy())

	if out == "" {
		f, _ := normalizeFormat(inFormat)
		if f == "" {
			f = "png"
		}
		out = strings.TrimSuffix(in, filepath.Ext(in)) + "_enhanced." + f
	}
	if _, err := formatFromPath(out); err != nil {
		return "", err // fail before spending time on enhancement
	}

	useAI := false
	switch engine {
	case "ai":
		useAI = true
	case "auto":
		_, err := exec.LookPath(aiBin)
		useAI = err == nil
	case "classic":
	default:
		return "", fmt.Errorf("unknown engine %q", engine)
	}

	var result image.Image
	if useAI {
		fmt.Println("engine: ai (Real-ESRGAN)")
		result, err = enhanceAI(context.Background(), src, aiBin, aiModel, opts, os.Stdout)
		if err != nil {
			return "", err
		}
	} else {
		fmt.Println("engine: classic")
		result = enhance.Enhance(src, opts)
	}
	return out, saveImage(out, result, quality)
}

// enhanceAI runs Real-ESRGAN (fixed 4x), resamples to the requested scale and
// applies a gentle classic sharpen pass on top.
// Real-ESRGAN progress output goes to logw; on failure its tail is included in
// the returned error. Cancelling ctx kills the subprocess.
func enhanceAI(ctx context.Context, src image.Image, bin, model string, opts enhance.Options, logw io.Writer) (image.Image, error) {
	if _, err := exec.LookPath(bin); err != nil {
		return nil, fmt.Errorf("Real-ESRGAN binary %q not found; install from "+
			"https://github.com/xinntao/Real-ESRGAN/releases or use -engine classic", bin)
	}
	if logw == nil {
		logw = io.Discard
	}
	dir, err := os.MkdirTemp("", "enhance-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)

	tmpIn := filepath.Join(dir, "in.png")
	tmpOut := filepath.Join(dir, "out.png")
	if err := saveImage(tmpIn, src, 100); err != nil {
		return nil, err
	}

	fmt.Fprintf(logw, "  • running %s -n %s -s 4\n", bin, model)
	var tail tailBuffer
	cmd := exec.CommandContext(ctx, bin, "-i", tmpIn, "-o", tmpOut, "-n", model, "-s", "4")
	cmd.Stdout = logw
	cmd.Stderr = io.MultiWriter(logw, &tail)
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("real-esrgan failed: %w: %s", err, tail.String())
	}
	up, _, err := loadImage(tmpOut)
	if err != nil {
		return nil, err
	}

	// AI output is already clean: skip denoise, rescale 4x -> requested scale,
	// and only lightly sharpen.
	b := src.Bounds()
	post := opts
	post.Denoise = 0
	post.Scale = float64(b.Dx()) * opts.Scale / float64(up.Bounds().Dx())
	post.Sharpen = opts.Sharpen * 0.3
	post.Clarity = opts.Clarity * 0.5
	post.ChromaFix = false // model output has clean chroma already
	return enhance.Enhance(up, post), nil
}

// Model aliases: "fast" is the compact SRVGG general model (~10x faster),
// "quality" the original RRDB x4plus model.
var modelAliases = map[string]string{
	"fast":    "realesr-general-x4v3",
	"quality": "realesrgan-x4plus",
}

func resolveModel(name string) string {
	if m, ok := modelAliases[name]; ok {
		return m
	}
	return name
}

// resolveAIBin returns bin if it is on PATH, otherwise falls back to the
// bundled copy in tools/realesrgan (next to the working dir or the executable).
func resolveAIBin(bin string) string {
	if p, err := exec.LookPath(bin); err == nil {
		return p
	}
	rel := filepath.Join("tools", "realesrgan", "realesrgan-ncnn-vulkan")
	candidates := []string{rel}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), rel))
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			if abs, err := filepath.Abs(c); err == nil {
				return abs
			}
		}
	}
	return bin
}

// tailBuffer keeps only the last 2 KB written to it.
type tailBuffer struct{ b []byte }

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.b = append(t.b, p...)
	if len(t.b) > 2048 {
		t.b = t.b[len(t.b)-2048:]
	}
	return len(p), nil
}

func (t *tailBuffer) String() string { return strings.TrimSpace(string(t.b)) }

// loadImage decodes path and returns the image with its detected format
// name ("png", "jpeg", "webp", ...).
func loadImage(path string) (image.Image, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", err
	}
	defer f.Close()
	img, format, err := image.Decode(f)
	if err != nil {
		return nil, "", fmt.Errorf("decode %s: %w", path, err)
	}
	return img, format, nil
}

func saveImage(path string, img image.Image, quality int) error {
	format, err := formatFromPath(path)
	if err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	err = encodeImage(f, img, format, quality)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}
