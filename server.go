package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"Image-enhancement/enhance"
)

type serverConfig struct {
	Addr      string
	AIBin     string
	AIModel   string
	Defaults  enhance.Options
	MaxUpload int64 // bytes
	MaxSide   int   // max input width/height
	Quality   int
}

// Real-ESRGAN saturates the GPU, so jobs run one at a time; a bounded number
// of requests may wait in line before new ones are rejected with 503.
const (
	maxConcurrentJobs = 1
	maxQueuedJobs     = 8
	jobTimeout        = 5 * time.Minute
)

// uploadFormats maps accepted upload MIME types (sniffed from the bytes) to
// the output format used when the client does not request one.
var uploadFormats = map[string]string{
	"image/png":  "png",
	"image/jpeg": "jpg",
	"image/webp": "webp",
}

// apiFormats are the values accepted in the "format" field.
var apiFormats = map[string]string{"png": "png", "jpg": "jpg", "jpeg": "jpg", "webp": "webp"}

type server struct {
	cfg     serverConfig
	slots   chan struct{}
	waiting atomic.Int32
}

func runServer(cfg serverConfig) error {
	if _, err := exec.LookPath(cfg.AIBin); err != nil {
		return fmt.Errorf("Real-ESRGAN binary %q not found (use -ai-bin)", cfg.AIBin)
	}
	s := &server{cfg: cfg, slots: make(chan struct{}, maxConcurrentJobs)}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", s.handleIndex)
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /enhance", s.handleEnhance)
	mux.HandleFunc("GET /openapi.yaml", handleSpec)
	mux.HandleFunc("GET /docs", handleDocs)

	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      jobTimeout + 2*time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	errc := make(chan error, 1)
	go func() {
		log.Printf("listening on %s (model %s, binary %s)", cfg.Addr, cfg.AIModel, cfg.AIBin)
		log.Printf("swagger UI: http://localhost%s/docs", cfg.Addr)
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
		log.Print("shutting down…")
		shutCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		return srv.Shutdown(shutCtx)
	}
}

// POST /enhance
//
//	multipart/form-data:
//	  image   (required) png, jpg/jpeg or webp file
//	  scale   (optional) 1–4, default 4
//	  model   (optional) fast | quality, default server's -ai-model
//	  format  (optional) png | jpg | webp, default same as the upload
//	  quality (optional) jpg/webp quality 1–100
//
// Fields may also be given as query parameters. Responds with the image bytes.
func (s *server) handleEnhance(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	r.Body = http.MaxBytesReader(w, r.Body, s.cfg.MaxUpload+1<<20) // + room for form overhead
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		var mbe *http.MaxBytesError
		if errors.As(err, &mbe) {
			writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("upload exceeds %d MB", s.cfg.MaxUpload>>20))
			return
		}
		writeError(w, http.StatusBadRequest, "expected multipart/form-data with an 'image' file field")
		return
	}
	defer r.MultipartForm.RemoveAll()

	file, header, err := r.FormFile("image")
	if err != nil {
		writeError(w, http.StatusBadRequest, "missing 'image' file field")
		return
	}
	defer file.Close()
	if header.Size > s.cfg.MaxUpload {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Sprintf("upload exceeds %d MB", s.cfg.MaxUpload>>20))
		return
	}

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(file); err != nil {
		writeError(w, http.StatusBadRequest, "could not read upload")
		return
	}
	data := buf.Bytes()

	// Trust the bytes, not the filename or client-supplied Content-Type.
	ctype := http.DetectContentType(data)
	if _, ok := uploadFormats[ctype]; !ok {
		writeError(w, http.StatusUnsupportedMediaType, fmt.Sprintf("unsupported file type %q (allowed: png, jpg, jpeg, webp)", ctype))
		return
	}

	scale := 4.0
	if v := r.FormValue("scale"); v != "" {
		scale, err = strconv.ParseFloat(v, 64)
		if err != nil || scale < 1 || scale > 4 {
			writeError(w, http.StatusBadRequest, "scale must be a number between 1 and 4")
			return
		}
	}
	model := s.cfg.AIModel
	if v := r.FormValue("model"); v != "" {
		m, ok := modelAliases[strings.ToLower(v)]
		if !ok {
			writeError(w, http.StatusBadRequest, "model must be fast or quality")
			return
		}
		model = m
	}
	// Output keeps the uploaded format unless the client asks otherwise.
	format := uploadFormats[ctype]
	if v := r.FormValue("format"); v != "" {
		f, ok := apiFormats[strings.ToLower(v)]
		if !ok {
			writeError(w, http.StatusBadRequest, "format must be png, jpg or webp")
			return
		}
		format = f
	}
	quality := s.cfg.Quality
	if v := r.FormValue("quality"); v != "" {
		quality, err = strconv.Atoi(v)
		if err != nil || quality < 1 || quality > 100 {
			writeError(w, http.StatusBadRequest, "quality must be an integer between 1 and 100")
			return
		}
	}

	// Check dimensions before fully decoding, so huge images are rejected cheaply.
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not decode image: "+err.Error())
		return
	}
	if cfg.Width > s.cfg.MaxSide || cfg.Height > s.cfg.MaxSide {
		writeError(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("image is %dx%d; max allowed side is %d px", cfg.Width, cfg.Height, s.cfg.MaxSide))
		return
	}
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		writeError(w, http.StatusBadRequest, "could not decode image: "+err.Error())
		return
	}

	release, err := s.acquire(r.Context())
	if err != nil {
		if errors.Is(err, errQueueFull) {
			w.Header().Set("Retry-After", "30")
			writeError(w, http.StatusServiceUnavailable, "server busy, try again later")
		}
		return // client went away
	}
	defer release()

	ctx, cancel := context.WithTimeout(r.Context(), jobTimeout)
	defer cancel()

	opts := s.cfg.Defaults
	opts.Scale = scale
	result, err := enhanceAI(ctx, src, s.cfg.AIBin, model, opts, nil)
	if err != nil {
		switch {
		case errors.Is(err, context.DeadlineExceeded):
			writeError(w, http.StatusGatewayTimeout, "enhancement timed out")
		case errors.Is(err, context.Canceled):
			// client disconnected; nothing to send
		default:
			log.Printf("enhance failed: %v", err)
			writeError(w, http.StatusInternalServerError, "enhancement failed")
		}
		return
	}

	var out bytes.Buffer
	if err := encodeImage(&out, result, format, quality); err != nil {
		log.Printf("encode %s failed: %v", format, err)
		writeError(w, http.StatusInternalServerError, "could not encode result")
		return
	}

	rb := result.Bounds()
	name := strings.TrimSuffix(filepath.Base(header.Filename), filepath.Ext(header.Filename))
	if name == "" || name == "." {
		name = "image"
	}
	w.Header().Set("Content-Type", formatMIME[format])
	w.Header().Set("Content-Length", strconv.Itoa(out.Len()))
	w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=%q", name+"_enhanced."+format))
	w.Header().Set("X-Input-Size", fmt.Sprintf("%dx%d", cfg.Width, cfg.Height))
	w.Header().Set("X-Output-Size", fmt.Sprintf("%dx%d", rb.Dx(), rb.Dy()))
	w.Header().Set("X-Model", model)
	w.Header().Set("X-Processing-Time", time.Since(start).Round(time.Millisecond).String())
	w.Write(out.Bytes())
}

var errQueueFull = errors.New("queue full")

// acquire waits for a GPU slot. It fails fast when too many requests are
// already waiting, and gives up if the client disconnects.
func (s *server) acquire(ctx context.Context) (func(), error) {
	if s.waiting.Add(1) > maxQueuedJobs {
		s.waiting.Add(-1)
		return nil, errQueueFull
	}
	defer s.waiting.Add(-1)
	select {
	case s.slots <- struct{}{}:
		return func() { <-s.slots }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":  "ok",
		"model":   s.cfg.AIModel,
		"busy":    len(s.slots),
		"waiting": s.waiting.Load(),
	})
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

type statusRecorder struct {
	http.ResponseWriter
	code int
}

func (s *statusRecorder) WriteHeader(code int) {
	s.code = code
	s.ResponseWriter.WriteHeader(code)
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, code: http.StatusOK}
		next.ServeHTTP(rec, r)
		if errors.Is(r.Context().Err(), context.Canceled) {
			rec.code = 499 // client closed request
		}
		log.Printf("%s %s %d %s", r.Method, r.URL.Path, rec.code, time.Since(start).Round(time.Millisecond))
	})
}

//go:embed openapi.yaml
var openAPISpec []byte

func handleSpec(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/yaml")
	w.Write(openAPISpec)
}

func handleDocs(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(swaggerHTML))
}

// Swagger UI is loaded from the jsDelivr CDN, so /docs needs internet access
// in the browser; the API itself does not.
const swaggerHTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Enhancer API</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head><body>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
SwaggerUIBundle({url: '/openapi.yaml', dom_id: '#swagger-ui', deepLinking: true, tryItOutEnabled: true});
</script>
</body></html>`

func (s *server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(indexHTML))
}

const indexHTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Image Enhancer</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 16px;background:#f6f6f4;color:#222}
form{display:flex;gap:.75rem;flex-wrap:wrap;align-items:center;background:#fff;padding:1rem;border-radius:10px;border:1px solid #ddd}
button{padding:.5rem 1.1rem;border:0;border-radius:6px;background:#1f6feb;color:#fff;cursor:pointer}
button:disabled{opacity:.5}
.row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}
.row figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:10px;padding:.5rem}
img{width:100%;display:block}
#status{margin-top:1rem;min-height:1.5em}
@media (max-width:700px){.row{grid-template-columns:1fr}}
</style></head><body>
<h1>Image Enhancer</h1>
<form id="f">
  <input type="file" name="image" accept="image/png,image/jpeg,image/webp" required>
  <label>Scale <select name="scale"><option>2</option><option>3</option><option selected>4</option></select></label>
  <label>Model <select name="model"><option value="fast" selected>fast (~5s)</option><option value="quality">quality (~50s)</option></select></label>
  <label>Format <select name="format"><option value="" selected>same as input</option><option>png</option><option>jpg</option><option>webp</option></select></label>
  <button>Enhance</button>
  <a href="/docs" style="margin-left:auto">API docs</a>
</form>
<div id="status"></div>
<div class="row">
  <figure><figcaption>Original</figcaption><img id="before"></figure>
  <figure><figcaption>Enhanced <a id="dl" hidden>download</a></figcaption><img id="after"></figure>
</div>
<script>
const f=document.getElementById('f'), st=document.getElementById('status');
f.image.onchange=()=>{const file=f.image.files[0]; if(file) document.getElementById('before').src=URL.createObjectURL(file)};
f.onsubmit=async e=>{
  e.preventDefault(); const btn=f.querySelector('button'); btn.disabled=true;
  st.textContent='Enhancing… (this can take a minute)'; const t=performance.now();
  try{
    const res=await fetch('/enhance',{method:'POST',body:new FormData(f)});
    if(!res.ok){const j=await res.json().catch(()=>({error:res.statusText})); throw new Error(j.error)}
    const url=URL.createObjectURL(await res.blob());
    document.getElementById('after').src=url;
    const dl=document.getElementById('dl'); dl.href=url; dl.download=(/filename="([^"]+)"/.exec(res.headers.get('Content-Disposition')||'')||[,'enhanced'])[1]; dl.hidden=false;
    st.textContent='Done: '+res.headers.get('X-Input-Size')+' → '+res.headers.get('X-Output-Size')+' in '+((performance.now()-t)/1000).toFixed(1)+'s';
  }catch(err){st.textContent='Error: '+err.message}
  btn.disabled=false;
};
</script>
</body></html>`
