// Package enhance implements a classical (non-AI) image restoration and
// super-resolution pipeline: artifact removal, Lanczos upscaling and
// halo-limited sharpening, all operating on float32 planes.
package enhance

import (
	"image"
	"image/color"
	"runtime"
	"sync"
)

// Plane is a single-channel float32 image.
type Plane struct {
	W, H int
	Pix  []float32
}

func NewPlane(w, h int) *Plane {
	return &Plane{W: w, H: h, Pix: make([]float32, w*h)}
}

// At returns the pixel at (x, y), clamping coordinates to the plane edges.
func (p *Plane) At(x, y int) float32 {
	x = clampInt(x, 0, p.W-1)
	y = clampInt(y, 0, p.H-1)
	return p.Pix[y*p.W+x]
}

func (p *Plane) Clone() *Plane {
	c := NewPlane(p.W, p.H)
	copy(c.Pix, p.Pix)
	return c
}

// YCC holds an image split into luma, chroma and (optional) alpha planes.
// Y is in [0,1], Cb/Cr in [-0.5,0.5], A in [0,1].
type YCC struct {
	Y, Cb, Cr, A *Plane
}

// FromImage converts any image.Image to full-range BT.601 YCbCr planes.
func FromImage(img image.Image) *YCC {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := &YCC{Y: NewPlane(w, h), Cb: NewPlane(w, h), Cr: NewPlane(w, h)}
	hasAlpha := !opaque(img)
	if hasAlpha {
		out.A = NewPlane(w, h)
	}
	// set stores one pixel given 8-bit straight (non-premultiplied) RGBA.
	set := func(i int, r8, g8, b8, a8 uint8) {
		r := float32(r8) / 255
		g := float32(g8) / 255
		bl := float32(b8) / 255
		yy := 0.299*r + 0.587*g + 0.114*bl
		out.Y.Pix[i] = yy
		out.Cb.Pix[i] = (bl - yy) * 0.564
		out.Cr.Pix[i] = (r - yy) * 0.713
		if hasAlpha {
			out.A.Pix[i] = float32(a8) / 255
		}
	}

	// Fast paths for the common concrete types avoid a per-pixel interface
	// call and colour-model conversion.
	switch src := img.(type) {
	case *image.NRGBA:
		parallelRows(h, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				o := src.PixOffset(b.Min.X, b.Min.Y+y)
				for x := 0; x < w; x++ {
					p := src.Pix[o+4*x : o+4*x+4 : o+4*x+4]
					set(y*w+x, p[0], p[1], p[2], p[3])
				}
			}
		})
		return out
	case *image.RGBA:
		if !hasAlpha {
			parallelRows(h, func(y0, y1 int) {
				for y := y0; y < y1; y++ {
					o := src.PixOffset(b.Min.X, b.Min.Y+y)
					for x := 0; x < w; x++ {
						p := src.Pix[o+4*x : o+4*x+4 : o+4*x+4]
						set(y*w+x, p[0], p[1], p[2], 255)
					}
				}
			})
			return out
		}
	case *image.YCbCr:
		parallelRows(h, func(y0, y1 int) {
			for y := y0; y < y1; y++ {
				for x := 0; x < w; x++ {
					yi := src.YOffset(b.Min.X+x, b.Min.Y+y)
					ci := src.COffset(b.Min.X+x, b.Min.Y+y)
					r, g, bl := color.YCbCrToRGB(src.Y[yi], src.Cb[ci], src.Cr[ci])
					set(y*w+x, r, g, bl, 255)
				}
			}
		})
		return out
	}

	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for x := 0; x < w; x++ {
				c := color.NRGBA64Model.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA64)
				r := float32(c.R) / 65535
				g := float32(c.G) / 65535
				bl := float32(c.B) / 65535
				i := y*w + x
				yy := 0.299*r + 0.587*g + 0.114*bl
				out.Y.Pix[i] = yy
				out.Cb.Pix[i] = (bl - yy) * 0.564
				out.Cr.Pix[i] = (r - yy) * 0.713
				if hasAlpha {
					out.A.Pix[i] = float32(c.A) / 65535
				}
			}
		}
	})
	return out
}

// ToImage converts the planes back to an 8-bit NRGBA image.
func (c *YCC) ToImage() *image.NRGBA {
	w, h := c.Y.W, c.Y.H
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for x := 0; x < w; x++ {
				i := y*w + x
				yy, cb, cr := c.Y.Pix[i], c.Cb.Pix[i], c.Cr.Pix[i]
				r := yy + 1.403*cr
				g := yy - 0.344*cb - 0.714*cr
				b := yy + 1.773*cb
				a := float32(1)
				if c.A != nil {
					a = c.A.Pix[i]
				}
				o := img.PixOffset(x, y)
				img.Pix[o+0] = to8(r)
				img.Pix[o+1] = to8(g)
				img.Pix[o+2] = to8(b)
				img.Pix[o+3] = to8(a)
			}
		}
	})
	return img
}

func opaque(img image.Image) bool {
	if o, ok := img.(interface{ Opaque() bool }); ok {
		return o.Opaque()
	}
	return false
}

func to8(v float32) uint8 {
	v = v*255 + 0.5
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampF(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// parallelRows splits [0,h) into chunks and runs fn on each chunk concurrently.
func parallelRows(h int, fn func(y0, y1 int)) {
	workers := runtime.GOMAXPROCS(0)
	if workers > h {
		workers = h
	}
	if workers <= 1 {
		fn(0, h)
		return
	}
	chunk := (h + workers - 1) / workers
	var wg sync.WaitGroup
	for y0 := 0; y0 < h; y0 += chunk {
		y1 := min(y0+chunk, h)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(y0, y1)
		}()
	}
	wg.Wait()
}
