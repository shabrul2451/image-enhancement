package enhance

import (
	"image"
	"math"
	"time"
)

// Options tunes the enhancement pipeline. Zero values disable a stage.
type Options struct {
	Scale     float64 // upscale factor (1 = keep size)
	Denoise   float64 // artifact/noise removal strength, ~0..0.1
	ChromaFix bool    // realign blurry/blocky colour to luma edges
	Sharpen   float64 // fine-detail sharpening amount, ~0..3
	Clarity   float64 // mid-scale local contrast amount, ~0..1
	HaloLimit float64 // allowed overshoot past local min/max, ~0..0.2
	Logf      func(format string, args ...any)
}

func DefaultOptions() Options {
	return Options{
		Scale:     2,
		Denoise:   0.03,
		ChromaFix: true,
		Sharpen:   2.0,
		Clarity:   0.3,
		HaloLimit: 0.03,
	}
}

// Enhance runs the full restoration pipeline on img:
//
//  1. Edge-preserving denoise (guided filter) to remove JPEG/WebP blocking,
//     ringing and noise before they get magnified.
//  2. Luma-guided chroma filtering to fix smeared, blocky colour edges caused
//     by chroma subsampling.
//  3. Lanczos-3 upscale.
//  4. Multi-scale unsharp masking on luma only (no colour fringing), with a
//     noise threshold and local min/max clamping to suppress halos.
func Enhance(img image.Image, o Options) image.Image {
	logf := o.Logf
	if logf == nil {
		logf = func(string, ...any) {}
	}
	// stage logs a step together with how long it took.
	stage := func(format string, args ...any) func() {
		start := time.Now()
		return func() {
			logf(format+" (%s)", append(args, time.Since(start).Round(time.Millisecond))...)
		}
	}

	done := stage("convert to YCbCr")
	c := FromImage(img)
	w, h := c.Y.W, c.Y.H
	done()

	if o.Denoise > 0 {
		done := stage("denoise / deblock (strength %.3f)", o.Denoise)
		eps := float32(o.Denoise * o.Denoise)
		// Two radii: a small one keeps fine edges, a larger one flattens blocks.
		fine := GuidedFilter(c.Y, c.Y, 1, eps)
		c.Y = GuidedFilter(fine, fine, 3, eps*0.5)
		done()
	}

	scale := o.Scale
	if scale <= 0 {
		scale = 1
	}
	nw, nh := int(math.Round(float64(w)*scale)), int(math.Round(float64(h)*scale))
	resize := nw != w || nh != h

	chromaFix := func(r int) {
		done := stage("chroma realignment")
		c.Cb, c.Cr = GuidedFilter2(c.Y, c.Cb, c.Cr, r, 1e-4)
		done()
	}

	// Fix blocky chroma at the input resolution unless we are about to
	// downscale, in which case it is cheaper to do it afterwards.
	if o.ChromaFix && scale >= 1 {
		chromaFix(4)
	}

	if resize {
		done := stage("resize %dx%d -> %dx%d (Lanczos-3)", w, h, nw, nh)
		c.Y = Resize(c.Y, nw, nh)
		c.Cb = Resize(c.Cb, nw, nh)
		c.Cr = Resize(c.Cr, nw, nh)
		if c.A != nil {
			c.A = Resize(c.A, nw, nh)
		}
		done()
		// Re-align resampled chroma to the resampled luma.
		if o.ChromaFix {
			chromaFix(max(1, int(math.Round(scale))))
		}
	}

	if o.Sharpen > 0 || o.Clarity > 0 {
		done := stage("sharpen (amount %.2f, clarity %.2f)", o.Sharpen, o.Clarity)
		c.Y = sharpenLuma(c.Y, max(scale, 1), o)
		done()
	}

	done = stage("convert to RGB")
	defer done()
	return c.ToImage()
}

// sharpenLuma applies fine + medium unsharp masks and a broad clarity boost,
// then clamps each pixel close to the local min/max of the unsharpened luma so
// that edges become crisp without bright/dark halos.
func sharpenLuma(y *Plane, scale float64, o Options) *Plane {
	s := math.Max(1, scale)
	fine := GaussianBlur(y, 0.6*s)
	medium := GaussianBlur(y, 1.6*s)
	var broad *Plane
	if o.Clarity > 0 {
		broad = GaussianBlur(y, 12*s)
	}

	lo, hi := MinMax(y, max(1, int(math.Round(s))))

	const noiseThreshold = 0.004 // below this, detail is treated as noise
	amtFine := float32(o.Sharpen)
	amtMed := float32(o.Sharpen * 0.35)
	amtClar := float32(o.Clarity)
	halo := float32(o.HaloLimit)

	out := NewPlane(y.W, y.H)
	parallelRows(y.H, func(y0, y1 int) {
		for yy := y0; yy < y1; yy++ {
			for x := 0; x < y.W; x++ {
				i := yy*y.W + x
				v := y.Pix[i]
				df := softThreshold(v-fine.Pix[i], noiseThreshold)
				dm := softThreshold(v-medium.Pix[i], noiseThreshold)
				r := v + amtFine*df + amtMed*dm

				l, h := lo.Pix[i], hi.Pix[i]
				span := h - l
				r = clampF(r, l-halo*span-0.002, h+halo*span+0.002)

				if broad != nil {
					r += amtClar * (v - broad.Pix[i])
				}
				out.Pix[i] = clampF(r, 0, 1)
			}
		}
	})
	return out
}

// softThreshold attenuates small differences (noise) smoothly instead of
// cutting them off hard.
func softThreshold(d, t float32) float32 {
	a := d
	if a < 0 {
		a = -a
	}
	if a >= t {
		return d
	}
	return d * (a / t)
}
