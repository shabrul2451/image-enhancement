package enhance

import "math"

// BoxBlur computes the mean over a (2r+1)x(2r+1) window in O(N) using
// separable running sums with clamped borders.
func BoxBlur(p *Plane, r int) *Plane {
	if r <= 0 {
		return p.Clone()
	}
	tmp := NewPlane(p.W, p.H)
	boxH(p, tmp, r)
	out := NewPlane(p.W, p.H)
	boxV(tmp, out, r)
	return out
}

// boxH runs a horizontal running-sum box filter from src into dst.
func boxH(src, dst *Plane, r int) {
	w := src.W
	norm := 1 / float32(2*r+1)
	parallelRows(src.H, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			row := src.Pix[y*w : (y+1)*w]
			out := dst.Pix[y*w : (y+1)*w]
			var sum float32
			for k := -r; k <= r; k++ {
				sum += row[clampInt(k, 0, w-1)]
			}
			for x := 0; x < w; x++ {
				out[x] = sum * norm
				sum += row[min(x+r+1, w-1)] - row[max(x-r, 0)]
			}
		}
	})
}

// boxV runs a vertical running-sum box filter from src into dst. It walks
// rows (not columns) so memory access stays sequential.
func boxV(src, dst *Plane, r int) {
	w, h := src.W, src.H
	norm := 1 / float32(2*r+1)
	parallelRows(w, func(x0, x1 int) { // split by column bands
		sum := make([]float32, x1-x0)
		row := func(y int) []float32 {
			y = clampInt(y, 0, h-1)
			return src.Pix[y*w+x0 : y*w+x1]
		}
		for k := -r; k <= r; k++ {
			for i, v := range row(k) {
				sum[i] += v
			}
		}
		for y := 0; y < h; y++ {
			out := dst.Pix[y*w+x0 : y*w+x1]
			add, sub := row(y+r+1), row(y-r)
			for i := range sum {
				out[i] = sum[i] * norm
				sum[i] += add[i] - sub[i]
			}
		}
	})
}

// GaussianBlur applies a Gaussian blur with the given sigma. Large sigmas use
// three successive box blurs, which approximate a Gaussian closely at O(N)
// cost regardless of radius.
func GaussianBlur(p *Plane, sigma float64) *Plane {
	if sigma <= 0 {
		return p.Clone()
	}
	if sigma > 3 {
		return boxGaussian(p, sigma)
	}
	r := int(math.Ceil(sigma * 3))
	k := make([]float32, 2*r+1)
	var total float64
	for i := -r; i <= r; i++ {
		v := math.Exp(-float64(i*i) / (2 * sigma * sigma))
		k[i+r] = float32(v)
		total += v
	}
	for i := range k {
		k[i] /= float32(total)
	}

	w, h := p.W, p.H
	tmp := NewPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			row := p.Pix[y*w : (y+1)*w]
			out := tmp.Pix[y*w : (y+1)*w]
			for x := 0; x < w; x++ {
				var s float32
				if x >= r && x < w-r { // interior: no bounds clamping
					win := row[x-r : x+r+1]
					for i, kv := range k {
						s += kv * win[i]
					}
				} else {
					for i, kv := range k {
						s += kv * row[clampInt(x+i-r, 0, w-1)]
					}
				}
				out[x] = s
			}
		}
	})

	out := NewPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			dst := out.Pix[y*w : (y+1)*w]
			for i, kv := range k {
				yy := clampInt(y+i-r, 0, h-1)
				src := tmp.Pix[yy*w : (yy+1)*w]
				for x := range dst {
					dst[x] += kv * src[x]
				}
			}
		}
	})
	return out
}

// boxGaussian approximates a Gaussian with three box passes (Wells, 1986).
func boxGaussian(p *Plane, sigma float64) *Plane {
	// Box radius whose 3-fold convolution has variance ~sigma^2.
	r := max(1, int(math.Round((math.Sqrt(4*sigma*sigma+1)-1)/2)))
	a, b := p.Clone(), NewPlane(p.W, p.H)
	for range 3 {
		boxH(a, b, r)
		boxV(b, a, r)
	}
	return a
}

// GuidedFilter is the edge-preserving filter of He et al. (2010). The output
// follows the structure of guide while smoothing src; eps controls how strong
// an edge must be to survive (larger = smoother). With guide == src it acts as
// an edge-preserving denoiser, which is effective against compression blocks.
func GuidedFilter(guide, src *Plane, r int, eps float32) *Plane {
	g := newGuide(guide, r, eps)
	return g.filter(src)
}

// GuidedFilter2 filters two planes with the same guide, sharing the guide
// statistics (used for Cb and Cr).
func GuidedFilter2(guide, a, b *Plane, r int, eps float32) (*Plane, *Plane) {
	g := newGuide(guide, r, eps)
	return g.filter(a), g.filter(b)
}

type guideStats struct {
	I, meanI, varI *Plane
	r              int
	eps            float32
}

func newGuide(I *Plane, r int, eps float32) *guideStats {
	ii := NewPlane(I.W, I.H)
	for i, v := range I.Pix {
		ii.Pix[i] = v * v
	}
	meanI := BoxBlur(I, r)
	varI := BoxBlur(ii, r) // becomes variance below
	for i, m := range meanI.Pix {
		varI.Pix[i] -= m * m
	}
	return &guideStats{I: I, meanI: meanI, varI: varI, r: r, eps: eps}
}

func (g *guideStats) filter(src *Plane) *Plane {
	n := len(src.Pix)
	ip := NewPlane(src.W, src.H)
	for i := 0; i < n; i++ {
		ip.Pix[i] = g.I.Pix[i] * src.Pix[i]
	}
	meanP := BoxBlur(src, g.r)
	corrIP := BoxBlur(ip, g.r)

	a, b := corrIP, meanP // reuse buffers
	for i := 0; i < n; i++ {
		mI, mP := g.meanI.Pix[i], meanP.Pix[i]
		av := (corrIP.Pix[i] - mI*mP) / (g.varI.Pix[i] + g.eps)
		a.Pix[i] = av
		b.Pix[i] = mP - av*mI
	}
	meanA := BoxBlur(a, g.r)
	meanB := BoxBlur(b, g.r)

	for i := 0; i < n; i++ {
		meanA.Pix[i] = meanA.Pix[i]*g.I.Pix[i] + meanB.Pix[i]
	}
	return meanA
}

// MinMax returns the local minimum and maximum over a (2r+1)x(2r+1) window.
func MinMax(p *Plane, r int) (lo, hi *Plane) {
	w, h := p.W, p.H
	tmpLo, tmpHi := NewPlane(w, h), NewPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			row := p.Pix[y*w : (y+1)*w]
			for x := 0; x < w; x++ {
				win := row[max(x-r, 0):min(x+r+1, w)]
				mn, mx := win[0], win[0]
				for _, v := range win[1:] {
					mn, mx = min(mn, v), max(mx, v)
				}
				tmpLo.Pix[y*w+x], tmpHi.Pix[y*w+x] = mn, mx
			}
		}
	})
	lo, hi = NewPlane(w, h), NewPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			l := lo.Pix[y*w : (y+1)*w]
			u := hi.Pix[y*w : (y+1)*w]
			copy(l, tmpLo.Pix[y*w:(y+1)*w])
			copy(u, tmpHi.Pix[y*w:(y+1)*w])
			for yy := max(y-r, 0); yy <= min(y+r, h-1); yy++ {
				if yy == y {
					continue
				}
				sl := tmpLo.Pix[yy*w : (yy+1)*w]
				su := tmpHi.Pix[yy*w : (yy+1)*w]
				for x := range l {
					l[x] = min(l[x], sl[x])
					u[x] = max(u[x], su[x])
				}
			}
		}
	})
	return lo, hi
}
