package enhance

import "math"

const lanczosA = 3

func lanczos(x float64) float64 {
	if x == 0 {
		return 1
	}
	if x <= -lanczosA || x >= lanczosA {
		return 0
	}
	px := math.Pi * x
	return lanczosA * math.Sin(px) * math.Sin(px/lanczosA) / (px * px)
}

// contrib is the list of source taps and weights for one output coordinate.
type contrib struct {
	start   int
	weights []float32
}

// buildContribs precomputes normalized Lanczos-3 weights for resampling a
// 1-D axis of length src to length dst.
func buildContribs(src, dst int) []contrib {
	scale := float64(dst) / float64(src)
	support := float64(lanczosA)
	filterScale := 1.0
	if scale < 1 { // widen the kernel when downscaling to avoid aliasing
		filterScale = 1 / scale
		support *= filterScale
	}
	out := make([]contrib, dst)
	for i := range dst {
		center := (float64(i)+0.5)/scale - 0.5
		start := int(math.Floor(center - support))
		end := int(math.Ceil(center + support))
		ws := make([]float32, 0, end-start+1)
		var total float64
		for j := start; j <= end; j++ {
			w := lanczos((float64(j) - center) / filterScale)
			ws = append(ws, float32(w))
			total += w
		}
		for k := range ws {
			ws[k] /= float32(total)
		}
		out[i] = contrib{start: start, weights: ws}
	}
	return out
}

// Resize resamples p to w x h with a separable Lanczos-3 filter.
func Resize(p *Plane, w, h int) *Plane {
	cx := buildContribs(p.W, w)
	cy := buildContribs(p.H, h)

	tmp := NewPlane(w, p.H)
	parallelRows(p.H, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			row := p.Pix[y*p.W : (y+1)*p.W]
			out := tmp.Pix[y*w : (y+1)*w]
			for x := range w {
				c := cx[x]
				var s float32
				if c.start >= 0 && c.start+len(c.weights) <= p.W {
					win := row[c.start : c.start+len(c.weights)]
					for k, wt := range c.weights {
						s += wt * win[k]
					}
				} else {
					for k, wt := range c.weights {
						s += wt * row[clampInt(c.start+k, 0, p.W-1)]
					}
				}
				out[x] = s
			}
		}
	})
	// Vertical pass accumulates whole rows so memory access stays sequential.
	out := NewPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			dst := out.Pix[y*w : (y+1)*w]
			c := cy[y]
			for k, wt := range c.weights {
				yy := clampInt(c.start+k, 0, p.H-1)
				src := tmp.Pix[yy*w : (yy+1)*w]
				for x := range dst {
					dst[x] += wt * src[x]
				}
			}
		}
	})
	return out
}
