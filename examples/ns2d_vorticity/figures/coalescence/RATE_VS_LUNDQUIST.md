# `rate_vs_lundquist` — the Stage-2 comparison figure

Built with `pdflatex` + `pdf2svg` from `pgf/rate_vs_lundquist.tex`. PDF for
LaTeX in `pgf/out/`, SVG for HTML beside this file.

## Provenance

Every plotted value is a window mean already reported elsewhere; the figure
introduces no measurement.

**Orszag–Tang** (grey squares) — the admitted comparator, copied from
`articles/resistive-mhd-reconnection/data/scaling.csv` on `origin/master` of
`ahojukka5/research`:

| eta | N | S_local | R | R*S |
|---|---|---:|---:|---:|
| 0.0100 | 256 | 66.75 | 0.7736 | 43.72 |
| 0.0050 | 512 | 137.9 | 0.3403 | 41.70 |
| 0.0025 | 1024 | 281.0 | 0.1602 | 40.82 |

An OLS log-log fit of those three points gives -1.0956, which is the
"descriptive slope near -1.10" the article states.

**Island coalescence** (blue circles) — this work, from
`scripts/coalescence_ladder.py` over the common window `p` in [0.30, 0.38]
with `--degen-ratio 0.01` and an Ohm-residual gate of 1e-3:

| eta | N | S_local | R | R*S | dumps |
|---|---|---:|---:|---:|---:|
| 0.0100 | 256 | 18.14 | 0.07652 | 1.39 | 9 |
| 0.0050 | 256 | 40.23 | 0.12241 | 4.92 | 9 |
| 0.0025 | 512 | 126.36 | 0.11316 | 14.30 | 7 |

The window is [0.30, 0.38] rather than the Stage-0 bracket [0.30, 0.60]
because tracking truncates earlier at lower resistivity; all three points
are recomputed on the common window so the means are comparable.

## What the two panels say

Panel (a): the dashed guide is `S^-1` through the first Orszag–Tang point.
It is a reference line, not a fit.

Panel (b) is the same data as `R * S_local`, which is constant exactly when
`R ~ 1/S`. Orszag–Tang holds it to within 7% over a 4.2x range in
`S_local`; coalescence grows it 10x over a 7x range. That is the result.

No error bars are drawn. The Orszag–Tang source carries an `R_std` column
(0.2633, 0.0893, 0.0365) and the coalescence points do not yet have a
matching within-window spread, so drawing one family's uncertainty and not
the other's would mislead.
