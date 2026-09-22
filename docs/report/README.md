<!--
SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Applications report artifacts

This directory keeps CSV and figure artifacts produced by the
applications. It does not store a manuscript.

| Path | Role |
|---|---|
| `data/*.csv` | CSV writers from study binaries and transcribed LUMI timings |
| `figures/*.py` | regenerate committed SVGs from `data/` or from field-demo output |
| `figures/run_field_demos.sh` | run recipe for the field-visualisation figures |

Do not add Quarto chapters here. New catalog prose belongs in the
research article. After a study writes a CSV, import a reviewed copy
into the research article before treating the number as admitted
there.

## Regenerating scalability figures

```bash
python3 docs/report/figures/make_figures.py
```

Copy reviewed SVGs into the research article if the catalog figures
should change.

## Field-visualisation figures

See the comments in `figures/run_field_demos.sh`. Build OpenPFC,
run the demos, then render with `figures/make_field_figures.py`.
The research article consumes the committed SVGs/PNGs, not the
raw `.vti`/`.bin` dumps.
