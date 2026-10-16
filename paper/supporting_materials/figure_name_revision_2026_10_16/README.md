# Online comparison figure: display-name revision, 2026-10-16

This folder preserves the exact frozen numerical inputs and a portable
generator for the complete manuscript's online service comparison figure
(`figures/mfd_structure_online_runtime.pdf`). The revision changes the
displayed method names in the legend to:

- All three enclosures: target 1600
- Signed: target 1625
- Matched nominal replays (overlap)

The manuscript caption identifies the three enclosures as Base interval,
Positivity/queue, and Signed. The plotted scientific arrays, curves, axes,
ticks, line styles, and numerical results remain unchanged. The generator
performs no simulation, optimization, integration, or network/API call.

The historical run identifiers containing `original`, `positive_queue`,
or `affine_v3` are preserved inside the input JSON. These identifiers link
the frozen arrays to the archived experiment records; the displayed
`affine` method name is now `Signed`.

## Regenerate

Use Python with NumPy and Matplotlib installed:

```sh
python regenerate_figure.py --output-dir generated_figure
```

The default input is `frozen_figuredata.json` beside the script, so this
command can run from any working directory. `--output-dir` is required and
specifies the local destination for the PDF, PNG, and
`FIGURE_NAME_REVISION_QA.json`. An optional `--data` path selects a copy of
the same frozen JSON.

The verification record reports the input SHA-256, a canonical hash of all
scientific arrays, and the hashes of the generated PDF and PNG. PDF hashes
may vary because the plotting library writes creation timestamps and
platform-dependent font metadata. The scientific-array hash is the stable
check that the plotted numerical inputs are unchanged.

Frozen input SHA-256:
`58ca8c31ad1d11e8f9b750a21f2470c60030728b225335c0807b63feb77052c0`

Scientific arrays SHA-256:
`896700e4a97526792af0fcf07f421d6c7c60568ffd80085d7b999fca7915c249`

The frozen JSON is an exact copy of the previously archived
`FOUNDATION_MFD_PARTB_CONSISTENCY_REVISION_2026-09-25/figures/`
`mfd_structure_online_runtime_figuredata.json`. Its historical provenance
fields are retained as recorded.
