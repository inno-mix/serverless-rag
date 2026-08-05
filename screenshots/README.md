# Screenshots

Demo captures of the deployed UI, for the project README and for linking from a
CV or portfolio page.

| File | What it shows |
|---|---|
| [`desktop.png`](desktop.png) | The full app at 1280px: document list with live status, an answer, and a citation expanded to the passage it came from |
| [`mobile.png`](mobile.png) | The same app at 390px. The four-column table becomes a two-line block per document, and every control clears a 44px touch target |
| [`similarity-floor.png`](similarity-floor.png) | An out-of-scope question refused **without a model call** — the behaviour ADR-003 exists for |
| [`dark-mode.png`](dark-mode.png) | The dark theme, which follows `prefers-color-scheme` |

## How these were produced

The real `frontend/` sources, served locally against a stub `/v1` API with
representative data, rendered in Chromium at the stated viewport widths.

They are **not** captures of live traffic, and the numbers in them — latency,
token counts, similarity scores — are plausible values rather than measurements.
Two reasons that matters here:

- The similarity scores shown (0.751, 0.622) are in the range the golden-set
  eval actually measures for answerable questions, but they were not produced by
  a Bedrock call.
- The latency figures are representative of the p50 target in PRD NFR-1, not a
  timing of these specific requests.

If you want captures of genuine traffic instead, deploy, upload a document, and
re-shoot against the CloudFront URL:

```bash
make deploy
```

Everything except the data source is identical — same HTML, same CSS, same
JavaScript, same API response shapes.

## Re-shooting after a UI change

These are static files and will drift as the frontend changes. There is no
automation regenerating them, so re-capture after any visible change to
`frontend/`, at 1280px and 390px.
