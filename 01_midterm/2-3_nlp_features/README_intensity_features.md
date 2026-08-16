# Text intensity / severity features

## Goal

Distinguish a mild complaint such as “slightly warm” from a severe safety or
failure signal such as “burned my hand” or “stopped working”.

## Source and leakage rule

- Source: `data/interim/clean_reviews.parquet`
- Text: `text_norm`
- Unit: product × month
- Only reviews from the current month or preceding months are used.
- `*_p3` is the mean over the previous three months and excludes the current
  month; `*_delta` is current-month rate minus that previous-month mean.
- The next-month surge label is never used to construct a feature.

## Severity hierarchy

| Level | Signal | Score |
| --- | --- | ---: |
| Mild | mild qualifier + issue word | 1 |
| Severe | strong qualifier + issue word | 2 |
| Safety | burns, shocks, fires, smoke, toxic risks | 3 |
| Consequence | stopped working, injury, destroyed, leaked everywhere | 2 |
