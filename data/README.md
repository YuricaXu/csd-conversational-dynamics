# Raw data

This directory bundles the two primary data sources used in the project. See
the top-level `data_sources.txt` for full citations, URLs, and licenses.

## Files

| File | Source | Size |
|------|--------|------|
| `emotion-detection-trn.json` | Emory NLP, S1–S4 train split | 4.0 MB |
| `emotion-detection-dev.json` | Emory NLP, S1–S4 dev split | 550 KB |
| `emotion-detection-tst.json` | Emory NLP, S1–S4 test split | 543 KB |
| `NRC-VAD-Lexicon-v2.1.txt`   | NRC VAD Lexicon v2.1 | 1.5 MB |

## Note on splits

This project does not train a model; it applies a deterministic detector and
performs manual validation. The Emory train / dev / test split is therefore
collapsed into one corpus on load. See `code/model/data_loader.py`.

## Note on the lexicon

The NRC VAD file is the *unigram + bigram* combined table. Trigrams in the
upstream distribution are not used (the bigram-first greedy strategy in
`code/model/vad_engine.py` covers >95% of matched tokens).

## Reproducibility

If for some reason these files are missing or stale, fetch them fresh from:

- https://github.com/emorynlp/emotion-detection
- https://saifmohammad.com/WebPages/nrc-vad.html

and place them with the exact filenames shown above.
