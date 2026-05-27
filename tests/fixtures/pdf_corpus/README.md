# PDF Corpus

This folder holds optional real-paper PDF regression fixtures.

The files are intentionally not committed by default because they can be large and may have source-specific access limits. Download each entry from `manifest.json` into this folder using the exact `filename` value, then run:

```powershell
python -m unittest tests.test_pdf_corpus -v
```

The corpus test skips missing files and validates every PDF that is present.
