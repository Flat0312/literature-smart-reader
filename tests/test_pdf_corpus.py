from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from services.paper_parse_service import build_paper_result
from services.pdf_service import extract_pdf_context


CORPUS_DIR = Path(__file__).parent / "fixtures" / "pdf_corpus"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"


class PdfCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def without_relay(self):
        return patch.dict(
            os.environ,
            {"RELAY_API_KEY": "", "RELAY_BASE_URL": "", "RELAY_MODEL": ""},
            clear=False,
        )

    def test_manifest_has_real_pdf_targets(self) -> None:
        self.assertGreaterEqual(len(self.manifest), 5)
        for item in self.manifest:
            self.assertTrue(item["filename"].endswith(".pdf"))
            self.assertTrue(item["url"].startswith("https://"))

    def test_available_pdf_corpus_items_parse(self) -> None:
        available_items = [
            item
            for item in self.manifest
            if (CORPUS_DIR / item["filename"]).exists()
        ]
        if not available_items:
            self.skipTest("no local PDF corpus files downloaded yet")

        failures: list[str] = []
        with self.without_relay():
            for item in available_items:
                path = CORPUS_DIR / item["filename"]
                try:
                    pdf_result = extract_pdf_context(path.read_bytes())
                    paper_result = build_paper_result(path.name, pdf_result)
                except Exception as exc:  # pragma: no cover - failure aggregation
                    failures.append(f"{item['id']}: parse failed: {exc}")
                    continue

                expected_title = str(item.get("expected_title_contains", "")).strip()
                if expected_title and expected_title not in paper_result.title:
                    failures.append(
                        f"{item['id']}: title mismatch: expected contains {expected_title!r}, got {paper_result.title!r}"
                    )

                keywords = " ".join(paper_result.filtered_keywords())
                expected_keywords = [str(value) for value in item.get("expected_keywords_any", [])]
                if expected_keywords and not any(keyword.lower() in keywords.lower() for keyword in expected_keywords):
                    failures.append(
                        f"{item['id']}: keyword mismatch: expected one of {expected_keywords!r}, got {keywords!r}"
                    )

                min_structured_fields = int(item.get("min_structured_fields", 0))
                if paper_result.structured_field_count() < min_structured_fields:
                    failures.append(
                        f"{item['id']}: structured fields {paper_result.structured_field_count()} < {min_structured_fields}"
                    )

                preflight = pdf_result.preflight
                if not preflight.extraction_strategy or preflight.quality_score <= 0:
                    failures.append(
                        f"{item['id']}: invalid preflight strategy={preflight.extraction_strategy!r}, score={preflight.quality_score}"
                    )

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
