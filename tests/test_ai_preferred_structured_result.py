from __future__ import annotations

import unittest
from unittest.mock import patch

from services.llm_service import RelayConfigError
from services.paper_parse_service import _resolve_structured_result
from services.structured_rewrite_service import (
    StructuredFieldCandidate,
    StructuredRewriteResult,
    build_structured_rewrite_request,
)


class AiPreferredStructuredResultTests(unittest.TestCase):
    def _rule_complete_request(self):
        return build_structured_rewrite_request(
            raw_text="这是一段可用于结构化抽取的论文摘要。",
            title="文化遗产数字画像体系弹性保护视域下我国文化遗产分类体系重构",
            candidates={
                "research_question": StructuredFieldCandidate(
                    field_name="research_question",
                    text="本文旨在解决文化遗产分类体系过于线性的问题。",
                ),
                "research_method": StructuredFieldCandidate(
                    field_name="research_method",
                    text="本文基于数字画像方法构建分群重构框架。",
                ),
                "core_conclusion": StructuredFieldCandidate(
                    field_name="core_conclusion",
                    text="研究表明数字画像可以支撑文化遗产系统性保护。",
                ),
            },
            preferred_backend="relay_llm",
            debug_info={
                "explicit_abstract_labels_found": True,
                "rule_candidate_char_count": 72,
                "abstract_fallback_text": "本文从文化遗产数字画像角度讨论分类体系重构。",
            },
        )

    def test_ai_rewrite_is_attempted_even_when_rule_fields_are_complete(self) -> None:
        request = self._rule_complete_request()
        llm_debug_seed = {
            "rule_candidate_char_count": 72,
            "abstract_char_count": 25,
            "abstract_fallback_enabled": False,
            "llm_input_source": "正文候选",
        }

        def fake_rewrite(_request):
            return StructuredRewriteResult(
                research_question="AI 改写后的研究问题。",
                research_method="AI 改写后的研究方法。",
                core_conclusion="AI 改写后的核心结论。",
                confidence="中",
                note="模型已基于规则候选完成重写。",
                backend="relay_chat",
                candidates=_request.candidates,
                debug_info={"backend": "relay_chat", "valid_return": True},
            )

        with patch("services.paper_parse_service.rewrite_structured_result", side_effect=fake_rewrite) as rewrite_mock:
            result, notice, debug = _resolve_structured_result(request, llm_debug_seed)

        self.assertEqual("relay_chat", result.backend)
        self.assertEqual("AI 改写后的研究问题。", result.research_question)
        self.assertIn("模型", notice)
        self.assertEqual("relay_chat", debug["backend"])
        rewrite_mock.assert_called_once_with(request)

    def test_rule_complete_request_falls_back_to_local_rules_when_ai_config_is_missing(self) -> None:
        request = self._rule_complete_request()
        llm_debug_seed = {
            "rule_candidate_char_count": 72,
            "abstract_char_count": 25,
            "abstract_fallback_enabled": False,
            "llm_input_source": "正文候选",
        }

        with patch(
            "services.paper_parse_service.rewrite_structured_result",
            side_effect=RelayConfigError("缺少 RELAY_MODEL 配置，已回退到规则结果。"),
        ):
            result, notice, debug = _resolve_structured_result(request, llm_debug_seed)

        self.assertEqual("local_rule", result.backend)
        self.assertEqual("relay_config_error", debug["backend"])
        self.assertIn("回退", notice)


if __name__ == "__main__":
    unittest.main()
