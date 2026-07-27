from __future__ import annotations

import re
from collections import defaultdict
from importlib.resources import files
from pathlib import Path

from pydantic import ValidationError

from powerhouse_ai.errors import PowerhouseError
from powerhouse_ai.llm import LLMClient
from powerhouse_ai.schemas import Domain, DomainScore, RouteDecision, ToolName

DOMAIN_PATTERNS: dict[Domain, tuple[str, ...]] = {
    Domain.PROGRAMMING: (
        r"\bcode\b",
        r"\bpython\b",
        r"\b(?:java|type)script\b",
        r"\bprogram(?:ming)?\b",
        r"\bsoftware\b",
        r"\bapi\b",
        r"\bfastapi\b",
        r"\bsql\b",
        r"\bdatabase\b",
        r"\bwebsite\b",
        r"\bdebug\w*\b",
        r"\bfunction\b",
        r"\balgorithm\b",
        r"\bimplement\w*\b",
        r"\brefactor\w*\b",
        r"\bfilename\b",
        r"\bcontext manager\b",
        r"\bretr(?:y|ies)\b",
        r"\bdef\s+\w+",
    ),
    Domain.BIOLOGY: (
        r"\bbiolog\w*\b",
        r"\bbacter\w*\b",
        r"\bcells?\b",
        r"\bgenes?\b",
        r"\bdna\b",
        r"\bproteins?\b",
        r"\benzym\w*\b",
        r"\batp(?: synthase)?\b",
        r"\borganism\w*\b",
        r"\bevolution\w*\b",
        r"\becolog\w*\b",
        r"\bpopulation\b",
        r"\bplants?\b",
        r"\bnatural selection\b",
        r"\bgrowth curves?\b",
    ),
    Domain.CHEMISTRY: (
        r"\bchemist\w*\b",
        r"\bchemical\w*\b",
        r"\bmoles?\b",
        r"\bmolecules?\b",
        r"\breaction\b",
        r"\bacids?\b",
        r"\balkali\w*\b",
        r"\bph\b",
        r"\bcompounds?\b",
        r"\bstoichiometr\w*\b",
        r"\bconcentrations?\b",
        r"\bcombustion\b",
        r"\bmethane\b",
        r"\bdilut\w*\b",
        r"\bsolute\b",
        r"\bmolar\b",
    ),
    Domain.MATHEMATICS: (
        r"\bmath\w*\b",
        r"\bequations?\b",
        r"\bcalculus\b",
        r"\balgebra\w*\b",
        r"\bodes?\b",
        r"\bdifferential\b",
        r"\bdifferentiat\w*\b",
        r"\bintegrals?\b",
        r"\bprobability of\b",
        r"\blogistic\b",
        r"\bnewton(?:'s|s) method\b",
        r"\bsolve\b",
        r"\bderive\b",
        r"\b(?:show|include|check) (?:the )?units?\b",
    ),
    Domain.STATISTICS: (
        r"\bstatistic\w*\b",
        r"\bdatasets?\b",
        r"\bregression\b",
        r"\bvariance\b",
        r"\bsampling\b",
        r"\bstandard error\b",
        r"\bconfidence interval\b",
        r"\bhypothesis\b",
        r"\bdistribution\b",
        r"\bcorrelation\b",
        r"\bconfound\w*\b",
        r"\bp-hacking\b",
        r"\brandomi[sz]\w*\b",
        r"\breplicates?\b",
        r"\bcontrolled experiment\b",
        r"\b(?:control|treatment) groups?\b",
        r"\bobservational study\b",
        r"\buncertainty\b",
    ),
    Domain.WRITING: (
        r"\brewrite\b",
        r"\bedit(?:or|ing)?\b",
        r"\bessay\b",
        r"\bemail\b",
        r"\bstory\b",
        r"\btone\b",
        r"\bgrammar\b",
        r"\bproposal\b",
        r"\b(?:write|draft|prepare|produce|create) (?:a )?(?:\w+ )?report\b",
        r"\bturn\b.+\binto (?:a )?(?:\w+ )?report\b",
        r"\bstyle guide\b",
    ),
    Domain.RESEARCH: (
        r"\bresearch\b",
        r"\bsources?\b",
        r"\bevidence\b",
        r"\bliterature\b",
        r"\bpapers?\b",
        r"\binvestigat\w*\b",
        r"\bcitations?\b",
        r"\bscientific study\b",
        r"\bscientific source\b",
        r"\bp-hacking\b",
        r"\bobservational study\b",
        r"\btrustworthy\b",
    ),
}


class Router:
    def __init__(self, llm: LLMClient | None = None, prompt_path: Path | None = None) -> None:
        self.llm = llm
        if prompt_path is not None:
            self.prompt = prompt_path.read_text(encoding="utf-8")
        else:
            self.prompt = (
                files("powerhouse_ai")
                .joinpath("resources", "prompts", "router.txt")
                .read_text(encoding="utf-8")
            )

    def heuristic(self, user_prompt: str) -> RouteDecision:
        lowered = user_prompt.lower()
        words = set(re.findall(r"\b[\w'-]+\b", lowered))

        memory_phrases = (
            "recall",
            "remember",
            "saved decision",
            "stored for this project",
            "project memory",
            "previous decision",
        )
        if any(phrase in lowered for phrase in memory_phrases):
            return RouteDecision(
                goal=user_prompt.strip()[:500],
                domains=[DomainScore(name=Domain.GENERAL, importance=1.0)],
                tools=[ToolName.DOCUMENT_RETRIEVAL],
                complexity=0.35,
                requires_multiple_steps=False,
                confidence=0.9,
                source="heuristic",
            )

        scores: defaultdict[Domain, int] = defaultdict(int)
        for domain, patterns in DOMAIN_PATTERNS.items():
            scores[domain] = sum(bool(re.search(pattern, lowered)) for pattern in patterns)
        calculation_requested = bool(
            re.search(r"\bcalculat\w*\b|\b(?:add|subtract|multiply|divide)\b", lowered)
            or re.search(r"\d\s*[-+*/^]\s*\d", lowered)
        )
        if calculation_requested and not scores[Domain.STATISTICS]:
            scores[Domain.MATHEMATICS] += 1
        unit_conversion_requested = bool(
            re.search(
                r"\d+(?:\.\d+)?\s*(?:mm|cm|m|km|mg|g|kg|ml|l|s|min|h)"
                r"\s+(?:to|into|in)\s+(?:mm|cm|m|km|mg|g|kg|ml|l|s|min|h)\b",
                lowered,
            )
        )
        if unit_conversion_requested:
            scores[Domain.MATHEMATICS] += 1

        ranked = [(domain, score) for domain, score in scores.items() if score > 0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        had_domain_match = bool(ranked)
        if not ranked:
            ranked = [(Domain.GENERAL, 1)]
        max_score = max(score for _, score in ranked)
        domains = [
            DomainScore(name=domain, importance=min(1.0, 0.35 + 0.65 * score / max_score))
            for domain, score in ranked[:4]
        ]

        selected = {item.name for item in domains}
        tools: list[ToolName] = []
        needs_calculator = bool(
            selected & {Domain.MATHEMATICS, Domain.STATISTICS, Domain.CHEMISTRY}
        ) and not (unit_conversion_requested and not calculation_requested)
        if needs_calculator:
            tools.append(ToolName.CALCULATOR)
        if unit_conversion_requested:
            tools.append(ToolName.UNIT_CONVERSION)
        if Domain.PROGRAMMING in selected:
            tools.extend([ToolName.PYTHON, ToolName.CODE_TESTS])
        if selected & {
            Domain.BIOLOGY,
            Domain.CHEMISTRY,
            Domain.RESEARCH,
            Domain.PROGRAMMING,
        }:
            tools.append(ToolName.DOCUMENT_RETRIEVAL)

        multi = len(domains) > 1 or len(user_prompt) > 500
        complexity = min(
            0.95,
            0.2 + 0.14 * len(domains) + (0.2 if multi else 0.0) + min(len(words) / 200, 0.25),
        )
        if not had_domain_match:
            confidence = 0.8
        elif len(ranked) == 1:
            confidence = 0.88
        else:
            confidence = 0.72
        return RouteDecision(
            goal=user_prompt.strip()[:500],
            domains=domains,
            tools=list(dict.fromkeys(tools)),
            complexity=complexity,
            requires_multiple_steps=multi,
            confidence=confidence,
            source="heuristic",
        )

    def route(
        self,
        user_prompt: str,
        *,
        force_llm: bool = False,
        allow_llm: bool = True,
    ) -> RouteDecision:
        initial = self.heuristic(user_prompt)
        should_use_llm = allow_llm and (
            force_llm
            or (self.llm is not None and initial.confidence < 0.6)
        )
        if not should_use_llm or self.llm is None:
            return initial

        try:
            data = self.llm.chat_json(
                [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1200,
            )
            decision = RouteDecision.model_validate(data)
            decision.source = "model"
            if not decision.domains:
                decision.domains = initial.domains
            return decision
        except (PowerhouseError, ValidationError, ValueError):
            return initial
