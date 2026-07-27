from __future__ import annotations

import ast
import json
import math
import operator
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class SafeCalculator:
    _max_ast_nodes = 128
    _max_absolute_result = 10**100
    _binary: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    _functions: dict[str, Callable[..., float]] = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
    }
    _constants = {"pi": math.pi, "e": math.e}

    def evaluate(self, expression: str) -> float | int:
        if len(expression) > 500:
            raise ValueError("Expression is too long.")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > self._max_ast_nodes:
            raise ValueError("Expression is too complex.")
        return self._visit(tree.body)

    def _guard_number(self, value: Any) -> float | int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Expression did not produce a real number.")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Expression produced a non-finite result.")
        if abs(value) > self._max_absolute_result:
            raise ValueError("Result is outside the safe numeric range.")
        return value

    def _visit(self, node: ast.AST) -> float | int:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return self._guard_number(node.value)
        if isinstance(node, ast.Name) and node.id in self._constants:
            return self._constants[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            left = self._visit(node.left)
            right = self._visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100:
                    raise ValueError("Exponent is outside the safe range.")
                if right > 0 and abs(left) > 1:
                    result_digits = float(right) * math.log10(abs(left))
                    if result_digits > 100:
                        raise ValueError("Result is outside the safe numeric range.")
            return self._guard_number(self._binary[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._guard_number(self._unary[type(node.op)](self._visit(node.operand)))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._functions
        ):
            if node.keywords or len(node.args) > 8:
                raise ValueError("Unsupported function arguments.")
            result = self._functions[node.func.id](*(self._visit(arg) for arg in node.args))
            return self._guard_number(result)
        raise ValueError("Expression contains unsupported operations.")


UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "mm": ("length", 0.001),
    "cm": ("length", 0.01),
    "m": ("length", 1.0),
    "km": ("length", 1000.0),
    "mg": ("mass", 0.000001),
    "g": ("mass", 0.001),
    "kg": ("mass", 1.0),
    "ml": ("volume", 0.000001),
    "l": ("volume", 0.001),
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
}

_CALCULATION_FUNCTION = re.compile(
    r"\b(?:sqrt|log10|log|exp|sin|cos|tan|abs)\s*\([^()\r\n]{1,100}\)"
    r"(?:\s*(?:\*\*|//|[+\-*/%^])\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))*",
    flags=re.IGNORECASE,
)
_CALCULATION_ARITHMETIC = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?:\*\*|//|[+\-*/%^])\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+))+"
)


def _calculation_match(text: str) -> re.Match[str] | None:
    return _CALCULATION_FUNCTION.search(text) or _CALCULATION_ARITHMETIC.search(text)


def _is_complete_calculation_match(text: str, match: re.Match[str]) -> bool:
    """Reject matches that are visibly part of a larger or second expression."""
    before = text[: match.start()].rstrip()
    after = text[match.end() :].lstrip()
    continuation = "()+-*/%^"
    if (before and before[-1] in continuation) or (after and after[0] in continuation):
        return False
    if re.search(r"\d[,.]$", before) or re.match(r"^[,.]\d", after):
        return False

    for candidate in [
        *_CALCULATION_FUNCTION.finditer(text),
        *_CALCULATION_ARITHMETIC.finditer(text),
    ]:
        overlaps = candidate.start() < match.end() and match.start() < candidate.end()
        if not overlaps:
            return False
    return True


def extract_calculation(text: str) -> str | None:
    """Extract only an unambiguous calculator expression from natural-language text."""
    match = _calculation_match(text)
    if match is None or not _is_complete_calculation_match(text, match):
        return None
    return match.group(0).lower().replace("^", "**")


def _normalized_wrapper(text: str) -> str:
    return re.sub(r"[\s,;:.!?]+", " ", text.strip().lower()).strip()


def _has_exact_tool_wrapper(
    text: str,
    match: re.Match[str],
    *,
    prefix_pattern: str,
    tool_pattern: str,
) -> bool:
    prefix = _normalized_wrapper(text[: match.start()])
    suffix = _normalized_wrapper(text[match.end() :])
    if prefix and re.fullmatch(prefix_pattern, prefix) is None:
        return False
    result_clause = r"(?:state|give|return|show) (?:the )?(?:exact )?result"
    suffix_pattern = (
        rf"(?:(?:using|with) (?:the )?(?:{tool_pattern})"
        rf"(?: and {result_clause})?|and {result_clause}|{result_clause})?(?: ?please)?"
    )
    return re.fullmatch(suffix_pattern, suffix) is not None


def is_exact_calculation_request(text: str) -> bool:
    """Return whether the whole request asks only for one extractable calculation."""
    match = _calculation_match(text)
    if match is None:
        return False
    prefix_pattern = (
        r"(?:please )?(?:(?:calculate|compute|evaluate|solve)"
        r"(?: (?:the )?(?:value|result) of)?|what is|find (?:the )?(?:value|result) of)"
    )
    return _has_exact_tool_wrapper(
        text,
        match,
        prefix_pattern=prefix_pattern,
        tool_pattern=r"calculator(?: tool)?",
    )


def validate_python_blocks(text: str) -> dict[str, Any] | None:
    openings = list(
        re.finditer(
            r"```(?:python|py)[ \t]*\r?\n",
            text,
            flags=re.IGNORECASE,
        )
    )
    blocks = re.findall(
        r"```(?:python|py)[ \t]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not openings:
        return None
    errors = []
    for index, block in enumerate(blocks, start=1):
        try:
            ast.parse(block)
        except SyntaxError as exc:
            errors.append(
                {
                    "block": index,
                    "line": exc.lineno,
                    "offset": exc.offset,
                    "error": exc.msg,
                }
            )
    if len(blocks) < len(openings):
        errors.append(
            {
                "block": len(blocks) + 1,
                "line": None,
                "offset": None,
                "error": "Python code fence is not closed.",
            }
        )
    return {"ok": not errors, "blocks_checked": len(blocks), "errors": errors}


def _unit_conversion_match(text: str) -> re.Match[str] | None:
    units = "|".join(sorted(UNIT_FACTORS, key=len, reverse=True))
    return re.search(
        rf"(?<![\w.])(\d+(?:\.\d+)?)\s*({units})\s+(?:to|into|in)\s+({units})\b",
        text,
        flags=re.IGNORECASE,
    )


def extract_unit_conversion(text: str) -> tuple[float, str, str] | None:
    match = _unit_conversion_match(text)
    if not match:
        return None
    return float(match.group(1)), match.group(2).lower(), match.group(3).lower()


def is_exact_unit_conversion_request(text: str) -> bool:
    """Return whether the whole request asks only for one extractable unit conversion."""
    match = _unit_conversion_match(text)
    if match is None:
        return False
    return _has_exact_tool_wrapper(
        text,
        match,
        prefix_pattern=r"(?:please )?(?:convert|what is)",
        tool_pattern=r"unit conversion(?: tool)?|converter",
    )


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    source = UNIT_FACTORS.get(from_unit.lower())
    target = UNIT_FACTORS.get(to_unit.lower())
    if source is None or target is None:
        raise ValueError("Unsupported unit.")
    if source[0] != target[0]:
        raise ValueError("Units describe different dimensions.")
    return value * source[1] / target[1]


def run_python(code: str, timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Run trusted code with isolation flags. This is not a hostile-code security boundary."""
    if len(code) > 50_000:
        raise ValueError("Code is too large.")
    with tempfile.TemporaryDirectory(prefix="powerhouse-") as directory:
        script = Path(directory) / "main.py"
        script.write_text(code, encoding="utf-8")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={},
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout[-20_000:],
                "stderr": result.stderr[-20_000:],
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": 124,
                "stdout": (exc.stdout or "")[-20_000:],
                "stderr": "Execution timed out.",
                "duration_seconds": round(time.perf_counter() - started, 3),
            }


class ToolRegistry:
    def __init__(self, *, enable_python: bool = False) -> None:
        self.enable_python = enable_python
        self.calculator = SafeCalculator()

    def definitions(self, allowed_names: list[str] | None = None) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a numerical expression accurately.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "maxLength": 500},
                        },
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "unit_conversion",
                    "description": "Convert a value between compatible metric units.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "from_unit": {"type": "string"},
                            "to_unit": {"type": "string"},
                        },
                        "required": ["value", "from_unit", "to_unit"],
                    },
                },
            },
        ]
        if self.enable_python:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "python",
                        "description": (
                            "Execute trusted standard-library Python code with a timeout. "
                            "No third-party packages are available."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string"}},
                            "required": ["code"],
                        },
                    },
                }
            )
        if allowed_names is not None:
            allowed = set(allowed_names)
            definitions = [
                item for item in definitions if item["function"]["name"] in allowed
            ]
        return definitions

    def execute(self, name: str, arguments: str | dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            if name == "calculator":
                result = self.calculator.evaluate(str(parsed["expression"]))
            elif name in {"unit_conversion", "convert_unit"}:
                result = convert_unit(
                    float(parsed["value"]),
                    str(parsed["from_unit"]),
                    str(parsed["to_unit"]),
                )
            elif name == "python" and self.enable_python:
                return {"ok": True, "result": run_python(str(parsed["code"]))}
            else:
                raise ValueError(f"Unknown or disabled tool: {name}")
            return {"ok": True, "result": result}
        except (
            ArithmeticError,
            SyntaxError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            return {"ok": False, "error": str(exc)}
