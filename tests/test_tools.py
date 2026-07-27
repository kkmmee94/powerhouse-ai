import pytest

from powerhouse_ai.tools import (
    SafeCalculator,
    ToolRegistry,
    convert_unit,
    extract_calculation,
    extract_unit_conversion,
    is_exact_calculation_request,
    is_exact_unit_conversion_request,
    run_python,
    validate_python_blocks,
)


def test_calculator_uses_operator_precedence() -> None:
    assert SafeCalculator().evaluate("2 + 3 * 4") == 14


def test_calculator_supports_math_functions() -> None:
    assert SafeCalculator().evaluate("sqrt(81) + round(pi, 2)") == pytest.approx(12.14)


def test_calculator_rejects_code_execution() -> None:
    with pytest.raises(ValueError):
        SafeCalculator().evaluate("__import__('os').system('whoami')")


@pytest.mark.parametrize("expression", ["1/0", "1+", "exp(1000)"])
def test_calculator_errors_are_returned_as_tool_results(expression: str) -> None:
    result = ToolRegistry().execute("calculator", {"expression": expression})
    assert result["ok"] is False
    assert result["error"]


def test_calculator_rejects_excessive_intermediate_results() -> None:
    with pytest.raises(ValueError, match="safe numeric range"):
        SafeCalculator().evaluate("(10**100)**100")


def test_unit_conversion_checks_dimensions() -> None:
    assert convert_unit(2.5, "kg", "g") == 2500
    with pytest.raises(ValueError):
        convert_unit(1, "kg", "m")


def test_extracts_unit_conversion_request() -> None:
    assert extract_unit_conversion("Convert 2.5 kg to g.") == (2.5, "kg", "g")


def test_python_tool_is_disabled_by_default() -> None:
    result = ToolRegistry().execute("python", {"code": "print('unsafe')"})
    assert result["ok"] is False


def test_python_tool_runs_only_when_explicitly_enabled() -> None:
    result = ToolRegistry(enable_python=True).execute(
        "python",
        {"code": "print(6 * 7)"},
    )
    assert result["ok"] is True
    assert result["result"]["exit_code"] == 0
    assert result["result"]["stdout"].strip() == "42"


def test_python_tool_reports_a_timeout() -> None:
    result = run_python("while True: pass", timeout_seconds=0.1)
    assert result["exit_code"] == 124
    assert result["stderr"] == "Execution timed out."


def test_extracts_only_unambiguous_calculations() -> None:
    assert extract_calculation("Calculate sqrt(144), please.") == "sqrt(144)"
    assert extract_calculation("Calculate sqrt(144) + 3^2.") == "sqrt(144) + 3**2"
    assert extract_calculation("What is 12 * 17?") == "12 * 17"
    assert extract_calculation("Explain a confidence interval.") is None
    assert extract_calculation("Calculate (2+3)*4.") is None
    assert extract_calculation("Calculate 2+2 and 3+3.") is None


def test_exact_tool_requests_reject_additional_tasks() -> None:
    assert is_exact_calculation_request(
        "Calculate sqrt(144) + 3^2 using the calculator tool and state the exact result."
    )
    assert not is_exact_calculation_request(
        "Calculate 2+2 and tell me the capital of France."
    )
    assert is_exact_unit_conversion_request(
        "Convert 2.5 kg to g using the unit conversion tool and state the exact result."
    )
    assert not is_exact_unit_conversion_request(
        "Convert 2.5 kg to g and write a poem about the result."
    )


def test_tool_definitions_can_be_scoped_to_the_plan() -> None:
    definitions = ToolRegistry().definitions(["calculator"])
    assert [item["function"]["name"] for item in definitions] == ["calculator"]


def test_python_code_blocks_are_syntax_checked_without_execution() -> None:
    valid = validate_python_blocks("```python\nvalue = 2 + 2\n```")
    valid_unlabelled = validate_python_blocks("```\nvalue = 2 + 2\n```")
    unlabelled_sql = validate_python_blocks("```\nSELECT * FROM users;\n```")
    invalid = validate_python_blocks("```python\nif True print('bad')\n```")
    assert valid == {"ok": True, "blocks_checked": 1, "errors": []}
    assert valid_unlabelled is None
    assert unlabelled_sql is None
    assert invalid is not None
    assert invalid["ok"] is False


def test_unclosed_python_code_fence_is_invalid() -> None:
    result = validate_python_blocks("```python\nvalue = 2 + 2")
    assert result is not None
    assert result["ok"] is False
    assert result["errors"] == [
        {
            "block": 1,
            "line": None,
            "offset": None,
            "error": "Python code fence is not closed.",
        }
    ]
