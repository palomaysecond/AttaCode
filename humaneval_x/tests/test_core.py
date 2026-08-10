from __future__ import annotations

import shutil
import unittest

from rq5_reproduction.cpp_runner import compile_and_run
from rq5_reproduction.dataset import HumanEvalTask
from rq5_reproduction.generator import GenerationContext, extract_cpp_code, generate_task_stages
from rq5_reproduction.llm_clients import LLMClient, LLMResponse
from rq5_reproduction.metrics import aggregate_results, render_latex_rows
from rq5_reproduction.rules import RULES, build_rule_manifest
from rq5_reproduction.syntax import CppSyntaxChecker


def benchmark_fixture() -> HumanEvalTask:
    return HumanEvalTask(
        task_id="CPP/test",
        prompt="Return the sum.",
        declaration="#include <cassert>\nint add(int a, int b){\n",
        canonical_solution="    int value = a + b;\n    return value;\n}\n",
        test="#undef NDEBUG\n#include <assert.h>\nint main(){ assert(add(2, 3) == 5); }\n",
        example_test="",
    )


class FixedResponseClient(LLMClient):
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def generate(self, prompt: str) -> LLMResponse:
        return LLMResponse(text=self.response_text)


class DatasetAndRulesTests(unittest.TestCase):
    def test_source_assembly_and_entrypoint(self) -> None:
        task = benchmark_fixture()
        self.assertEqual(task.entrypoint, "add")
        self.assertIn("int add", task.source_code)
        self.assertIn("int main", task.evaluation_program)

    def test_manifest_is_deterministic(self) -> None:
        task = benchmark_fixture()
        self.assertEqual(len(RULES), 25)
        first = build_rule_manifest([task], {task.task_id}, stages=3, seed=42)
        second = build_rule_manifest([task], {task.task_id}, stages=3, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]["rules"]), 3)


class GenerationUtilityTests(unittest.TestCase):
    def test_extracts_cpp_fence(self) -> None:
        response = "Explanation\n```cpp\nint f(){ return 1; }\n```\n"
        self.assertEqual(extract_cpp_code(response), "int f(){ return 1; }")

    def test_syntax_checker(self) -> None:
        checker = CppSyntaxChecker()
        self.assertTrue(checker.check("int f(){ return 1; }").valid)
        self.assertFalse(checker.check("int f(){ return 1;").valid)

    def test_unchanged_generation_blocks_dependent_stages(self) -> None:
        task = benchmark_fixture()
        manifest = build_rule_manifest([task], {task.task_id}, stages=3, seed=42)[0]
        template = (
            "Stage {stage}; rule {rule_name}: {rule_description}; entry {entrypoint}; "
            "location {location}\n{source_code}"
        )
        context = GenerationContext(
            templates=[template],
            checker=CppSyntaxChecker(),
            generation_config={"max_attempts": 2, "refusal_keywords": []},
        )
        records = generate_task_stages(
            task,
            "controlled-client",
            FixedResponseClient(task.source_code),
            manifest,
            context,
        )
        self.assertEqual(
            [record["generation_status"] for record in records],
            ["generation_failed", "blocked_by_previous_stage", "blocked_by_previous_stage"],
        )
        self.assertEqual(records[0]["output_code"], task.source_code)
        self.assertEqual(len(records[0]["attempts"]), 2)


class MetricsTests(unittest.TestCase):
    def test_metric_denominators(self) -> None:
        baseline = [{"task_id": "CPP/0", "passed": True}, {"task_id": "CPP/1", "passed": True}]
        evaluation = [
            {"task_id": "CPP/0", "model_id": "m", "stage": 1, "compiled": True, "passed": True, "changed": True},
            {"task_id": "CPP/1", "model_id": "m", "stage": 1, "compiled": True, "passed": False, "changed": True, "failure_kind": "test_failure"},
        ]
        summary = aggregate_results(baseline, evaluation, [{"id": "m", "display_name": "M"}], stages=1)
        group = summary["groups"][0]
        self.assertEqual(group["CSR"], 1.0)
        self.assertEqual(group["TPR"], 0.5)
        self.assertEqual(group["BPR"], 0.5)
        self.assertIn("1/2 (50.0%)", render_latex_rows(summary))


class CompilerTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ is not available")
    def test_compile_and_run(self) -> None:
        result = compile_and_run(
            benchmark_fixture().evaluation_program,
            {
                "executable": "g++",
                "standard": "gnu++17",
                "flags": ["-O0"],
                "compile_timeout_seconds": 30,
                "run_timeout_seconds": 5,
                "memory_limit_mb": None,
                "max_log_characters": 2000,
            },
        )
        self.assertTrue(result["compiled"], result["compile_stderr"])
        self.assertTrue(result["passed"], result["run_stderr"])


if __name__ == "__main__":
    unittest.main()
