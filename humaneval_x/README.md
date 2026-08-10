# HumanEval-X behavioral-preservation reproduction for AttaCode RQ5

This directory contains the complete reproduction pipeline for the executable part of RQ5. It evaluates whether sequential AttaCode-generated C++ transformations preserve the behavior exercised by the official HumanEval-X tests.

## Experiment protocol

For every HumanEval-X C++ task, the artifact:

1. constructs the reference implementation as `declaration + canonical_solution`;
2. appends the official `test` field only inside the evaluator;
3. retains original programs that compile and pass all tests;
4. builds one ordered, task-specific sequence of three applicable rules;
5. gives the same sequence to every configured generator;
6. applies one rule per stage, with stage `S - 1` output becoming stage `S` input;
7. never provides compiler or test feedback to a generator; and
8. reports CSR, TPR, and BPR using the fixed baseline set as the denominator.

The official tests are deliberately absent from prompts. Tree-sitter is a required syntax filter only; compilation and execution provide the behavioral evidence. If a stage cannot produce an accepted, changed implementation within the retry budget, that stage and its dependent later stages remain failures in the fixed denominator.

## Repository layout

```text
humaneval_x/
├── humaneval_cpp.jsonl
├── run_pipeline.py
├── experiment_config.json
├── requirements.txt
├── prompts/
├── rq5_reproduction/
└── tests/
```

Runtime outputs are written below the configured `output_dir` and ignored by Git.

## Installation

Run the following commands from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

A C++ compiler must also be available. The committed experiment configuration uses `g++ -std=gnu++17`; check `g++ --version` because `/usr/bin/g++` is an Apple Clang alias on macOS. HumanEval-X tasks also reference Boost.Any and OpenSSL headers, so their development packages must be visible to the selected compiler. The paper run retained 162 programs; the pipeline deliberately derives this count again instead of forcing it. If the local count differs, `baseline.jsonl` gives the task-level reason and `metadata.json` records the compiler, platform, dataset hash, and resulting count.

## Running the experiment

The version-controlled `experiment_config.json` contains the compiler settings, generation parameters, model identifiers, rule scheduling policy, and environment-variable names used by the pipeline. API credentials remain outside the repository and are read from the process environment.

```bash
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export GEMINI_API_KEY="..."
```

Custom endpoints can be supplied through `OPENAI_BASE_URL`, `DEEPSEEK_BASE_URL`, and `GEMINI_BASE_URL`. The committed configuration already contains the public DeepSeek and Gemini OpenAI-compatible endpoints; environment variables override them when a gateway is used.

Run the stages separately when conducting the full experiment:

```bash
python run_pipeline.py baseline  --config experiment_config.json
python run_pipeline.py manifest  --config experiment_config.json
python run_pipeline.py generate  --config experiment_config.json
python run_pipeline.py evaluate  --config experiment_config.json
python run_pipeline.py aggregate --config experiment_config.json
```

Use `--models gpt4o` to execute a selected configured generator. `--overwrite` explicitly replaces the output of the requested stage.

## Rule ordering

RQ5 uses a frozen task-specific rule schedule so that GPT-4o, DeepSeek-V3, and Gemini-2.5-Pro receive exactly the same ordered transformations. The configured `deterministic` strategy constructs this schedule by applying a stable SHA-256 ordering to rules that are applicable to each original program. Freezing the schedule isolates differences caused by the code generators rather than differences in rule selection.

To use attribution-guided ordering, set `rule_selection.strategy` to `importance` and set `rule_selection.importance_manifest` to a relative JSON or JSONL path. Supported rows include the format produced by the AttaCode importance analyzer:

```json
{
  "index": 0,
  "patterns": [
    {"transformation_name": "ReturnViaTempVariable", "score": 2.31},
    {"transformation_name": "ExtractSubexpression", "score": 1.77}
  ]
}
```

Rows may instead use `task_id` and `rules`; rule entries may use `rule_name`, `rule`, or `transformation_name`. When an attribution file provides fewer than three applicable scored rules, the remaining stages are completed using the frozen deterministic ordering. The generated `rule_manifest.jsonl` is the auditable record proving that all generators received the same task-specific order.

## Outputs

The pipeline produces:

```text
outputs/
├── metadata.json
├── baseline.jsonl
├── rule_manifest.jsonl
├── generations/<model_id>.jsonl
├── evaluation.jsonl
├── summary.json
├── summary.csv
└── table_rows.tex
```

- `baseline.jsonl` records compilation and official-test results for the unmodified reference implementations.
- `rule_manifest.jsonl` records the shared rule name, score, source, and target location for every stage.
- `generations/*.jsonl` retains prompts, raw responses, accepted code, rejected-code hashes, syntax decisions, token usage, and retry information. API keys are never logged.
- `evaluation.jsonl` contains compilation and execution outcomes without exposing test code to the generator.
- `table_rows.tex` can be pasted into the body of the RQ5 LaTeX table after the results have been reviewed.

## Metrics

For each generator and stage, let `N` be the number of original programs that compile and pass all official tests, `N_compiled` the number of transformed programs that compile, and `N_passed` the number that compile and pass every test:

```text
CSR = N_compiled / N
TPR = N_passed / N_compiled
BPR = N_passed / N
```

Generation, validation, and scheduling failures remain in the fixed denominator and are reported as failures. Later stages are not generated after an earlier stage fails, because they would no longer represent the stated sequential transformation depth.

## Reproducibility notes

- Pin or record dated model versions whenever the provider exposes them. Mutable aliases such as `gpt-4o` can drift.
- Archive `metadata.json`, `rule_manifest.jsonl`, and raw generation records for the paper run.
- Record the API access date and provider outside the code when a gateway maps public aliases to internal model revisions.
- A passing official test suite is execution-based evidence for the behavior exercised by HumanEval-X; it is not presented as a formal equivalence proof.
- Rule applicability is screened before scheduling, the benchmark entry point is protected in every prompt, and deviations introduced by a generator remain visible in the compilation and test logs.

## Tests

```bash
python -m unittest discover -s tests -v
```
