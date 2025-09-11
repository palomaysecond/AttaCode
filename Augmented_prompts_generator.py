"""
@author: Xiongjiawei
@contact:213223741@seu.edu.cn
@version: 1.0.0
@time: 2025/6/17 14:08
"""
import json
from collections import defaultdict
import random

# Transformation rule descriptions for Java and C/C++
RULE_DESCRIPTIONS = {
    "java": {
        "ForToWhileConversion": "Replace 'for-loop' with equivalent 'while-loop'.",
        "WhileToForRefactoring": "Replace 'while-loop' with equivalent 'for-loop'.",
        "DoWhileToWhileConversion": "Replace 'do-while-loop' with equivalent 'while-loop'.",
        "InlineLoopDeclaration": "Move loop variable declaration into 'for' header.",
        "ExtractLoopDeclaration": "Move loop variable declaration out of 'for' header.",
        "IfElseBranchSwap": "Swap 'if' and 'else' blocks using negated conditions.",
        "ElseIfToNestedIf": "Convert 'else if' into 'else { if }' structure.",
        "NestedIfToElseIf": "Merge 'else { if }' into 'else if' structure.",
        "WrapStatementInBlock": "Add curly braces '{}' to single-statement blocks.",
        "UnwrapRedundantBlock": "Remove redundant '{}' from single-statement blocks.",
        "SplitCompoundCondition": "Split compound conditions into nested if-statements.",
        "AddRedundantStatement": "Insert no-op code like 'x = x;' to maintain semantics.",
        "WrapWithConstantCondition": "Wrap statements in always-true conditional blocks.",
        "ExtractSubexpression": "Extract part of complex expressions into temporary variables.",
        "SwitchToIfElse": "Convert 'switch' statements to equivalent 'if-else' chains.",
        "ReturnViaTempVariable": "Assign return value to a temporary variable before returning.",
        "NegateWithReversedOperator": "Replace comparisons with negated, reversed operators.",
        "ExpandCompoundAssign": "Expand compound assignments into full expressions (e.g., 'x += 1' → 'x = x + 1').",
        "ExpandUnaryOP": "Expand unary increment/decrement operators into explicit assignments.",
        "PromoteIntToLong": "Change 'int' declarations to 'long' where semantics allow.",
        "PromoteFloatToDouble": "Change 'float' declarations to 'double'.",
        "AddUnusedParameter": "Add unused parameters to method signatures.",
        "RefactorOutputAPI": "Split output API calls into separate stream references and method calls.",
        "RenameVariable": "Rename variables consistently without affecting program logic.",
        "RenameClassAndMethod": "Rename class and method identifiers while preserving behavior."
    },
    "cpp": {
        "ForToWhileConversion": "Replace 'for-loop' with equivalent 'while-loop'.",
        "WhileToForRefactoring": "Replace 'while-loop' with equivalent 'for-loop'.",
        "DoWhileToWhileConversion": "Replace 'do-while-loop' with equivalent 'while-loop'.",
        "InlineLoopDeclaration": "Move loop variable declaration into 'for' header.",
        "ExtractLoopDeclaration": "Move loop variable declaration out of 'for' header.",
        "IfElseBranchSwap": "Swap 'if' and 'else' blocks using negated conditions.",
        "ElseIfToNestedIf": "Convert 'else if' into 'else { if }' structure.",
        "NestedIfToElseIf": "Merge 'else { if }' into 'else if' structure.",
        "WrapStatementInBlock": "Add curly braces '{}' to single-statement blocks.",
        "UnwrapRedundantBlock": "Remove redundant '{}' from single-statement blocks.",
        "SplitCompoundCondition": "Split compound conditions into nested if-statements.",
        "AddRedundantStatement": "Insert no-op code like 'x = x;' to maintain semantics.",
        "WrapWithConstantCondition": "Wrap statements in always-true conditional blocks.",
        "ExtractSubexpression": "Extract part of complex expressions into temporary variables.",
        "SwitchToIfElse": "Convert 'switch' statements to equivalent 'if-else' chains.",
        "ReturnViaTempVariable": "Assign return value to a temporary variable before returning.",
        "NegateWithReversedOperator": "Replace comparisons with negated, reversed operators.",
        "ExpandCompoundAssign": "Expand compound assignments into full expressions (e.g., 'x += 1' → 'x = x + 1').",
        "ExpandUnaryOP": "Expand unary increment/decrement operators into explicit assignments.",
        "PromoteIntToLong": "Change 'int' declarations to 'long' where semantics allow.",
        "PromoteFloatToDouble": "Change 'float' declarations to 'double'.",
        "AddUnusedParameter": "Add unused parameters to function signatures.",
        "RefactorOutputAPI": "Split output API calls into separate stream references and method calls.",
        "RenameVariable": "Rename variables consistently without affecting program logic.",
        "RenameClassAndMethod": "Rename class and function identifiers while preserving behavior."
    }
}


PROMPT_TEMPLATES = {
    "java": [
        # Template 0 - Original
        '''You are a powerful AI code transformer specializing in generating adversarial variants of source code that are indistinguishable in behavior but significantly altered in structure. Your mission is to help bypass static vulnerability detectors by rewriting code using multiple semantics-preserving transformations.

Your refactored code must apply all given transformations accurately and thoroughly, modifying every occurrence and minimizing superficial or trivial changes. The goal is to reshape the code structure while strictly preserving functionality and ensuring it compiles without errors.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored Java code, enclosed in a single Java markdown block (` ```java ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 1
        '''You are an expert AI in code restructuring, specializing in generating behavior-preserving but structurally divergent variants of source code. Your task is to apply a set of semantic-preserving transformations to the input code to help evade static vulnerability detection tools.

You must accurately and thoroughly perform all specified transformations, changing every relevant structure while preserving the exact behavior and ensuring the resulting code compiles correctly.

Focus on deep structural rewrites rather than superficial edits. Your transformed code should be as different as possible in form while remaining functionally identical to the original.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored Java code, enclosed in a single Java markdown block (` ```java ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 2
        '''You are a high-performance adversarial code generator. Your job is to apply a series of semantics-preserving transformations to the given source code, producing a variant that maintains identical functionality but appears significantly different to static analysis tools.

Every transformation listed must be fully implemented wherever applicable. Avoid trivial rewrites — aim for maximal structural change without introducing errors or altering program behavior.

Your output must be compilable, behaviorally equivalent, and structurally altered to evade detection by conventional vulnerability scanners.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored Java code, enclosed in a single Java markdown block (` ```java ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
'''
    ],
    "cpp": [
        # Template 0 - Original
        '''You are a powerful AI code transformer specializing in generating adversarial variants of source code that are indistinguishable in behavior but significantly altered in structure. Your mission is to help bypass static vulnerability detectors by rewriting code using multiple semantics-preserving transformations.

Your refactored code must apply all given transformations accurately and thoroughly, modifying every occurrence and minimizing superficial or trivial changes. The goal is to reshape the code structure while strictly preserving functionality and ensuring it compiles without errors.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored C++ code, enclosed in a single C++ markdown block (` ```cpp ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 1
        '''You are an expert AI in code restructuring, specializing in generating behavior-preserving but structurally divergent variants of source code. Your task is to apply a set of semantic-preserving transformations to the input code to help evade static vulnerability detection tools.

You must accurately and thoroughly perform all specified transformations, changing every relevant structure while preserving the exact behavior and ensuring the resulting code compiles correctly.

Focus on deep structural rewrites rather than superficial edits. Your transformed code should be as different as possible in form while remaining functionally identical to the original.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored C++ code, enclosed in a single C++ markdown block (` ```cpp ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
''',
        # Template 2
        '''You are a high-performance adversarial code generator. Your job is to apply a series of semantics-preserving transformations to the given source code, producing a variant that maintains identical functionality but appears significantly different to static analysis tools.

Every transformation listed must be fully implemented wherever applicable. Avoid trivial rewrites — aim for maximal structural change without introducing errors or altering program behavior.

Your output must be compilable, behaviorally equivalent, and structurally altered to evade detection by conventional vulnerability scanners.

### INSTRUCTIONS ###
1. Carefully analyze the `### ORIGINAL SOURCE CODE ###` provided below.
2. Apply ALL transformations listed in the `### REQUIRED TRANSFORMATIONS ###` section. Each transformation is chosen because it has a high impact on the detection model's analysis.
3. The resulting code MUST be semantically equivalent to the original. It must compile and have the exact same functionality. Do not add or remove any features.
4. Your final output MUST BE ONLY the complete, refactored C++ code, enclosed in a single C++ markdown block (` ```cpp ... ``` `). Do not include ANY text, explanation, or commentary before or after the code block.

### ORIGINAL SOURCE CODE ###
{code_str}

### REQUIRED TRANSFORMATIONS ###
{transformation_block}
'''
    ]
}




def normalize_language(language: str) -> str:

    language = language.lower()
    if language in {"c", "cpp", "c++"}:
        return "cpp"
    if language == "java":
        return "java"
    raise ValueError(f"Unsupported language: {language}")


def format_transformation_rule(rule_name, loc_desc, language):

    lang_key = normalize_language(language)
    description = RULE_DESCRIPTIONS[lang_key].get(rule_name, "No description provided.")
    return f"- {rule_name} at {loc_desc}: {description}"


def generate_augmented_prompt(code_str, transformations, language="java", template_index=0):

    # Generate a single augmented prompt given code and transformations.

    lang_key = normalize_language(language)
    transformation_lines = [
        format_transformation_rule(t["transformation_name"], t.get("location", "unspecified"), lang_key)
        for t in transformations
    ]
    transformation_block = "\n".join(transformation_lines)

    header_template = PROMPT_TEMPLATES[lang_key][template_index % len(PROMPT_TEMPLATES[lang_key])]
    prompt = header_template.format(
        transformation_block=transformation_block,
        code_str=code_str.strip()
    )
    return prompt


def generate_diverse_prompts(code_str, transformations, language="java", num_variants=3):

    return [
        generate_augmented_prompt(code_str, transformations, language, template_index=i)
        for i in range(num_variants)
    ]


def generate_augmented_prompts_from_files(code_file, transformation_file, output_file, language="java"):

    lang_key = normalize_language(language)

    def read_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f]

    code_map = {item["idx"]: item for item in read_jsonl(code_file)}
    transformations_map = {item["index"]: item["patterns"] for item in read_jsonl(transformation_file)}

    results = []
    for idx, transformations in transformations_map.items():
        code_entry = code_map.get(idx)
        if not code_entry:
            continue

        try:
            prompt = generate_augmented_prompt(
                code_entry["func"], transformations, language=lang_key
            )
            results.append({"idx": idx, "prompt": prompt})
        except Exception as e:
            print(f"Failed to generate prompt for idx {idx}: {e}")

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in results:
            out_f.write(json.dumps(item, ensure_ascii=False) + '\n')

# test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_file', type=str, required=True)
    parser.add_argument('--score_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--language', type=str, choices=["java", "cpp"], default="java")
    args = parser.parse_args()

    prompts = generate_augmented_prompts_from_files(
        args.code_file, args.score_file, topk=args.topk, language=args.language
    )
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f" {len(prompts)} prompts written to {args.output_file}")
    if prompts:
        avg_complexity = sum(p["complexity_score"] for p in prompts) / len(prompts)
        avg_transformations = sum(p["num_transformations"] for p in prompts) / len(prompts)
        print(f"[Info] Avg complexity: {avg_complexity:.2f}, Avg transformations: {avg_transformations:.2f}")