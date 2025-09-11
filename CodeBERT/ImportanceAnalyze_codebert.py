import numpy as np
import inspect
import json
import argparse
from GetAST import generateASt
from tqdm import tqdm

def create_token_byte_map(source_code: str, token_list: list, scores: np.ndarray):

    token_byte_map = []
    source_bytes = source_code.encode('utf-8')
    current_byte_pos = 0

    for i, token in enumerate(token_list):
        raw_token = token.lstrip('Ġ_')  # 处理CodeBERT token前缀
        if not raw_token or raw_token in {'<s>', '</s>', '<cls>', '<sep>', '<pad>'}:
            continue

        score = float(scores[i]) if i < len(scores) else 0.0

        try:
            token_bytes = raw_token.encode('utf-8')
            byte_start = source_bytes.find(token_bytes, current_byte_pos)
            if byte_start != -1:
                byte_end = byte_start + len(token_bytes)
                token_byte_map.append({
                    "token": raw_token,
                    "start_byte": byte_start,
                    "end_byte": byte_end,
                    "score": score
                })
                current_byte_pos = byte_end
        except Exception:
            pass

    return token_byte_map


def calculate_score_for_range(token_byte_map: list, pattern_start_byte: int, pattern_end_byte: int):

    total_score = sum(
        token_info['score']
        for token_info in token_byte_map
        if pattern_start_byte <= token_info['start_byte'] < token_info['end_byte'] <= pattern_end_byte
    )
    return total_score


def load_preprocessed_data(jsonl_file, tokens_file, scores_file):

    with open(jsonl_file, 'r') as f:
        source_codes = [json.loads(line) for line in f]

    with open(tokens_file, 'r') as f:
        tokens_by_idx = json.load(f)

    scores_by_idx = np.load(scores_file, allow_pickle=True)

    return source_codes, tokens_by_idx, scores_by_idx


def walk_tree(node):

    yield node
    for child in getattr(node, 'children', []):
        yield from walk_tree(child)


def load_assess_functions(language: str):

    if language == 'java':
        import Java_FindTransformations as tf_module
    elif language == 'c':
        import C_Cpp_FindTransformations as tf_module
    else:
        raise ValueError(f"Unsupported language: {language}")

    assess_functions = [
        (name, func)
        for name, func in inspect.getmembers(tf_module, inspect.isfunction)
        if name.startswith('Assess')
    ]

    if not assess_functions:
        raise RuntimeError(f"No Assess functions found in module for language {language}")

    print(f"[INFO] Loaded {len(assess_functions)} transformations for {language}")
    return assess_functions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_codes_path", default="dataset/Devignn/test1.jsonl", type=str,
                        help="Path to test data JSONL file.")
    parser.add_argument("--tokens_list_path", default="GraphCodeBERT/tokens.json", type=str,
                        help="Path to tokens.json file.")
    parser.add_argument("--tokens_scores_path", default="GraphCodeBERT/token_grad_norms.npz", type=str,
                        help="Path to token_grad_norms.npz file.")
    parser.add_argument("--language", default="java", choices=["java", "c"],
                        help="Programming language of the source code.")
    parser.add_argument("--top_n", default=10, type=int,
                        help="Number of top transformations to save per sample.")
    parser.add_argument("--saved_filename", default="./transformation_importance_scores1.json", type=str,
                        help="Output JSON file path.")
    args = parser.parse_args()

    source_codes, tokens_by_idx, scores_by_idx = load_preprocessed_data(
        args.source_codes_path,
        args.tokens_list_path,
        args.tokens_scores_path
    )

    assess_functions = load_assess_functions(args.language)

    all_samples_results = []

    for source_code_info in tqdm(source_codes, desc="Analyzing Transformations"):
        sample_idx = source_code_info.get('idx')
        if sample_idx is None:
            continue

        str_sample_idx = str(sample_idx)
        if str_sample_idx not in tokens_by_idx or str_sample_idx not in scores_by_idx:
            print(f"[WARN] Data missing for idx {sample_idx}, skipping.")
            continue

        current_code = source_code_info['func']
        current_tokens = tokens_by_idx[str_sample_idx]
        current_scores = scores_by_idx[str_sample_idx]

        if len(current_tokens) >= 512:
            print(f"[INFO] idx:{sample_idx} token length >=512, skipping.")
            continue

        token_map = create_token_byte_map(current_code, current_tokens, current_scores)
        root_node = generateASt(current_code, args.language)

        if not root_node:
            print(f"[WARN] Failed to parse AST for idx:{sample_idx}, skipping.")
            continue

        sample_patterns = {
            "index": sample_idx,
            "target": source_code_info.get('target', -1),
            "patterns": []
        }

        for node in walk_tree(root_node):
            for name, func in assess_functions:
                transformation_info = func(node, current_code)
                if transformation_info:
                    start_byte, end_byte = transformation_info[1], transformation_info[2]
                    pattern_score = calculate_score_for_range(token_map, start_byte, end_byte)

                    sample_patterns["patterns"].append({
                        "transformation_name": name,
                        "score": pattern_score,
                        "start_point": transformation_info[3],
                        "end_point": transformation_info[4],
                        "code_snippet": current_code[start_byte:end_byte]
                    })

        if sample_patterns["patterns"]:
            sample_patterns["patterns"].sort(key=lambda x: x['score'], reverse=True)
            sample_patterns["patterns"] = sample_patterns["patterns"][:args.top_n]
            all_samples_results.append(sample_patterns)

    with open(args.saved_filename, 'w', encoding='utf-8') as f:
        json.dump(all_samples_results, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Completed analysis. Saved {len(all_samples_results)} samples to {args.saved_filename}")


if __name__ == '__main__':
    main()
