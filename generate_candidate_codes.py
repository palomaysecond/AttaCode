import argparse
import json
import re
from tqdm import tqdm
from collections import defaultdict
from Model.GPT import GPT
from Model.Gemini import Gemini
from Model.DeepSeek import DeepSeek
from Model.Qwen import Qwen
from utils import read_jsonl, write_jsonl, load_config, get_api_key, get_model_config


def get_model(model_name, api_key=None, model_name_config=None, config=None):

    model_name_lower = model_name.lower()

    if config is None:
        try:
            config = load_config()
        except:
            config = {}

    if api_key is None:
        api_key = get_api_key(model_name, config)

    if model_name_config is None:
        model_config = get_model_config(model_name, config)
        model_name_config = model_config.get('model_name', 'default')

    if model_name_lower == "gpt":
        return GPT(api_key=api_key, model_name=model_name_config)
    elif model_name_lower == "gemini":
        return Gemini(api_key=api_key, model_name=model_name_config)
    elif model_name_lower == "deepseek":
        return DeepSeek(api_key=api_key, model_name=model_name_config)
    elif model_name_lower == "qwen":
        return Qwen(api_key=api_key, model_name=model_name_config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def clean_markdown_code_block(code_str):

    if not code_str or not isinstance(code_str, str):
        return ""

    code_str = code_str.strip()

    if "```" in code_str:
        start_idx = code_str.find("```")
        end_idx = code_str.find("```", start_idx + 3)
        if end_idx != -1:
            first_line_end = code_str.find("\n", start_idx)
            if first_line_end != -1 and first_line_end < end_idx:
                code_content = code_str[first_line_end + 1:end_idx]
            else:
                code_content = code_str[start_idx + 3:end_idx]
            return code_content.strip()

    if "#include" in code_str or "int main" in code_str or ";" in code_str:
        return code_str

    return ""



def extract_java_code(text):

    if not text:
        return ""

    cleaned = clean_markdown_code_block(text)
    if cleaned:
        return cleaned

    java_patterns = [
        r'(public\s+class\s+\w+.*?)(?=\n\n|\Z)',
        r'(class\s+\w+.*?)(?=\n\n|\Z)',
        r'(public\s+\w+.*?\{.*?\})',
        r'(\w+\s+\w+\s*\(.*?\)\s*\{.*?\})'
    ]
    
    for pattern in java_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            return matches[0].strip()

    return text.strip()


def validate_generated_code(code):

    if not code or len(code.strip()) < 10:
        return False, "Code too short"

    if not any(keyword in code for keyword in ['class', 'public', 'private', 'void', 'int', 'String']):
        return False, "No Java keywords found"

    open_braces = code.count('{')
    close_braces = code.count('}')
    if abs(open_braces - close_braces) > 2:  
        return False, "Unmatched braces"
    
    return True, "Valid"


def generate_single_candidate(model, prompt, max_retries=3):

    for attempt in range(max_retries):
        try:
            output = model.generate(prompt)
            cleaned_code = extract_java_code(output)

            is_valid, reason = validate_generated_code(cleaned_code)
            if is_valid:
                return cleaned_code, True
            else:
                print(f"    Attempt {attempt+1} failed validation: {reason}")
                
        except Exception as e:
            print(f"    Attempt {attempt+1} failed with error: {e}")
    
    return "", False


def generate_multiple_candidates_batch(prompt, models, num_candidates=3):

    all_candidates = set()
    successful_generations = 0
    
    for model_name, model in models.items():
        if len(all_candidates) >= num_candidates:
            break
            
        try:

            for i in range(min(3, num_candidates - len(all_candidates) + 1)):
                candidate, success = generate_single_candidate(model, prompt)
                if success and candidate:
                    all_candidates.add(candidate)
                    successful_generations += 1
                    print(f"    Generated candidate {len(all_candidates)} using {model_name}")
                
                if len(all_candidates) >= num_candidates:
                    break
                    
        except Exception as e:
            print(f"    Model {model_name} failed: {e}")
    
    return list(all_candidates)[:num_candidates]


def generate_candidate_codes(prompt_file, output_file, model_names, config_path='config.json'):

    try:
        config = load_config(config_path)
        print(f"Configuration file loaded successfully.")
    except Exception as e:
        print(f"Failed to load configuration file: {e}")
        return

    models = {}
    for model_name in model_names:
        try:
            models[model_name] = get_model(model_name, config=config)
            print(f"{model_name} model initialized successfully.")
        except Exception as e:
            print(f"{model_name} model initialization failed: {e}")
    
    if not models:
        print("No models available")
        return

    data = read_jsonl(prompt_file)
    results = []
    
    success_count = 0
    total_count = len(data)

    for item in tqdm(data, desc="Generating candidate codes"):
        index = item["index"]
        prompt = item["prompt"]
        
        print(f"\n[Processing] Index={index}")
        
        # 生成候选代码
        candidates = generate_multiple_candidates_batch(prompt, models, num_candidates=3)
        
        if candidates:
            success_count += 1
            print(f"Generated {len(candidates)} candidates for index={index}")
        else:
            print(f"Failed to generate for index={index}")
        
        results.append({
            "index": index,
            "candidate_codes": candidates,
            "num_candidates": len(candidates)
        })


    write_jsonl(output_file, results)

    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"\n[Statistics]")
    print(f"Total samples: {total_count}")
    print(f"Successful generations: {success_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Results written to: {output_file}")


def test_model_generation(model_name, test_prompt="Write a simple Java hello world program.", config_path='config.json'):

    try:
        config = load_config(config_path)
        model = get_model(model_name, config=config)
        
        print(f"Testing {model_name} model...")
        candidate, success = generate_single_candidate(model, test_prompt)
        
        if success:
            print(f" {model_name} test successful")
            print(f"Generated code length: {len(candidate)}")
            print(f"First 200 chars: {candidate[:200]}...")
        else:
            print(f" {model_name} test failed")
            
    except Exception as e:
        print(f" {model_name} test error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate candidate code using multiple LLMs")
    parser.add_argument('--prompt_file', type=str, help='Path to augmented_prompts.jsonl')
    parser.add_argument('--output_file', type=str, help='Path to output candidate_codes.jsonl')
    parser.add_argument('--models', nargs='+', default=["gpt", "deepseek", "claude", "qwen"], 
                       help='List of models to use')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--test', type=str, help='Test a specific model')
    args = parser.parse_args()

    if args.test:
        # test
        test_model_generation(args.test, config_path=args.config)
    elif args.prompt_file and args.output_file:
        # generate candidate codes
        generate_candidate_codes(args.prompt_file, args.output_file, args.models, args.config)
    else:
        print("Please provide either --test MODEL_NAME or both --prompt_file and --output_file")