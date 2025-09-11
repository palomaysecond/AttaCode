import os
import random
import numpy as np
import torch
import json

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  
    random.seed(seed)  
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  



def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(file_path, data_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file {config_path} does not exist. Please ensure that config.json is in the project root directory.")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 验证配置文件结构
    if 'api_keys' not in config:
        raise ValueError("The configuration file is missing the 'api_keys' section.")
    if 'model_configs' not in config:
        raise ValueError("The configuration file is missing the 'model_configs' section.")
    
    return config

def get_api_key(model_name, config=None):
    if config is None:
        config = load_config()
    
    model_name_lower = model_name.lower()
    api_key = config['api_keys'].get(model_name_lower)
    
    if not api_key:
        raise ValueError(f"The configuration file is missing the API key for the model '{model_name}'.")
    
    return api_key

def get_model_config(model_name, config=None):
    if config is None:
        config = load_config()
    
    model_name_lower = model_name.lower()
    model_config = config['model_configs'].get(model_name_lower)
    
    if not model_config:
        raise ValueError(f"The configuration file is missing the configuration for the model '{model_name}'.")
    
    return model_config

api_key = "sk-xxx-your-real-key-here"  # 作为后备使用