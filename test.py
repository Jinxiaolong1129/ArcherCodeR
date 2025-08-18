#!/usr/bin/env python3
"""
测试 DeepSeek-R1-Distill-Qwen-1.5B 模型访问
"""

from transformers import AutoTokenizer, AutoModel

def main():
    """主函数：测试模型tokenizer的加载"""
    try:
        # 从预训练模型加载tokenizer
        model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        # 成功提示
        print('Model access successful!')
        print(f'Successfully loaded tokenizer for: {model_name}')
        
    except Exception as e:
        print(f'Error loading model: {e}')

if __name__ == "__main__":
    main()