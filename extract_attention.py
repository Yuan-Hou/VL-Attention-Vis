"""
Qwen3-VL Attention 提取工具

用于从 Qwen3-VL 模型中提取输入图像和文本的 attention 权重，
并将结果保存为 JSON 格式，供可视化使用。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import base64
from io import BytesIO
import numpy as np

class AttentionExtractor:
    """提取 Qwen3-VL 模型的 attention 权重"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        image_start_id: int = 151652,
        image_end_id: int = 151653,
        image_pad_id: int = 151655,
        use_pad_mode: bool = False
    ):
        """
        初始化模型和分词器
        
        Args:
            model_name: 模型名称或路径
            image_start_id: 图像起始token ID（默认: 151652）
            image_end_id: 图像结束token ID（默认: 151653）
            image_pad_id: 图像填充token ID（默认: 151655）
            use_pad_mode: 是否使用pad模式识别图像token（默认: False，使用start+end模式）
        """
        print(f"正在加载模型: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载分词器和模型
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            output_attentions=True  # 启用 attention 输出
        )
        
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("模型加载完成")
        
        # 图像token识别参数
        self.image_start_id = image_start_id
        self.image_end_id = image_end_id
        self.image_pad_id = image_pad_id
        self.use_pad_mode = use_pad_mode
    
    def extract_attention(
        self, 
        image_path: str, 
        text_prompt: str
    ) -> Dict[str, Any]:
        """
        提取给定图像和文本的 attention 权重
        
        Args:
            image_path: 图像文件路径
            text_prompt: 文本提示
            
        Returns:
            包含输入、输出和 attention 权重的字典
        """
        print(f"处理图像: {image_path}")
        print(f"文本输入: {text_prompt}")
        
        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 图像编码为 base64 字符串
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8").strip()
        print(img_str[:30] + "...")  # 打印部分编码以确认
        
        # 准备输入
        # TODO: 根据 Qwen3-VL 的实际 API 调整输入格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{img_str}"},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        # 应用聊天模板
        inputs = self.processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(self.device)
        
        text = self.processor.decode(
            inputs.input_ids[0],
            skip_special_tokens=True
        )
        
        # 前向传播，获取输出和 attention
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                output_attentions=True,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
        
        image_token_h, image_token_w = inputs.image_grid_thw[0].cpu().tolist()[1:]
        image_token_h, image_token_w = int(image_token_h / 2), int(image_token_w / 2)
        print(f"图像 token 大小: {image_token_h} x {image_token_w}")

        # 提取结果
        all_ids = outputs.sequences
        attentions = outputs.attentions  # tuple of tuples: (layers, heads, seq_len, seq_len)
        
        # 解码输出
        output_ids = all_ids[0][inputs.input_ids.shape[1]:]
        output_text = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )
        
        # 获取输入 tokens
        input_tokens = self.processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # 获取输出 tokens
        all_tokens = self.processor.tokenizer.convert_ids_to_tokens(all_ids[0])
        
        print(f"生成完成，输出长度: {len(all_tokens)} tokens")
        
        # 识别图像token范围
        all_token_ids = all_ids[0].cpu().tolist()
        image_token_range = self._find_image_token_range(all_token_ids)
        
        # 识别输出token范围（输入之后的就是输出）
        input_length = len(inputs.input_ids[0])
        output_token_range = [input_length, len(all_token_ids) - 1] if len(all_token_ids) > input_length else None
        
        # 构建token映射信息
        token_mappings = self._build_token_mappings(
            all_tokens=all_tokens,
            input_text=text,
            input_tokens=input_tokens,
            input_length=input_length,
        )
        
        # 整理 attention 数据
        layer_count = len(attentions[0])
        head_count = attentions[0][0].shape[1]
        attention_data = []
        for layer_idx in range(layer_count):
            layer_attentions = []
            for head_idx in range(head_count):
                layer_attentions.append(attentions[0][layer_idx][0, head_idx, -len(output_ids):, image_token_range[0]:image_token_range[1]+1].cpu().tolist())
            attention_data.append(layer_attentions)
        attention_data = np.array(attention_data)
        # 转置为 (output_len, layers, heads, image_tokens)
        attention_data = attention_data.transpose(2, 0, 1, 3)
        print(f"原始 attention 数据形状: {attention_data.shape}")
        # 变形为 (output_len, layers, heads, image_h, image_w)
        attention_data = attention_data.reshape(
            attention_data.shape[0],
            attention_data.shape[1],
            attention_data.shape[2],
            image_token_h,
            image_token_w
        )
        # 对heads和layers求平均
        attention_data_average = attention_data.mean(axis=2).mean(axis=1)  # 形状 (output_len, image_h, image_w)
        attention_data_min = attention_data.min()
        attention_data_max = attention_data.max()
        normalized_attention_data = (attention_data - attention_data_min) / (attention_data_max - attention_data_min + 1e-12) * 100.0
        normalized_attention_data = normalized_attention_data.astype(np.int8).tolist()
        normalized_attention_data_average = (attention_data_average - attention_data_min) / (attention_data_max - attention_data_min + 1e-12) * 100.0

        tokenwize_attention_data_average = normalized_attention_data_average.mean(axis=(1,2))

        normalized_attention_data_average = normalized_attention_data_average.astype(np.int8).tolist()
        
        for m in token_mappings:
            if m['type'] == 'output':
                idx = m['output_token_idx']
                m['average_attention'] = tokenwize_attention_data_average[idx].item()




        print(f"output length: {len(output_ids)} tokens, total length: {len(all_token_ids)} tokens;\nattention map shape: {len(attentions)} {len(attentions[0])} {attentions[0][0].shape}")
        result = {
            "metadata": {
                "image_path": image_path,
                "image_size": list(image.size),
                "image_base64": f"data:image/png;base64,{img_str}",
                "text_prompt": text_prompt,
                "model": self.model.config.name_or_path,
                "device": str(self.device),
                "image_token_range": image_token_range,
                "output_token_range": output_token_range,
                "total_tokens": len(all_token_ids),
            },
            "input": {
                "text": text,
                # "tokens": input_tokens,
                # "token_ids": inputs.input_ids[0].cpu().tolist(),
            },
            "output": {
                "text": output_text,
                # "tokens": all_tokens,
                # "token_ids": all_ids[0].cpu().tolist(),
            },
            "token_mappings": token_mappings,
            "attention_data": {
                "layers": layer_count,
                "heads": head_count,
                "image_token_size": [image_token_h, image_token_w],
                "output_token_size": len(output_ids),
                "min": attention_data_min,
                "max": attention_data_max,
                "data": normalized_attention_data,  # 量化到 0-100 的整数
                "average_data": normalized_attention_data_average  # 量化到 0-100 的整数，heads和layers平均
            } 
        }
        
        return result
    
    def _find_image_token_range(self, token_ids: List[int]) -> Optional[List[int]]:
        """
        在token序列中查找图像token的范围
        
        Args:
            token_ids: token ID序列
            
        Returns:
            图像token的范围 [start_idx, end_idx]，如果未找到则返回None
        """
        if self.use_pad_mode:
            # 使用pad模式：查找所有pad token
            pad_indices = [i for i, tid in enumerate(token_ids) if tid == self.image_pad_id]
            if pad_indices:
                return [pad_indices[0], pad_indices[-1]]
        else:
            # 使用start+end模式：查找start和end之间的范围
            try:
                start_idx = token_ids.index(self.image_start_id)
                # 从start之后查找end
                end_idx = token_ids.index(self.image_end_id, start_idx + 1)
                return [start_idx+1, end_idx-1]  # 不包括start和end本身
            except ValueError:
                pass
        
        print("警告: 未找到图像token")
        return None
    
    def _build_token_mappings(self, 
                             all_tokens: List[str], 
                             input_text: str,
                             input_tokens: List[str],
                             input_length: int) -> List[Dict[str, Any]]:
        """
        构建每个token的映射信息
        
        Args:
            all_tokens: 所有token列表（包含输入和输出）
            input_text: 输入文本
            input_tokens: 输入token列表
            input_length: 输入长度
            generated_text: 生成的文本
            
        Returns:
            每个token的映射信息列表
        """
        mappings = []
        
        # 解码每个token以获取对应文本
        input_token_ids = self.processor.tokenizer.convert_tokens_to_ids(input_tokens)
        
        # 处理输入tokens
        current_pos = 0
        for idx in range(input_length):
            token = all_tokens[idx]
            token_id = input_token_ids[idx] if idx < len(input_token_ids) else -1
            
            # 尝试找到token在文本中的位置
            # 这是一个简化的实现，实际可能需要更复杂的对齐逻辑
            token_text = self.processor.tokenizer.convert_tokens_to_string([token])
            
            # 在输入文本中查找
            char_start = input_text.find(token_text, current_pos) if token_text.strip() else current_pos
            char_end = char_start + len(token_text) if char_start >= 0 else current_pos
            
            if char_start >= 0:
                current_pos = char_end
            
            # if char_start >= 0:
            #     mappings.append({
            #         "token_idx": idx,
            #         "token": token,
            #         "token_id": token_id,
            #         "type": "input",
            #         "char_range": [char_start, char_end] if char_start >= 0 else None,
            #         "text": token_text
            #     })
        
        # 处理输出tokens（生成的新tokens）
        output_tokens = all_tokens[input_length:]
        output_token_ids = self.processor.tokenizer.convert_tokens_to_ids(output_tokens)
        
        # 解码输出部分以获取准确的文本
        output_text = self.processor.tokenizer.decode(
            output_token_ids,
            skip_special_tokens=False
        )
        
        current_pos = 0
        for local_idx, token in enumerate(output_tokens):
            idx = input_length + local_idx
            token_id = output_token_ids[local_idx] if local_idx < len(output_token_ids) else -1
            
            token_text = self.processor.tokenizer.convert_tokens_to_string([token])
            
            # 在输出文本中查找
            char_start = output_text.find(token_text, current_pos) if token_text.strip() else current_pos
            char_end = char_start + len(token_text) if char_start >= 0 else current_pos
            
            if char_start >= 0:
                current_pos = char_end
            
            mappings.append({
                "token_idx": idx,
                "output_token_idx": local_idx,
                "type": "output",
                "char_range": [char_start, char_end] if char_start >= 0 else None,
                "text": token_text
            })
        
        return mappings
        
        return processed
    
    def save_to_json(self, data: Dict[str, Any], output_path: str):
        """
        保存数据到 JSON 文件，attention数据分离存储为npz格式
        
        Args:
            data: 要保存的数据
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 提取attention数据
        attention_data = data.pop('attention_data', None)
        
        # 先保存attention数据为npz格式
        if attention_data:
            npz_path = output_path.with_suffix('.npz')
            attention_array = np.array(attention_data['data'], dtype=np.int8)
            
            np.savez_compressed(
                npz_path,
                attention=attention_array,
                layers=attention_data['layers'],
                heads=attention_data['heads'],
                image_token_size=attention_data['image_token_size'],
                output_token_size=attention_data['output_token_size'],
                min_val=attention_data['min'],
                max_val=attention_data['max'],
                average_data=np.array(attention_data['average_data'], dtype=np.int8)
            )
            
            npz_size = npz_path.stat().st_size / 1024 / 1024
            print(f"Attention数据已保存到: {npz_path}")
            print(f"NPZ文件大小: {npz_size:.2f} MB")
            
            # 在JSON中记录npz文件路径
            data['attention_file'] = npz_path.name
        
        # 保存JSON元数据（包含attention_file引用）
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        json_size = output_path.stat().st_size / 1024 / 1024
        print(f"元数据已保存到: {output_path}")
        print(f"JSON文件大小: {json_size:.2f} MB")
        
        if attention_data:
            print(f"总大小: {json_size + npz_size:.2f} MB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从 Qwen3-VL 模型中提取 attention 权重"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        required=True,
        help="输入图像路径"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        required=True,
        help="文本提示"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/attention_output.json",
        help="输出 JSON 文件路径（默认: data/attention_output.json）"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="模型名称或路径（默认: Qwen/Qwen3-VL-4B-Instruct）"
    )
    parser.add_argument(
        "--image-start-id",
        type=int,
        default=151652,
        help="图像起始token ID（默认: 151652）"
    )
    parser.add_argument(
        "--image-end-id",
        type=int,
        default=151653,
        help="图像结束token ID（默认: 151653）"
    )
    parser.add_argument(
        "--image-pad-id",
        type=int,
        default=151655,
        help="图像填充token ID（默认: 151655）"
    )
    parser.add_argument(
        "--use-pad-mode",
        action="store_true",
        help="使用pad模式识别图像token（默认使用start+end模式）"
    )
    
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    # 创建提取器
    extractor = AttentionExtractor(
        model_name=args.model,
        image_start_id=args.image_start_id,
        image_end_id=args.image_end_id,
        image_pad_id=args.image_pad_id,
        use_pad_mode=args.use_pad_mode
    )
    
    # 提取 attention
    result = extractor.extract_attention(
        image_path=args.image,
        text_prompt=args.text
    )
    
    # 保存结果
    extractor.save_to_json(result, args.output)
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
