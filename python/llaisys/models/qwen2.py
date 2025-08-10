from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..tensor import Tensor
import json
import numpy as np

from pathlib import Path
import safetensors


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        self.weights = {}
        # Load model weights from safetensors files
        # Tempo
        for file in sorted(self.model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="torch", device="cpu")
            for name in data.keys():
                tensor = data.get_tensor(name)
                print(tensor)
                llaisys_tensor = Tensor(
                    list(tensor.shape),
                    dtype=DataType.BF16,
                    device=DeviceType.CPU,
                    device_id=0,
                )
                print(tensor.data_ptr())
                llaisys_tensor.load(tensor.data_ptr())
                print(tensor)
                # llaisys_tensor.debug()
                break
        
        # Load model configuration from config.json
        config_path = self.model_path / "config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Initialize model parameters from configuration
        self.eos_token_id = self.config.get("eos_token_id")
        self.hidden_size = self.config.get("hidden_size")
        self.intermediate_size = self.config.get("intermediate_size")
        self.max_position_embeddings = self.config.get("max_position_embeddings")
        self.num_attention_heads = self.config.get("num_attention_heads")
        self.num_hidden_layers = self.config.get("num_hidden_layers")
        self.num_key_value_heads = self.config.get("num_key_value_heads")
        self.rms_norm_eps = self.config.get("rms_norm_eps")
        self.rope_theta = self.config.get("rope_theta")
        self.torch_dtype = self.config.get("torch_dtype")
        self.vocab_size = self.config.get("vocab_size")

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        embeddings_weight = self.weights["embeddings.word_embeddings.weight"]
        input_embeds = embeddings_weight[input_ids]


        return logits

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 1. 初始化上下文tokens列表
        generated = list(inputs)

        for _ in range(max_new_tokens):
            # 2. 构造模型输入，比如把 generated 转成模型需要的tensor格式
            input_ids = np.array(generated, dtype=np.int64)  # 这里用numpy示范

            # 3. 调用模型前向推理，获得下一个token的logits（[vocab_size]）
            logits = self.forward(input_ids)
            
            # 4. 对 logits 做温度调整
            logits = logits / temperature

            # 5. 应用 Top-k 采样
            if top_k > 0:
                indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
                logits[indices_to_remove] = -float('Inf')

            # 6. 应用 Top-p (nucleus) 采样
            if top_p < 1.0:
                sorted_logits = np.sort(logits)[::-1]
                sorted_indices = np.argsort(logits)[::-1]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
                cutoff = np.searchsorted(cumulative_probs, top_p)
                if cutoff < len(logits):
                    threshold = sorted_logits[cutoff]
                    logits[logits < threshold] = -float('Inf')

            # 7. 转成概率分布
            exp_logits = np.exp(logits - np.max(logits))  # 减max防止溢出
            probs = exp_logits / np.sum(exp_logits)

            # 8. 根据概率采样下一个token
            next_token = np.random.choice(len(probs), p=probs)

            # 9. 追加token
            generated.append(next_token)

            # 10. 遇到eos则终止生成
            if next_token == self.eos_token_id:
                break

        return []
