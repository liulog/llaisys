from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys.models import load_qwen2, LlaisysQwen2Meta
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import llaisysTensor_t
import json
import ctypes
from modelscope.hub.snapshot_download import snapshot_download

from pathlib import Path
import safetensors
import torch    # Used for transferring weights from bf16 to fp32

load_qwen2(LIB_LLAISYS)

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # Currently, only CPU is supported
        # assert(device == DeviceType.CPU), "Only CPU is supported now."

        if model_path is not None and Path(model_path).exists():
            self.model_path = Path(model_path)
            print(f"Using local model path: {self.model_path}", flush=True)
        else:
            print("Model path not provided or does not exist. Downloading from ModelScope...", flush=True)
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.model_path = Path(snapshot_download(model_id))
            print(f"Model downloaded to: {self.model_path}", flush=True)
        
        # Check if related model files exist
        config_file = self.model_path / "config.json"
        weights_file = self.model_path / "model.safetensors"
        if not config_file.exists():
            raise FileNotFoundError(f"{config_file} not found!")
        if not weights_file.exists():
            raise FileNotFoundError(f"{weights_file} not found!")

        # Initialize model...
        self.device = device
        self.device_id = 0  # means CPU
        # Load model configuration from config.json
        config_path = self.model_path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Initialize model parameters from configuration
        self.eos_token_id = config.get("eos_token_id")
        self.hidden_size = config.get("hidden_size")
        self.intermediate_size = config.get("intermediate_size")
        self.max_position_embeddings = config.get("max_position_embeddings")
        self.num_attention_heads = config.get("num_attention_heads")
        self.num_hidden_layers = config.get("num_hidden_layers")
        self.num_key_value_heads = config.get("num_key_value_heads")
        self.rms_norm_eps = config.get("rms_norm_eps")
        self.rope_theta = config.get("rope_theta")
        self.torch_dtype = config.get("torch_dtype")
        self.vocab_size = config.get("vocab_size")
        self.per_head_dim = self.hidden_size // self.num_attention_heads
        self.per_kvhead_dim = self.per_head_dim # for Qwen2, dv = d

        # Currently, only bfloat16 is supported
        assert self.torch_dtype == "bfloat16", "Only bfloat16 is supported currently."
        # self.data_type = DataType.BF16

        if self.device == DeviceType.CPU:
            print("⚙️ CPU detected: forcing torch_dtype=float32 for better performance.", flush=True)
            self.data_type = DataType.F32  # CPU uses float32 for better performance
        else:
            assert config.get("torch_dtype") == "bfloat16", "Only bfloat16 is supported currently."
            self.data_type = DataType.BF16

        meta = LlaisysQwen2Meta(
            dtype=self.data_type,
            nlayer=self.num_hidden_layers,
            hs=self.hidden_size,
            nh=self.num_attention_heads,
            nkvh=self.num_key_value_heads,
            dh=self.per_head_dim,
            di=self.intermediate_size,
            maxseq=self.max_position_embeddings,
            voc=self.vocab_size,
            epsilon=self.rms_norm_eps,
            theta=self.rope_theta,
            end_token=self.eos_token_id
        )

        device_ids = (ctypes.c_int * 1)(0)

        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            ctypes.c_int(device),
            device_ids,
            ctypes.c_int(1)
        )

        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")

        # Get native weights struct pointer
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not weights:
            raise RuntimeError("Failed to get Qwen2 weights.")

        print("📦 Qwen2: Loading weights...", flush=True)

        for file in sorted(self.model_path.glob("*.safetensors")):
            print(f"📂 Loading file: {file.name}", flush=True)
            data = safetensors.safe_open(file, framework="torch", device="cpu")

            def maybe_cast_tensor(tensor):
                if self.device == DeviceType.CPU:
                    return tensor.to(torch.float32).contiguous()
                return tensor

            # Load embedding and output layers
            print("🔄 Qwen2: Loading Input/Output Embedding layer weights", flush=True)
            for name, field in [
                ("model.embed_tokens.weight", "in_embed"),
                ("lm_head.weight", "out_embed"),
                ("model.norm.weight", "out_norm_w")
            ]:
                tensor = maybe_cast_tensor(data.get_tensor(name))
                LIB_LLAISYS.tensorLoad(getattr(weights.contents, field), tensor.data_ptr())

            def load_layer_array(field_name, base_name):
                arr_ptr = getattr(weights.contents, field_name)
                arr_type = llaisysTensor_t * self.num_hidden_layers
                arr = ctypes.cast(arr_ptr, ctypes.POINTER(arr_type)).contents

                for i in range(self.num_hidden_layers):
                    tensor_name = f"model.layers.{i}.{base_name}"
                    tensor = maybe_cast_tensor(data.get_tensor(tensor_name))
                    LIB_LLAISYS.tensorLoad(arr[i], tensor.data_ptr())

            print("🔄 Qwen2: Loading Self-Attention layer weights", flush=True)
            load_layer_array("attn_norm_w", "input_layernorm.weight")
            load_layer_array("attn_q_w", "self_attn.q_proj.weight")
            load_layer_array("attn_q_b", "self_attn.q_proj.bias")
            load_layer_array("attn_k_w", "self_attn.k_proj.weight")
            load_layer_array("attn_k_b", "self_attn.k_proj.bias")
            load_layer_array("attn_v_w", "self_attn.v_proj.weight")
            load_layer_array("attn_v_b", "self_attn.v_proj.bias")
            load_layer_array("attn_o_w", "self_attn.o_proj.weight")

            print("🔄 Qwen2: Loading MLP layer weights", flush=True)
            load_layer_array("mlp_norm_w", "post_attention_layernorm.weight")
            load_layer_array("mlp_gate_w", "mlp.gate_proj.weight")
            load_layer_array("mlp_up_w", "mlp.up_proj.weight")
            load_layer_array("mlp_down_w", "mlp.down_proj.weight")

        print("🎉 Qwen2: All weights loaded successfully!", flush=True)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        use_cache=True,
    ):
        """
        Input:
            inputs: List of input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            temperature: Sampling temperature
        Output:
            generated: List of token IDs including the input and generated tokens
        """    
        # 1. Init generated tokens list
        generated = list(inputs)

        # 2. Create KV cache tensors
        if use_cache:
            kcache_array = (llaisysTensor_t * self.num_hidden_layers)()
            vcache_array = (llaisysTensor_t * self.num_hidden_layers)()

            for i in range(self.num_hidden_layers):
                shape_arr = (ctypes.c_size_t * 3)(
                    max_new_tokens + len(generated),  # sequence length
                    self.num_key_value_heads,         # num heads
                    self.per_kvhead_dim               # per-head dimension
                )
                kcache_array[i] = LIB_LLAISYS.tensorCreate(shape_arr, 3, self.data_type, self.device, self.device_id)
                vcache_array[i] = LIB_LLAISYS.tensorCreate(shape_arr, 3, self.data_type, self.device, self.device_id)
        else:
            kcache_array = ctypes.POINTER(llaisysTensor_t)()
            vcache_array = ctypes.POINTER(llaisysTensor_t)()

        # 3. Prefill
        ntokens = len(generated)
        TokenArrayType = ctypes.c_int64 * ntokens
        input_token_array = TokenArrayType(*generated)

        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            input_token_array,
            ctypes.c_size_t(ntokens),
            kcache_array,
            vcache_array,
            ctypes.c_size_t(0)   # past_len
        )
        generated.append(next_token)

        for _ in range(max_new_tokens - 1):
            print(generated, flush=True)
            if next_token == self.eos_token_id:
                break
            if use_cache:
                TokenArrayType = ctypes.c_int64 * 1
                past_len = len(generated) - 1   # past_len is calculated index.
                input_token_array = TokenArrayType(next_token)
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model,
                    input_token_array,
                    ctypes.c_size_t(1),
                    kcache_array,
                    vcache_array,
                    ctypes.c_size_t(past_len)   # [past_len, past_len + 1]
                )
            else:
                ntokens = len(generated)
                TokenArrayType = ctypes.c_int64 * ntokens
                input_token_array = TokenArrayType(*generated)
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model,
                    input_token_array,
                    ctypes.c_size_t(ntokens),
                    kcache_array, # nullptr
                    vcache_array, # nullptr
                    ctypes.c_size_t(0)      # [0, 0 + ntokens]
                )
            generated.append(next_token)

        return generated

    def generate_stream(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        use_cache=True,
    ):
        """
        Used for streaming generation.
        """
        # 1. Init generated tokens list
        generated = list(inputs)

        # 2. Create KV cache tensors
        if use_cache:
            kcache_array = (llaisysTensor_t * self.num_hidden_layers)()
            vcache_array = (llaisysTensor_t * self.num_hidden_layers)()

            for i in range(self.num_hidden_layers):
                shape_arr = (ctypes.c_size_t * 3)(
                    max_new_tokens + len(generated),  # sequence length
                    self.num_key_value_heads,         # num heads
                    self.per_kvhead_dim               # per-head dimension
                )
                kcache_array[i] = LIB_LLAISYS.tensorCreate(shape_arr, 3, self.data_type, self.device, self.device_id)
                vcache_array[i] = LIB_LLAISYS.tensorCreate(shape_arr, 3, self.data_type, self.device, self.device_id)
        else:
            kcache_array = ctypes.POINTER(llaisysTensor_t)()
            vcache_array = ctypes.POINTER(llaisysTensor_t)()

        # 3. Prefill
        ntokens = len(generated)
        TokenArrayType = ctypes.c_int64 * ntokens
        input_token_array = TokenArrayType(*generated)

        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            input_token_array,
            ctypes.c_size_t(ntokens),
            kcache_array,
            vcache_array,
            ctypes.c_size_t(0)   # past_len
        )
        generated.append(next_token)
        yield f"{next_token}"  # 每个 token 后添加换行符

        for _ in range(max_new_tokens - 1):
            # print(generated, flush=True)
            if next_token == self.eos_token_id:
                break
            if use_cache:
                TokenArrayType = ctypes.c_int64 * 1
                past_len = len(generated) - 1   # past_len is calculated index.
                input_token_array = TokenArrayType(next_token)
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model,
                    input_token_array,
                    ctypes.c_size_t(1),
                    kcache_array,
                    vcache_array,
                    ctypes.c_size_t(past_len)   # [past_len, past_len + 1]
                )
            else:
                ntokens = len(generated)
                TokenArrayType = ctypes.c_int64 * ntokens
                input_token_array = TokenArrayType(*generated)
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self.model,
                    input_token_array,
                    ctypes.c_size_t(ntokens),
                    kcache_array, # nullptr
                    vcache_array, # nullptr
                    ctypes.c_size_t(0)
                )
            generated.append(next_token)
            print(next_token, flush=True)
            yield f"{next_token}"  # 每个 token 后添加换行符

[151646, 151646, 151644, 4340, 311, 3960, 23071, 30, 151645, 151648, 198, 32313, 11, 773, 358, 1366, 311, 3960, 33789, 11, 714, 358, 2776, 537, 2167, 2704, 1380, 311, 1191, 13, 358, 3003, 6617, 264, 2696, 911, 432, 11, 714, 358, 1513, 944, 1414, 1753, 911, 432, 13, 6771, 752, 1430, 311, 7071, 419, 700, 3019, 553]
[151646, 151646, 151644, 4340, 311, 3960, 23071, 30, 151645, 151648, 198, 32313, 11, 773, 358, 1366, 311, 3960, 33789, 15473, 11, 714, 358, 2776, 537, 2167, 2704, 1380, 311, 1191, 13, 358, 3003, 6617, 1251, 3061, 911, 432, 1660, 264, 2699, 2155, 504, 1128, 358, 2776, 1483, 311, 11, 714, 358, 2776, 12035, 311, 29863, 1119, 432, 13, 6771, 752, 1744, 1526, 847, 3381, 1882, 438, 4325, 879, 594, 1101, 5916, 700, 382, 5338, 11, 358, 7942, 358, 1184, 311, 3535, 1128, 33789, 374, 13, 5542, 1128, 358, 1414, 11, 432, 594, 264, 5942, 15473, 4128, 11, 1290, 30, 2055, 432, 594, 911, 11589, 3347, 11591, 7525, 11, 1075, 4938, 323, 78010, 13, 2938, 3363, 432, 594, 5020, 2155, 504, 15459, 1075, 13027, 476, 12914, 11, 892, 525, 803, 1550, 11591, 13, 1988, 358, 1744, 279, 16170, 525, 4428]