from transformers import AutoTokenizer
import os
import llaisys

def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    else:
        raise ValueError(f"Unsupported device name: {device_name}")

def device_name(llaisys_device: llaisys.DeviceType):
    if llaisys_device == llaisys.DeviceType.CPU:
        return "cpu"
    elif llaisys_device == llaisys.DeviceType.NVIDIA:
        return "nvidia"
    else:
        raise ValueError(f"Unsupported llaisys device: {llaisys_device}")

def load_hf_tokenizer(model_path=None, device_name="cpu"):
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer

def load_llaisys_model(model_path, device_name="cpu"):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model

def llaisys_infer(conversation, tokenizer, model, max_new_tokens=1024, top_p=0.8, top_k=50, temperature=0.8):
    input_content = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    print("Input content:", input_content)
    inputs = tokenizer.encode(input_content)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        use_cache=True,
    )
    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)
    
def llaisys_infer_stream(conversation, tokenizer, model, max_new_tokens=1024, top_p=0.8, top_k=50, temperature=0.8):
    input_content = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    tokens_without_generation_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=False,
        tokenize=False,
    )
    print("Input content:", input_content)
    inputs = tokenizer.encode(input_content)

    for token in model.generate_stream(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        use_cache=True,
        tokens_num=len(tokenizer.encode(tokens_without_generation_prompt))
    ):
        yield f"{tokenizer.decode([int(token)], skip_special_tokens=True)}"