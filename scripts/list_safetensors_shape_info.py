import sys
from safetensors import safe_open

def list_safetensor_tensors(file_path: str):
    with safe_open(file_path, framework="torch", device="cpu") as f:
        print(f"📂 File: {file_path}")
        print(f"🔢 Total tensors: {len(f.keys())}")
        print("=" * 60)
        for name in f.keys():
            tensor = f.get_tensor(name)
            print(f"{name:<60} shape={tensor.shape} dtype={tensor.dtype}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_safetensors_shape_info.py <model.safetensors>")
        sys.exit(1)

    safetensor_path = sys.argv[1]
    list_safetensor_tensors(safetensor_path)
