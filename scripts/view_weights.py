import argparse
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser(description="Inspect a weight tensor in a .safetensors file")
    parser.add_argument("file", type=str, help="Path to the .safetensors file")
    parser.add_argument("key", type=str, help="Name of the tensor to inspect")
    args = parser.parse_args()

    tensors = load_file(args.file)

    if args.key not in tensors:
        print(f"Key '{args.key}' not found in {args.file}")
        print("Available keys:")
        for k in tensors.keys():
            print(" -", k)
        return

    tensor = tensors[args.key]
    print(f"Tensor '{args.key}':")
    print("Shape:", tensor.shape)
    print("Dtype:", tensor.dtype)
    print("Values:")
    print(tensor)

if __name__ == "__main__":
    main()
