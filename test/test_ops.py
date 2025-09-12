import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    # ops 目录相对 run_ops.py 所在目录
    script_dir = Path(__file__).resolve().parent
    ops_dir = script_dir / "ops"

    # 只收集普通 .py 文件，排除 __init__.py / __main__.py
    py_files = sorted(
        f for f in ops_dir.glob("*.py")
        if f.name not in {"__init__.py", "__main__.py"}
    )

    if not py_files:
        print(f"⚠️ 在 {ops_dir} 下没有找到任何可执行的 .py 文件")
        return

    print(f"🔍 Found {len(py_files)} scripts under {ops_dir}:")
    for f in py_files:
        print(f"   - {f.name}")

    for script in py_files:
        cmd = [sys.executable, str(script), "--device", args.device]
        if args.profile:
            cmd.append("--profile")

        print(f"\n🚀 Running {script.name} with args: {cmd[2:]}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ {script.name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n\033[92mAll ops tests passed!\033[0m")

if __name__ == "__main__":
    main()
