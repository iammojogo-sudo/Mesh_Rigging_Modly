import json
import platform
import subprocess
import sys
from pathlib import Path

IS_WIN = platform.system() == "Windows"

def venv_python(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")

def pip(venv, *args):
    subprocess.run([str(venv_python(venv)), "-m", "pip"] + list(args), check=True)

def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"
    if not venv.exists():
        print("creating venv...")
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("venv exists, skipping creation")

    if gpu_sm >= 100:
        index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0"]
    elif gpu_sm >= 70:
        index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0"]
    else:
        index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1"]

    print("installing torch (sm%d)..." % gpu_sm)
    pip(venv, "install", *torch_pkgs, "--index-url", index)

    print("installing dependencies...")
    pip(venv, "install",
        "diffusers>=0.29.0",
        "transformers>=4.51.0,<4.52.0" if gpu_sm < 100 else "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.20.0",
        "sentencepiece",
        "safetensors",
        "Pillow",
        "numpy",
        "tiktoken",
        "protobuf",
    )

    print("installing xformers...")
    try:
        old_index = "https://download.pytorch.org/whl/cu118"
        pip(venv, "install",
            "xformers==0.0.29.post3" if gpu_sm >= 70 else "xformers==0.0.28.post2",
            "--index-url", index if gpu_sm >= 70 else old_index,
        )
    except subprocess.CalledProcessError:
        print("xformers failed, skipping")

    print("done")

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) == 2:
        a = json.loads(sys.argv[1])
        setup(Path(a["python_exe"]), Path(a["ext_dir"]), int(a["gpu_sm"]))
    else:
        print("usage: setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)