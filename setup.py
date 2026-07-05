import os
import subprocess
import sys
from pathlib import Path

try:
    from setuptools import setup
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
    from setuptools import setup

def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'transformers>=4.35.0',
        'accelerate>=0.21.0',
        'accrue>=0.1.5',
        'opencv-python',
        'pillow',
        'numpy',
        'scipy',
        'easydict',
        'tqdm',
        'einops',
        'timm',
        'decord',
        'imageio[ffmpeg]',
        'huggingface-hub',
        'pyyaml',
        'open3d',
        'pymeshlab'
    ]

setup(
    name="riganything-motion-extension",
    version="1.0.0",
    description="Modly extension for auto-rigging 3D models and applying text-to-motion animations",
    author="iammojogo",
    packages=[],
    install_requires=read_requirements(),
    python_requires=">=3.10",
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.ckpt', '*.glb', '*.obj', '*.npz'],
    }
)