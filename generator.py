import io
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from services.generators.base import BaseGenerator

_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

RIGANYTHING_HF_REPO   = "Isabellaliu/RigAnything"
RIGANYTHING_GITHUB    = "https://github.com/Isabella98Liu/RigAnything/archive/refs/heads/main.zip"


class RigAnythingGenerator(BaseGenerator):
    MODEL_ID     = "riganything"
    DISPLAY_NAME = "RigAnything"
    VRAM_GB      = 4

    def is_downloaded(self) -> bool:
        check = self.download_check
        if check:
            return (self.model_dir / check).exists()
        return (self.model_dir / "riganything_ckpt.pt").exists()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()

        self._device = self._resolve_device()
        print(f"[RigAnything] Using device: {self._device}")

        self._ensure_riganything_source()

        self._model = True
        print("[RigAnything] Ready.")

    def unload(self) -> None:
        self._model = None
        self._device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self._model is None:
            self.load()

        if self._device != "cuda":
            raise RuntimeError(
                "RigAnything requires a CUDA-capable GPU. "
                f"Detected device: {self._device}. "
                "Please run on a system with an NVIDIA GPU."
            )

        params = params or {}
        simplify_flag = _int(params.get("simplify_mesh"), 1)
        target_faces  = _int(params.get("target_face_count"), 8192)

        self._report(progress_cb, 2, "Preparing mesh")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mesh_path = self._write_input_mesh(image_bytes, tmp)
            out_dir = tmp / "output"
            out_dir.mkdir(exist_ok=True)

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 10, "Simplifying mesh")
            simplified = self._maybe_simplify(mesh_path, simplify_flag, target_faces, tmp)

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 25, "Rigging skeleton")
            rigged_npz = self._run_riganything(simplified, out_dir)
            self._check_cancelled(cancel_event)
            self._report(progress_cb, 55, "Exporting rigged mesh")
            rigged_glb = self._build_rigged_mesh(rigged_npz, simplified, out_dir)

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 95, "Saving final output")
            out_path = self._finalize_output(rigged_glb, out_dir)

        self._report(progress_cb, 100, "Done")
        return out_path

    def _write_input_mesh(self, data: bytes, tmp: Path) -> Path:
        path = tmp / "input_mesh.glb"
        with open(str(path), "wb") as f:
            f.write(data)
        return path

    def _maybe_simplify(
        self, mesh_path: Path, enable: int, target_faces: int, tmp: Path,
    ) -> Path:
        if not enable:
            return mesh_path
        script = self._rig_src_dir / "inference_utils" / "mesh_simplify.py"
        if not script.exists():
            print("[RigAnything] mesh_simplify.py not found; skipping simplification")
            return mesh_path
        cmd = [
            sys.executable, script.as_posix(),
            "--data_path", mesh_path.as_posix(),
            "--mesh_simplify", "1",
            "--simplify_count", str(target_faces),
            "--output_path", tmp.as_posix(),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"[RigAnything] Mesh simplification failed (stderr): {result.stderr.strip()}")
                print("[RigAnything] Skipping simplification, using original mesh.")
                return mesh_path
        except Exception as e:
            print(f"[RigAnything] Mesh simplification error: {e}")
            print("[RigAnything] Skipping simplification, using original mesh.")
            return mesh_path
        result_path = tmp / f"{mesh_path.stem}_simplified.glb"
        if result_path.exists():
            return result_path
        return mesh_path

    def _run_riganything(self, mesh_path: Path, out_dir: Path) -> Path:
        ckpt = self.model_dir / "riganything_ckpt.pt"
        if not ckpt.exists():
            raise RuntimeError(
                "RigAnything checkpoint not found at " + str(ckpt) + ". "
                "Try re-downloading the model weights in Modly."
            )

        config = self._rig_src_dir / "config.yaml"
        infer  = self._rig_src_dir / "inference.py"

        cmd = [
            sys.executable, infer.as_posix(),
            "--config", config.as_posix(),
            "--load", ckpt.as_posix(),
            "-s", "inference", "true",
            "-s", "inference_out_dir", out_dir.as_posix(),
            "--mesh_path", mesh_path.as_posix(),
        ]
        print(f"[RigAnything] Running inference on {mesh_path.name} ...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[RigAnything] Inference stderr: {result.stderr.strip()}")
                raise RuntimeError(f"RigAnything inference failed (exit {result.returncode})")
        except subprocess.TimeoutExpired:
            raise RuntimeError("RigAnything inference timed out after 600s")
        items = list(out_dir.glob("*.npz"))
        if not items:
            raise RuntimeError("RigAnything produced no .npz output")
        return items[0]

    def _build_rigged_mesh(self, npz_path: Path, mesh_path: Path, out_dir: Path) -> Path:
        vis = self._rig_src_dir / "inference_utils" / "vis_skel.py"
        if not vis.exists():
            raise RuntimeError("vis_skel.py not found in RigAnything source")
        cmd = [
            sys.executable, vis.as_posix(),
            "--data_path", npz_path.as_posix(),
            "--save_path", out_dir.as_posix(),
            "--mesh_path", mesh_path.as_posix(),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
        rigged = npz_path.with_name(f"{npz_path.stem}_rig.glb")
        if rigged.exists():
            return rigged
        candidates = list(out_dir.glob("*_rig.glb"))
        if candidates:
            return candidates[0]
        return mesh_path

    def _finalize_output(self, rigged_glb: Path, tmp_dir: Path) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"rigged_{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        out = self.outputs_dir / name
        shutil.copy(str(rigged_glb), str(out))
        print(f"[RigAnything] Saved {out}")
        return out

    @property
    def _rig_src_dir(self) -> Path:
        return self.model_dir / "_riganything_src"

    def _ensure_riganything_source(self) -> None:
        src = self._rig_src_dir
        if (src / "inference.py").exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            self._patch_vis_skel(src)
            return
        print("[RigAnything] Downloading RigAnything source code from GitHub ...")
        self._download_github_zip(RIGANYTHING_GITHUB, src, "RigAnything-main")
        self._patch_vis_skel(src)

    @staticmethod
    def _patch_vis_skel(src: Path) -> None:
        vis = src / "inference_utils" / "vis_skel.py"
        if not vis.exists():
            return
        text = vis.read_text(encoding="utf-8")
        original = text
        text = text.replace(
            'bone.tail = bone.head + Vector([0, 0, 0.1])',
            'bone.tail = bone.head + Vector([0, 0, 1.0])',
        )
        text = text.replace(
            "modifier.object = armature\n\n        # Assign vertex groups",
            "modifier.object = armature\n        obj.parent = armature\n\n        # Assign vertex groups",
        )
        if text != original:
            vis.write_text(text, encoding="utf-8")
            print("[RigAnything] Patched vis_skel.py for Godot compatibility.")

    def _download_github_zip(self, url: str, dest: Path, top_dir: str) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[RigAnything] Downloading {url} ...")
        with urllib.request.urlopen(url, timeout=180) as resp:
            data = resp.read()
        prefix = f"{top_dir}/"
        strip  = f"{top_dir}/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        print(f"[RigAnything] Extracted to {dest}.")
        if str(dest) not in sys.path:
            sys.path.insert(0, str(dest))

    def _auto_download(self) -> None:
        self._download_weights()

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)

        if not self.is_downloaded():
            print(f"[RigAnything] Downloading RigAnything weights from {RIGANYTHING_HF_REPO} ...")
            snapshot_download(
                repo_id=RIGANYTHING_HF_REPO,
                local_dir=str(self.model_dir),
                ignore_patterns=["*.md", "LICENSE", "NOTICE", ".gitattributes"],
            )
            print("[RigAnything] RigAnything weights downloaded.")
        else:
            print("[RigAnything] RigAnything weights already present.")

    def _resolve_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch, "backends", None) and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


def _int(val, default):
    try:
        return int(val)
    except Exception:
        return default
