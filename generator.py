"""
RigAnything + HY-Motion-1.0-Lite extension for Modly.

Auto-rigs any 3D mesh (human, quad, multi) using RigAnything,
then optionally applies text-to-motion animation from HY-Motion-1.0-Lite.
"""
import io
import json
import os
import random
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

from services.generators.base import BaseGenerator, smooth_progress

_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

RIGANYTHING_HF_REPO   = "Isabellaliu/RigAnything"
RIGANYTHING_GITHUB    = "https://github.com/Isabella98Liu/RigAnything/archive/refs/heads/main.zip"
HY_MOTION_HF_REPO     = "tencent/HY-Motion-1.0"
HY_MOTION_SUBFOLDER   = "HY-Motion-1.0-Lite"
HY_MOTION_GITHUB      = "https://github.com/Tencent-Hunyuan/HY-Motion-1.0/archive/refs/heads/master.zip"


class RigAnythingMotionGenerator(BaseGenerator):
    MODEL_ID     = "riganything_motion"
    DISPLAY_NAME = "RigAnything Motion"
    VRAM_GB      = 4

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        check = self.download_check
        if check:
            return (self.model_dir / check).exists()
        rig_ckpt = self.model_dir / "riganything" / "latest.ckpt"
        motion_ckpt = self.model_dir / "motion" / HY_MOTION_SUBFOLDER / "latest.ckpt"
        return rig_ckpt.exists() or motion_ckpt.exists()

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()

        self._device = self._resolve_device()
        print(f"[RigAnything] Using device: {self._device}")

        self._ensure_riganything_source()
        self._ensure_hymotion_source()

        self._model = True  # mark as loaded
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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self._model is None:
            self.load()

        params = params or {}
        prompt = params.get("prompt", "").strip()
        simplify_flag = _int(params.get("simplify_mesh"), 1)
        target_faces  = _int(params.get("target_face_count"), 8192)
        enable_motion = _int(params.get("enable_motion"), 1)
        guidance      = _float(params.get("guidance_scale"), 5.0)
        steps         = _int(params.get("num_inference_steps"), 25)
        seed          = _int(params.get("seed"), -1)
        if seed == -1:
            seed = random.randint(0, 2**31 - 1)

        if enable_motion and not prompt:
            print("[RigAnything] Motion enabled but no prompt provided; rigging only.")
            enable_motion = 0

        self._report(progress_cb, 2, "Preparing mesh")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mesh_path = self._write_input_mesh(image_bytes, tmp)
            out_dir = tmp / "output"
            out_dir.mkdir(exist_ok=True)

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 10, "Simplifying mesh")
            simplified = self._maybe_simplify(mesh_path, simplify_flag, target_faces, tmp, progress_cb)

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 25, "Rigging skeleton")
            rigged_npz = self._run_riganything(simplified, out_dir)
            self._check_cancelled(cancel_event)
            self._report(progress_cb, 55, "Exporting rigged mesh")
            rigged_glb = self._build_rigged_mesh(rigged_npz, simplified, out_dir)

            if enable_motion:
                self._check_cancelled(cancel_event)
                self._report(progress_cb, 65, "Generating motion")
                motion_data = self._generate_motion(
                    prompt, guidance, steps, seed, out_dir, progress_cb
                )
                if motion_data:
                    self._check_cancelled(cancel_event)
                    self._report(progress_cb, 80, "Applying motion to rig")
                    self._apply_motion_to_rig(rigged_glb, motion_data, out_dir)

            self._check_cancelled(cancel_event)
            self._report(progress_cb, 95, "Saving final output")
            out_path = self._finalize_output(rigged_glb, out_dir)

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _write_input_mesh(self, data: bytes, tmp: Path) -> Path:
        path = tmp / "input_mesh.glb"
        with open(str(path), "wb") as f:
            f.write(data)
        return path

    def _maybe_simplify(
        self, mesh_path: Path, enable: int, target_faces: int,
        tmp: Path, progress_cb=None,
    ) -> Path:
        if not enable:
            return mesh_path
        simplified = tmp / "simplified.glb"
        script = self._rig_src_dir / "inference_utils" / "mesh_simplify.py"
        if not script.exists():
            print("[RigAnything] mesh_simplify.py not found; skipping simplification")
            return mesh_path
        cmd = [
            sys.executable, str(script),
            "--data_path", str(mesh_path),
            "--mesh_simplify", "1",
            "--simplify_count", str(target_faces),
            "--output_path", str(tmp),
        ]
        subprocess.run(cmd, check=True)
        result = tmp / f"{mesh_path.stem}_simplified.glb"
        if result.exists():
            return result
        return mesh_path

    def _run_riganything(self, mesh_path: Path, out_dir: Path) -> Path:
        ckpt = self.model_dir / "riganything" / "latest.ckpt"
        if not ckpt.exists():
            ckpt = self.model_dir / "riganything" / "riganything_ckpt.pt"
            if not ckpt.exists():
                raise RuntimeError(
                    "RigAnything checkpoint not found. "
                    "Try re-downloading the model weights in Modly."
                )

        config = self._rig_src_dir / "config.yaml"
        infer  = self._rig_src_dir / "inference.py"

        cmd = [
            sys.executable, str(infer),
            "--config", str(config),
            "--load", str(ckpt),
            "-s", "inference", "true",
            "-s", "inference_out_dir", str(out_dir),
            "--mesh_path", str(mesh_path),
        ]
        print(f"[RigAnything] Running inference on {mesh_path.name} ...")
        subprocess.run(cmd, check=True)
        items = list(out_dir.glob("*.npz"))
        if not items:
            raise RuntimeError("RigAnything produced no .npz output")
        return items[0]

    def _build_rigged_mesh(self, npz_path: Path, mesh_path: Path, out_dir: Path) -> Path:
        vis = self._rig_src_dir / "inference_utils" / "vis_skel.py"
        if not vis.exists():
            raise RuntimeError("vis_skel.py not found in RigAnything source")
        cmd = [
            sys.executable, str(vis),
            "--data_path", str(npz_path),
            "--save_path", str(out_dir),
            "--mesh_path", str(mesh_path),
        ]
        subprocess.run(cmd, check=True)
        rigged = npz_path.with_name(f"{npz_path.stem}_rig.glb")
        if rigged.exists():
            return rigged
        candidates = list(out_dir.glob("*_rig.glb"))
        if candidates:
            return candidates[0]
        return mesh_path

    def _generate_motion(
        self, prompt: str, guidance: float, steps: int,
        seed: int, out_dir: Path, progress_cb=None,
    ) -> Optional[dict]:
        try:
            return self._run_hymotion(prompt, guidance, steps, seed, out_dir)
        except Exception as e:
            print(f"[RigAnything] HY-Motion generation failed: {e}")
            print("[RigAnything] Returning rigged mesh without motion.")
            return None

    def _apply_motion_to_rig(self, rigged_glb: Path, motion: dict, out_dir: Path) -> None:
        print("[RigAnything] Motion data generated, saving alongside rigged mesh.")
        motion_path = out_dir / "motion_data.json"
        with open(str(motion_path), "w") as f:
            json.dump(motion, f, indent=2)

    def _finalize_output(self, rigged_glb: Path, tmp_dir: Path) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"rigged_{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        out = self.outputs_dir / name
        shutil.copy(str(rigged_glb), str(out))
        print(f"[RigAnything] Saved {out}")
        return out

    # ------------------------------------------------------------------
    # HY-Motion-1.0-Lite inference
    # ------------------------------------------------------------------

    def _run_hymotion(
        self, prompt: str, guidance: float, steps: int,
        seed: int, out_dir: Path,
    ) -> dict:
        motion_dir = self.model_dir / "motion" / HY_MOTION_SUBFOLDER
        ckpt = motion_dir / "latest.ckpt"
        config = motion_dir / "config.yml"
        if not ckpt.exists():
            raise FileNotFoundError(f"HY-Motion checkpoint not found at {ckpt}")

        sys.path.insert(0, str(self._hymotion_src_dir))
        try:
            from hymotion.utils.t2m_runtime import T2MRuntime
        except ImportError:
            print("[RigAnything] HY-Motion package not importable; trying local import")
            raise

        runtime = T2MRuntime(
            config_path=str(config),
            ckpt_name=str(ckpt),
            device_ids=[0] if self._device == "cuda" else None,
            disable_prompt_engineering=True,
        )

        seeds_csv = ",".join(str(seed + i) for i in range(4))
        _, fbx_files, _ = runtime.generate_motion(
            text=prompt,
            seeds_csv=seeds_csv,
            duration=5.0,
            cfg_scale=guidance,
            output_format="dict",
            original_text=prompt,
            output_dir=str(out_dir),
            output_filename="motion",
        )
        for f in fbx_files:
            dst = out_dir / os.path.basename(f)
            if not dst.exists():
                shutil.copy(f, str(dst))

        return {"prompt": prompt, "seed": seed, "files": fbx_files}

    # ------------------------------------------------------------------
    # Source code bootstrapping (RigAnything)
    # ------------------------------------------------------------------

    @property
    def _rig_src_dir(self) -> Path:
        return self.model_dir / "_riganything_src"

    def _ensure_riganything_source(self) -> None:
        src = self._rig_src_dir
        if (src / "inference.py").exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return
        print("[RigAnything] Downloading RigAnything source code from GitHub ...")
        self._download_github_zip(RIGANYTHING_GITHUB, src, "RigAnything-main")

    # ------------------------------------------------------------------
    # Source code bootstrapping (HY-Motion-1.0)
    # ------------------------------------------------------------------

    @property
    def _hymotion_src_dir(self) -> Path:
        return self.model_dir / "_hymotion_src"

    def _ensure_hymotion_source(self) -> None:
        src = self._hymotion_src_dir
        if (src / "hymotion").exists():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            return
        print("[RigAnything] Downloading HY-Motion-1.0 source code from GitHub ...")
        self._download_github_zip(HY_MOTION_GITHUB, src, "HY-Motion-1.0-master")

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

    # ------------------------------------------------------------------
    # Weight download
    # ------------------------------------------------------------------

    def _auto_download(self) -> None:
        self._download_weights()

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)

        rig_dir = self.model_dir / "riganything"
        if not rig_dir.exists() or not any(rig_dir.iterdir()):
            rig_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RigAnything] Downloading RigAnything weights from {RIGANYTHING_HF_REPO} ...")
            snapshot_download(
                repo_id=RIGANYTHING_HF_REPO,
                local_dir=str(rig_dir),
                ignore_patterns=["*.md", "LICENSE", "NOTICE", ".gitattributes"],
            )
            print("[RigAnything] RigAnything weights downloaded.")
        else:
            print("[RigAnything] RigAnything weights already present.")

        motion_dir = self.model_dir / "motion"
        if not motion_dir.exists() or not any(motion_dir.iterdir()):
            motion_dir.mkdir(parents=True, exist_ok=True)
            print(f"[RigAnything] Downloading HY-Motion-1.0-Lite from {HY_MOTION_HF_REPO} ...")
            snapshot_download(
                repo_id=HY_MOTION_HF_REPO,
                local_dir=str(motion_dir),
                allow_patterns=[f"{HY_MOTION_SUBFOLDER}/**"],
                ignore_patterns=["*.md", "LICENSE", "NOTICE", ".gitattributes"],
            )
            print("[RigAnything] HY-Motion-1.0-Lite downloaded.")
        else:
            print("[RigAnything] HY-Motion weights already present.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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


def _float(val, default):
    try:
        return float(val)
    except Exception:
        return default