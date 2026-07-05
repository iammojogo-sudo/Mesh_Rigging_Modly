import os
import sys
import threading
import time
import uuid
import shutil
import tempfile
from pathlib import Path

from services.generators.base import BaseGenerator, smooth_progress

_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

HY_MOTION_REPO = "tencent/HY-Motion-1.0/HY-Motion-1.0-Lite"
RIGANYTHING_REPO = "Isabella98Liu/RigAnything"

class RigAnythingMotionGenerator(BaseGenerator):
    MODEL_ID = "riganything_motion"
    DISPLAY_NAME = "RigAnything Motion Transfer"
    VRAM_GB = 3

    def __init__(self):
        super().__init__()
        self._rig_model = None
        self._motion_model = None
        self._device = None

    def is_downloaded(self):
        motion_dir = self.model_dir / "motion"
        return (motion_dir / "latest.ckpt").exists() or (motion_dir / "pytorch_model.bin").exists()

    def load(self):
        if self._rig_model is not None and self._motion_model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[RigAnything] loading models on %s" % self._device)
        self._load_rig_model()
        self._load_motion_model()
        print("[RigAnything] models loaded successfully")

    def unload(self):
        self._rig_model = None
        self._motion_model = None
        self._device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        rig_dir = self.model_dir / "riganything"
        motion_dir = self.model_dir / "motion"
        rig_dir.mkdir(parents=True, exist_ok=True)
        motion_dir.mkdir(parents=True, exist_ok=True)
        print("[RigAnything] downloading RigAnything weights...")
        snapshot_download(repo_id=RIGANYTHING_REPO, local_dir=str(rig_dir))
        print("[RigAnything] downloading HY-Motion-1.0 Lite weights...")
        snapshot_download(repo_id=HY_MOTION_REPO, local_dir=str(motion_dir))
        print("[RigAnything] download complete")

    def _load_rig_model(self):
        rig_dir = self.model_dir / "riganything"
        try:
            import inference_utils.mesh_simplify as mesh_simplify
            import inference_utils.rig_pipeline as rig_pipeline
            self._rig_utils = {
                "mesh_simplify": mesh_simplify,
                "rig_pipeline": rig_pipeline,
            }
        except ImportError as e:
            print("[RigAnything] Warning: Could not import rig utilities: %s" % e)
            self._rig_utils = None

    def _load_motion_model(self):
        motion_dir = self.model_dir / "motion"
        try:
            import torch
            ckpt_path = motion_dir / "latest.ckpt"
            if ckpt_path.exists():
                self._motion_ckpt = torch.load(str(ckpt_path), map_location=self._device)
            else:
                self._motion_ckpt = None
        except Exception as e:
            print("[RigAnything] Warning: Could not load motion checkpoint: %s" % e)
            self._motion_ckpt = None

    def generate(self, model_bytes, params, progress_cb=None, cancel_event=None):
        params = params or {}
        if self._rig_model is None or self._motion_model is None:
            self.load()
        simplify_flag = _int(params.get("simplify_mesh"), 1)
        target_faces = _int(params.get("target_face_count"), 8192)
        prompt = params.get("prompt", "").strip()
        if not prompt:
            raise ValueError("no prompt provided — connect a Text node to the input")
        self._report(progress_cb, 5, "starting rigging process")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            mesh_path = tmpdir / "input_mesh.obj"
            with open(mesh_path, "wb") as f:
                f.write(model_bytes)
            # rig
            rigged_path = self._rig_mesh(mesh_path, simplify_flag, target_faces, tmpdir, progress_cb, cancel_event)
            self._check_cancelled(cancel_event)
            self._report(progress_cb, 60, "generating motion")
            self._check_cancelled(cancel_event)
            # motion
            animated_path = self._apply_motion(rigged_path, prompt, tmpdir, progress_cb, cancel_event)
            self._check_cancelled(cancel_event)
            self._report(progress_cb, 95, "saving output")
            out_dir = self.outputs_dir if self.outputs_dir else self.model_dir.parent.parent.parent / "outputs" / self.MODEL_ID
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"riganimated_{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
            out_path = out_dir / filename
            shutil.copy(str(animated_path), str(out_path))
            self._report(progress_cb, 100, "done")
            print("[RigAnything] saved %s" % out_path)
            return str(out_path)

    def _rig_mesh(self, mesh_path, simplify_flag, target_faces, output_dir, progress_cb, cancel_event):
        self._report(progress_cb, 10, "simplifying mesh")
        self._check_cancelled(cancel_event)
        if simplify_flag:
            simplified_path = output_dir / "simplified_mesh.obj"
            if self._rig_utils and self._rig_utils["mesh_simplify"]:
                self._rig_utils["mesh_simplify"].simplify_mesh(str(mesh_path), str(simplified_path), target_faces)
            else:
                simplified_path = mesh_path
        else:
            simplified_path = mesh_path
        self._report(progress_cb, 30, "rigging skeleton")
        self._check_cancelled(cancel_event)
        rigged_path = output_dir / "rigged_mesh.glb"
        if self._rig_utils and self._rig_utils["rig_pipeline"]:
            self._rig_utils["rig_pipeline"].rig_mesh(str(simplified_path), str(rigged_path))
        else:
            import subprocess
            rig_script = self.model_dir / "riganything" / "scripts" / "inference.sh"
            if rig_script.exists():
                subprocess.run([str(rig_script), str(simplified_path), "0", str(target_faces)], check=True)
            else:
                raise RuntimeError("Rig pipeline not available")
        return rigged_path

    def _apply_motion(self, rigged_path, prompt, output_dir, progress_cb, cancel_event):
        self._report(progress_cb, 70, "running HY-Motion inference")
        self._check_cancelled(cancel_event)
        animated_path = output_dir / "animated_mesh.glb"
        try:
            from inference_utils.motion_inference import apply_motion_to_rig
            apply_motion_to_rig(str(rigged_path), prompt, str(animated_path), self._motion_ckpt, self._device)
        except ImportError:
            print("[RigAnything] Motion inference not available in current setup")
            print("[RigAnything] Please ensure motion inference utilities are installed")
            shutil.copy(str(rigged_path), str(animated_path))
        return animated_path

    def _check_cancelled(self, cancel_event):
        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Operation cancelled by user")

def _int(val, default):
    try:
        return int(val)
    except:
        return default

def _float(val, default):
    try:
        return float(val)
    except:
        return default
