#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert SMPL pkl data directly into per-walk HumanML3D .npy files.

This script is based on data/preprocessing/smpl2humanml3d.py and npz2npy.py:
- keeps the original SMPL -> joints -> HumanML3D 263D conversion path
- keeps downsampling to 30 fps by producing down0/down1/... variants
- removes mirror augmentation, so no *_M.npy files are generated

Accepted pkl layouts:
1. CARE-PD style:
   {
       subject_id: {
           walk_id: {"pose": (T,72), "beta": (10,) or (T,10), "trans": (T,3)}
       }
   }
2. Single sequence:
   {"pose": (T,72), "beta"/"betas": (10,) or (T,10), "trans": (T,3)}
"""

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from tqdm import tqdm


def patch_numpy_for_chumpy():
    """chumpy expects NumPy aliases removed in NumPy >= 1.24."""
    np.bool = np.bool_
    np.int = int
    np.float = float
    np.complex = complex
    np.object = np.object_
    np.unicode = np.str_
    np.str = np.str_


patch_numpy_for_chumpy()

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PREPROCESSING_ROOT = os.path.join(PROJECT_ROOT, "data", "preprocessing")
sys.path.insert(0, PREPROCESSING_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from const.const import _DEVICE, DATASET_ORIGINAL_FPS
from data.preprocessing.create_redundant_representation import process_file
from data.preprocessing.human_body_prior.body_model.body_model import BodyModel
from data.preprocessing.trajectory_correction_amass import transform_seq_so_it_has_no_slope_AMASS
from data.preprocessing.transforms.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
from data.preprocessing.transforms.skeleton import Skeleton


SLOPE_CORRECTION_DATASETS = {"T-LTC", "T-SDU", "T-SDU-PD"}


def sanitize_filename(name):
    name = str(name).replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._") or "sample"


def get_betas(sequence):
    if "beta" in sequence:
        return sequence["beta"]
    if "betas" in sequence:
        return sequence["betas"]
    raise KeyError("SMPL sequence must contain 'beta' or 'betas'.")


def normalize_smpl_sequence(sequence):
    if "pose" not in sequence or "trans" not in sequence:
        raise KeyError("SMPL sequence must contain 'pose' and 'trans'.")

    pose = np.asarray(sequence["pose"])
    beta = np.asarray(get_betas(sequence))
    trans = np.asarray(sequence["trans"])

    if pose.ndim == 3:
        pose = pose.reshape(pose.shape[0], -1)
    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"Expected pose shape (T,72) or (T,24,3), got {pose.shape}.")
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Expected trans shape (T,3), got {trans.shape}.")
    if pose.shape[0] != trans.shape[0]:
        raise ValueError(f"pose and trans frame counts differ: {pose.shape[0]} vs {trans.shape[0]}.")

    if beta.ndim == 1:
        beta = np.tile(beta[None, :], (pose.shape[0], 1))
    elif beta.shape[0] != pose.shape[0]:
        beta = np.tile(beta[:1], (pose.shape[0], 1))

    return {"pose": pose, "beta": beta, "trans": trans}


def iter_smpl_sequences(data):
    if isinstance(data, dict) and {"pose", "trans"}.issubset(data.keys()):
        yield "sample", "walk", normalize_smpl_sequence(data)
        return

    if not isinstance(data, dict):
        raise TypeError("Expected pkl to contain a dict.")

    for subject_id, subject_data in data.items():
        if isinstance(subject_data, dict) and {"pose", "trans"}.issubset(subject_data.keys()):
            yield subject_id, "walk", normalize_smpl_sequence(subject_data)
            continue

        if not isinstance(subject_data, dict):
            continue

        for walk_id, sequence in subject_data.items():
            if not isinstance(sequence, dict):
                continue
            yield subject_id, walk_id, normalize_smpl_sequence(sequence)


def generate_smpl_to_pose(bm, sequence, down_sample_rate, down):
    frame_number = sequence["pose"].shape[0]
    pose_world = sequence["pose"].reshape(-1, 24, 3)
    betas = sequence["beta"]
    world_trans = sequence["trans"]

    global_orient = torch.tensor(pose_world[:, 0:1, :], dtype=torch.float32)
    body_pose = torch.tensor(pose_world[:, 1:24, :], dtype=torch.float32)
    betas = torch.tensor(betas, dtype=torch.float32)

    if down_sample_rate > 1:
        global_orient = global_orient[down::down_sample_rate, ...]
        body_pose = body_pose[down::down_sample_rate, ...]
        betas = betas[down::down_sample_rate, ...]
        world_trans = world_trans[down::down_sample_rate, ...]
        frame_number = body_pose.shape[0]

    body_parms = {
        "root_orient": global_orient.reshape(frame_number, -1).to(_DEVICE),
        "pose_body": body_pose.reshape(frame_number, -1).to(_DEVICE),
        "trans": torch.tensor(world_trans, dtype=torch.float32).to(_DEVICE),
        "betas": betas.reshape(frame_number, -1).to(_DEVICE),
        "pose_hand": None,
    }

    with torch.no_grad():
        body = bm(**body_parms)

    return body.Jtr.detach().cpu().numpy()[:, :22]


def build_cfg(args):
    cfg = SimpleNamespace()
    cfg.input = Path(args.input)
    cfg.output_dir = Path(args.output)
    cfg.dataset = args.dataset
    cfg.fps = args.fps if args.fps is not None else DATASET_ORIGINAL_FPS.get(args.dataset, 30)
    cfg.exfps = args.target_fps
    cfg.down_sample_rate = max(1, int(cfg.fps / cfg.exfps))
    cfg.slope_correction = args.slope_correction or args.dataset in SLOPE_CORRECTION_DATASETS
    cfg.model_path = Path(args.model_path)
    cfg.joints_num = 22

    cfg.l_idx1, cfg.l_idx2 = 5, 8
    cfg.fid_r, cfg.fid_l = [8, 11], [7, 10]
    cfg.face_joint_indx = [2, 1, 17, 16]
    cfg.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    cfg.kinematic_chain = t2m_kinematic_chain

    example_data = np.load(args.template_path)
    example_data = torch.from_numpy(example_data.reshape(len(example_data), -1, 3))
    tgt_skel = Skeleton(cfg.n_raw_offsets, cfg.kinematic_chain, "cpu")
    cfg.tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    cfg.feet_thre = args.feet_thre
    return cfg


def convert_pkl_to_npy(cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    all_smpls = joblib.load(cfg.input)
    bm = BodyModel(bm_fname=str(cfg.model_path), num_betas=10).to(_DEVICE)

    written = 0
    skipped = 0
    sequences = list(iter_smpl_sequences(all_smpls))

    for subject_id, walk_id, smpl_data in tqdm(sequences, desc="SMPL pkl -> HumanML3D npy"):
        if smpl_data["pose"].shape[0] < 30:
            skipped += 1
            continue
        if "Trimmed" in str(walk_id):
            skipped += 1
            continue

        for down in range(cfg.down_sample_rate):
            if cfg.down_sample_rate == 1:
                motion_id = f"{subject_id}__{walk_id}"
            else:
                motion_id = f"{subject_id}__{walk_id}_down{down}"

            try:
                pose = generate_smpl_to_pose(bm, smpl_data, cfg.down_sample_rate, down)
                if cfg.slope_correction:
                    pose = transform_seq_so_it_has_no_slope_AMASS(pose)

                motion_263, _, _, _ = process_file(
                    pose,
                    cfg.feet_thre,
                    cfg.tgt_offsets,
                    cfg.face_joint_indx,
                    cfg.fid_l,
                    cfg.fid_r,
                    cfg.l_idx1,
                    cfg.l_idx2,
                    cfg.n_raw_offsets,
                    cfg.kinematic_chain,
                )

                out_path = cfg.output_dir / f"{sanitize_filename(motion_id)}.npy"
                np.save(out_path, motion_263)
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"[ERROR] {motion_id}: {exc}")

    print(f"Done. Wrote {written} .npy files to: {cfg.output_dir}")
    if skipped:
        print(f"Skipped/failed sequences: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL pkl data into per-walk HumanML3D .npy files without mirror augmentation."
    )
    parser.add_argument("-i", "--input", required=True, help="Input SMPL .pkl path.")
    parser.add_argument("-o", "--output", required=True, help="Output directory for .npy files.")
    parser.add_argument("-db", "--dataset", default="BMCLab", help="Dataset name, used to infer original fps.")
    parser.add_argument("--fps", type=int, default=None, help="Original SMPL/video fps. Overrides dataset default.")
    parser.add_argument("--target_fps", type=int, default=30, help="Target fps after downsampling.")
    parser.add_argument("--slope_correction", action="store_true", help="Force slope correction.")
    parser.add_argument("--feet_thre", type=float, default=0.002, help="Foot contact velocity threshold.")
    parser.add_argument(
        "--model_path",
        default="data/preprocessing/common/body_models/smpl/SMPL_NEUTRAL.pkl",
        help="SMPL neutral body model path.",
    )
    parser.add_argument(
        "--template_path",
        default="data/preprocessing/transforms/000021.npy",
        help="HumanML3D target skeleton template .npy path.",
    )
    args = parser.parse_args()

    cfg = build_cfg(args)
    convert_pkl_to_npy(cfg)


if __name__ == "__main__":
    main()
