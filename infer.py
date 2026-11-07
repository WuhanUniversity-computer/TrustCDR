#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TrustCDR Inference Script

Performs multi-CDR antibody design inference using the contact_downonly_l0.10
configuration (Contact Reliability Gate).

Usage:
    python infer.py --pdb <input.pdb> --output_dir <output_dir> \
                    --checkpoint <checkpoint.pt> [--gpu <id>] \
                    [--num_seeds 5] [--num_samples 10]

Example:
    python infer.py \
        --pdb ./data/test/7V0P_H_L_A/clean.pdb \
        --output_dir ./inference_results/7V0P \
        --checkpoint ./checkpoints/250000.pt \
        --gpu 0
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffab.utils.misc import seed_all, load_config
from diffab.utils.inference import RemoveNative
from diffab.utils.transforms import Compose, MaskMultipleCDRs, MergeChains, PatchAroundAnchor
from diffab.datasets.custom import preprocess_antibody_structure
from diffab.utils.protein.writers import save_pdb
from diffab.models import get_model


DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), 'configs', 'contact_downonly_l0.10.yaml')
DEFAULT_CDRS = ['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3']


def run_inference(pdb_path, output_dir, checkpoint_path, gpu_id=0,
                  num_seeds=5, num_samples=10, cdrs=None):
    """Run TrustCDR inference on a single PDB structure.

    Args:
        pdb_path: Path to input antibody-antigen complex PDB.
        output_dir: Directory to save generated PDB files.
        checkpoint_path: Path to model checkpoint.
        gpu_id: GPU device ID.
        num_seeds: Number of design seeds.
        num_samples: Number of samples per seed.
        cdrs: List of CDRs to redesign (default: all 6 CDRs).
    """
    if cdrs is None:
        cdrs = DEFAULT_CDRS

    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    # Load config and model
    config, _ = load_config(DEFAULT_CONFIG)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = get_model(ckpt['config'].model).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint: {checkpoint_path}')

    # Preprocess input structure
    data = preprocess_antibody_structure(pdb_path)

    # Apply transforms: mask CDRs, merge chains, extract patch
    transform = Compose([
        MaskMultipleCDRs(selection=cdrs, augmentation=False),
        MergeChains(),
    ])
    data_tmpl = transform(data)
    data_native = MergeChains()(data)

    # Inference transforms
    inference_tfm = Compose([
        PatchAroundAnchor(),
        RemoveNative(
            remove_structure=config.sampling.sample_structure,
            remove_sequence=config.sampling.sample_sequence,
        ),
    ])

    from diffab.utils.data import PaddingCollate, apply_patch_to_tensor
    collate_fn = PaddingCollate(eight=False)

    total_count = 0
    for seed in range(1, num_seeds + 1):
        seed_all(seed)
        for sample_idx in range(num_samples):
            # Apply inference transform
            data_input = inference_tfm(data_tmpl.copy())

            # Collate and move to device
            batch = collate_fn([data_input])
            for key in batch:
                if hasattr(batch[key], 'to'):
                    batch[key] = batch[key].to(device)

            # Sample
            with torch.no_grad():
                samples = model.sample(batch)

            # Save generated PDB
            gen_path = os.path.join(output_dir, f'seed{seed}_sample{sample_idx}.pdb')
            save_pdb(samples, gen_path)
            total_count += 1
            print(f'  [{total_count}/{num_seeds * num_samples}] seed={seed}, sample={sample_idx} -> {gen_path}')

    print(f'\nDone. Generated {total_count} structures in {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='TrustCDR Inference')
    parser.add_argument('--pdb', type=str, required=True, help='Input PDB file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (default: 0)')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of design seeds (default: 5)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per seed (default: 10)')
    parser.add_argument('--cdrs', nargs='+', default=None,
                        help='CDRs to redesign (default: all 6 CDRs)')

    args = parser.parse_args()

    run_inference(
        pdb_path=args.pdb,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        gpu_id=args.gpu,
        num_seeds=args.num_seeds,
        num_samples=args.num_samples,
        cdrs=args.cdrs,
    )


if __name__ == '__main__':
    main()
