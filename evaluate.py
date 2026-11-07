"""
Evaluation script for TrustCDR generated structures.
Computes per-CDR RMSD and AAR (Amino Acid Recovery) metrics.

Usage:
    python evaluate.py --results_dir ./results --output_dir ./eval_output
"""
import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import protein_letters_3to1 as three_to_one


# Chothia CDR range definitions
CHOTHIA_CDR_RANGES = {
    'H_CDR1': ('H', 26, 32),
    'H_CDR2': ('H', 52, 56),
    'H_CDR3': ('H', 95, 102),
    'L_CDR1': ('L', 24, 34),
    'L_CDR2': ('L', 50, 56),
    'L_CDR3': ('L', 89, 97),
}


def reslist_rmsd(res_list1, res_list2):
    """Calculate RMSD between two residue lists using dynamic programming alignment."""
    res_short, res_long = (res_list1, res_list2) if len(res_list1) < len(res_list2) else (res_list2, res_list1)
    M, N = len(res_short), len(res_long)

    def d(i, j):
        coord_i = np.array(res_short[i]['CA'].get_coord())
        coord_j = np.array(res_long[j]['CA'].get_coord())
        return ((coord_i - coord_j) ** 2).sum()

    SD = np.full([M, N], np.inf)
    for i in range(M):
        j = N - (M - i)
        if j < N:
            SD[i, j] = sum([d(i+k, j+k) for k in range(N-j)])

    for j in range(N):
        SD[M-1, j] = d(M-1, j)

    for i in range(M-2, -1, -1):
        for j in range((N-(M-i))-1, -1, -1):
            SD[i, j] = min(
                d(i, j) + SD[i+1, j+1],
                SD[i, j+1]
            )

    min_SD = SD[0, :N-M+1].min() if (N-M+1) > 0 else SD[0, 0]
    best_RMSD = np.sqrt(min_SD / M)
    return best_RMSD


def entity_to_seq(entity):
    """Convert a Bio.PDB entity to a sequence string."""
    seq = ''
    for res in Selection.unfold_entities(entity, 'R'):
        try:
            seq += three_to_one(res.get_resname())
        except KeyError:
            pass
    return seq


def reslist_seqid(res_list1, res_list2):
    """Calculate sequence identity (AAR) between two residue lists."""
    seq1 = entity_to_seq(res_list1)
    seq2 = entity_to_seq(res_list2)

    if not seq1 or not seq2:
        return 0.0

    min_len = min(len(seq1), len(seq2))
    identical = sum(seq1[i] == seq2[i] for i in range(min_len))
    return (100.0 * identical) / min_len


def get_cdr_residues(model, chain_id, start, end):
    """Extract residues in a CDR range from a structure model."""
    if chain_id not in model:
        return []
    chain = model[chain_id]
    residues = []
    for res in chain:
        res_num = res.id[1]
        if start <= res_num <= end:
            residues.append(res)
    return residues


def evaluate_sample(result_dir, parser):
    """Evaluate a single sample directory."""
    # Locate reference and generated PDB files
    ref_file = None
    gen_dir = None

    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file == 'REF1.pdb' or file == 'reference.pdb':
                ref_file = os.path.join(root, file)
        for dir_name in dirs:
            if dir_name == 'MultipleCDRs':
                gen_dir = os.path.join(root, dir_name)

    if not ref_file:
        return []

    # Find generated PDBs
    pdb_files = []
    search_dir = gen_dir if gen_dir else os.path.dirname(ref_file)
    if os.path.isdir(search_dir):
        for file in os.listdir(search_dir):
            if re.match(r'^\d+\.pdb$', file):
                pdb_files.append(os.path.join(search_dir, file))

    if not pdb_files:
        return []

    # Parse reference
    ref_model = parser.get_structure('reference', ref_file)[0]
    structure_id = os.path.basename(result_dir)

    results = []

    for gen_file in sorted(pdb_files, key=lambda f: int(os.path.basename(f).split('.')[0])):
        try:
            gen_model = parser.get_structure('generated', gen_file)[0]
            gen_idx = int(os.path.basename(gen_file).split('.')[0])

            for cdr_name, (chain_id, start, end) in CHOTHIA_CDR_RANGES.items():
                ref_residues = get_cdr_residues(ref_model, chain_id, start, end)
                gen_residues = get_cdr_residues(gen_model, chain_id, start, end)

                if not ref_residues or not gen_residues:
                    continue

                rmsd = reslist_rmsd(ref_residues, gen_residues)
                aar = reslist_seqid(ref_residues, gen_residues)

                results.append({
                    'structure': structure_id,
                    'cdr': cdr_name,
                    'generation_idx': gen_idx,
                    'rmsd': rmsd,
                    'aar': aar,
                })
        except Exception as e:
            print(f"Warning: error evaluating {gen_file}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate TrustCDR generated structures')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing inference results')
    parser.add_argument('--output_dir', type=str, default='./eval_output',
                        help='Directory to save evaluation results')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top candidates per CDR for statistics')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pdb_parser = PDBParser(QUIET=True)
    all_results = []

    for sample_dir in sorted(os.listdir(args.results_dir)):
        sample_path = os.path.join(args.results_dir, sample_dir)
        if os.path.isdir(sample_path):
            print(f"Processing {sample_path}")
            sample_results = evaluate_sample(sample_path, pdb_parser)
            all_results.extend(sample_results)

    if not all_results:
        print("No results found for evaluation.")
        return

    df = pd.DataFrame(all_results)

    # Select top-k per CDR per structure by RMSD
    top_k_df = (
        df.sort_values('rmsd')
        .groupby(['structure', 'cdr'])
        .head(args.top_k)
        .reset_index(drop=True)
    )

    # Per-CDR statistics
    cdr_stats = top_k_df.groupby('cdr').agg(
        rmsd_mean=('rmsd', 'mean'),
        rmsd_std=('rmsd', 'std'),
        aar_mean=('aar', 'mean'),
        aar_std=('aar', 'std'),
    ).round(4)

    # Overall statistics
    overall = top_k_df.agg(
        rmsd_mean=('rmsd', 'mean'),
        rmsd_std=('rmsd', 'std'),
        aar_mean=('aar', 'mean'),
        aar_std=('aar', 'std'),
    ).round(4)

    # Print
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Top-{} per CDR per structure)".format(args.top_k))
    print("="*60)
    print("\nOverall:")
    print(overall.to_string())
    print("\nPer-CDR:")
    print(cdr_stats.to_string())

    # Save
    df.to_csv(os.path.join(args.output_dir, 'all_results.csv'), index=False)
    top_k_df.to_csv(os.path.join(args.output_dir, 'top{}_results.csv'.format(args.top_k)), index=False)
    cdr_stats.to_csv(os.path.join(args.output_dir, 'cdr_statistics.csv'))
    overall.to_csv(os.path.join(args.output_dir, 'overall_statistics.csv'))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
