"""Training-time antigen perturbation for perturbation-aware training.

Supports two modes:
1. single_forward_replace: Replace clean antigen with perturbed antigen in 30% of batches.
2. consistency: Dual forward (clean + perturbed) with consistency loss.

Uses smooth_perturbation from diffab.utils.perturbation for coherent epitope deformation.
"""

import copy
import torch
import torch.nn.functional as F
from diffab.utils.protein.constants import BBHeavyAtom
from diffab.utils.perturbation import smooth_perturbation


def identify_epitope_from_batch(batch, contact_cutoff=6.0, expand_cutoff=8.0):
    """Identify epitope residues from a training batch.

    Args:
        batch: Training batch dict with pos_heavyatom, fragment_type, generate_flag.
        contact_cutoff: Distance cutoff for core epitope (any heavy atom).
        expand_cutoff: CA-CA distance cutoff for expanded epitope.

    Returns:
        epi_plus_masks: (N, L_ag) bool tensor per sample, True for E+ residues.
            L_ag is the number of antigen residues per sample.
    """
    pos_heavyatom = batch['pos_heavyatom']  # (N, L, n_atoms, 3)
    mask_heavyatom = batch['mask_heavyatom']  # (N, L, n_atoms)
    fragment_type = batch['fragment_type']  # (N, L)
    generate_flag = batch['generate_flag']  # (N, L)

    N = pos_heavyatom.shape[0]
    antigen_mask = (fragment_type == 3)
    results = []

    for n in range(N):
        am = antigen_mask[n]
        gm = generate_flag[n]

        if not am.any() or not gm.any():
            results.append(torch.zeros(am.sum(), dtype=torch.bool, device=pos_heavyatom.device))
            continue

        ag_indices = am.nonzero(as_tuple=False).squeeze(-1)
        cdr_indices = gm.nonzero(as_tuple=False).squeeze(-1)

        # Get CDR atom positions (all heavy atoms)
        cdr_pos = pos_heavyatom[n, cdr_indices]  # (L_cdr, n_atoms, 3)
        cdr_mask = mask_heavyatom[n, cdr_indices]  # (L_cdr, n_atoms)
        cdr_atoms = cdr_pos[cdr_mask]  # (n_cdr_atoms, 3)

        # Get antigen CA positions
        ag_ca = pos_heavyatom[n, ag_indices, BBHeavyAtom.CA]  # (L_ag, 3)

        L_ag = len(ag_indices)
        e_mask = torch.zeros(L_ag, dtype=torch.bool, device=pos_heavyatom.device)

        for i in range(L_ag):
            ag_atoms = pos_heavyatom[n, ag_indices[i]][mask_heavyatom[n, ag_indices[i]].bool()]
            if ag_atoms.shape[0] == 0:
                continue
            dists = torch.cdist(ag_atoms.unsqueeze(0), cdr_atoms.unsqueeze(0)).squeeze(0)
            if dists.min() <= contact_cutoff:
                e_mask[i] = True

        if e_mask.sum() == 0:
            e_mask[:] = True

        # Expand: E+ = residues within expand_cutoff of any E residue (CA-CA)
        e_plus_mask = e_mask.clone()
        e_indices = torch.where(e_mask)[0]
        for i in e_indices:
            ca_i = ag_ca[i]
            ca_dists = torch.norm(ag_ca - ca_i, dim=1)
            e_plus_mask = e_plus_mask | (ca_dists <= expand_cutoff)

        results.append(e_plus_mask)

    return results


def perturb_batch_antigen(batch, target_rmsd=0.8, sigma=4.0, seed=None):
    """Create a perturbed copy of the batch with epitope antigen perturbation.

    Only perturbs antigen epitope residues. CDR, antibody framework, and non-epitope
    antigen residues are unchanged. Amino acid sequences are unchanged.

    Args:
        batch: Training batch dict.
        target_rmsd: Target CA RMSD for perturbation (Angstroms).
        sigma: Gaussian smoothing kernel width.
        seed: Random seed for reproducibility.

    Returns:
        perturbed_batch: Deep copy of batch with perturbed antigen coordinates.
    """
    if seed is not None:
        torch.manual_seed(seed)

    perturbed_batch = copy.deepcopy(batch)
    pos_heavyatom = perturbed_batch['pos_heavyatom']
    fragment_type = perturbed_batch['fragment_type']
    antigen_mask = (fragment_type == 3)

    epi_plus_masks = identify_epitope_from_batch(batch)

    for n in range(pos_heavyatom.shape[0]):
        am = antigen_mask[n]
        if not am.any():
            continue

        ag_indices = am.nonzero(as_tuple=False).squeeze(-1)
        ag_ca = pos_heavyatom[n, ag_indices, BBHeavyAtom.CA]  # (L_ag, 3)

        epi_mask = epi_plus_masks[n]
        if epi_mask.sum() == 0:
            continue

        # Generate smooth perturbation
        displacements = smooth_perturbation(ag_ca, epi_mask, target_rmsd, sigma=sigma)

        # Apply displacements to all heavy atoms in epitope residues
        epi_indices = torch.where(epi_mask)[0]
        for idx in epi_indices:
            global_idx = ag_indices[idx]
            disp = displacements[idx]  # (3,)
            # Apply to all heavy atoms in this residue
            pos_heavyatom[n, global_idx] = pos_heavyatom[n, global_idx] + disp.unsqueeze(0)

    return perturbed_batch


def compute_consistency_loss(preds_perturbed, preds_clean, mask_generate):
    """Compute consistency loss between perturbed and clean predictions.

    L_consistency = MSE(v_perturbed, stopgrad(v_clean))

    Args:
        preds_perturbed: Dict with vel_s, vel_pos, vel_qua from perturbed forward.
        preds_clean: Dict with vel_s, vel_pos, vel_qua from clean forward.
        mask_generate: (N, L) mask for generated residues.

    Returns:
        consistency_loss: Scalar tensor.
    """
    mask = mask_generate.unsqueeze(-1).float()

    # Sequence velocity consistency
    loss_seq = F.mse_loss(preds_perturbed['vel_s'], preds_clean['vel_s'].detach(), reduction='none')
    loss_seq = (loss_seq.mean(dim=-1, keepdim=True) * mask).sum() / (mask.sum() + 1e-8)

    # Position velocity consistency
    loss_pos = F.mse_loss(preds_perturbed['vel_pos'], preds_clean['vel_pos'].detach(), reduction='none')
    loss_pos = (loss_pos.mean(dim=-1, keepdim=True) * mask).sum() / (mask.sum() + 1e-8)

    # Rotation velocity consistency
    loss_qua = F.mse_loss(preds_perturbed['vel_qua'], preds_clean['vel_qua'].detach(), reduction='none')
    loss_qua = (loss_qua.mean(dim=-1, keepdim=True) * mask).sum() / (mask.sum() + 1e-8)

    return loss_seq + loss_pos + loss_qua
