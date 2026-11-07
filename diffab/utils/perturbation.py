import torch
import copy
import numpy as np
from diffab.utils.protein.constants import BBHeavyAtom, ChothiaCDRRange


def _get_cdr_coords(structure):
    cdr_coords = []

    cdr_chains = [
        ('heavy', [ChothiaCDRRange.H1, ChothiaCDRRange.H2, ChothiaCDRRange.H3]),
        ('light', [ChothiaCDRRange.L1, ChothiaCDRRange.L2, ChothiaCDRRange.L3]),
    ]

    for chain_key, cdr_ranges in cdr_chains:
        chain = structure.get(chain_key)
        if chain is None:
            continue
        resseq = chain['resseq']
        pos = chain['pos_heavyatom']
        mask = chain['mask_heavyatom']

        for start, end in cdr_ranges:
            in_cdr = (resseq >= start) & (resseq <= end)
            for i in torch.where(in_cdr)[0]:
                coords = pos[i][mask[i].bool()]
                if coords.shape[0] > 0:
                    cdr_coords.append(coords)

    if not cdr_coords:
        return None
    return torch.cat(cdr_coords, dim=0)


def identify_epitope(structure, contact_cutoff=6.0, expand_cutoff=8.0):
    """
    Identify epitope residues in antigen.

    E = {i in antigen: min_{j in CDR} d(i,j) <= contact_cutoff}
    E+ = {k: d(CA_k, CA_i) <= expand_cutoff, i in E}

    Returns:
        epi_plus_mask: (L_ag,) bool tensor, True for residues in E+
        epi_mask: (L_ag,) bool tensor, True for residues in E (core epitope)
    """
    antigen = structure.get('antigen')
    if antigen is None:
        return None, None

    cdr_coords = _get_cdr_coords(structure)
    if cdr_coords is None:
        return None, None

    ag_pos = antigen['pos_heavyatom']
    ag_mask = antigen['mask_heavyatom']
    L_ag = ag_pos.shape[0]

    e_mask = torch.zeros(L_ag, dtype=torch.bool)
    for i in range(L_ag):
        ag_atoms = ag_pos[i][ag_mask[i].bool()]
        if ag_atoms.shape[0] == 0:
            continue
        dists = torch.cdist(ag_atoms.unsqueeze(0), cdr_coords.unsqueeze(0)).squeeze(0)
        if dists.min() <= contact_cutoff:
            e_mask[i] = True

    if e_mask.sum() == 0:
        e_mask[:] = True

    ag_ca = ag_pos[:, BBHeavyAtom.CA]
    e_plus_mask = e_mask.clone()
    e_indices = torch.where(e_mask)[0]

    for i in e_indices:
        ca_i = ag_ca[i]
        ca_dists = torch.norm(ag_ca - ca_i, dim=1)
        e_plus_mask = e_plus_mask | (ca_dists <= expand_cutoff)

    return e_plus_mask, e_mask


def smooth_perturbation(ag_ca_pos, epi_mask, target_rmsd, sigma=4.0, seed=None):
    """
    Generate smooth local coordinate perturbation on antigen Ca positions.

    Steps:
    1. Sample random 3D displacements for epitope residues
    2. Smooth with Gaussian kernel over Ca distances
    3. Normalize to target Ca RMSD

    Args:
        ag_ca_pos: (L_ag, 3) Ca positions
        epi_mask: (L_ag,) bool mask for epitope residues to perturb
        target_rmsd: target Ca RMSD in Angstroms
        sigma: Gaussian smoothing kernel width in Angstroms
        seed: random seed

    Returns:
        displacements: (L_ag, 3) displacement vectors for each antigen residue
    """
    if seed is not None:
        torch.manual_seed(seed)

    epi_indices = torch.where(epi_mask)[0]
    N_epi = len(epi_indices)

    if N_epi == 0:
        return torch.zeros_like(ag_ca_pos)

    device = ag_ca_pos.device
    raw_disp = torch.randn(N_epi, 3, device=device)

    epi_ca = ag_ca_pos[epi_indices]
    dists = torch.cdist(epi_ca, epi_ca)
    kernel = torch.exp(-dists ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)

    smoothed = kernel @ raw_disp

    current_rmsd = torch.sqrt((smoothed ** 2).sum(dim=1).mean())
    if current_rmsd > 1e-8:
        smoothed = smoothed * (target_rmsd / current_rmsd)

    displacements = torch.zeros(ag_ca_pos.shape[0], 3, device=device)
    displacements[epi_indices] = smoothed

    return displacements


def apply_perturbation_to_structure(structure, displacements):
    """
    Apply displacement vectors to all atoms of antigen residues.

    Args:
        structure: raw structure dict
        displacements: (L_ag, 3) per-residue displacement

    Returns:
        new_structure: deep copy with perturbed antigen positions
    """
    new_structure = copy.deepcopy(structure)
    antigen = new_structure.get('antigen')
    if antigen is None:
        return new_structure

    antigen['pos_heavyatom'] = antigen['pos_heavyatom'] + displacements.unsqueeze(1)
    return new_structure


def rigid_transform_structure(structure, rotation=None, translation=None, seed=None):
    """
    Apply random rigid transform to the entire complex.
    Control experiment 1: whole-complex rigid transform.

    Args:
        rotation: (3, 3) rotation matrix, if None random rotation
        translation: (3,) translation vector, if None random translation

    Returns:
        new_structure: deep copy with transformed coordinates
    """
    if seed is not None:
        torch.manual_seed(seed)

    if rotation is None:
        R = torch.randn(3, 3)
        U, _, V = torch.svd(R)
        rotation = U @ V.T
        if torch.det(rotation) < 0:
            rotation[:, -1] *= -1

    if translation is None:
        translation = torch.randn(3) * 10.0

    new_structure = copy.deepcopy(structure)

    for key in ['heavy', 'light', 'antigen']:
        chain = new_structure.get(key)
        if chain is not None and 'pos_heavyatom' in chain:
            pos = chain['pos_heavyatom']
            chain['pos_heavyatom'] = (pos @ rotation.T) + translation

    return new_structure


def identify_distant_patch(structure, epi_mask, min_distance=20.0):
    """
    Find a patch on antigen surface far from the epitope.
    Control experiment 2: distant patch perturbation.

    Args:
        structure: raw structure dict
        epi_mask: (L_ag,) bool mask for epitope residues
        min_distance: minimum Ca distance from epitope

    Returns:
        distant_mask: (L_ag,) bool mask for distant patch residues
    """
    antigen = structure.get('antigen')
    if antigen is None or epi_mask is None:
        return None

    ag_ca = antigen['pos_heavyatom'][:, BBHeavyAtom.CA]
    epi_ca = ag_ca[epi_mask]
    if epi_ca.shape[0] == 0:
        return None

    dist_to_epi = torch.cdist(ag_ca, epi_ca).min(dim=1)[0]

    distant_mask = dist_to_epi > min_distance

    if distant_mask.sum() < 3:
        distant_mask = ~epi_mask

    return distant_mask


def compute_ca_rmsd(coords1, coords2, mask=None):
    """
    Compute Ca RMSD between two coordinate sets.

    Args:
        coords1: (L, 3) or (L, N, 3)
        coords2: (L, 3) or (L, N, 3)
        mask: (L,) bool mask

    Returns:
        rmsd: float
    """
    if coords1.dim() == 3:
        coords1 = coords1[:, BBHeavyAtom.CA]
        coords2 = coords2[:, BBHeavyAtom.CA]

    diff = coords1 - coords2
    sq_diff = (diff ** 2).sum(dim=-1)
    if mask is not None:
        sq_diff = sq_diff[mask]
    return torch.sqrt(sq_diff.mean()).item()


def validate_perturbation(original_structure, perturbed_structure, epi_mask,
                          target_rmsd, rmsd_tolerance=0.1):
    """
    Validate the perturbed structure.

    Checks:
    1. Antigen sequence unchanged
    2. Epitope Ca RMSD within target range
    3. Antibody framework unchanged

    Returns:
        (ok, message) tuple
    """
    antigen_orig = original_structure.get('antigen')
    antigen_pert = perturbed_structure.get('antigen')

    if antigen_orig is None or antigen_pert is None:
        return False, "No antigen"

    if not torch.equal(antigen_orig['aa'], antigen_pert['aa']):
        return False, "Antigen sequence changed"

    ca_orig = antigen_orig['pos_heavyatom'][:, BBHeavyAtom.CA]
    ca_pert = antigen_pert['pos_heavyatom'][:, BBHeavyAtom.CA]

    if epi_mask is not None:
        epi_ca_diff = ca_pert[epi_mask] - ca_orig[epi_mask]
        epi_rmsd = torch.sqrt((epi_ca_diff ** 2).sum(dim=1).mean()).item()
        if abs(epi_rmsd - target_rmsd) > rmsd_tolerance:
            return False, f"Epitope RMSD {epi_rmsd:.3f} outside target range {target_rmsd}±{rmsd_tolerance}"
    else:
        epi_rmsd = 0.0

    for key in ['heavy', 'light']:
        chain_orig = original_structure.get(key)
        chain_pert = perturbed_structure.get(key)
        if chain_orig is not None and chain_pert is not None:
            if not torch.allclose(chain_orig['pos_heavyatom'], chain_pert['pos_heavyatom']):
                return False, f"Antibody {key} coordinates changed"

    return True, f"OK (epi_rmsd={epi_rmsd:.3f})"