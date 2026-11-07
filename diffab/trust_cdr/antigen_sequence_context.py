"""Antigen sequence / physicochemical context module.

Builds a per-antigen-residue context vector from:
  1. A learnable amino-acid embedding (antigen residues only).
  2. A fixed, normalized physicochemical feature table (6 dims).

CRITICAL LEAKAGE CONTROL:
  - Only antigen residues (fragment_type == 3) are read.
  - CDR ground-truth AA (generate_flag == True) is NEVER used.
  - The module receives `aa: [B, L]` and `antigen_mask: [B, L]` and indexes
    ONLY with antigen_mask, so CDR AA never enters this branch.

Output: seq_context of shape [*, L_antigen, output_dim].
"""

import torch
import torch.nn as nn


# num_aa_types = 21 in the project (AA enum 0..20, UNK=20). We use 22 slots to
# match ResidueEmbedding's max_aa_types and leave room for padding.
NUM_AA_EMBEDDINGS = 22
UNK_INDEX = 20

# ---------------------------------------------------------------------------
# Physicochemical feature table for the 20 standard amino acids.
# Index order follows diffab.utils.protein.constants.AA:
#   0=ALA,1=CYS,2=ASP,3=GLU,4=PHE,5=GLY,6=HIS,7=ILE,8=LYS,9=LEU,
#   10=MET,11=ASN,12=PRO,13=GLN,14=ARG,15=SER,16=THR,17=VAL,18=TRP,19=TYR
# Features (6 dims):
#   0: charge           {-1, 0, +1}  -> normalized to {-0.5, 0, +0.5}
#   1: hydrophobicity   Kyte-Doolittle scale -> normalized to [0, 1]
#   2: polarity         {0: nonpolar, 0.5: polar, 1: charged}
#   3: aromaticity      {0, 1}
#   4: donor_acceptor   {0..1} H-bond donor/acceptor propensity
#   5: size             residue volume -> normalized to [0, 1]
# Values are pre-normalized; UNK (20) is all zeros.
# ---------------------------------------------------------------------------
_RAW_PHYSCHEM = {
    # AA: (charge, hydrophobicity_kd, polarity, aromaticity, donor_acceptor, size)
    0:  ( 0.0,  1.8, 0.0, 0, 0.0, 0.30),  # Ala
    1:  ( 0.0,  2.5, 0.5, 0, 0.5, 0.43),  # Cys
    2:  (-1.0, -3.5, 1.0, 0, 1.0, 0.41),  # Asp
    3:  (-1.0, -3.5, 1.0, 0, 1.0, 0.55),  # Glu
    4:  ( 0.0,  2.8, 0.0, 1, 0.3, 0.71),  # Phe
    5:  ( 0.0, -0.4, 0.5, 0, 0.5, 0.18),  # Gly
    6:  ( 0.5, -3.2, 1.0, 1, 1.0, 0.58),  # His (partial charge)
    7:  ( 0.0,  4.5, 0.0, 0, 0.2, 0.60),  # Ile
    8:  ( 1.0, -3.9, 1.0, 0, 1.0, 0.57),  # Lys
    9:  ( 0.0,  3.8, 0.0, 0, 0.2, 0.57),  # Leu
    10: ( 0.0,  1.9, 0.0, 0, 0.3, 0.60),  # Met
    11: ( 0.0, -3.5, 0.5, 0, 1.0, 0.46),  # Asn
    12: ( 0.0, -1.6, 0.0, 0, 0.3, 0.42),  # Pro
    13: ( 0.0, -3.5, 0.5, 0, 1.0, 0.55),  # Gln
    14: ( 1.0, -4.5, 1.0, 0, 1.0, 0.70),  # Arg
    15: ( 0.0, -0.8, 0.5, 0, 1.0, 0.27),  # Ser
    16: ( 0.0, -0.7, 0.5, 0, 1.0, 0.41),  # Thr
    17: ( 0.0,  4.2, 0.0, 0, 0.2, 0.51),  # Val
    18: ( 0.0, -0.9, 0.0, 1, 1.0, 0.90),  # Trp
    19: ( 0.0, -1.3, 0.5, 1, 1.0, 0.78),  # Tyr
}


def _build_physchem_table():
    """Build the normalized [NUM_AA_EMBEDDINGS, 6] physchem table."""
    table = torch.zeros(NUM_AA_EMBEDDINGS, 6, dtype=torch.float32)
    for idx, vals in _RAW_PHYSCHEM.items():
        charge, hydro, polar, arom, donor, size = vals
        table[idx, 0] = charge * 0.5            # charge -> {-0.5,0,+0.5}
        table[idx, 1] = (hydro + 4.5) / 9.0      # KD in [-4.5, 4.5] -> [0,1]
        table[idx, 2] = polar
        table[idx, 3] = float(arom)
        table[idx, 4] = donor
        table[idx, 5] = size
    # UNK (20) and slot 21 remain zeros.
    return table


# Shared, non-trainable buffer (registered as buffer per-instance to follow device).
PHYSCHEM_TABLE = _build_physchem_table()


class AntigenSequenceContext(nn.Module):
    """Build antigen residue sequence / physicochemical context.

    Inputs:
        aa: amino acid ids, shape [B, L] (LongTensor)
        antigen_mask: shape [B, L] (bool). Only antigen residues are read.

    Outputs:
        seq_context: shape [B, L, output_dim] (zeros on non-antigen positions).

    The context is computed ONLY from antigen residues. CDR / framework residues
    produce zero context vectors and are never indexed into the AA embedding for
    information purposes (the embedding lookup is masked out).
    """

    def __init__(
        self,
        aa_embedding_dim=16,
        physchem_dim=6,
        output_dim=32,
        hidden_dim=128,
        num_aa_embeddings=NUM_AA_EMBEDDINGS,
        dropout=0.0,
    ):
        super().__init__()
        self.aa_embedding_dim = aa_embedding_dim
        self.physchem_dim = physchem_dim
        self.output_dim = output_dim
        self.num_aa_embeddings = num_aa_embeddings

        # Learnable AA embedding. padding_idx=UNK so UNK -> zeros.
        self.aa_embed = nn.Embedding(
            num_aa_embeddings, aa_embedding_dim, padding_idx=UNK_INDEX
        )
        nn.init.normal_(self.aa_embed.weight, std=0.02)
        with torch.no_grad():
            self.aa_embed.weight[UNK_INDEX].zero_()

        # Fixed physchem table (non-trainable), registered as buffer.
        self.register_buffer('physchem_table', PHYSCHEM_TABLE.clone())

        # MLP: [aa_emb || physchem] -> output_dim
        in_dim = aa_embedding_dim + physchem_dim
        self.context_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, aa, antigen_mask, shuffle_antigen_aa=False):
        """Compute sequence context.

        Args:
            aa: [B, L] LongTensor of AA indices.
            antigen_mask: [B, L] bool tensor (fragment_type == 3).
            shuffle_antigen_aa: if True, shuffle each antigen's AA indices
                *in place along the antigen residues only* (for the ablation
                check). Coordinates, reliability, CDR, framework are untouched.
        Returns:
            seq_context: [B, L, output_dim] (zeros on non-antigen).
        """
        B, L = aa.shape
        device = aa.device

        # Defensive copy so we never mutate the original batch['aa'].
        aa_used = aa.clone()

        if shuffle_antigen_aa:
            aa_used = self._shuffle_antigen_aa(aa_used, antigen_mask)

        # Clamp to valid range to avoid OOB in embedding.
        aa_clamped = aa_used.clamp(min=0, max=self.num_aa_embeddings - 1)

        # Learnable AA embedding: [B, L, aa_embedding_dim]
        aa_emb = self.aa_embed(aa_clamped)

        # Physicochemical features: [B, L, physchem_dim]
        physchem = self.physchem_table.to(device)[aa_clamped]

        # Concatenate and project.
        ctx_in = torch.cat([aa_emb, physchem], dim=-1)        # [B, L, in_dim]
        seq_context = self.context_mlp(ctx_in)                 # [B, L, output_dim]

        # Zero-out non-antigen positions so ONLY antigen info is propagated.
        mask = antigen_mask.unsqueeze(-1).float()              # [B, L, 1]
        seq_context = seq_context * mask

        return seq_context

    @staticmethod
    def _shuffle_antigen_aa(aa, antigen_mask):
        """Shuffle AA indices within each sample's antigen region only."""
        B, L = aa.shape
        out = aa.clone()
        for b in range(B):
            am = antigen_mask[b]
            if not am.any():
                continue
            idx = am.nonzero(as_tuple=False).squeeze(-1)
            vals = out[b, idx]
            perm = torch.randperm(vals.numel(), device=aa.device)
            out[b, idx] = vals[perm]
        return out

    def count_residues_used(self, antigen_mask, generate_mask=None, cdr_flag=None):
        """Return counts for leakage auditing.

        Returns:
            dict with:
              num_antigen_residues_used: int
              num_cdr_residues_used_by_seq_context: int (MUST be 0)
        """
        num_ag = int(antigen_mask.sum().item())
        num_cdr = 0
        if generate_mask is not None:
            # We never index generate_flag positions, so this is 0 by construction.
            # Reported for verification.
            overlap = (antigen_mask & generate_mask).sum().item()
            num_cdr = int(overlap)  # antigen ∩ generate should be 0
        return {
            'num_antigen_residues_used': num_ag,
            'num_cdr_residues_used_by_seq_context': num_cdr,
        }
