"""
data/dataset.py
---------------
Multi-modality dataset builder for BEP perturbation prediction.

Per-locus features extracted:
  1. DNA sequence embedding  (from frozen Enformer backbone)
  2. ATAC-seq baseline       (binned bedgraph, GFP condition)
  3. CG methylation          (allc format, GFP condition)
  4. dCas9 ChIP signal       (binned bedgraph)
  5. 25 histone marks        (GFP baseline, mean of 2 reps)

Targets (treatment BEP vs GFP control):
  - Δhistone log2FC (25,)
  - ΔATAC    log2FC scalar
  - Δmeth    scalar
  - ΔRNA     log2FC per gene (linked to nearest peak)
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)

EPS = 1e-6


# ────────────────────────────────────────────────────────────────────────────
# Bedgraph helpers
# ────────────────────────────────────────────────────────────────────────────

def load_bedgraph(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["chrom", "start", "end", "value"])
    return pd.read_csv(
        path, sep="\t", header=None,
        names=["chrom", "start", "end", "value"],
        dtype={"chrom": str, "start": int, "end": int, "value": float},
        comment="#",
    )


def query_signal(
    df: pd.DataFrame, chrom: str, center: int,
    window: int, n_bins: int,
) -> np.ndarray:
    """Extract mean signal in bins around center."""
    half     = window // 2
    w_start  = center - half
    w_end    = center + half
    bin_size = window / n_bins
    bins     = np.zeros(n_bins, dtype=np.float32)

    if df.empty:
        return bins

    mask = (
        (df["chrom"] == chrom)
        & (df["end"]   > w_start)
        & (df["start"] < w_end)
    )
    region = df.loc[mask]
    if region.empty:
        return bins

    for _, row in region.iterrows():
        r_s = max(row["start"], w_start)
        r_e = min(row["end"],   w_end)
        b_s = int((r_s - w_start) / bin_size)
        b_e = int(np.ceil((r_e - w_start) / bin_size))
        b_e = min(b_e, n_bins)
        if b_s < b_e:
            bins[b_s:b_e] += row["value"] * (r_e - r_s) / (
                (b_e - b_s) * bin_size
            )
    return bins


class BedgraphCache:
    """Load-once, reuse many bedgraph file cache."""
    def __init__(self):
        self._d: Dict[str, pd.DataFrame] = {}

    def get(self, path: str) -> pd.DataFrame:
        if path not in self._d:
            self._d[path] = load_bedgraph(path)
        return self._d[path]


# ────────────────────────────────────────────────────────────────────────────
# Allc methylation reader
# ────────────────────────────────────────────────────────────────────────────

def load_allc(path: str, min_cov: int = 1) -> pd.DataFrame:
    """Load allc, keep CG context only."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["chrom", "pos0", "meth_frac"])
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["chrom", "pos", "strand", "mc_class", "mc_count", "cov", "is_meth"],
        dtype={"chrom": str, "pos": int, "strand": str,
               "mc_class": str, "mc_count": int, "cov": int},
        comment="#",
    )
    df = df[df["mc_class"].str.startswith("CG") & (df["cov"] >= min_cov)].copy()
    df["meth_frac"] = df["mc_count"] / df["cov"]
    df["pos0"] = df["pos"] - 1   # 0-based
    return df[["chrom", "pos0", "meth_frac", "cov"]]


def query_meth(
    df: pd.DataFrame, chrom: str, center: int,
    window: int, n_bins: int,
) -> Tuple[np.ndarray, float]:
    """Returns (binned_meth_level, global_mean)."""
    half    = window // 2
    w_start = center - half
    w_end   = center + half
    bin_size = window / n_bins
    sums  = np.zeros(n_bins, dtype=np.float64)
    cnts  = np.zeros(n_bins, dtype=np.int32)

    if not df.empty:
        sub = df[
            (df["chrom"] == chrom)
            & (df["pos0"] >= w_start)
            & (df["pos0"] < w_end)
        ]
        for _, row in sub.iterrows():
            b = int((row["pos0"] - w_start) / bin_size)
            b = min(b, n_bins - 1)
            sums[b] += row["meth_frac"]
            cnts[b] += 1

    meth_bins  = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.5).astype(np.float32)
    global_val = float(sums.sum() / max(cnts.sum(), 1)) if cnts.sum() > 0 else 0.5
    return meth_bins, global_val


# ────────────────────────────────────────────────────────────────────────────
# RNA-seq loader (gene-level TPM)
# ────────────────────────────────────────────────────────────────────────────

def load_rna(path: str) -> pd.Series:
    """Load gene-level TPM. Returns pd.Series indexed by gene_id."""
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, sep="\t", dtype={"gene_id": str, "TPM": float})
    if "gene_id" not in df.columns or "TPM" not in df.columns:
        # try common column names
        df.columns = ["gene_id", "TPM"][:len(df.columns)]
    return df.set_index("gene_id")["TPM"]


def compute_rna_log2fc(
    ctrl_tpm: pd.Series, treat_tpm: pd.Series,
) -> pd.Series:
    """log2((treat + 1) / (ctrl + 1)) per gene."""
    genes = ctrl_tpm.index.union(treat_tpm.index)
    ctrl  = ctrl_tpm.reindex(genes, fill_value=0.0) + 1.0
    treat = treat_tpm.reindex(genes, fill_value=0.0) + 1.0
    return np.log2(treat / ctrl)


# ────────────────────────────────────────────────────────────────────────────
# Peak → nearest TSS mapper (for RNA linkage)
# ────────────────────────────────────────────────────────────────────────────

def build_peak_gene_map(
    peaks_df: pd.DataFrame, gtf_path: str, max_dist: int = 50_000,
) -> pd.Series:
    """
    Returns pd.Series: peak_index → gene_id of nearest TSS within max_dist.
    Uses simple distance approach; replace with ABC model for publication.
    """
    if not os.path.exists(gtf_path):
        logger.warning("GTF not found; peak→gene mapping skipped.")
        return pd.Series(dtype=str)

    # Parse TSS from GTF (gene lines only)
    tss_rows = []
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            chrom = cols[0]
            start = int(cols[3]) - 1
            end   = int(cols[4])
            strand = cols[6]
            tss = start if strand == "+" else end
            # extract gene_id
            info = cols[8]
            gid = ""
            for part in info.split(";"):
                part = part.strip()
                if part.startswith("gene_id"):
                    gid = part.split('"')[1]
                    break
            if gid:
                tss_rows.append({"chrom": chrom, "tss": tss, "gene_id": gid})

    tss_df = pd.DataFrame(tss_rows)
    if tss_df.empty:
        return pd.Series(dtype=str)

    # Group by chrom for speed
    tss_by_chrom = {c: g for c, g in tss_df.groupby("chrom")}
    mapping = {}

    for idx, row in peaks_df.iterrows():
        chrom  = row["chrom"]
        center = int(row["center"])
        if chrom not in tss_by_chrom:
            continue
        sub  = tss_by_chrom[chrom]
        dist = np.abs(sub["tss"].values - center)
        best = dist.argmin()
        if dist[best] <= max_dist:
            mapping[idx] = sub.iloc[best]["gene_id"]

    return pd.Series(mapping)


# ────────────────────────────────────────────────────────────────────────────
# Main dataset builder
# ────────────────────────────────────────────────────────────────────────────

class BEPDatasetBuilder:
    """
    Builds per-locus feature arrays for all BEPs and modalities.
    Output dict keys match EpiDataset below.
    """

    def __init__(self, cfg: dict, cell_type: str = "hek293t"):
        self.cfg        = cfg
        self.cell_type  = cell_type.lower()
        self.dc         = cfg["data"][cell_type.lower()]
        self.marks      = cfg["histone_marks"]
        self.beps       = [b["name"] for b in cfg["beps"]]
        self.ctrl       = cfg["control_bep"]
        self.window     = cfg["data"]["peak_window"]
        self.n_bins     = cfg["data"]["n_signal_bins"]
        self.cache      = BedgraphCache()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _fmt(self, template: str, **kw) -> str:
        root = self.dc["root"]
        return template.format(root=root, **kw)

    def _bg_path(self, bep: str, mark: str, rep: int) -> str:
        return self._fmt(self.dc["histone_template"], bep=bep, mark=mark, rep=rep)

    def _atac_path(self, bep: str, rep: int) -> str:
        return self._fmt(self.dc["atac_template"], bep=bep, rep=rep)

    def _meth_path(self, bep: str) -> str:
        return self._fmt(self.dc["methyl_template"], bep=bep)

    def _rna_path(self, bep: str, rep: int) -> str:
        return self._fmt(self.dc["rna_template"], bep=bep, rep=rep)

    def _rep_mean_hist(self, bep: str, mark: str,
                       chrom: str, center: int) -> float:
        sigs = []
        for rep in (1, 2):
            p = self._bg_path(bep, mark, rep)
            df = self.cache.get(p)
            sigs.append(query_signal(df, chrom, center, self.window, self.n_bins).mean())
        return float(np.mean(sigs))

    def _rep_mean_atac(self, bep: str, chrom: str, center: int) -> float:
        sigs = []
        for rep in (1, 2):
            p = self._atac_path(bep, rep)
            df = self.cache.get(p)
            sigs.append(query_signal(df, chrom, center, self.window, self.n_bins).mean())
        return float(np.mean(sigs))

    def _rep_mean_atac_bins(self, bep: str, chrom: str, center: int) -> np.ndarray:
        sigs = []
        for rep in (1, 2):
            p = self._atac_path(bep, rep)
            df = self.cache.get(p)
            sigs.append(query_signal(df, chrom, center, self.window, self.n_bins))
        return np.mean(sigs, axis=0).astype(np.float32)

    # ── peak loading ────────────────────────────────────────────────────────

    def _load_peaks(self) -> pd.DataFrame:
        path = self._fmt(self.dc["dcas9_peaks"])
        peaks = pd.read_csv(
            path, sep="\t", header=None, usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
            dtype={"chrom": str, "start": int, "end": int},
            comment="#",
        )
        peaks["center"] = (peaks["start"] + peaks["end"]) // 2
        return peaks

    # ── RNA loading ──────────────────────────────────────────────────────────

    def _load_rna_both_reps(self, bep: str) -> pd.Series:
        """Average TPM across 2 reps."""
        s1 = load_rna(self._rna_path(bep, 1))
        s2 = load_rna(self._rna_path(bep, 2))
        genes = s1.index.union(s2.index)
        v1 = s1.reindex(genes, fill_value=0.0)
        v2 = s2.reindex(genes, fill_value=0.0)
        return (v1 + v2) / 2.0

    # ── main build ───────────────────────────────────────────────────────────

    def build(self, gtf_path: Optional[str] = None) -> dict:
        """
        Returns dict of arrays, shape (N_loci, ...).
        Call build() once; cache to disk with pickle.
        """
        peaks     = self._load_peaks()
        dcas9_df  = self.cache.get(self._fmt(self.dc["dcas9_bg"]))
        ctrl_meth = load_allc(self._meth_path(self.ctrl))

        # RNA: load ctrl & all treatment BEPs
        ctrl_rna = self._load_rna_both_reps(self.ctrl)
        treat_beps = [b for b in self.beps if b != self.ctrl]
        treat_rna  = {b: self._load_rna_both_reps(b) for b in treat_beps}

        # Peak → gene map
        peak_gene = build_peak_gene_map(
            peaks, gtf_path or self.cfg["data"]["genome"]["annotation_gtf"]
        )

        N       = len(peaks)
        n_marks = len(self.marks)
        n_treat = len(treat_beps)
        thr     = self.cfg["model"]["output"]["log2fc_threshold"]

        # allocate
        dcas9_arr   = np.zeros((N, self.n_bins),         dtype=np.float32)
        atac_bins   = np.zeros((N, self.n_bins),         dtype=np.float32)
        meth_bins   = np.zeros((N, self.n_bins),         dtype=np.float32)
        glob_meth   = np.zeros(N,                        dtype=np.float32)
        hist_ctrl   = np.zeros((N, n_marks),             dtype=np.float32)
        atac_ctrl   = np.zeros(N,                        dtype=np.float32)

        # targets
        hist_log2fc = np.zeros((N, n_treat, n_marks),    dtype=np.float32)
        hist_cls    = np.ones( (N, n_treat, n_marks),    dtype=np.int64)
        atac_log2fc = np.zeros((N, n_treat),             dtype=np.float32)
        meth_delta  = np.zeros((N, n_treat),             dtype=np.float32)
        rna_log2fc  = np.zeros((N, n_treat),             dtype=np.float32)

        keep = np.ones(N, dtype=bool)
        locus_ids: List[Tuple[str, int]] = []
        min_sig = self.cfg["data"]["min_dcas9_signal"]

        logger.info(f"Building {self.cell_type.upper()} dataset: {N} loci …")

        for i, row in peaks.iterrows():
            chrom, center = row["chrom"], int(row["center"])
            locus_ids.append((chrom, center))

            # dCas9 signal
            dcas9_arr[i] = query_signal(
                dcas9_df, chrom, center, self.window, self.n_bins
            )
            if dcas9_arr[i].mean() < min_sig:
                keep[i] = False
                continue

            # ATAC baseline (ctrl)
            atac_bins[i] = self._rep_mean_atac_bins(self.ctrl, chrom, center)
            atac_ctrl[i] = atac_bins[i].mean()

            # Methylation baseline (ctrl)
            meth_bins[i], glob_meth[i] = query_meth(
                ctrl_meth, chrom, center, self.window, self.n_bins
            )

            # Histone baseline (ctrl, 25 marks)
            for j, mark in enumerate(self.marks):
                hist_ctrl[i, j] = self._rep_mean_hist(self.ctrl, mark, chrom, center)

            # Targets per treatment BEP
            for t, bep in enumerate(treat_beps):
                # Histone log2FC
                for j, mark in enumerate(self.marks):
                    v_t = self._rep_mean_hist(bep, mark, chrom, center)
                    v_c = hist_ctrl[i, j]
                    fc  = np.log2((v_t + EPS) / (v_c + EPS))
                    hist_log2fc[i, t, j] = fc
                    if   fc >  thr: hist_cls[i, t, j] = 2
                    elif fc < -thr: hist_cls[i, t, j] = 0

                # ATAC log2FC
                atac_t = self._rep_mean_atac(bep, chrom, center)
                atac_log2fc[i, t] = np.log2((atac_t + EPS) / (atac_ctrl[i] + EPS))

                # Meth delta
                meth_t_df = load_allc(self._meth_path(bep))
                _, glob_t  = query_meth(meth_t_df, chrom, center, self.window, self.n_bins)
                meth_delta[i, t] = glob_t - glob_meth[i]

                # RNA log2FC (nearest TSS)
                gene = peak_gene.get(i)
                if gene and gene in ctrl_rna.index and gene in treat_rna[bep].index:
                    fc_rna = np.log2(
                        (treat_rna[bep][gene] + 1.0) / (ctrl_rna[gene] + 1.0)
                    )
                    rna_log2fc[i, t] = float(fc_rna)

            if (i + 1) % 500 == 0:
                logger.info(f"  {i+1}/{N} loci processed")

        n_kept = keep.sum()
        logger.info(f"Loci after QC: {n_kept}/{N}")

        return dict(
            dcas9_signal = dcas9_arr[keep],
            atac_bins    = atac_bins[keep],
            meth_bins    = meth_bins[keep],
            glob_meth    = glob_meth[keep],
            hist_ctrl    = hist_ctrl[keep],
            atac_ctrl    = atac_ctrl[keep],
            hist_log2fc  = hist_log2fc[keep],
            hist_cls     = hist_cls[keep],
            atac_log2fc  = atac_log2fc[keep],
            meth_delta   = meth_delta[keep],
            rna_log2fc   = rna_log2fc[keep],
            locus_ids    = [locus_ids[i] for i in range(N) if keep[i]],
            treat_beps   = treat_beps,
            marks        = self.marks,
        )


# ────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ────────────────────────────────────────────────────────────────────────────

class BEPDataset(Dataset):
    """
    One sample = one locus × one treatment BEP.
    bep_id: integer index into full bep list (including GFP=0).
    """

    def __init__(self, data: dict, bep_to_idx: dict, bep_role_idx: dict):
        self.data = data
        treat_beps = data["treat_beps"]
        n_loci     = data["dcas9_signal"].shape[0]
        self.index = [(li, ti)
                      for li in range(n_loci)
                      for ti in range(len(treat_beps))]
        self.bep_ids  = [bep_to_idx[b]    for b in treat_beps]
        self.role_ids = [bep_role_idx[b]  for b in treat_beps]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        li, ti = self.index[idx]
        d = self.data
        ds = d["dcas9_signal"][li]
        return {
            # ── inputs ────────────────────────────────────────────
            "dcas9_signal" : torch.tensor(ds,                    dtype=torch.float32),
            "dcas9_scalar" : torch.tensor([ds.mean(), ds.max()], dtype=torch.float32),
            "atac_bins"    : torch.tensor(d["atac_bins"][li],    dtype=torch.float32),
            "atac_scalar"  : torch.tensor([d["atac_ctrl"][li]],  dtype=torch.float32),
            "meth_bins"    : torch.tensor(d["meth_bins"][li],    dtype=torch.float32),
            "glob_meth"    : torch.tensor([d["glob_meth"][li]],  dtype=torch.float32),
            "hist_ctrl"    : torch.tensor(d["hist_ctrl"][li],    dtype=torch.float32),
            "bep_id"       : torch.tensor(self.bep_ids[ti],      dtype=torch.long),
            "role_id"      : torch.tensor(self.role_ids[ti],     dtype=torch.long),
            # ── targets ───────────────────────────────────────────
            "hist_log2fc"  : torch.tensor(d["hist_log2fc"][li, ti],  dtype=torch.float32),
            "hist_cls"     : torch.tensor(d["hist_cls"][li, ti],     dtype=torch.long),
            "atac_log2fc"  : torch.tensor([d["atac_log2fc"][li, ti]], dtype=torch.float32),
            "meth_delta"   : torch.tensor([d["meth_delta"][li, ti]],  dtype=torch.float32),
            "rna_log2fc"   : torch.tensor([d["rna_log2fc"][li, ti]],  dtype=torch.float32),
        }


def build_bep_role_idx(cfg: dict) -> Tuple[dict, dict]:
    """Returns (bep_to_idx, bep_role_idx). role: control=0, repressor=1, activator=2."""
    role_map = {"control": 0, "repressor": 1, "activator": 2}
    bep_to_idx   = {b["name"]: b["index"] for b in cfg["beps"]}
    bep_role_idx = {b["name"]: role_map[b["role"]] for b in cfg["beps"]}
    return bep_to_idx, bep_role_idx


def build_dataloaders(
    cfg: dict, data: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    bep_to_idx, bep_role_idx = build_bep_role_idx(cfg)
    dataset = BEPDataset(data, bep_to_idx, bep_role_idx)
    N   = len(dataset)
    gen = torch.Generator().manual_seed(cfg["project"]["seed"])
    n_tr = int(N * cfg["data"]["train_ratio"])
    n_va = int(N * cfg["data"]["val_ratio"])
    n_te = N - n_tr - n_va
    tr, va, te = random_split(dataset, [n_tr, n_va, n_te], generator=gen)
    bs = cfg["training"]["stage2"]["batch_size"]
    kw = dict(num_workers=4, pin_memory=True)
    return (
        DataLoader(tr, batch_size=bs, shuffle=True,  **kw),
        DataLoader(va, batch_size=bs, shuffle=False, **kw),
        DataLoader(te, batch_size=bs, shuffle=False, **kw),
    )
