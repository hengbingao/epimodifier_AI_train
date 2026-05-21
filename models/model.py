"""
models/model.py
---------------
Three-stage model for BEP epigenome perturbation prediction.

Stage 1 — Enformer backbone (frozen):
  DNA sequence → 3072-dim per-bin embedding (196kbp context)

Stage 2 — BEP perturbation head (trained on HEK293T):
  Inputs:  Enformer embeddings + dCas9 ChIP + ATAC baseline
           + meth baseline + 25 histone marks baseline
           + BEP identity (FiLM) + role (activator/repressor)
  Outputs: Δhistone(25) · ΔATAC · Δmeth · ΔRNA

Stage 3 — Cell-type adapter (K562 transfer):
  Adapts cell embedding layer to K562 ATAC profile.
  Backbone + BEP head frozen; only adapter + output heads update.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────
# Optional: try to import enformer-pytorch for pretrained backbone
# Falls back to a lightweight stub if not installed
# ─────────────────────────────────────────────────────────────────
try:
    from enformer_pytorch import Enformer as _Enformer
    ENFORMER_AVAILABLE = True
except ImportError:
    ENFORMER_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
# Enformer backbone wrapper
# ────────────────────────────────────────────────────────────────────────────

class EnformerBackbone(nn.Module):
    """
    Wraps enformer-pytorch and returns the trunk embedding
    (before the final output head) — shape (B, n_bins, 3072).

    If enformer-pytorch is not installed, uses a lightweight
    CNN+Transformer stub that produces the same output shape.
    Install the real model with:
        pip install enformer-pytorch
    Then load pretrained weights:
        model = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
    """

    def __init__(self, cfg: dict, use_pretrained: bool = True):
        super().__init__()
        self.seq_output_dim = cfg["backbone"]["seq_output_dim"]   # 3072
        self.n_bins         = cfg["data"]["n_bins"]               # 1024

        if ENFORMER_AVAILABLE and use_pretrained:
            hub_name = cfg["backbone"]["enformer_pytorch_hub"]
            try:
                self.enformer = _Enformer.from_pretrained(hub_name)
                self._mode = "enformer"
                print(f"[backbone] Loaded pretrained Enformer from {hub_name}")
            except Exception as e:
                print(f"[backbone] Enformer load failed ({e}), using stub.")
                self._mode = "stub"
                self.enformer = self._build_stub()
        else:
            self._mode = "stub"
            self.enformer = self._build_stub()

    def _build_stub(self) -> nn.Module:
        """
        Lightweight stub: 3-layer dilated CNN + 2-layer Transformer.
        Output: (B, n_bins, seq_output_dim).
        Useful for development without downloading the full model.
        """
        n_bins = self.n_bins
        d = self.seq_output_dim

        class Stub(nn.Module):
            def __init__(self, n_bins, d):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(4, 256, 11, padding=5), nn.GELU(),
                    nn.Conv1d(256, 512, 11, padding=5, dilation=2), nn.GELU(),
                    nn.AdaptiveAvgPool1d(n_bins),
                    nn.Conv1d(512, d, 1),
                )
                enc = nn.TransformerEncoderLayer(d, 8, d*2, dropout=0.1,
                                                 batch_first=True, norm_first=True)
                self.tr = nn.TransformerEncoder(enc, num_layers=2)

            def forward(self, x):
                # x: (B, 4, L)
                z = self.conv(x)            # (B, d, n_bins)
                z = z.permute(0, 2, 1)      # (B, n_bins, d)
                return self.tr(z)           # (B, n_bins, d)

        return Stub(n_bins, d)

    def forward(self, seq_onehot: torch.Tensor) -> torch.Tensor:
        """
        seq_onehot : (B, 4, L)  one-hot DNA sequence
        Returns    : (B, n_bins, seq_output_dim)
        """
        if self._mode == "enformer":
            # enformer-pytorch returns dict; grab 'human' trunk embedding
            out = self.enformer(seq_onehot.permute(0, 2, 1),
                                return_only_embeddings=True)
            return out   # (B, n_bins, 3072)
        else:
            return self.enformer(seq_onehot)   # (B, n_bins, d)


# ────────────────────────────────────────────────────────────────────────────
# Shared utility layers
# ────────────────────────────────────────────────────────────────────────────

def mlp(dims: List[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class CNN1DEncoder(nn.Module):
    """Encode a 1-D signal profile → fixed-size vector."""
    def __init__(self, in_ch: int = 1, n_bins: int = 100, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, padding=4),    nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 15, padding=7),  nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.proj = mlp([128 * 8, 512, out_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_bins)
        return self.proj(self.net(x.unsqueeze(1)).flatten(1))


class HistoneTransformerEncoder(nn.Module):
    """Self-attention over 25 histone marks → (B, out_dim) + (B, 25, d) tokens."""
    def __init__(self, n_marks: int, d: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 256,
                 dropout: float = 0.1, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.pos  = nn.Embedding(n_marks, d)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, dim_ff, dropout,
                                         batch_first=True, norm_first=True)
        self.tr  = nn.TransformerEncoder(enc, num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n = x.shape
        t = self.proj(x.unsqueeze(-1)) + self.pos(torch.arange(n, device=x.device))
        t = self.tr(torch.cat([self.cls.expand(B, -1, -1), t], 1))
        return self.out(t[:, 0]), t[:, 1:]   # (B, out_dim), (B, n, d)


class MethEncoder(nn.Module):
    """Encode binned methylation (level + coverage) → (B, out_dim)."""
    def __init__(self, n_bins: int = 100, d: int = 64, nhead: int = 4,
                 num_layers: int = 2, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(1, d)
        self.pos  = nn.Embedding(n_bins, d)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, nhead, d * 2, batch_first=True, norm_first=True)
        self.tr  = nn.TransformerEncoder(enc, num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))

    def forward(self, meth: torch.Tensor) -> torch.Tensor:
        B, n = meth.shape
        t = self.proj(meth.unsqueeze(-1)) + self.pos(torch.arange(n, device=meth.device))
        t = self.tr(torch.cat([self.cls.expand(B, -1, -1), t], 1))
        return self.out(t[:, 0])


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for BEP identity conditioning."""
    def __init__(self, n_bep: int, n_role: int, d: int):
        super().__init__()
        # BEP embedding
        self.bep_emb  = nn.Embedding(n_bep, d * 2)
        # Role embedding (activator / repressor / control)
        self.role_emb = nn.Embedding(n_role, d * 2)
        nn.init.ones_(self.bep_emb.weight[:, :d])
        nn.init.zeros_(self.bep_emb.weight[:, d:])
        nn.init.ones_(self.role_emb.weight[:, :d])
        nn.init.zeros_(self.role_emb.weight[:, d:])
        # Add small random noise so different BEP indices differ at init
        with torch.no_grad():
            self.bep_emb.weight.add_(torch.randn_like(self.bep_emb.weight) * 0.02)
            self.role_emb.weight.add_(torch.randn_like(self.role_emb.weight) * 0.02)

    def forward(self, z: torch.Tensor,
                bep_id: torch.Tensor,
                role_id: torch.Tensor) -> torch.Tensor:
        gb_bep  = self.bep_emb(bep_id)
        gb_role = self.role_emb(role_id)
        gb = gb_bep + gb_role
        gamma, beta = gb.chunk(2, -1)
        return gamma * z + beta


class GatedCrossAttn(nn.Module):
    """dCas9 feature as Query; other modalities as KV."""
    def __init__(self, d: int = 256, nhead: int = 4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, nhead, batch_first=True, dropout=0.1)
        self.gate  = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.norm  = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, query: torch.Tensor,
                modalities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gates = torch.sigmoid(self.gate(modalities))  # (B, M, 1)
        kv    = modalities * gates
        q     = query.unsqueeze(1)
        out, w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        out = self.norm(out + q)
        out = self.norm2(out + self.ff(out))
        return out.squeeze(1), w.squeeze(1)   # (B, d), (B, M)


# ────────────────────────────────────────────────────────────────────────────
# Full model
# ────────────────────────────────────────────────────────────────────────────

MODALITY_NAMES = ["dcas9", "atac", "methylation", "histone", "seq_summary"]


class BEPPerturbationModel(nn.Module):
    """
    Full three-stage model.

    Inputs (per sample)
    -------------------
    seq_onehot   : (B, 4, L)       DNA sequence (L = 131072 for Enformer)
    dcas9_signal : (B, n_bins)     binned dCas9 ChIP coverage
    dcas9_scalar : (B, 2)          mean + max of dCas9 signal
    atac_bins    : (B, n_bins)     GFP baseline ATAC-seq
    atac_scalar  : (B, 1)          mean ATAC
    meth_bins    : (B, n_bins)     GFP baseline CG methylation
    glob_meth    : (B, 1)
    hist_ctrl    : (B, n_marks)    GFP baseline histone marks
    bep_id       : (B,)            BEP index
    role_id      : (B,)            role (0=ctrl, 1=repressor, 2=activator)

    Outputs
    -------
    hist_log2fc  : (B, n_marks)
    hist_cls     : (B, n_marks, 3)
    atac_log2fc  : (B, 1)
    meth_delta   : (B, 1)
    rna_log2fc   : (B, 1)
    modal_attn   : (B, 5)          interpretable cross-modal attention
    """

    MODALITY_NAMES = MODALITY_NAMES

    def __init__(self, cfg: dict, use_pretrained_backbone: bool = True):
        super().__init__()
        mc      = cfg["model"]
        d       = mc["d_model"]
        n_marks = len(cfg["histone_marks"])
        n_bep   = len(cfg["beps"])
        n_bins  = cfg["data"]["n_signal_bins"]
        film_d  = mc["bep_conditioning"]["film_dim"]
        oc      = mc["output"]

        # ── Backbone (Stage 1, frozen) ────────────────────────────────────
        self.backbone = EnformerBackbone(cfg, use_pretrained=use_pretrained_backbone)
        # Pool Enformer trunk to d
        seq_d = cfg["backbone"]["seq_output_dim"]
        self.seq_pool = nn.Sequential(
            nn.Linear(seq_d, 512), nn.GELU(), nn.Linear(512, d)
        )

        # ── Signal encoders (Stage 2) ─────────────────────────────────────
        self.dcas9_enc = CNN1DEncoder(1, n_bins, d)
        self.atac_enc  = CNN1DEncoder(1, n_bins, d)
        self.meth_enc  = MethEncoder(n_bins, d=mc["meth_encoder"]["d_model"],
                                     nhead=mc["meth_encoder"]["nhead"],
                                     num_layers=mc["meth_encoder"]["num_layers"],
                                     out_dim=d)
        self.hist_enc  = HistoneTransformerEncoder(
            n_marks, d=mc["histone_encoder"]["d_model"],
            nhead=mc["histone_encoder"]["nhead"],
            num_layers=mc["histone_encoder"]["num_layers"],
            dim_ff=mc["histone_encoder"]["dim_feedforward"],
            dropout=mc["histone_encoder"]["dropout"],
            out_dim=d,
        )
        # Scalar auxiliary projections
        self.dcas9_scalar_proj = mlp([2, 64, d])
        self.atac_scalar_proj  = mlp([1, 64, d])
        self.meth_scalar_proj  = mlp([1, 64, d])

        # ── Fusion ───────────────────────────────────────────────────────
        self.cross_attn = GatedCrossAttn(d=d, nhead=mc["cross_attn"]["nhead"])

        # Concat all streams (5 modalities + dcas9 CNN = 6×d) → film_d
        self.pre_film = nn.Sequential(
            nn.Linear(d * 6, film_d), nn.GELU(), nn.LayerNorm(film_d)
        )
        self.film = FiLMLayer(
            n_bep=n_bep,
            n_role=3,
            d=film_d,
        )

        # ── Cell-type adapter (Stage 3) ───────────────────────────────────
        # Lightweight adapter: ATAC cell embedding → d-dim shift
        self.cell_adapter = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        )

        # ── Output heads ─────────────────────────────────────────────────
        self.hist_reg_head  = mlp([film_d] + oc["histone_hidden"] + [n_marks])
        self.hist_cls_head  = mlp([film_d] + oc["histone_hidden"] + [n_marks * 3])
        self.atac_head      = mlp([film_d] + oc["atac_hidden"] + [1])
        self.meth_head      = mlp([film_d] + oc["meth_hidden"] + [1])
        self.rna_head       = mlp([film_d] + oc["rna_hidden"] + [1])

        self.n_marks = n_marks
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_for_transfer(self):
        """Freeze everything except cell_adapter and output heads."""
        for p in self.parameters():
            p.requires_grad = False
        for mod in [self.cell_adapter,
                    self.hist_reg_head, self.hist_cls_head,
                    self.atac_head, self.meth_head, self.rna_head]:
            for p in mod.parameters():
                p.requires_grad = True

    def get_param_count(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    def forward(
        self,
        seq_onehot:   Optional[torch.Tensor],   # (B, 4, L) — None to skip
        dcas9_signal: torch.Tensor,
        dcas9_scalar: torch.Tensor,
        atac_bins:    torch.Tensor,
        atac_scalar:  torch.Tensor,
        meth_bins:    torch.Tensor,
        glob_meth:    torch.Tensor,
        hist_ctrl:    torch.Tensor,
        bep_id:       torch.Tensor,
        role_id:      torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # ── Encode each modality ──────────────────────────────────────────
        # 1. DNA sequence (Enformer trunk → pool to d)
        if seq_onehot is not None:
            with torch.no_grad() if not self.backbone.training else torch.enable_grad():
                seq_emb = self.backbone(seq_onehot)   # (B, n_bins, seq_d)
            z_seq = self.seq_pool(seq_emb.mean(dim=1))  # (B, d)
        else:
            # Skip backbone (faster for ablation / debugging)
            z_seq = torch.zeros(dcas9_signal.size(0), self.seq_pool[-1].out_features,
                                device=dcas9_signal.device)

        # 2. dCas9 ChIP
        z_dcas9 = self.dcas9_enc(dcas9_signal) + self.dcas9_scalar_proj(dcas9_scalar)

        # 3. ATAC baseline
        z_atac = self.atac_enc(atac_bins) + self.atac_scalar_proj(atac_scalar)
        # Cell-type adapter applied to ATAC stream
        z_atac = z_atac + self.cell_adapter(z_atac)

        # 4. Methylation
        z_meth = self.meth_enc(meth_bins) + self.meth_scalar_proj(glob_meth)

        # 5. Histone marks (baseline)
        z_hist, hist_tokens = self.hist_enc(hist_ctrl)

        # ── Cross-modal attention (dCas9 as query) ────────────────────────
        mod_stack = torch.stack([z_atac, z_meth, z_hist, z_seq], dim=1)  # (B, 4, d)
        z_cross, modal_attn_4 = self.cross_attn(z_dcas9, mod_stack)

        # Pad modality attention to include dCas9 itself = 5 entries
        attn_dcas9  = torch.zeros(z_dcas9.size(0), 1, device=z_dcas9.device)
        modal_attn  = torch.cat([attn_dcas9, modal_attn_4], dim=-1)  # (B, 5)

        # ── Fuse & condition ──────────────────────────────────────────────
        z_cat    = torch.cat([z_cross, z_atac, z_meth, z_hist, z_seq, z_dcas9], dim=-1)
        z_fused  = self.pre_film(z_cat)
        z_fused  = self.film(z_fused, bep_id, role_id)

        # ── Output heads ──────────────────────────────────────────────────
        hist_fc  = self.hist_reg_head(z_fused)                          # (B, n_marks)
        hist_cls = self.hist_cls_head(z_fused).view(-1, self.n_marks, 3)  # (B, n_marks, 3)
        atac_fc  = self.atac_head(z_fused)                              # (B, 1)
        meth_d   = self.meth_head(z_fused)                              # (B, 1)
        rna_fc   = self.rna_head(z_fused)                               # (B, 1)

        return {
            "hist_log2fc": hist_fc,
            "hist_cls":    hist_cls,
            "atac_log2fc": atac_fc,
            "meth_delta":  meth_d,
            "rna_log2fc":  rna_fc,
            "modal_attn":  modal_attn,   # (B, 5) — interpretable
        }
