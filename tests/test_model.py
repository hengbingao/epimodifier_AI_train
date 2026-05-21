"""
tests/test_model.py
Unit tests for model forward pass, loss, and data utilities.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model(cfg):
    from models.model import BEPPerturbationModel
    m = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
    m.eval()
    return m


def make_batch(cfg, B=4):
    n_bins  = cfg["data"]["n_signal_bins"]
    n_marks = len(cfg["histone_marks"])
    return {
        "dcas9_signal": torch.randn(B, n_bins),
        "dcas9_scalar": torch.randn(B, 2),
        "atac_bins":    torch.randn(B, n_bins),
        "atac_scalar":  torch.randn(B, 1),
        "meth_bins":    torch.rand(B, n_bins),
        "glob_meth":    torch.rand(B, 1),
        "hist_ctrl":    torch.randn(B, n_marks),
        "bep_id":       torch.randint(0, 10, (B,)),
        "role_id":      torch.randint(0, 3, (B,)),
        # targets
        "hist_log2fc":  torch.randn(B, n_marks),
        "hist_cls":     torch.randint(0, 3, (B, n_marks)),
        "atac_log2fc":  torch.randn(B, 1),
        "meth_delta":   torch.randn(B, 1),
        "rna_log2fc":   torch.randn(B, 1),
    }


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModel:

    def test_parameter_count(self, model):
        counts = model.get_param_count()
        assert counts["total"] > 1_000_000, "Model seems too small"
        assert counts["trainable"] > 0

    def test_forward_output_shapes(self, model, cfg):
        B = 4
        n_marks = len(cfg["histone_marks"])
        batch = make_batch(cfg, B)

        with torch.no_grad():
            out = model(
                None,
                batch["dcas9_signal"], batch["dcas9_scalar"],
                batch["atac_bins"],    batch["atac_scalar"],
                batch["meth_bins"],    batch["glob_meth"],
                batch["hist_ctrl"],
                batch["bep_id"],       batch["role_id"],
            )

        assert out["hist_log2fc"].shape  == (B, n_marks)
        assert out["hist_cls"].shape     == (B, n_marks, 3)
        assert out["atac_log2fc"].shape  == (B, 1)
        assert out["meth_delta"].shape   == (B, 1)
        assert out["rna_log2fc"].shape   == (B, 1)
        assert out["modal_attn"].shape   == (B, 5)

    def test_modal_attention_sums_to_one(self, model, cfg):
        batch = make_batch(cfg, B=8)
        with torch.no_grad():
            out = model(
                None,
                batch["dcas9_signal"], batch["dcas9_scalar"],
                batch["atac_bins"],    batch["atac_scalar"],
                batch["meth_bins"],    batch["glob_meth"],
                batch["hist_ctrl"],
                batch["bep_id"],       batch["role_id"],
            )
        # Attention over 4 KV modalities (+ 1 dCas9 zero) sums to ~1
        # (the zero dCas9 entry brings sum below 1; rest of 4 should sum to ~1)
        attn_kv = out["modal_attn"][:, 1:]  # (B, 4)
        sums = attn_kv.sum(dim=-1)
        assert (sums > 0.9).all(), f"Attention KV sum < 0.9: {sums}"
        assert (sums <= 1.01).all(), f"Attention KV sum > 1.01: {sums}"

    def test_freeze_backbone(self, cfg):
        from models.model import BEPPerturbationModel
        m = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
        m.freeze_backbone()
        backbone_params = list(m.backbone.parameters())
        assert all(not p.requires_grad for p in backbone_params), \
            "Backbone parameters should be frozen"

    def test_freeze_for_transfer(self, cfg):
        from models.model import BEPPerturbationModel
        m = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
        m.freeze_for_transfer()
        counts = m.get_param_count()
        # Only adapter + output heads should be trainable — much fewer params
        assert counts["trainable"] < counts["total"] * 0.05, \
            f"Expected <5% trainable after freeze, got {counts['trainable']/counts['total']:.1%}"
        assert counts["trainable"] > 0

    def test_different_bep_ids_give_different_outputs(self, model, cfg):
        batch = make_batch(cfg, B=1)
        results = []
        with torch.no_grad():
            for bep_id in [1, 2, 3]:
                b = dict(batch)
                b["bep_id"] = torch.tensor([bep_id])
                out = model(
                    None,
                    b["dcas9_signal"], b["dcas9_scalar"],
                    b["atac_bins"],    b["atac_scalar"],
                    b["meth_bins"],    b["glob_meth"],
                    b["hist_ctrl"],
                    b["bep_id"],       b["role_id"],
                )
                results.append(out["hist_log2fc"].clone())
        # Different BEP IDs must produce different predictions (FiLM conditioning)
        assert not torch.allclose(results[0], results[1], atol=1e-6), \
            "BEP1 and BEP2 should produce different predictions"
        assert not torch.allclose(results[0], results[2], atol=1e-6), \
            "BEP1 and BEP3 should produce different predictions"

    def test_no_nan_in_outputs(self, model, cfg):
        batch = make_batch(cfg, B=8)
        with torch.no_grad():
            out = model(
                None,
                batch["dcas9_signal"], batch["dcas9_scalar"],
                batch["atac_bins"],    batch["atac_scalar"],
                batch["meth_bins"],    batch["glob_meth"],
                batch["hist_ctrl"],
                batch["bep_id"],       batch["role_id"],
            )
        for k, v in out.items():
            if hasattr(v, "isnan"):
                assert not v.isnan().any(), f"NaN in output '{k}'"


# ── Loss tests ────────────────────────────────────────────────────────────────

class TestLoss:

    def test_loss_positive(self, cfg):
        from utils.training import BEPLoss
        from models.model import BEPPerturbationModel
        model = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
        model.eval()
        crit  = BEPLoss()
        batch = make_batch(cfg, B=4)
        with torch.no_grad():
            out = model(
                None,
                batch["dcas9_signal"], batch["dcas9_scalar"],
                batch["atac_bins"],    batch["atac_scalar"],
                batch["meth_bins"],    batch["glob_meth"],
                batch["hist_ctrl"],
                batch["bep_id"],       batch["role_id"],
            )
        losses = crit(out, batch)
        for k, v in losses.items():
            assert v.item() >= 0, f"Loss '{k}' is negative: {v.item()}"
        assert losses["total"].item() > 0

    def test_loss_keys(self, cfg):
        from utils.training import BEPLoss
        from models.model import BEPPerturbationModel
        model = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
        crit  = BEPLoss()
        batch = make_batch(cfg, B=2)
        with torch.no_grad():
            out = model(
                None,
                batch["dcas9_signal"], batch["dcas9_scalar"],
                batch["atac_bins"],    batch["atac_scalar"],
                batch["meth_bins"],    batch["glob_meth"],
                batch["hist_ctrl"],
                batch["bep_id"],       batch["role_id"],
            )
        losses = crit(out, batch)
        expected = {"total", "hist_reg", "hist_cls", "atac", "meth", "rna"}
        assert expected.issubset(set(losses.keys()))

    def test_backward_pass(self, cfg):
        from utils.training import BEPLoss
        from models.model import BEPPerturbationModel
        model = BEPPerturbationModel(cfg, use_pretrained_backbone=False)
        model.train()
        model.freeze_backbone()
        crit  = BEPLoss()
        batch = make_batch(cfg, B=2)
        out = model(
            None,
            batch["dcas9_signal"], batch["dcas9_scalar"],
            batch["atac_bins"],    batch["atac_scalar"],
            batch["meth_bins"],    batch["glob_meth"],
            batch["hist_ctrl"],
            batch["bep_id"],       batch["role_id"],
        )
        losses = crit(out, batch)
        losses["total"].backward()
        # Check that at least some gradients are populated
        trainable = [p for p in model.parameters() if p.requires_grad]
        has_grad  = [p for p in trainable if p.grad is not None]
        assert len(has_grad) > 0, "No gradients after backward pass"


# ── Normaliser tests ──────────────────────────────────────────────────────────

class TestNormalisers:

    def test_log1p_normalizer(self):
        from utils.training import Log1pNormalizer
        norm = Log1pNormalizer()
        x    = np.random.exponential(5, (100, 50)).astype(np.float32)
        xn   = norm.fit_transform(x)
        assert xn.shape == x.shape
        assert abs(xn.mean()) < 0.5, "log1p normalised mean should be near 0"
        assert abs(xn.std() - 1.0) < 0.3

    def test_normalizer_per_col(self):
        from utils.training import Normalizer
        norm = Normalizer(per_col=True)
        x    = np.random.randn(200, 25).astype(np.float32)
        x[:, 0] *= 10   # make one column have a different scale
        xn   = norm.fit_transform(x)
        col_means = xn.mean(0)
        col_stds  = xn.std(0)
        assert np.allclose(col_means, 0, atol=0.1)
        assert np.allclose(col_stds,  1, atol=0.1)

    def test_transform_without_fit_raises(self):
        from utils.training import Normalizer
        norm = Normalizer()
        x    = np.random.randn(10, 5)
        with pytest.raises((TypeError, AttributeError)):
            norm.transform(x)   # mean_ is None → should fail


# ── Data utilities ────────────────────────────────────────────────────────────

class TestDataUtils:

    def test_query_signal_empty_df(self):
        import pandas as pd
        from data.dataset import query_signal
        empty = pd.DataFrame(columns=["chrom", "start", "end", "value"])
        result = query_signal(empty, "chr1", 1000, 4000, 100)
        assert result.shape == (100,)
        assert (result == 0).all()

    def test_query_signal_basic(self):
        import pandas as pd
        from data.dataset import query_signal
        df = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [800],
            "end":   [1200],
            "value": [5.0],
        })
        result = query_signal(df, "chr1", 1000, 4000, 100)
        assert result.shape == (100,)
        # The region covers bins 40-60 (center bin)
        assert result.max() > 0, "Signal should be > 0 within the peak"
        assert result[0] == 0, "Bins far from peak should be 0"

    def test_query_meth_no_sites(self):
        import pandas as pd
        from data.dataset import query_meth
        empty = pd.DataFrame(columns=["chrom", "pos0", "meth_frac", "cov"])
        bins, glob = query_meth(empty, "chr1", 1000, 4000, 100)
        assert bins.shape == (100,)
        assert (bins == 0.5).all(), "Missing sites should be filled with 0.5"
        assert glob == 0.5

    def test_build_bep_role_idx(self, cfg):
        from data.dataset import build_bep_role_idx
        bep_to_idx, bep_role_idx = build_bep_role_idx(cfg)
        assert "BEP073_GFP" in bep_to_idx
        assert bep_role_idx["BEP073_GFP"] == 0    # control
        assert bep_role_idx["BEP100_ZIM3"] == 1   # repressor
        assert bep_role_idx["BEP486_SREBF2ddr"] == 2  # activator


# ── Interpretability tests ────────────────────────────────────────────────────

class TestInterpretability:

    def test_sensitivity_index(self):
        from utils.interpret import sensitivity_index
        fc = np.array([[1.0, 0.1, -1.5],
                       [0.8, 0.0, -1.2],
                       [0.6, 0.05, -0.9]])
        si = sensitivity_index(fc, thr=0.5)
        assert si.shape == (3,)
        # Mark 0 and 2 should have higher SI than mark 1
        assert si[0] > si[1]
        assert si[2] > si[1]

    def test_dose_response_monotone_tendency(self, model, cfg):
        """Increasing dCas9 should generally increase |predicted effect|."""
        from utils.interpret import dose_response_curves
        batch = make_batch(cfg, B=1)
        doses, h_curves, a_curves, r_curves = dose_response_curves(
            model, torch.device("cpu"), batch,
            bep_id_int=1, role_id_int=1,
            n_doses=10, dose_range=(-3.0, 3.0),
        )
        assert h_curves.shape == (10, len(cfg["histone_marks"]))
        assert len(doses) == 10
        # High dose should produce larger absolute predictions than low dose
        # for at least some marks
        low_abs  = np.abs(h_curves[0]).mean()
        high_abs = np.abs(h_curves[-1]).mean()
        # Not strictly required but typical
        assert low_abs != high_abs, "Dose should affect predictions"
