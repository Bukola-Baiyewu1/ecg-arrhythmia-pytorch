import math

import numpy as np
import torch
import yaml

from ecg.metrics import bootstrap_ci, summary
from ecg.model import ECGNet
from ecg.train import class_weights, run


def test_output_shape_and_raw_logits():
    model = ECGNet().eval()
    out = model(torch.randn(8, 200))
    assert out.shape == (8, 5)
    assert not torch.allclose(out.softmax(1).sum(1), out.sum(1))  # logits, not probabilities


def test_initial_loss_is_about_log_num_classes():
    torch.manual_seed(0)
    model = ECGNet().eval()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(256, 200)), torch.randint(0, 5, (256,)))
    assert abs(loss.item() - math.log(5)) < 0.5


def test_can_overfit_one_small_batch():
    torch.manual_seed(0)
    model = ECGNet(dropout=0.0)
    x, y = torch.randn(10, 200), torch.arange(10) % 5
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward(); opt.step()
    assert loss.item() < 0.05


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 900 + [1] * 90 + [2] * 10)
    w = class_weights(y, 5, "inverse")
    assert w[2] > w[1] > w[0]
    assert class_weights(y, 5, "none") is None


def test_metrics_perfect_and_ci():
    y = np.array([0, 1, 2, 3, 4] * 20)
    s = summary(y, y)
    assert s["accuracy"] == 1.0 and s["macro_f1"] == 1.0
    lo, hi = bootstrap_ci(y, y, n_boot=50)
    assert lo == hi == 1.0


def test_end_to_end_synthetic_run(tmp_path):
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["epochs"] = 2
    cfg["data"]["synthetic_beats_per_record"] = 40
    cfg["eval"]["n_bootstrap"] = 20
    res = run(cfg, "patient", seed=0, synthetic=True, out_root=tmp_path)
    assert res["records_in_train_and_test"] == 0
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert (tmp_path / "runs_v2.csv").exists()


def test_model_with_rr_branch():
    model = ECGNet(n_rr=4).eval()
    assert model(torch.randn(3, 200), torch.randn(3, 4)).shape == (3, 5)


def test_end_to_end_with_rr_features(tmp_path):
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["use_rr"] = True
    cfg["train"]["epochs"] = 2
    cfg["data"]["synthetic_beats_per_record"] = 40
    cfg["eval"]["n_bootstrap"] = 20
    res = run(cfg, "patient", seed=0, synthetic=True, out_root=tmp_path)
    assert res["use_rr"] is True and 0.0 <= res["macro_f1_nsvf"] <= 1.0
