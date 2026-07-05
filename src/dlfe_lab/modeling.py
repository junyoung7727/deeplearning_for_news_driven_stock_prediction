"""실제 모델 구조 확인 + 캐시된 실데이터 특징으로 짧은 학습."""
from __future__ import annotations

from typing import Any

import numpy as np

from .paths import ART, bootstrap


def build_ntn():
    bootstrap()
    import config as C
    from s4_ntn import NTN

    return NTN(C.WORD_DIM, C.NTN_K)


def describe_model(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{model}\n\ntotal params: {total:,} | trainable: {trainable:,}"


def load_features(rep: str = "EB") -> dict[str, Any]:
    """feats_{rep}.npz + samples.parquet 를 torch 텐서/인덱스로 로드."""
    bootstrap()
    import torch

    from .data import load_samples

    z = np.load(ART / f"feats_{rep}.npz")
    samples = load_samples()
    split = samples.split.values
    y_pm1 = samples.label.values.astype(int)
    return {
        "short": torch.from_numpy(z["short"]),
        "mid": torch.from_numpy(z["mid"]),
        "long": torch.from_numpy(z["long"]),
        "samples": samples,
        "tr": np.where(split == "train")[0].astype(np.int64),
        "dv": np.where(split == "dev")[0].astype(np.int64),
        "te": np.where(split == "test")[0].astype(np.int64),
        "y_pm1": y_pm1,
        "y01": ((y_pm1 + 1) // 2).astype(int),
    }


def quick_train(
    rep: str = "EB",
    nn_only: bool = True,
    epochs: int = 6,
    lr: float = 1e-3,
    batch: int = 128,
    seed: int = 13,
) -> dict[str, Any]:
    """실제 s6_models.DenseModel 을 몇 에폭만 학습하고 loss/dev MCC 기록.

    전체 재현(s7_train_eval)은 4-seed 앙상블 + early stop 이지만, 여기서는
    학생이 학습 루프 자체를 관찰할 수 있게 축약한 버전이다.
    """
    bootstrap()
    import torch
    import torch.nn as nn
    from sklearn.metrics import matthews_corrcoef

    from s6_models import DenseModel, metrics, predict

    f = load_features(rep)
    short, mid, long_ = f["short"], f["mid"], f["long"]
    tr, dv, te = f["tr"], f["dv"], f["te"]
    y01, y_pm1 = f["y01"], f["y_pm1"]

    def inputs(ix):
        return short[ix], mid[ix], long_[ix]

    torch.manual_seed(seed)
    model = DenseModel(int(short.shape[1]), nn_only)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.BCELoss()
    yt = torch.from_numpy(y01.astype(np.float32))

    train_loss: list[float] = []
    dev_mcc: list[float] = []
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(tr)
        losses = []
        for s in range(0, len(perm), batch):
            sub = torch.from_numpy(perm[s : s + batch])
            out = model(*inputs(sub))
            loss = lossf(out, yt[sub])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss.append(float(np.mean(losses)))
        dp = predict(model, lambda ix: inputs(ix), dv)
        pred = (dp > 0.5).astype(int)
        mcc = matthews_corrcoef(y01[dv], pred) if len(np.unique(pred)) > 1 else 0.0
        dev_mcc.append(float(mcc))

    test_prob = predict(model, lambda ix: inputs(ix), te)
    test_acc, test_mcc = metrics(y_pm1[te], test_prob)
    return {
        "rep": rep,
        "nn_only": nn_only,
        "train_loss": train_loss,
        "dev_mcc": dev_mcc,
        "test_acc": float(test_acc),
        "test_mcc": float(test_mcc),
        "test_prob": test_prob,
        "model": model,
    }
