"""dlfe_lab 실데이터 스모크 테스트 (alphamale venv python으로 실행)."""
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dlfe_lab import backtest, data, embeddings, kr, modeling, paths, viz  # noqa: E402


class PathsTests(unittest.TestCase):
    def test_paths_and_bootstrap(self):
        self.assertTrue(paths.ROOT.exists())
        self.assertTrue(paths.ART.exists())
        paths.bootstrap()
        n = len(sys.path)
        paths.bootstrap()
        self.assertEqual(n, len(sys.path))


class DataTests(unittest.TestCase):
    def test_loaders(self):
        samples = data.load_samples()
        self.assertEqual(len(samples), 2075)
        self.assertGreater(len(data.load_news()), 50_000)
        self.assertGreater(len(data.load_prices()), 2_000)
        ev = data.load_events()
        ok = data.load_event_ok()
        self.assertEqual(len(ev), len(ok))
        st = data.news_stats(data.load_news().head(5_000))
        self.assertIn("per_ticker", st)


class EmbeddingTests(unittest.TestCase):
    def test_ntn_inference(self):
        ev = data.load_events()
        ok = data.load_event_ok()
        ntn = embeddings.load_ntn()
        n_params = sum(p.numel() for p in ntn.parameters())
        self.assertGreater(n_params, 1_000_000)
        idx = list(map(int, ev.index[ok][:2]))
        out = embeddings.embed_events(ntn, ev, idx)
        self.assertEqual(out.shape, (2, 100))
        self.assertGreater(float(abs(out).sum()), 0.0)

    def test_word_neighbors_and_pca(self):
        vocab, w2i, W = embeddings.load_word_vectors()
        self.assertEqual(W.shape, (len(vocab), 100))
        nb = embeddings.nearest_words("nvidia", k=5)
        self.assertEqual(len(nb), 5)
        xy = embeddings.pca_2d(W[:200])
        self.assertEqual(xy.shape, (200, 2))


class ModelingTests(unittest.TestCase):
    def test_quick_train_one_epoch(self):
        hist = modeling.quick_train(rep="EB", nn_only=True, epochs=1)
        self.assertEqual(len(hist["train_loss"]), 1)
        self.assertEqual(len(hist["dev_mcc"]), 1)
        self.assertEqual(len(hist["test_prob"]), 366)
        self.assertIn("test_acc", hist)


class BacktestTests(unittest.TestCase):
    def test_paper_simulate_and_randomization(self):
        teb = backtest.load_teb_daily()
        sim = backtest.paper_simulate(teb.prob_up.values)
        self.assertEqual(len(sim["daily"]), 366)
        self.assertIsInstance(sim["total"], float)
        dist, p = backtest.randomization(sim["total"], n=200)
        self.assertEqual(len(dist), 200)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)


class KrTests(unittest.TestCase):
    def test_kr_loaders(self):
        ko = kr.load_kr_ohlcv()
        rates = kr.exceedance_rates(ko)
        self.assertEqual(len(rates), 6)
        for v in rates.values():
            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)
        curves = kr.load_equity_curves()
        self.assertEqual(set(curves), {"kr50", "kr51", "kr52"})
        import numpy as np
        for name, df_ in curves.items():
            eq = df_.equity.astype(float).values
            self.assertTrue(np.isfinite(eq).all(), name)
            self.assertLess(float(eq.max()), 100.0, name)  # 복리 폭발 회귀 방지
        sc = kr.load_kr_scores()
        top1 = kr.daily_topk_hit(sc, k=1)
        self.assertGreater(top1["hit_rate"], top1["base_rate"])
        self.assertIn("variants", kr.load_gap_results())
        self.assertIn("xai_best_days", kr.load_xai())
        self.assertIn("seed_rows", kr.load_survival())


class VizTests(unittest.TestCase):
    def test_viz_smoke(self):
        news = data.load_news().head(3_000)
        fig1 = viz.news_overview(news)
        self.assertIsNotNone(fig1)
        hist = {"rep": "EB", "nn_only": True, "train_loss": [0.7, 0.69], "dev_mcc": [0.0, 0.05]}
        fig2 = viz.training_curves(hist)
        self.assertIsNotNone(fig2)
        fig3 = viz.exceedance_bars({"UP2%": 0.3, "UP5%": 0.08, "DN5%": 0.06})
        self.assertIsNotNone(fig3)


if __name__ == "__main__":
    unittest.main()
