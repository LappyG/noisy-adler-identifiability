"""Behavioral checks for fitting observed Adler phase records."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fit_phase_csv import fit_phase_record, main, read_phase_csv, wrap_angle


class FitPhaseCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20)
        cls.dt = np.where(np.arange(8000) % 2 == 0, 0.08, 0.12)
        cls.time = np.r_[0.0, np.cumsum(cls.dt)]
        cls.gap = np.zeros(len(cls.time))
        for i, step in enumerate(cls.dt):
            cls.gap[i + 1] = (
                cls.gap[i]
                + (0.12 - 0.08 * np.sin(cls.gap[i])) * step
                + 0.04 * np.sqrt(step) * rng.normal()
            )

    def test_irregular_fit_matches_independent_weighted_regression(self):
        result = fit_phase_record(self.time, self.gap, phase_format="unwrapped")
        x = np.column_stack((np.ones(len(self.dt)), -np.sin(self.gap[:-1])))
        whitened_x = np.sqrt(self.dt)[:, None] * x
        whitened_y = np.diff(self.gap) / np.sqrt(self.dt)
        reference = np.linalg.lstsq(whitened_x, whitened_y, rcond=None)[0]
        np.testing.assert_allclose(
            [result["D_rad_per_h"], result["K_rad_per_h"]], reference, rtol=1e-12
        )
        self.assertAlmostEqual(result["D_rad_per_h"], 0.12, delta=0.01)
        self.assertAlmostEqual(result["K_rad_per_h"], 0.08, delta=0.01)
        self.assertAlmostEqual(result["sigma_hat_rad_per_sqrt_h"], 0.04, delta=0.003)
        self.assertLess(result["D_ci95_rad_per_h"][0], result["D_rad_per_h"])
        self.assertGreater(result["D_ci95_rad_per_h"][1], result["D_rad_per_h"])

    def test_wrapped_gap_and_clock_phase_give_same_fit(self):
        unwrapped = fit_phase_record(self.time, self.gap, phase_format="unwrapped")
        wrapped = fit_phase_record(self.time, wrap_angle(self.gap))
        cue_origin = 0.7
        clock = wrap_angle(cue_origin + 2 * np.pi * self.time / 24 - self.gap)
        from_clock = fit_phase_record(
            self.time, clock, phase_kind="clock", cue_phase_at_zero_rad=cue_origin
        )
        for key in ("D_rad_per_h", "K_rad_per_h", "sigma_hat_rad_per_sqrt_h"):
            self.assertAlmostEqual(unwrapped[key], wrapped[key], places=11)
            self.assertAlmostEqual(unwrapped[key], from_clock[key], places=11)
        known = fit_phase_record(self.time, self.gap, phase_format="unwrapped", known_sigma=0.04)
        self.assertEqual(known["sigma_source"], "known")
        self.assertEqual(known["interval_method"], "profile likelihood with known sigma")

    def test_rejects_uninformative_or_bad_records(self):
        with self.assertRaisesRegex(ValueError, "do not vary enough"):
            fit_phase_record(np.arange(8.0), np.full(8, 0.2))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            fit_phase_record([0, 1, 1, 2], [0, 0.1, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_phase_record([0, 1, 2, 3], [0, np.nan, 0.2, 0.3])
        with self.assertRaisesRegex(ValueError, "positive"):
            fit_phase_record(self.time, self.gap, known_sigma=0)
        with self.assertRaisesRegex(ValueError, "largest sampling gap"):
            fit_phase_record(self.time[::240], self.gap[::240], phase_format="unwrapped")
        coarse = fit_phase_record(
            self.time[::240], self.gap[::240], phase_format="unwrapped", allow_coarse=True
        )
        self.assertTrue(any("Coarse-sampling override" in text for text in coarse["warnings"]))
        fast_time = np.arange(101) * 0.1
        fast_phase = 3 * fast_time + 0.01 * np.sin(fast_time)
        with self.assertRaisesRegex(ValueError, "large relative"):
            fit_phase_record(fast_time, fast_phase, phase_format="unwrapped")
        fast = fit_phase_record(
            fast_time, fast_phase, phase_format="unwrapped", allow_coarse=True
        )
        self.assertGreater(fast["max_fitted_rate_times_step"], 0.2)

    def test_csv_cli_writes_json_and_reports_wrap_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "phase.csv"
            output = Path(directory) / "fit.json"
            source.write_text(
                "time_h,phase_rad\n"
                + "".join(
                    f"{t:.12f},{p:.12f}\n"
                    for t, p in zip(self.time[:1001], wrap_angle(self.gap[:1001]))
                ),
                encoding="utf-8",
            )
            times, phases = read_phase_csv(source)
            self.assertEqual(len(times), 1001)
            self.assertEqual(len(phases), 1001)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source), "--output", str(output)]), 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["n_observations"], 1001)
            self.assertEqual(saved["phase_format"], "wrapped")
            self.assertTrue(any("full turns" in text for text in saved["warnings"]))
            bad = Path(directory) / "bad.csv"
            bad.write_text("time,phase\n0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "header columns"):
                read_phase_csv(bad)


if __name__ == "__main__":
    unittest.main()
