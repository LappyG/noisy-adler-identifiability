#!/usr/bin/env python3
"""Fit the noisy Adler drift to a CSV of observed phase values.

This is a conditional Gaussian Euler fit to phase increments. It does not
extract phase from a signal or account for error in the measured phase.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2


TWO_PI = 2.0 * np.pi
MAX_DEFAULT_STEP_H = 1.0
MAX_DEFAULT_RATE_STEP = 0.2


def wrap_angle(value):
    """Return principal circular differences in [-pi, pi)."""
    return (value + np.pi) % TWO_PI - np.pi


def read_phase_csv(path):
    """Read time_h and phase_rad columns, retaining the file's row order."""
    times, phases = [], []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"time_h", "phase_rad"}.issubset(reader.fieldnames):
            raise ValueError("CSV must have time_h and phase_rad header columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row["time_h"]))
                phases.append(float(row["phase_rad"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid time_h or phase_rad at CSV line {line_number}") from exc
    return np.asarray(times, dtype=float), np.asarray(phases, dtype=float)


def fit_phase_record(
    time_h,
    phase_rad,
    *,
    phase_kind="gap",
    phase_format="wrapped",
    cue_period_h=24.0,
    cue_phase_at_zero_rad=0.0,
    known_sigma=None,
    allow_coarse=False,
):
    """Estimate detuning D and signed coupling K from directly observed phase.

    For each interval i, delta_psi_i is Gaussian with mean
    (D - K sin(psi_i)) * delta_t_i and variance sigma^2 * delta_t_i.
    Irregular time steps are handled by the corresponding weighted likelihood.
    Marginal 95% intervals use asymptotic chi-square profile thresholds.
    """
    time_h = np.asarray(time_h, dtype=float)
    phase_rad = np.asarray(phase_rad, dtype=float)
    if time_h.ndim != 1 or phase_rad.ndim != 1 or len(time_h) != len(phase_rad):
        raise ValueError("time_h and phase_rad must be one-dimensional arrays of equal length")
    if len(time_h) < 4:
        raise ValueError("at least four phase observations are required")
    if not np.all(np.isfinite(time_h)) or not np.all(np.isfinite(phase_rad)):
        raise ValueError("time_h and phase_rad must contain only finite numbers")
    dt_h = np.diff(time_h)
    if np.any(dt_h <= 0):
        raise ValueError("time_h values must be strictly increasing")
    if np.max(dt_h) > MAX_DEFAULT_STEP_H + 1e-12 and not allow_coarse:
        raise ValueError(
            f"largest sampling gap exceeds {MAX_DEFAULT_STEP_H:g} h; a single Euler step "
            "can give biased estimates at coarse cadence. Use --allow-coarse only "
            "if you accept this model limitation"
        )
    if phase_kind not in ("gap", "clock"):
        raise ValueError("phase_kind must be 'gap' or 'clock'")
    if phase_format not in ("wrapped", "unwrapped"):
        raise ValueError("phase_format must be 'wrapped' or 'unwrapped'")
    if not np.isfinite(cue_period_h) or cue_period_h <= 0:
        raise ValueError("cue_period_h must be positive and finite")
    if not np.isfinite(cue_phase_at_zero_rad):
        raise ValueError("cue_phase_at_zero_rad must be finite")
    if known_sigma is not None and (not np.isfinite(known_sigma) or known_sigma <= 0):
        raise ValueError("known_sigma must be positive and finite")

    gap = phase_rad.copy()
    if phase_kind == "clock":
        gap = cue_phase_at_zero_rad + TWO_PI * time_h / cue_period_h - phase_rad
    if phase_format == "wrapped":
        gap = wrap_angle(gap)
        increments = wrap_angle(np.diff(gap))
    else:
        increments = np.diff(gap)

    # G and h are the normal equations for the Euler likelihood with unequal dt.
    sine = np.sin(gap[:-1])
    gram = np.array(
        [
            [np.sum(dt_h), -np.sum(dt_h * sine)],
            [-np.sum(dt_h * sine), np.sum(dt_h * sine**2)],
        ]
    )
    rhs = np.array([np.sum(increments), -np.sum(sine * increments)])
    gram_condition = float(np.linalg.cond(gram))
    if not np.isfinite(gram_condition) or gram_condition > 1e10:
        raise ValueError(
            "phase values do not vary enough to estimate D and K separately "
            "(ill-conditioned design)"
        )
    theta = np.linalg.solve(gram, rhs)
    rate_step = float(np.max(dt_h) * np.max(np.abs(theta)))
    if rate_step > MAX_DEFAULT_RATE_STEP and not allow_coarse:
        raise ValueError(
            "sampling steps are large relative to the fitted D or K rate; "
            "a single Euler step may be biased. Use --allow-coarse only if "
            "you accept this model limitation"
        )
    inverse = np.linalg.inv(gram)
    residual = increments - dt_h * (theta[0] - theta[1] * sine)
    rss = float(np.sum(residual**2 / dt_h))
    if not np.isfinite(rss):
        raise ValueError("the residual sum of squares is not finite")
    n_steps = len(dt_h)
    if known_sigma is None and rss <= 0:
        raise ValueError("process noise cannot be estimated from a zero-residual record")
    sigma_hat = float(np.sqrt(rss / n_steps))
    sigma_used = float(known_sigma if known_sigma is not None else sigma_hat)

    cutoff = float(chi2.ppf(0.95, 1))
    if known_sigma is None:
        width = np.sqrt(rss * np.expm1(cutoff / n_steps) * np.diag(inverse))
        interval_method = "profile likelihood with estimated sigma"
    else:
        width = np.sqrt(cutoff * sigma_used**2 * np.diag(inverse))
        interval_method = "profile likelihood with known sigma"
    if not np.all(np.isfinite(width)):
        raise ValueError("uncertainty intervals could not be calculated")

    covariance = sigma_used**2 * inverse
    correlation = float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))
    omega = float(TWO_PI / cue_period_h - theta[0])
    free_period = float(TWO_PI / omega) if omega > 0 else None
    lag_one = None
    standardized = residual / np.sqrt(dt_h)
    if n_steps > 2 and np.std(standardized[:-1]) > 0 and np.std(standardized[1:]) > 0:
        lag_one = float(np.corrcoef(standardized[:-1], standardized[1:])[0, 1])

    near_pi_count = int(np.count_nonzero(np.abs(increments) >= 0.8 * np.pi))
    warnings = []
    if np.max(dt_h) > MAX_DEFAULT_STEP_H + 1e-12 or rate_step > MAX_DEFAULT_RATE_STEP:
        warnings.append(
            "Coarse-sampling override is active: the Euler fit and its 95% intervals "
            "may be biased even with unwrapped phase."
        )
    elif np.max(dt_h) > 0.1 + 1e-12:
        warnings.append(
            "The largest sampling gap exceeds the 0.1 h step used in this repo's "
            "simulation study; check Euler approximation error for this cadence."
        )
    if n_steps < 100:
        warnings.append(
            "Fewer than 100 increments were fitted; asymptotic 95% interval "
            "coverage may be unreliable."
        )
    if phase_format == "wrapped":
        warnings.append(
            "Wrapped samples cannot reveal full turns between observations; "
            "verify that every true phase increment has magnitude below pi radians."
        )
        if near_pi_count:
            warnings.append(
                f"{near_pi_count} observed increment(s) are within 20% of pi radians; "
                "winding ambiguity is especially plausible there."
            )
    if theta[1] < 0:
        warnings.append("The unconstrained coupling estimate K is negative; check the phase convention and model fit.")
    if free_period is None:
        warnings.append("The fitted natural angular speed is nonpositive, so no positive natural period is reported.")

    return {
        "model": "conditional Gaussian Euler noisy Adler fit",
        "phase_kind": phase_kind,
        "phase_format": phase_format,
        "cue_period_h": float(cue_period_h),
        "cue_phase_at_zero_rad": float(cue_phase_at_zero_rad),
        "n_observations": int(len(time_h)),
        "n_increments": int(n_steps),
        "duration_h": float(time_h[-1] - time_h[0]),
        "dt_h": {"min": float(dt_h.min()), "median": float(np.median(dt_h)), "max": float(dt_h.max())},
        "max_abs_observed_increment_rad": float(np.max(np.abs(increments))),
        "near_pi_increment_count": near_pi_count,
        "gram_condition_number": gram_condition,
        "max_fitted_rate_times_step": rate_step,
        "D_rad_per_h": float(theta[0]),
        "K_rad_per_h": float(theta[1]),
        "D_ci95_rad_per_h": [float(theta[0] - width[0]), float(theta[0] + width[0])],
        "K_ci95_rad_per_h": [float(theta[1] - width[1]), float(theta[1] + width[1])],
        "interval_method": interval_method,
        "sigma_hat_rad_per_sqrt_h": sigma_hat,
        "sigma_used_rad_per_sqrt_h": sigma_used,
        "sigma_source": "known" if known_sigma is not None else "estimated",
        "parameter_correlation": correlation,
        "residual_lag1_correlation": lag_one,
        "natural_angular_speed_rad_per_h": omega,
        "natural_period_h": free_period,
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="CSV with time_h and phase_rad columns")
    parser.add_argument("--output", type=Path, help="write full fit results as JSON")
    parser.add_argument("--phase-kind", choices=("gap", "clock"), default="gap")
    parser.add_argument("--phase-format", choices=("wrapped", "unwrapped"), default="wrapped")
    parser.add_argument("--cue-period-h", type=float, default=24.0)
    parser.add_argument("--cue-phase-at-zero-rad", type=float, default=0.0)
    parser.add_argument("--known-sigma", type=float, help="known process-noise SD in rad/sqrt(h)")
    parser.add_argument(
        "--allow-coarse", action="store_true",
        help="override step-size checks despite possible Euler bias and misleading intervals",
    )
    args = parser.parse_args(argv)
    if args.output is not None and args.output.resolve() == args.csv_file.resolve():
        parser.error("--output must differ from the input CSV path")
    try:
        times, phases = read_phase_csv(args.csv_file)
        result = fit_phase_record(
            times,
            phases,
            phase_kind=args.phase_kind,
            phase_format=args.phase_format,
            cue_period_h=args.cue_period_h,
            cue_phase_at_zero_rad=args.cue_phase_at_zero_rad,
            known_sigma=args.known_sigma,
            allow_coarse=args.allow_coarse,
        )
        result["input_csv"] = str(args.csv_file.resolve())
        if args.output is not None:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"fit_phase_csv.py: error: {exc}\n")

    print(f"Fit {result['n_observations']} observations over {result['duration_h']:.3g} h")
    print("Asymptotic 95% intervals under the direct-phase Euler model")
    print(
        f"D = {result['D_rad_per_h']:.6g} rad/h "
        f"(95% CI {result['D_ci95_rad_per_h'][0]:.6g} to {result['D_ci95_rad_per_h'][1]:.6g})"
    )
    print(
        f"K = {result['K_rad_per_h']:.6g} rad/h "
        f"(95% CI {result['K_ci95_rad_per_h'][0]:.6g} to {result['K_ci95_rad_per_h'][1]:.6g})"
    )
    print(f"sigma = {result['sigma_used_rad_per_sqrt_h']:.6g} rad/sqrt(h) ({result['sigma_source']})")
    if result["natural_period_h"] is not None:
        print(f"Natural period = {result['natural_period_h']:.6g} h")
    for warning in result["warnings"]:
        print(f"Note: {warning}")
    if args.output is not None:
        print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
