#!/usr/bin/env python3
"""Identifiability of detuning and coupling in the noisy Adler equation.

Run with ``python script.py`` (or ``python3 script.py``). Requires NumPy, SciPy,
and Matplotlib. Writes PNG/PDF figures, captions, parameter recovery metrics,
raw estimates and a methods/results report beside this file. The baseline
Fisher calculation assumes known sigma; recovery also profiles sigma out.
"""

from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a display.
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2


# All times are hours; D and K are angular rates (rad/h).
DT = 0.1
T = 24.0 * 30.0
SIGMA = 0.06  # rad / sqrt(h), known rather than estimated
N_PHASE = 512
N_REPLICATES = 12
N_CI_REPLICATES = 256
N_BOOTSTRAP = 500
SEED = 20261002
K_CROSS = 0.08
NOISE_LEVELS = (0.04, 0.06, 0.08, 0.10)
OBSERVATION_DAYS = (7, 30, 90)
RECOVERY_REPLICATES = 512
RECOVERY_DAYS = (7, 30, 90)
MEASUREMENT_SD = (0.0, 0.01, 0.03, 0.06)  # radians, independent at each observation
OUT_DIR = Path(__file__).resolve().parent


def phase_quadrature(n_phase=N_PHASE, n_shift=None):
    """Prepare an n_phase-point circle grid and periodic integral quadrature."""
    period = 2.0 * np.pi
    phase = period * np.arange(n_phase) / n_phase
    # Gauss-Legendre nodes cluster near both ends of the tilted-potential
    # integral, resolving narrow boundary layers that a uniform grid misses.
    if n_shift is None:
        n_shift = max(128, n_phase // 2)
    nodes, weights = np.polynomial.legendre.leggauss(n_shift)
    shift = np.pi * (nodes + 1.0)
    log_weight = np.log(weights)  # The shared factor pi cancels on normalization.
    cosine_change = np.cos(phase[:, None] + shift[None, :]) - np.cos(
        phase[:, None]
    )
    return phase, shift, log_weight, cosine_change


def stationary_density(D, K, sigma, quadrature):
    r"""Return the normalized stationary density, including nonzero current.

    With a = sigma^2/2 and A'(psi) = (D - K sin(psi))/a, the periodic
    Fokker--Planck solution is

       p(psi) proportional to exp(A(psi)) * integral[psi, psi+2*pi]
                                                 exp(-A(y)) dy.

    Substituting y = psi + s gives the exponent below. The log-sum-exp
    calculation avoids overflow for small noise and either sign of D.
    For D = 0, the integral is constant and the formula reduces to the
    zero-current Boltzmann density.
    """
    phase, shift, log_weight, cosine_change = quadrature
    diffusion = 0.5 * sigma**2
    log_integrand = (
        -D * shift[None, :] - K * cosine_change
    ) / diffusion + log_weight[None, :]
    log_unnormalized = logsumexp(log_integrand, axis=1)
    # The common quadrature scale cancels in the normalization.
    phase_mass = np.exp(log_unnormalized - logsumexp(log_unnormalized))
    return phase_mass / (2.0 * np.pi / len(phase))


def fisher_from_density(density, phase, sigma=SIGMA, duration=T):
    """F = T/sigma^2 * E[[1, -sin(psi)], [-sin(psi), sin^2(psi)]]."""
    mass = density * (2.0 * np.pi / len(phase))
    sine = np.sin(phase)
    mean_sine = np.sum(mass * sine)
    mean_sine_sq = np.sum(mass * sine**2)
    scale = duration / sigma**2
    return scale * np.array(
        [[1.0, -mean_sine], [-mean_sine, mean_sine_sq]], dtype=float
    )


def analytic_grid(D_values, K_values, quadrature, sigma=SIGMA, duration=T, keep_cdf=False):
    """Evaluate density, FIM eigenvalues, and condition over a parameter grid."""
    phase = quadrature[0]
    fim = np.empty((len(K_values), len(D_values), 2, 2))
    n_phase = len(phase)
    cdf = np.empty((len(K_values), len(D_values), n_phase)) if keep_cdf else None
    for i, K in enumerate(K_values):
        for j, D in enumerate(D_values):
            density = stationary_density(D, K, sigma, quadrature)
            fim[i, j] = fisher_from_density(density, phase, sigma, duration)
            if keep_cdf:
                cdf[i, j] = np.cumsum(density) * (2.0 * np.pi / n_phase)
    eigenvalues = np.linalg.eigvalsh(fim)
    condition = eigenvalues[..., 1] / eigenvalues[..., 0]
    return fim, eigenvalues, condition, cdf


def monte_carlo_fim(D_values, K_values, cdf, phase, rng, n_replicates=N_REPLICATES):
    """Simulate stationary Euler paths and average score outer products.

    The Gaussian Euler score for one increment is
       (residual / sigma^2) * (1, -sin(psi)).
    We retain the unwrapped phase to recover that increment exactly. With
    this dt and sigma, wrapping ambiguity in observed increments is negligible.
    """
    n_k, n_d = len(K_values), len(D_values)
    D_flat = np.tile(D_values, n_k)
    K_flat = np.repeat(K_values, n_d)
    n_parameters = len(D_flat)
    D_paths = np.repeat(D_flat, n_replicates)
    K_paths = np.repeat(K_flat, n_replicates)
    cdf_paths = np.repeat(cdf.reshape(n_parameters, len(phase)), n_replicates, axis=0)

    # Draw each path's initial phase from its numerical stationary density.
    uniforms = rng.random(len(D_paths))
    initial_index = np.sum(uniforms[:, None] > cdf_paths, axis=1)
    psi = phase[np.minimum(initial_index, N_PHASE - 1)].copy()

    f00 = np.zeros(len(psi))
    f01 = np.zeros(len(psi))
    f11 = np.zeros(len(psi))
    n_steps = round(T / DT)
    noise_step = SIGMA * np.sqrt(DT)
    for _ in range(n_steps):
        sine = np.sin(psi)
        residual = noise_step * rng.standard_normal(len(psi))
        score_d = residual / SIGMA**2
        score_k = -score_d * sine
        f00 += score_d * score_d
        f01 += score_d * score_k
        f11 += score_k * score_k
        psi += (D_paths - K_paths * sine) * DT + residual

    # Keep independent path estimates for path-level bootstrap uncertainty.
    pathwise = np.empty((n_parameters, n_replicates, 2, 2))
    pathwise[:, :, 0, 0] = f00.reshape(n_parameters, n_replicates)
    pathwise[:, :, 0, 1] = f01.reshape(n_parameters, n_replicates)
    pathwise[:, :, 1, 0] = pathwise[:, :, 0, 1]
    pathwise[:, :, 1, 1] = f11.reshape(n_parameters, n_replicates)
    return (
        pathwise.mean(axis=1).reshape(n_k, n_d, 2, 2),
        pathwise.reshape(n_k, n_d, n_replicates, 2, 2),
    )


def sensitivity_at_fixed_K(D_values, quadrature):
    """Sweep known noise; duration rescales F without changing its condition."""
    by_noise = {}
    for sigma in NOISE_LEVELS:
        _, eig, cond, _ = analytic_grid(
            D_values, np.array([K_CROSS]), quadrature, sigma=sigma
        )
        by_noise[sigma] = (eig[0], cond[0])
    return by_noise


def convergence_check():
    """Check phase-grid and tilted-potential integral convergence separately."""
    detunings = np.array([0.0, 1.0 / 15.0, K_CROSS, 0.12])
    sigmas = (0.04, SIGMA, 0.10)
    quadratures = {n: phase_quadrature(n) for n in (256, 512, 1024)}
    shift_quadratures = {
        n: phase_quadrature(N_PHASE, n_shift=n) for n in (16, 32, 64, 128, 256)
    }
    rows = []
    for sigma in sigmas:
        for D in detunings:
            results = {}
            for n, quad in quadratures.items():
                density = stationary_density(D, K_CROSS, sigma, quad)
                lam = np.linalg.eigvalsh(
                    fisher_from_density(density, quad[0], sigma)
                )[0]
                results[n] = (density, lam)
            reference_density, reference_lam = results[1024]
            row = {"sigma": sigma, "D": D, "ratio": D / K_CROSS}
            for n in (256, 512):
                density, lam = results[n]
                l1 = (2.0 * np.pi / n) * np.abs(
                    density - reference_density[:: 1024 // n]
                ).sum()
                row[f"density_l1_{n}"] = l1
                row[f"lambda_relative_{n}"] = abs(lam - reference_lam) / reference_lam
            for n_shift, quad in shift_quadratures.items():
                density = stationary_density(D, K_CROSS, sigma, quad)
                lam = np.linalg.eigvalsh(
                    fisher_from_density(density, quad[0], sigma)
                )[0]
                row[f"shift_relative_{n_shift}"] = abs(lam - reference_lam) / reference_lam
            rows.append(row)
    return rows


def bootstrap_edge_cases(quadrature, rng):
    """Independent-path bootstrap CIs at the center, bottleneck, edge, outside."""
    selected_D = np.array([0.0, 1.0 / 15.0, K_CROSS, 0.12])
    analytical, _, _, cdf = analytic_grid(
        selected_D, np.array([K_CROSS]), quadrature, keep_cdf=True
    )
    empirical, pathwise = monte_carlo_fim(
        selected_D,
        np.array([K_CROSS]),
        cdf,
        quadrature[0],
        rng,
        n_replicates=N_CI_REPLICATES,
    )
    intervals = np.empty((len(selected_D), 2))
    for j in range(len(selected_D)):
        indices = rng.integers(
            0, N_CI_REPLICATES, size=(N_BOOTSTRAP, N_CI_REPLICATES)
        )
        bootstrap_fim = pathwise[0, j, indices].mean(axis=1)
        bootstrap_small = np.linalg.eigvalsh(bootstrap_fim)[:, 0]
        intervals[j] = np.percentile(bootstrap_small, [2.5, 97.5])
    return (
        selected_D,
        np.linalg.eigvalsh(analytical)[0, :, 0],
        np.linalg.eigvalsh(empirical)[0, :, 0],
        intervals,
    )


def save_figure(fig, stem):
    """Keep both a high-resolution raster and a vector original."""
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_main(D_values, K_values, eigenvalues, condition, mc_D, mc_K, mc_eigenvalues):
    """Show condition, weakest information direction, and tongue cross-section."""
    plt.rcParams.update(
        {
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fffefd",
            "axes.edgecolor": "#b9c2c0",
            "axes.labelcolor": "#34443f",
            "text.color": "#34443f",
            "xtick.color": "#586b65",
            "ytick.color": "#586b65",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "#fbfaf7",
        }
    )
    soft = LinearSegmentedColormap.from_list(
        "soft_teal", ["#f0f5ef", "#c8dfd5", "#83b7ad", "#3d827f", "#235660"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), constrained_layout=True)
    extent = [D_values[0], D_values[-1], K_values[0], K_values[-1]]
    image0 = axes[0].imshow(
        np.log10(condition), origin="lower", extent=extent, aspect="auto", cmap=soft
    )
    image1 = axes[1].imshow(
        np.log10(eigenvalues[..., 0]),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=soft,
    )
    for ax in axes[:2]:
        ax.plot(K_values, K_values, color="#da8b69", lw=1.8, label=r"$|D|=K$")
        ax.plot(-K_values, K_values, color="#da8b69", lw=1.8)
        ax.set_xlabel(r"Detuning $D$ (rad/h)")
        ax.set_ylabel(r"Coupling $K$ (rad/h)")
        ax.set_xlim(D_values[0], D_values[-1])
        ax.set_ylim(K_values[0], K_values[-1])
    axes[0].set_title("A  FIM condition number")
    axes[1].set_title("B  Smallest FIM eigenvalue")
    axes[0].legend(frameon=False, loc="upper left")
    fig.colorbar(image0, ax=axes[0], shrink=0.83, label=r"$\log_{10}(\lambda_{max}/\lambda_{min})$")
    fig.colorbar(
        image1,
        ax=axes[1],
        shrink=0.83,
        label=r"$\log_{10}[\lambda_{min}/(\mathrm{h}^2\,\mathrm{rad}^{-2})]$",
    )

    cross_index = np.argmin(np.abs(K_values - K_CROSS))
    cross_min = eigenvalues[cross_index, :, 0]
    mc_cross_index = np.argmin(np.abs(mc_K - K_CROSS))
    axes[2].plot(D_values, np.log10(cross_min), color="#356f70", lw=2.3, label="Stationary FIM")
    axes[2].scatter(
        mc_D,
        np.log10(mc_eigenvalues[mc_cross_index, :, 0]),
        s=18,
        facecolor="#da8b69",
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
        label="Monte Carlo",
    )
    axes[2].axvline(-K_values[cross_index], color="#da8b69", lw=1.4, ls="--")
    axes[2].axvline(K_values[cross_index], color="#da8b69", lw=1.4, ls="--", label=r"$|D|=K$")
    axes[2].set_title(fr"C  Cross-section at $K={K_values[cross_index]:.3f}$ rad/h")
    axes[2].set_xlabel(r"Detuning $D$ (rad/h)")
    axes[2].set_ylabel(r"$\log_{10}[\lambda_{min}/(\mathrm{h}^2\,\mathrm{rad}^{-2})]$")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Noisy Adler equation · 30 days of phase observations", fontsize=14)
    return save_figure(fig, "adler_fisher")


def plot_validation(analytic, empirical, edge_cases, convergence):
    """Show broad validation, bootstrap intervals, and grid convergence."""
    analytic_eigs = np.linalg.eigvalsh(analytic)[..., 0].ravel()
    empirical_eigs = np.linalg.eigvalsh(empirical)[..., 0].ravel()
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), constrained_layout=True)
    ax = axes[0]
    a = np.log10(analytic_eigs)
    b = np.log10(empirical_eigs)
    low = min(a.min(), b.min())
    high = max(a.max(), b.max())
    ax.scatter(a, b, s=13, color="#508f87", alpha=0.53, edgecolors="none")
    ax.plot([low, high], [low, high], color="#d58a6a", lw=1.5)
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel=r"Stationary $\log_{10}[\lambda_{min}/(\mathrm{h}^2\,\mathrm{rad}^{-2})]$",
        ylabel=r"Monte Carlo $\log_{10}[\lambda_{min}/(\mathrm{h}^2\,\mathrm{rad}^{-2})]$",
    )
    ax.set_title("A  25 × 25 Monte Carlo comparison")

    selected_D, analytic_small, mc_small, intervals = edge_cases
    ax = axes[1]
    locations = np.arange(len(selected_D))
    ax.errorbar(
        locations,
        mc_small / analytic_small,
        yerr=np.vstack(
            ((mc_small - intervals[:, 0]) / analytic_small,
             (intervals[:, 1] - mc_small) / analytic_small)
        ),
        fmt="o",
        color="#356f70",
        ecolor="#84aca4",
        capsize=3,
        markersize=5,
        label="Monte Carlo, 95% bootstrap CI",
    )
    ax.axhline(1.0, color="#d58a6a", lw=1.5, label="Stationary FIM")
    ax.set_xticks(locations, [f"{D:.3f}" for D in selected_D])
    ax.set_xlabel(r"Detuning $D$ (rad/h), $K=0.080$ rad/h")
    ax.set_ylabel(r"$\lambda_{min}^{MC}/\lambda_{min}^{stationary}$")
    ax.set_title("B  Independent-path uncertainty")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    colors = {0.04: "#416f87", SIGMA: "#447c70", 0.10: "#ad775b"}
    shift_counts = (16, 32, 64, 128, 256)
    for sigma, color in colors.items():
        subset = [row for row in convergence if row["sigma"] == sigma]
        ax.plot(
            shift_counts,
            [
                max(row[f"shift_relative_{n}"] for row in subset)
                for n in shift_counts
            ],
            marker="o",
            lw=1.8,
            color=color,
            label=fr"$\sigma={sigma:.2f}$",
        )
    ax.axvline(256, color="#d58a6a", ls="--", lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(shift_counts, [str(n) for n in shift_counts])
    ax.set_xlabel("Gauss–Legendre nodes in periodic integral")
    ax.set_ylabel(r"Worst relative $\lambda_{min}$ error")
    ax.set_title("C  Periodic integral convergence")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Validation of the stationary Fisher calculation", fontsize=14)
    return save_figure(fig, "adler_mc_validation")


def plot_sensitivity(D_values, by_noise, base_eigenvalues):
    """Noise changes the stationary phase distribution; T rescales F exactly."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7), constrained_layout=True)
    colors = {0.04: "#416f87", 0.06: "#447c70", 0.08: "#ad775b", 0.10: "#766c95"}
    for sigma, (eigenvalues, condition) in by_noise.items():
        color = colors[sigma]
        axes[0].plot(D_values, np.log10(eigenvalues[:, 0]), color=color, lw=2, label=fr"$\sigma={sigma:.2f}$")
        axes[1].plot(D_values, np.log10(condition), color=color, lw=2, label=fr"$\sigma={sigma:.2f}$")
    for days in OBSERVATION_DAYS:
        scaled = base_eigenvalues[:, 0] * days / 30.0
        axes[2].plot(D_values, np.log10(scaled), lw=2, label=f"{days} days")
    for ax in axes:
        ax.axvline(-K_CROSS, color="#d58a6a", ls="--", lw=1.2)
        ax.axvline(K_CROSS, color="#d58a6a", ls="--", lw=1.2)
        ax.set_xlabel(r"Detuning $D$ (rad/h)")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("A  Noise and weakest information")
    axes[0].set_ylabel(r"$\log_{10}[\lambda_{min}/(\mathrm{h}^2\,\mathrm{rad}^{-2})]$")
    axes[1].set_title("B  Noise and condition number")
    axes[1].set_ylabel(r"$\log_{10}(\lambda_{max}/\lambda_{min})$")
    axes[2].set_title(r"C  Observation duration at $\sigma=0.06$")
    axes[2].set_ylabel(r"$\log_{10}[\lambda_{min}/(\mathrm{h}^2\,\mathrm{rad}^{-2})]$")
    fig.suptitle(r"Sensitivity at fixed coupling $K=0.080$ rad/h", fontsize=14)
    return save_figure(fig, "adler_sensitivity")


def wrap_angle(value):
    """Principal circular difference; winding errors are counted in simulation."""
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def simulate_recovery(D, K, quadrature, rng, days=(30,), measurement_sd=(0.0,),
                      replicates=RECOVERY_REPLICATES):
    """Stream likelihood sufficient statistics from independent circular records.

    Observations are y_k = (psi_k + e_k) mod 2*pi, e_k ~ N(0, tau^2).
    No latent phase or innovations are supplied to the estimators. Each tau
    shares the same latent paths, making comparisons paired. For tau=0,
    principal increments give the Gaussian Euler likelihood provided that
    no winding ambiguity occurs; ambiguities are explicitly counted.
    """
    D, K = np.asarray(D), np.asarray(K)
    phase = quadrature[0]
    cdfs = np.array([
        np.cumsum(stationary_density(d, k, SIGMA, quadrature))
        * (2.0 * np.pi / len(phase)) for d, k in zip(D, K)
    ])
    psi = np.array([
        phase[np.minimum(np.searchsorted(cdf, rng.random(replicates)), len(phase)-1)]
        for cdf in cdfs
    ])
    taus = np.asarray(measurement_sd)[:, None, None]
    errors = rng.normal(size=(len(taus), len(D), replicates)) * taus
    observed = wrap_angle(psi[None] + errors)
    # sum sin(y), sin^2(y), cos(y), delta_y, sin(y)*delta_y, delta_y^2/dt
    stats = np.zeros((6, len(taus), len(D), replicates))
    snapshots = {}
    winding_counts = np.zeros(len(taus), dtype=np.int64)
    snapshot_steps = {round(day * 24 / DT): day for day in days}
    for step in range(1, max(snapshot_steps) + 1):
        sine = np.sin(observed)
        delta = (D[:, None] - K[:, None] * np.sin(psi)) * DT
        delta += SIGMA * np.sqrt(DT) * rng.standard_normal(psi.shape)
        psi = wrap_angle(psi + delta)
        next_errors = rng.normal(size=errors.shape) * taus
        next_observed = wrap_angle(psi[None] + next_errors)
        increments = wrap_angle(next_observed - observed)
        winding_counts += np.count_nonzero(
            np.abs(delta[None] + next_errors - errors) >= np.pi, axis=(1, 2)
        )
        stats[0] += sine
        stats[1] += sine**2
        stats[2] += np.cos(observed)
        stats[3] += increments
        stats[4] += sine * increments
        stats[5] += increments**2 / DT
        observed, errors = next_observed, next_errors
        if step in snapshot_steps:
            snapshots[snapshot_steps[step]] = stats.copy()
    return snapshots, winding_counts


def fit_increment_likelihood(stats, n_steps):
    """Conditional Gaussian Euler MLE, observed information and profile CIs.

    G = dt sum x_k x_k^T, h = sum x_k delta_y_k, x_k = (1,-sin(y_k)).
    theta_hat = G^-1 h and RSS = sum delta_y_k^2/dt - theta_hat^T h.
    Unknown diffusion has sigma_hat^2 = RSS/n. Profile LR intervals are
    asymptotic: random, state-dependent regressors preclude an exact t claim.
    The signed K estimate is unconstrained; boundary hits are reported.
    """
    s, s2, _, dy, sdy, dy2 = stats
    shape = s.shape
    gram = np.empty(shape + (2, 2))
    gram[..., 0, 0] = n_steps * DT
    gram[..., 0, 1] = gram[..., 1, 0] = -DT * s
    gram[..., 1, 1] = DT * s2
    rhs = np.stack((dy, -sdy), axis=-1)
    inverse = np.linalg.inv(gram)
    theta = np.einsum("...ij,...j->...i", inverse, rhs)
    rss = dy2 - np.sum(theta * rhs, axis=-1)
    if np.any(rss <= 0) or not np.all(np.isfinite(theta)):
        raise FloatingPointError("Invalid likelihood fit: check numerical precision.")
    variance_diagonal = np.diagonal(inverse, axis1=-2, axis2=-1)
    cutoff = chi2.ppf(0.95, 1)
    known_width = np.sqrt(cutoff * SIGMA**2 * variance_diagonal)
    unknown_width = np.sqrt(
        (rss * np.expm1(cutoff / n_steps))[..., None] * variance_diagonal
    )
    return dict(theta=theta, gram=gram, inverse=inverse, rss=rss,
                sigma=np.sqrt(rss / n_steps), known_width=known_width,
                unknown_width=unknown_width, n_steps=n_steps)


def fit_corrected_moments(stats, n_steps, tau):
    """Moment correction for a KNOWN wrapped Gaussian observation SD tau.

    a = exp(-tau^2/2), E sin(y)=a E sin(psi),
    E sin^2(psi)=(1-exp(2*tau^2) E cos(2*y))/2,
    E[sin(psi)*delta_psi] = (E[sin(y)*delta_y]+tau^2 E cos(y))/a.
    The last identity removes the correlation between the predictor's
    observation error and the differenced response. These expectation
    identities assume correctly recovered increments, not stationarity.
    This is a moment estimator, not a latent-state maximum likelihood fit.
    """
    s, s2, c, dy, sdy, _ = stats
    a = np.exp(-0.5 * tau**2)
    ms = s / (n_steps * a)
    ms2 = (1.0 - np.exp(2.0 * tau**2) * (1.0 - 2.0*s2/n_steps)) / 2.0
    variance = ms2 - ms**2
    # An estimated Gram matrix can cease to be positive definite. Report
    # those failures instead of silently clipping or regularizing them.
    valid = variance > 1e-12
    drift_mean = dy / (n_steps * DT)
    sine_drift = (sdy + tau**2*c) / (n_steps * DT * a)
    coupling = np.divide(ms*drift_mean-sine_drift, variance,
                         out=np.full_like(ms, np.nan), where=valid)
    detuning = drift_mean + coupling*ms
    return np.stack((detuning, coupling), axis=-1), valid


def recovery_metrics(fit, truths):
    error = fit['theta'] - truths[:, None, :]
    joint_distance = np.einsum('...i,...ij,...j->...', error, fit['gram'], error)
    joint_known = joint_distance / SIGMA**2 <= chi2.ppf(.95, 2)
    joint_unknown = fit['n_steps'] * np.log1p(joint_distance/fit['rss']) <= chi2.ppf(.95, 2)
    return dict(
        bias=error.mean(axis=1), rmse=np.sqrt(np.mean(error**2, axis=1)),
        known_coverage=(np.abs(error) <= fit['known_width']).mean(axis=1),
        unknown_coverage=(np.abs(error) <= fit['unknown_width']).mean(axis=1),
        joint_known=joint_known.mean(axis=1), joint_unknown=joint_unknown.mean(axis=1),
        negative_K=(fit['theta'][..., 1] < 0).mean(axis=1),
        sigma_bias=(fit['sigma']-SIGMA).mean(axis=1),
        sigma_rmse=np.sqrt(np.mean((fit['sigma']-SIGMA)**2, axis=1)),
        bias_mc_se=error.std(axis=1, ddof=1)/np.sqrt(error.shape[1]),
        median_absolute_error=np.median(np.abs(error), axis=1),
        p95_absolute_error=np.percentile(np.abs(error), 95, axis=1),
    )


def wilson_interval(proportion, n):
    """95% binomial Monte Carlo uncertainty for an empirical coverage rate."""
    z = 1.959963984540054
    center = (proportion + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def parameter_recovery_study(quadrature):
    """Separate seeded streams for grid recovery, duration, and observation error."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED+1).spawn(3)]
    d = np.unique(np.r_[np.linspace(-.16, .16, 17), -1/15, 1/15])
    k = np.linspace(.04, .12, 9)
    D, K = np.meshgrid(d, k)
    truths = np.column_stack((D.ravel(), K.ravel()))
    print('Fitting independent trajectories on a 19 x 9 recovery grid...', flush=True)
    snapshots, wraps = simulate_recovery(*truths.T, quadrature, streams[0])
    fit = fit_increment_likelihood(snapshots[30][:, 0], round(T/DT))
    metrics = recovery_metrics(fit, truths)
    analytic, _, _, _ = analytic_grid(d, k, quadrature)
    benchmark = np.sqrt(np.diagonal(np.linalg.inv(analytic), axis1=-2, axis2=-1))

    selected_D = np.array([0., 1/15, .08, .12])
    selected_truths = np.column_stack((selected_D, np.full(4, K_CROSS)))
    print('Fitting 7-, 30-, and 90-day records and circular-error stress cases...', flush=True)
    duration_stats, duration_wraps = simulate_recovery(
        *selected_truths.T, quadrature, streams[1], days=RECOVERY_DAYS,
        replicates=1024,
    )
    durations = {}
    bootstrap_rng = np.random.default_rng(SEED+2)
    for days, stats in duration_stats.items():
        current = fit_increment_likelihood(stats[:, 0], round(days*24/DT))
        current_metrics = recovery_metrics(current, selected_truths)
        intervals = []
        for j in range(len(selected_truths)):
            squared_error = (current['theta'][j, :, 1]-K_CROSS)**2
            indices = bootstrap_rng.integers(0, 1024, (N_BOOTSTRAP, 1024))
            boot = np.sqrt(np.mean(squared_error[indices], axis=1))
            intervals.append(np.percentile(boot, [2.5, 97.5]))
        current_metrics['K_rmse_bootstrap'] = np.array(intervals)
        durations[days] = (current, current_metrics)

    observation_stats, observation_wraps = simulate_recovery(
        *selected_truths.T, quadrature, streams[2], measurement_sd=MEASUREMENT_SD,
    )
    observation = []
    for j, tau in enumerate(MEASUREMENT_SD):
        stats = observation_stats[30][:, j]
        naive = fit_increment_likelihood(stats, round(T/DT))
        corrected, valid = fit_corrected_moments(stats, round(T/DT), tau)
        if tau == 0 and not np.allclose(corrected, naive['theta'], atol=1e-9):
            raise AssertionError('Zero-error correction must equal the Euler MLE.')
        observation.append(dict(
            tau=tau, naive=naive, corrected=corrected, valid=valid,
            metrics=recovery_metrics(naive, selected_truths),
        ))
    return dict(d=d, k=k, truths=truths, fit=fit, metrics=metrics,
                benchmark=benchmark, selected_truths=selected_truths,
                durations=durations, observation=observation,
                winding_counts=np.r_[wraps, duration_wraps, observation_wraps])


def plot_recovery(study):
    d, k, metrics = study['d'], study['k'], study['metrics']
    shape = (len(k), len(d))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for j, name in enumerate(('D', 'K')):
        mesh = axes[0, j].pcolormesh(d, k, metrics['rmse'][:, j].reshape(shape),
                                     shading='nearest', cmap='GnBu')
        fig.colorbar(mesh, ax=axes[0, j], label=f'{name} RMSE (rad/h)', shrink=.8)
        axes[0, j].set_title(f"{'AB'[j]}  Recovery error in {name}")
    mesh = axes[0, 2].pcolormesh(d, k, metrics['joint_known'].reshape(shape),
                                shading='nearest', cmap='YlGnBu', vmin=.7, vmax=1)
    fig.colorbar(mesh, ax=axes[0, 2], label='Joint coverage', shrink=.8, extend='min')
    axes[0, 2].set_title('C  Coverage of nominal 95% joint region')
    for ax in axes[0]:
        ax.plot(k, k, color='#b97454', lw=1.2)
        ax.plot(-k, k, color='#b97454', lw=1.2)
        ax.set(xlabel='Detuning D (rad/h)', ylabel='Coupling K (rad/h)',
               xlim=(d[0], d[-1]), ylim=(k[0], k[-1]))
    cross = np.argmin(abs(k-K_CROSS))
    row = slice(cross*len(d), (cross+1)*len(d))
    colors = ('#416f87', '#ad775b')
    for j, name in enumerate(('D', 'K')):
        axes[1, 0].plot(d, metrics['rmse'][row, j], color=colors[j], label=f'{name}: measured RMSE')
        axes[1, 0].plot(d, study['benchmark'][cross, :, j], '--', color=colors[j],
                       label=f'{name}: stationary benchmark')
    axes[1, 0].set(xlabel='Detuning D (rad/h)', ylabel='Error (rad/h)',
                   title='D  Fixed K = 0.080 rad/h')
    axes[1, 0].legend(frameon=False, fontsize=8)
    for name, label, color in (
        ('known_coverage', 'Known sigma', '#416f87'),
        ('unknown_coverage', 'Profiled sigma', '#ad775b'),
    ):
        coverage = metrics[name][row, 1]
        lo, hi = wilson_interval(coverage, RECOVERY_REPLICATES)
        axes[1, 1].plot(d, coverage, color=color, label=label)
        axes[1, 1].fill_between(d, lo, hi, color=color, alpha=.13)
    axes[1, 1].axhline(.95, color='#777777', ls=':', lw=1)
    axes[1, 1].set(xlabel='Detuning D (rad/h)', ylabel='Marginal K coverage',
                   title='E  95% profile-likelihood intervals', ylim=(.65, 1.01))
    axes[1, 1].legend(frameon=False, fontsize=8)
    for ax in axes[1, :2]:
        for edge in (-K_CROSS, K_CROSS):
            ax.axvline(edge, color='#b97454', ls=':', lw=1)
    for j, (D, _) in enumerate(study['selected_truths']):
        days = np.array(RECOVERY_DAYS)
        errors = np.array([study['durations'][int(t)][1]['rmse'][j, 1] for t in days])
        interval = np.array([study['durations'][int(t)][1]['K_rmse_bootstrap'][j] for t in days])
        axes[1, 2].errorbar(days, errors,
                           yerr=np.maximum(0, np.vstack((errors-interval[:, 0], interval[:, 1]-errors))),
                           fmt='o-', capsize=3, color=('#416f87','#ad775b','#447c70','#766c95')[j],
                           label=f'D = {D:.3f}')
    axes[1, 2].set(xscale='log', yscale='log', xlabel='Observation duration (days)',
                   ylabel='K RMSE (rad/h)', title='F  Recovery from longer records')
    axes[1, 2].set_xticks(RECOVERY_DAYS, [str(n) for n in RECOVERY_DAYS])
    axes[1, 2].legend(frameon=False, fontsize=8)
    fig.suptitle('Parameter recovery from observed circular phase · 512 records per grid point')
    return save_figure(fig, 'adler_recovery')


def plot_likelihoods(study, quadrature):
    fit = study['durations'][30][0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    for col, (j, label) in enumerate(((0, 'Locked center'), (1, 'Inside edge'), (3, 'Unlocked'))):
        D, K = study['selected_truths'][j]
        ax = axes[0, col]
        estimates = fit['theta'][j]
        ax.scatter(estimates[:, 0], estimates[:, 1], s=7, alpha=.22, color='#447c70',
                   rasterized=True, label='Independent estimates')
        ax.plot(D, K, '+', color='#ad775b', ms=11, mew=2, label='True parameters')
        density = stationary_density(D, K, SIGMA, quadrature)
        covariance = np.linalg.inv(fisher_from_density(density, quadrature[0]))
        values, vectors = np.linalg.eigh(covariance)
        circle = np.array([np.cos(np.linspace(0, 2*np.pi, 180)),
                           np.sin(np.linspace(0, 2*np.pi, 180))])
        ellipse = (vectors * np.sqrt(values*chi2.ppf(.95, 2))) @ circle
        ax.plot(D+ellipse[0], K+ellipse[1], color='#416f87', lw=1.5,
                label='Stationary 95% Gaussian ellipse')
        ax.set(xlabel='Estimated D (rad/h)', ylabel='Estimated K (rad/h)',
               title=f'{label}: D = {D:.3f}, K = {K:.3f}')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if col == 0:
            ax.legend(frameon=False, fontsize=7)
        # Record 0 was selected before seeing any likelihood or estimate.
        khat = fit['theta'][j, 0, 1]
        invkk = fit['inverse'][j, 0, 1, 1]
        width = fit['known_width'][j, 0, 1]
        domain = np.linspace(min(khat-2*width, K-width), max(khat+2*width, K+width), 300)
        increase = (domain-khat)**2 / invkk
        ax = axes[1, col]
        ax.plot(domain, increase/SIGMA**2, color='#416f87', label='Known sigma')
        ax.plot(domain, fit['n_steps']*np.log1p(increase/fit['rss'][j, 0]),
                '--', color='#ad775b', label='Profiled sigma')
        ax.axhline(chi2.ppf(.95, 1), color='#777777', ls=':', label='95% LR threshold')
        ax.axvline(K, color='#447c70', lw=1, label='True K')
        ax.set(xlabel='Candidate K (rad/h)', ylabel='Twice profile log-likelihood loss', ylim=(0, 12))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        if col == 0:
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Finite-record likelihoods · top: 1,024 estimates; bottom: preselected record 0')
    return save_figure(fig, 'adler_likelihoods')


def plot_observation_error(study):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), constrained_layout=True)
    for col, j in enumerate((0, 1, 3)):
        truth = study['selected_truths'][j]
        for row, (parameter, label) in enumerate(((0, 'D'), (1, 'K'))):
            ax = axes[row, col]
            for method, name, color, style in (
                ('naive', 'Ignore observation error', '#ad775b', '-'),
                ('corrected', 'Known-error moment correction', '#447c70', '--'),
            ):
                errors = [
                    (case['naive']['theta'][j, :, parameter] if method == 'naive'
                     else case['corrected'][j, :, parameter]) - truth[parameter]
                    for case in study['observation']
                ]
                bias = np.array([np.nanmean(e) for e in errors])
                sem = np.array([np.nanstd(e, ddof=1)/np.sqrt(np.isfinite(e).sum()) for e in errors])
                ax.errorbar(MEASUREMENT_SD, bias, yerr=1.96*sem,
                            fmt='o'+style, color=color, capsize=3, label=name)
            ax.axhline(0, color='#777777', ls=':', lw=1)
            ax.set(xlabel='Observation error SD (rad)', ylabel=f'{label} bias (rad/h)')
            if row == 0:
                ax.set_title(f'D = {truth[0]:.3f}, K = {truth[1]:.3f} rad/h')
            if row == 1 and col == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.suptitle('Circular observation error · bars show Monte Carlo uncertainty in mean bias')
    return save_figure(fig, 'adler_observation_error')


def save_recovery_results(study, quadrature):
    """Machine-readable estimates, summary metrics, and a bounded research claim."""
    rows = []
    for idx, (D, K) in enumerate(study['truths']):
        m = study['metrics']
        for p, name in enumerate(('D', 'K')):
            lo, hi = wilson_interval(m['known_coverage'][idx, p], RECOVERY_REPLICATES)
            rows.append(dict(D=D, K=K, parameter=name, bias=m['bias'][idx, p],
                rmse=m['rmse'][idx, p], known_sigma_coverage=m['known_coverage'][idx, p],
                bias_mc_se=m['bias_mc_se'][idx, p],
                median_absolute_error=m['median_absolute_error'][idx, p],
                p95_absolute_error=m['p95_absolute_error'][idx, p],
                coverage_mc_low=lo, coverage_mc_high=hi,
                unknown_sigma_coverage=m['unknown_coverage'][idx, p],
                joint_known_coverage=m['joint_known'][idx],
                joint_unknown_coverage=m['joint_unknown'][idx],
                negative_K_fraction=m['negative_K'][idx],
                sigma_bias=m['sigma_bias'][idx], sigma_rmse=m['sigma_rmse'][idx]))
    with (OUT_DIR/'adler_recovery_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    observation_rows = []
    for case in study['observation']:
        for j, (D, K) in enumerate(study['selected_truths']):
            for method in ('naive', 'corrected'):
                estimates = case['naive']['theta'][j] if method == 'naive' else case['corrected'][j]
                error = estimates - (D, K)
                observation_rows.append(dict(D=D, K=K, tau=case['tau'], method=method,
                    bias_D=np.nanmean(error[:, 0]), bias_K=np.nanmean(error[:, 1]),
                    rmse_D=np.sqrt(np.nanmean(error[:, 0]**2)),
                    rmse_K=np.sqrt(np.nanmean(error[:, 1]**2)),
                    failures=int(np.any(~np.isfinite(estimates), axis=1).sum()),
                    mean_naive_sigma=case['naive']['sigma'][j].mean()))
    with (OUT_DIR/'adler_observation_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(observation_rows[0]))
        writer.writeheader()
        writer.writerows(observation_rows)
    arrays = dict(truths=study['truths'], theta_hat=study['fit']['theta'],
                  sigma_hat=study['fit']['sigma'], gram=study['fit']['gram'],
                  known_width=study['fit']['known_width'], unknown_width=study['fit']['unknown_width'])
    arrays['selected_truths'] = study['selected_truths']
    for days, (fit, _) in study['durations'].items():
        arrays[f'theta_{days}days'] = fit['theta']
        arrays[f'gram_{days}days'] = fit['gram']
        arrays[f'rss_{days}days'] = fit['rss']
        arrays[f'K_rmse_bootstrap_{days}days'] = study['durations'][days][1]['K_rmse_bootstrap']
    for i, case in enumerate(study['observation']):
        arrays[f'observation_{i}_naive'] = case['naive']['theta']
        arrays[f'observation_{i}_corrected'] = case['corrected']
    np.savez_compressed(OUT_DIR/'adler_recovery_results.npz', **arrays)
    metadata = dict(seed=SEED+1, replicates=RECOVERY_REPLICATES, duration_replicates=1024,
                    dt=DT, sigma=SIGMA, duration_days=RECOVERY_DAYS,
                    observation_error_sd=MEASUREMENT_SD, numpy_version=np.__version__,
                    winding_ambiguities=int(study['winding_counts'].sum()))
    (OUT_DIR/'adler_recovery_metadata.json').write_text(json.dumps(metadata, indent=2))

    lines = [
        '# Parameter recovery under the noisy Adler model', '',
        'These are simulation results for directly observed phase and a Gaussian Euler '
        'transition. They establish estimator behavior under the stated model; they do '
        'not establish novelty, experimental validity, or general identifiability for '
        'latent circadian measurements.', '',
        f'Grid: {len(study["truths"])} parameter points; {RECOVERY_REPLICATES} independent '
        '30-day records per point. Four selected points have 1,024 records observed at '
        '7, 30 and 90 days. Starts are drawn from the continuous stationary density. '
        'The conditional likelihood omits information in the initial density.', '',
        '## Actual parameter recovery', '',
        '| D (rad/h) | Days | K bias | K RMSE (95% bootstrap interval) | Known-sigma K coverage | Profiled-sigma K coverage |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for j, (D, K) in enumerate(study['selected_truths']):
        for days, (_, m) in study['durations'].items():
            lo, hi = m['K_rmse_bootstrap'][j]
            lines.append(f'| {D:.4f} | {days} | {m["bias"][j,1]:.4g} | '
                         f'{m["rmse"][j,1]:.4g} ({lo:.4g}, {hi:.4g}) | {m["known_coverage"][j,1]:.3f} | '
                         f'{m["unknown_coverage"][j,1]:.3f} |')
    lines += ['', 'Nominal likelihood coverage is 95%. With 1,024 records, the binomial '
              'standard error near 95% is about 0.7 percentage points. Grid coverage '
              'uncertainty is recorded as Wilson intervals in the CSV. The signed K '
              'MLE is unconstrained; negative estimates are counted.', '',
              '## What the likelihood analysis means', '',
              'The conditional Euler likelihood is quadratic in (D,K), so its '
              'unconstrained MLE is an exact two-regressor least-squares fit. '
              'Known-sigma profiles are quadratic; with sigma profiled out, '
              '2 delta log L = n log(1 + delta RSS/RSS_hat). Nominal intervals '
              'use chi-square thresholds. They are asymptotic because the '
              'regressors depend on the evolving stochastic phase. Estimating '
              'sigma changes interval widths but leaves these drift MLEs unchanged.', '',
              'The stationary inverse FIM is an asymptotic covariance benchmark. '
              'Finite records can be biased and explore the phase distribution '
              'unevenly, especially near the locking boundary. A confidence interval '
              'for an ensemble-averaged FIM is not a parameter confidence interval.', '',
              '## Small-noise conditioning is not zero absolute information', '',
              'For fixed r=D/K strictly inside the tongue, let c=sqrt(1-r^2). '
              'Linearizing around the stable phase gives Var(psi) approximately '
              'sigma^2/(2 K c), hence Var(sin psi) approximately sigma^2 c/(2K). '
              'The resulting stationary-information asymptotics are:', '',
              r'$$\lambda_{\min}\simeq\frac{T c}{2K(1+r^2)},\qquad '
              r'\lambda_{\max}\simeq\frac{T(1+r^2)}{\sigma^2}.$$', '',
              'Thus the condition number diverges while the weakest information '
              'approaches a positive limit. In particular, det(F)=(T/sigma^2)^2 '
              'Var(sin psi) does not tend to zero in this limit: the sigma^-4 '
              'prefactor must be retained. These approximations are nonuniform '
              'at the edge (c=0) and do not apply to the exactly deterministic '
              'experiment. The rank loss occurs in the normalized moment matrix.', '',
              '## Circular observations', '',
              'Observed y = (psi + e) mod 2pi with independent Gaussian e of known '
              'SD tau. Fits use only observed phase differences and sin(y). '
              'All scenarios share the same latent paths. Naive fits treat '
              'observation error as process noise; corrected fits use the exact '
              'wrapped-Gaussian trigonometric moment identities and correct the '
              'correlation induced by differencing observation errors. The correction '
              'requires known tau and correctly recovered increments. It is a '
              'moment estimator, with no claimed individual confidence coverage. '
              'A non-positive corrected design variance is reported as a failure.', '',
              f'Observed winding ambiguities: {int(study["winding_counts"].sum())}. '
              'The experiment therefore does not assess low-frequency sampling with '
              'unresolved phase wraps. Unknown observation-noise variance would '
              'require a separate model or calibration.', '',
              '| D | tau (rad) | Method | D bias | K bias | K RMSE | Failed fits |',
              '|---:|---:|:---|---:|---:|---:|---:|']
    for r in observation_rows:
        lines.append(f'| {r["D"]:.4f} | {r["tau"]:.2f} | {r["method"]} | '
                     f'{r["bias_D"]:.4g} | {r["bias_K"]:.4g} | {r["rmse_K"]:.4g} | {r["failures"]} |')
    lines += ['', '## Scope of a manuscript claim', '',
              'The defensible claim is about joint recovery of detuning and coupling '
              'under a specified phase observation model, including the gap between '
              'stationary information and finite-record recovery. Measurement-error '
              'sensitivity is a separate, practically relevant failure mode. '
              'The noise-free equilibrium rank loss does not imply structural '
              'nonidentifiability at positive process noise: the stationary density '
              'has full support, so Var(sin psi) is positive.', '',
              'The Gaussian Euler model and stationary diffusion approximation are '
              'still idealizations. These results do not fit real data, validate '
              'phase extraction from a circadian signal, establish a new theorem, '
              'or certify a novel result relative to the literature.']
    (OUT_DIR/'adler_recovery_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return observation_rows


def main():
    quadrature = phase_quadrature()
    # Fine grid spans both tongue boundaries D = +/- K.
    D_values = np.linspace(-0.16, 0.16, 81)
    K_values = np.linspace(0.02, 0.14, 61)
    _, eigenvalues, condition, _ = analytic_grid(D_values, K_values, quadrature)

    # Independent 25 x 25 grid; start paths in stationarity to avoid burn-in bias.
    mc_D = np.linspace(D_values[0], D_values[-1], 25)
    mc_K = np.linspace(K_values[0], K_values[-1], 25)
    analytic_mc, _, _, cdf = analytic_grid(mc_D, mc_K, quadrature, keep_cdf=True)
    rng = np.random.default_rng(SEED)
    empirical_mc, _ = monte_carlo_fim(mc_D, mc_K, cdf, quadrature[0], rng)
    empirical_eigenvalues = np.linalg.eigvalsh(empirical_mc)

    by_noise = sensitivity_at_fixed_K(D_values, quadrature)
    convergence = convergence_check()
    edge_cases = bootstrap_edge_cases(quadrature, rng)

    relative_frobenius = np.linalg.norm(empirical_mc - analytic_mc, axis=(-2, -1)) / np.linalg.norm(
        analytic_mc, axis=(-2, -1)
    )
    analytic_small = np.linalg.eigvalsh(analytic_mc)[..., 0]
    relative_small = np.abs(empirical_eigenvalues[..., 0] - analytic_small) / analytic_small
    main_files = plot_main(D_values, K_values, eigenvalues, condition, mc_D, mc_K, empirical_eigenvalues)
    sensitivity_files = plot_sensitivity(D_values, by_noise, by_noise[SIGMA][0])
    validation_files = plot_validation(analytic_mc, empirical_mc, edge_cases, convergence)

    caption_text = (
        "Figure 1 (adler_fisher). Stationary conditional Fisher information for "
        "detuning D and coupling K in the noisy Adler equation. The stationary "
        "density is the periodic, current-carrying Fokker-Planck solution "
        "evaluated on 512 phase points with a 256-node Gauss-Legendre "
        "integral. T = 720 h, dt = 0.1 h, and known "
        "sigma = 0.06 rad/sqrt(h). A: base-10 logarithm of the condition "
        "number. B: base-10 logarithm of the smallest eigenvalue in "
        "h^2/rad^2. C: K = 0.080 rad/h cross-section; orange points are "
        f"means of {N_REPLICATES} independent 720 h Euler-Maruyama paths "
        "at each coarse-grid location. The orange lines mark the noiseless "
        "boundary |D| = K; noise rounds this boundary.\n\n"
        "Figure 2 (adler_sensitivity). At K = 0.080 rad/h, A and B vary "
        "the known noise amplitude across 0.04, 0.06, 0.08, and 0.10 "
        "rad/sqrt(h) with T = 720 h. C varies observation duration from "
        "7 to 90 days at sigma = 0.06 rad/sqrt(h). For a stationary start, "
        "the expected conditional FIM is proportional to T, so its "
        "condition number is independent of T. Dashed lines mark |D| = K.\n\n"
        "Figure 3 (adler_mc_validation). A: analytic versus empirical "
        f"smallest FIM eigenvalues on a 25 x 25 grid ({N_REPLICATES} "
        "independent paths per parameter point). B: empirical-to-analytic "
        f"eigenvalue ratios from {N_CI_REPLICATES} paths at each of four "
        f"selected detunings; error bars are percentile 95% intervals "
        f"from {N_BOOTSTRAP} bootstrap resamples of entire paths. "
        "C: worst relative eigenvalue error across four detunings as the "
        "number of Gauss-Legendre nodes grows, relative to a 1024-phase-point "
        "and 512-node reference. Bootstrap intervals describe "
        "Monte Carlo variation, not uncertainty in the SDE model.\n\n"
        "Methods note. The analytic FIM conditions on the initial phase "
        "and assumes stationarity. Simulations draw initial phases from "
        "that density. Their score uses the Gaussian Euler increment with "
        "an internally unwrapped phase; the drift uses sin(psi), hence is "
        "periodic. Phase wraps are negligible at this dt and sigma, but "
        "the exact wrapped-normal likelihood and information in the "
        "parameter-dependent initial density are outside this analysis. "
        "The continuous-time stationary density and Euler path law differ "
        "by discretization error of order dt.\n"
    )
    captions_file = OUT_DIR / "adler_figure_captions.txt"
    captions_file.write_text(caption_text, encoding="utf-8")

    inside = np.abs(D_values[None, :]) < K_values[:, None]
    outside = np.abs(D_values[None, :]) > K_values[:, None]
    strong_locked = inside & (K_values[:, None] >= 0.08) & (
        np.abs(D_values[None, :]) <= 0.8 * K_values[:, None]
    )
    edge_index = np.argmin(np.abs(D_values - K_CROSS))
    cross_index = np.argmin(np.abs(K_values - K_CROSS))
    near_edge = (D_values >= 0.0) & (D_values <= K_values[cross_index])
    near_edge_indices = np.flatnonzero(near_edge)
    worst_index = near_edge_indices[np.argmax(condition[cross_index, near_edge])]
    print(f"Grid: {len(D_values)} D × {len(K_values)} K; {N_PHASE} phase points")
    print(f"T = {T:.0f} h, dt = {DT:g} h, sigma = {SIGMA:g} rad/sqrt(h)")
    print(
        "Monte Carlo (25 × 25, "
        f"{N_REPLICATES} paths/point): median FIM error "
        f"{np.median(relative_frobenius):.1%}, 95th percentile "
        f"{np.percentile(relative_frobenius, 95):.1%}; "
        f"median smallest-eigenvalue error {np.median(relative_small):.1%}."
    )
    print(
        f"Condition number: median {np.median(condition[strong_locked]):.1f} "
        f"in the well-locked interior; median {np.median(condition[outside]):.1f} outside."
    )
    print(
        f"At K = {K_values[cross_index]:.3f} rad/h, the worst sampled "
        f"conditioning is {condition[cross_index, worst_index]:.1f} at "
        f"D = {D_values[worst_index]:.3f} rad/h, just inside the +tongue edge. "
        f"At the edge (D = {D_values[edge_index]:.3f}), it falls to "
        f"{condition[cross_index, edge_index]:.1f}."
    )
    for sigma, (_, sensitivity_condition) in by_noise.items():
        interior = (D_values >= 0.0) & (D_values <= K_CROSS)
        indices = np.flatnonzero(interior)
        peak = indices[np.argmax(sensitivity_condition[interior])]
        print(
            f"sigma={sigma:.2f}: peak condition {sensitivity_condition[peak]:.1f} "
            f"at D={D_values[peak]:.3f} rad/h (K={K_CROSS:.3f})."
        )
    print("Duration scaling: lambda_min is proportional to T; condition is unchanged.")
    print(
        "Grid convergence (512 vs 1024 phase points): max relative lambda_min "
        f"error {max(row['lambda_relative_512'] for row in convergence):.3g}; "
        f"max density L1 error {max(row['density_l1_512'] for row in convergence):.3g}."
    )
    selected_D, analytic_edge, mc_edge, intervals = edge_cases
    coverage = (intervals[:, 0] <= analytic_edge) & (analytic_edge <= intervals[:, 1])
    print(
        "Edge-case bootstrap: analytic lambda_min lies in the 95% path "
        f"interval at {coverage.sum()}/{len(coverage)} selected detunings."
    )
    for D, analytical, empirical, interval in zip(
        selected_D, analytic_edge, mc_edge, intervals
    ):
        print(
            f"  D={D:.3f}: analytic {analytical:.1f}, MC {empirical:.1f}, "
            f"95% CI [{interval[0]:.1f}, {interval[1]:.1f}]"
        )
    print(
        "Interpretation: locking concentrates sin(psi), making the FIM poorly "
        "conditioned relative to its strongest direction. The phase "
        "bottleneck can make identifiability worst just inside the edge; "
        "noise rounds the transition, and phase slips restore information "
        "across and beyond the boundary."
    )
    print(
        "Saved "
        + ", ".join(
            p.name for p in (*main_files, *sensitivity_files, *validation_files, captions_file)
        )
    )

    study = parameter_recovery_study(quadrature)
    recovery_files = plot_recovery(study)
    likelihood_files = plot_likelihoods(study, quadrature)
    observation_files = plot_observation_error(study)
    observation_rows = save_recovery_results(study, quadrature)
    with captions_file.open('a', encoding='utf-8') as handle:
        handle.write(
            '\nFigure 4 (adler_recovery). Conditional Gaussian Euler maximum '
            'likelihood fits to observed circular phase on a 19 x 9 parameter '
            'grid, using 512 independent 30-day records per point. A/B: root '
            'mean squared estimation error in D/K, in rad/h. C: actual coverage '
            'of nominal 95% joint likelihood regions with known sigma. D: RMSE '
            'and square root diagonal of the inverse stationary FIM at K=0.08. '
            'E: marginal K likelihood coverage with known or profiled sigma; '
            'shading shows 95% Wilson intervals for Monte Carlo coverage. '
            'F: K RMSE at 7, 30 and 90 days for four detunings, 1,024 paths per '
            'point; bars are percentile 95% intervals from 500 independent-path '
            'bootstrap resamples. Orange boundaries mark |D|=K. The drift MLE is unconstrained.\n\n'
            'Figure 5 (adler_likelihoods). Top: clouds of 1,024 independent '
            '(D,K) estimates, true values, and stationary-FIM 95% Gaussian '
            'reference ellipses for three regimes. Bottom: K profile likelihood '
            'from record 0, selected before inspecting the data, with D profiled '
            'out and sigma either known or profiled out. Profiles use observed '
            'trajectory information. Chi-square thresholds are asymptotic.\n\n'
            'Figure 6 (adler_observation_error). Bias from circular phase '
            'measurements with independent wrapped-Gaussian error of SD '
            '0, 0.01, 0.03 or 0.06 rad. Orange: Euler likelihood fits ignoring '
            'observation error. Green: corrected moment fits using the known '
            'observation SD and the error correlation induced by differencing. '
            'Each condition uses 512 records; error bars are 1.96 Monte Carlo '
            'standard errors of mean bias, not individual confidence intervals. '
            'The correction assumes no unresolved winding and is not a '
            'latent-state MLE. Complete results and failures are in the CSV.\n'
        )
    print('\nPARAMETER RECOVERY (30-day records; 95% nominal marginal K intervals)')
    for j, (D, _) in enumerate(study['selected_truths']):
        m = study['durations'][30][1]
        print(f'  D={D:.4f}: K bias={m["bias"][j,1]:.4g}, '
              f'RMSE={m["rmse"][j,1]:.4g}, '
              f'coverage known/estimated sigma={m["known_coverage"][j,1]:.1%}/'
              f'{m["unknown_coverage"][j,1]:.1%}')
    print(f'Unresolved winding increments across all recovery scenarios: '
          f'{study["winding_counts"].sum()}')
    for row in observation_rows:
        if row['D'] == 1/15 and row['tau'] == .03:
            print(f'  Observation SD=0.03 rad, near edge, {row["method"]}: '
                  f'K bias={row["bias_K"]:.4g}, K RMSE={row["rmse_K"]:.4g}')
    print('Saved ' + ', '.join(p.name for p in
          (*recovery_files, *likelihood_files, *observation_files)))
    print('Saved recovery report, CSV metrics, NPZ estimates, and metadata.')


if __name__ == "__main__":
    main()
