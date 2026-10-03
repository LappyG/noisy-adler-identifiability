# Noisy Adler equation: parameter identifiability

This project asks a simple question: when a daily cue pulls a noisy clock into rhythm, can we tell the clock's natural speed from the strength of that pull? It maps where those two parameters are easy or hard to estimate, then checks the prediction by fitting simulated phase records.

## The idea, step by step

**1. Follow the gap between two cycles.** Think of the zeitgeber as a daily cue, such as a light–dark cycle. It has phase $z(t)$; the clock has phase $\phi(t)$. Their gap is an angle on a circle:

$$
\psi(t)=z(t)-\phi(t), \qquad z(t)=\frac{2\pi}{24}t.
$$

An angle of $0$ and an angle of $2\pi$ describe the same alignment. Time is measured in hours, so $2\pi/24$ is the cue's angular speed in rad/h.

**2. Let the cue pull the clock.** If the clock's natural angular speed is $\omega$, its difference from the daily cue is

$$
D=\frac{2\pi}{24}-\omega.
$$

This difference, $D$, is called **detuning**. The noisy Adler equation describes how the phase gap changes:

$$
d\psi=(D-K\sin\psi)\,dt+\sigma\,dW.
$$

Here $K\geq0$ is coupling strength, and $\sigma$ controls random phase fluctuations. The $-K\sin\psi$ term tries to pull the phases toward a stable gap.

**3. Find the locking region.** Without noise, a locked clock holds a constant gap $\psi_{*}$. Setting the rate of change to zero gives

$$
0=D-K\sin\psi_{*}, \qquad \sin\psi_{*} = \frac{D}{K}.
$$

For $K>0$, a fixed gap exists when $|D|\leq K$. Draw that condition over detuning and coupling and it forms the **Arnold tongue**. Inside it, one branch of the fixed gap is stable. Outside it, the gap keeps drifting around the circle. Noise blurs the boundary and can cause occasional phase slips even inside it.

**4. Turn the model into observations.** If we observe the gap every $\Delta t$ hours, the Euler approximation is

$$
\Delta\psi_k \approx (D-K\sin\psi_k)\Delta t
  +\sigma\sqrt{\Delta t}\,\xi_k.
$$

Each $\xi_k$ is an independent standard-normal draw. The simulation uses $\Delta t=0.1$ hour and usually observes 30 days. It fits $D$ and $K$ from these increments and checks how often uncertainty intervals contain the values used to generate them.

**5. Ask whether the parameters can be told apart.** Changing $D$ affects every expected increment equally; changing $K$ has an effect proportional to $-\sin\psi_k$. If the clock stays near one phase, those effects can look almost the same. With a stationary phase distribution and known $\sigma$, the Fisher information is

$$
\mathcal I(D,K)=\frac{T}{\sigma^2}
\begin{pmatrix}
1 & -\mathbb E[\sin\psi] \\
-\mathbb E[\sin\psi] & \mathbb E[\sin^2\psi]
\end{pmatrix}.
$$

Here $T$ is observation time and $\mathbb E$ averages over the stationary phase distribution. The key relationship is

$$
\det\mathcal I=\left(\frac{T}{\sigma^2}\right)^2
\operatorname{Var}(\sin\psi).
$$

The script plots the ratio of the largest and smallest information eigenvalues. A large ratio means one combination of $D$ and $K$ is much harder to estimate than the other. This is **poor conditioning**, not automatically a complete loss of information.

**6. Check what happens when phase measurements are imperfect.** The base model treats phase as directly observed. A separate simulation adds measurement error:

$$
y_k=(\psi_k+\epsilon_k)\bmod 2\pi.
$$

Each $\epsilon_k$ is an independent normal measurement error with standard deviation $\tau$. The new quantity $\tau$ is different from $\sigma$, which describes fluctuations in the clock itself. Ignoring measurement error can make the estimated coupling much too large; the script also checks a correction when $\tau$ is known.

## Run

Requires Python 3, NumPy, SciPy, and Matplotlib. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy matplotlib
python script.py
```

The script uses fixed random seeds and writes its results beside `script.py`. A run replaces files with the same output names. The settings are constants near the top of the script.

## Fit your own phase record

Put directly measured phase observations in a CSV with a header and two columns:

```csv
time_h,phase_rad
0,0.20
0.1,0.21
0.2,0.19
0.3,0.23
```

These rows illustrate the format; use a much longer record for meaningful intervals. `time_h` is elapsed time in hours from the experiment's chosen origin; values must increase, but the spacing may be irregular. `phase_rad` is an angle in radians. By default it is the **phase gap** $\psi=z-\phi$ (cue phase minus clock phase), recorded modulo $2\pi$. Fit it with:

```bash
python fit_phase_csv.py record.csv --output fit.json
```

For a record of the **clock phase** $\phi$ instead, add `--phase-kind clock`. The fitter constructs the cue phase as $z(t)=z(0)+2\pi t/P$ and then uses $\psi=z-\phi$. `P` defaults to 24 hours and $z(0)$ to 0 radians; set them to match your experiment's time origin:

```bash
python fit_phase_csv.py clock.csv --phase-kind clock \
  --cue-period-h 24 --cue-phase-at-zero-rad 0 --output fit.json
```

If `phase_rad` preserves accumulated turns instead of wrapping each angle modulo $2\pi$, pass `--phase-format unwrapped`. By default the fitter estimates the process-noise level $\sigma$ from the record. If it is independently known, pass `--known-sigma` in rad/$\sqrt{\mathrm{h}}$ (for example, `--known-sigma 0.06`). The fit reports detuning $D$ and coupling $K$ in rad/h; the clock's natural angular speed is $\omega=2\pi/P-D$.

The fit uses Gaussian Euler phase increments and assumes that phase itself has already been extracted from the measurements. The 95% intervals are asymptotic and conditional on that model. It does not extract phase from a raw signal or model independent measurement error in those phase values. Wrapped observations can hide turns between samples, so the inferred increments are ambiguous when the gap moves by about $\pi$ radians or more between observations. Even unwrapped observations can give biased Euler fits when sampled too sparsely. By default the fitter rejects gaps over 1 hour or a fitted $\max(|D|,|K|)\,\max(\Delta t)$ over 0.2; these are conservative heuristics, not guarantees of accuracy. `--allow-coarse` overrides them and adds a warning to the result. Check the approximation at your actual sampling cadence, especially near phase slips.

## What the script does

1. Computes the stationary circular density with its nonzero probability current on a 512-point phase grid.
2. Maps the Fisher information matrix, its eigenvalues, and its condition number over detuning and coupling. It checks the calculation against Monte Carlo score averages and a finer integration grid.
3. Fits $D$ and $K$ to independent simulated records. It reports bias, root mean squared error, likelihood interval coverage, and the effect of estimating $\sigma$.
4. Tests circular phase measurement error. It compares a fit that ignores the error with a moment correction that requires the measurement-error standard deviation to be known.

## Outputs

Each figure is saved as both a PNG and a PDF. The filename stems are:

| Figure | Contents |
| --- | --- |
| `adler_fisher` | Fisher information across the tongue and a fixed-coupling cross-section |
| `adler_sensitivity` | Effects of process noise and observation duration |
| `adler_mc_validation` | Monte Carlo and numerical convergence checks |
| `adler_recovery` | Parameter error and interval coverage across the tongue |
| `adler_likelihoods` | Estimate clouds and example profile likelihoods |
| `adler_observation_error` | Bias caused by circular measurement error |

The script also writes [figure captions](adler_figure_captions.txt), a [results and methods report](adler_recovery_report.md), [recovery metrics](adler_recovery_metrics.csv), [observation-error metrics](adler_observation_metrics.csv), [individual estimates](adler_recovery_results.npz), and [run metadata](adler_recovery_metadata.json). It prints a short numerical summary to the terminal.

## Reading the results

With the default settings, coupling recovery is hardest just inside the noiseless tongue edge. At $K=0.08$ rad/h and $D\approx0.067$ rad/h, the 30-day coupling estimate has a root mean squared error of about 0.025 rad/h; at $D=0.12$ rad/h it is about 0.0034 rad/h. Ignoring 0.03-radian phase measurement error near the edge produces substantial coupling bias in this simulation. See the report for uncertainty estimates and the full parameter grid.

Poor Fisher-matrix conditioning should not be read as zero absolute information. For positive process noise the stationary phase density has full support. Inside the tongue, away from its edge, the smallest Fisher eigenvalue approaches a positive limit as process noise decreases even though the condition number grows. The report gives the approximation and its assumptions.

## Scope

The simulation results use directly observed phase under a Gaussian Euler transition model. The analytic Fisher calculation assumes a stationary start and conditions on the initial phase. The measurement-error correction assumes a known error level and resolved phase increments. The CSV fitter accepts phase records but does not include experimental circadian data or validate phase extraction from a measured signal.

## Why I like this work

I like what I am doing here because a familiar picture of synchronization leads to a deeper question: what can the data actually tell us about the clock? This project lets me connect an intuitive story about daily rhythms to dynamical systems, statistics, and reproducible code. I am especially interested in turning the weak spots on the map into better ways to design an experiment.
