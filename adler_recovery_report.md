# Parameter recovery under the noisy Adler model

These are simulation results for directly observed phase and a Gaussian Euler transition. They establish estimator behavior under the stated model; they do not establish novelty, experimental validity, or general identifiability for latent circadian measurements.

Grid: 171 parameter points; 512 independent 30-day records per point. Four selected points have 1,024 records observed at 7, 30 and 90 days. Starts are drawn from the continuous stationary density. The conditional likelihood omits information in the initial density.

## Actual parameter recovery

| D (rad/h) | Days | K bias | K RMSE (95% bootstrap interval) | Known-sigma K coverage | Profiled-sigma K coverage |
|---:|---:|---:|---:|---:|---:|
| 0.0000 | 7 | 0.02506 | 0.04554 (0.04345, 0.04795) | 0.932 | 0.926 |
| 0.0000 | 30 | 0.005543 | 0.01641 (0.01555, 0.01722) | 0.948 | 0.949 |
| 0.0000 | 90 | 0.001865 | 0.009112 (0.008608, 0.009653) | 0.946 | 0.946 |
| 0.0667 | 7 | 0.04936 | 0.08456 (0.0785, 0.0916) | 0.898 | 0.898 |
| 0.0667 | 30 | 0.01032 | 0.02516 (0.02384, 0.02645) | 0.938 | 0.941 |
| 0.0667 | 90 | 0.003495 | 0.01295 (0.01224, 0.0136) | 0.938 | 0.936 |
| 0.0800 | 7 | 0.02575 | 0.1181 (0.1012, 0.1387) | 0.908 | 0.909 |
| 0.0800 | 30 | 0.001842 | 0.01197 (0.009256, 0.01504) | 0.943 | 0.942 |
| 0.0800 | 90 | 0.0002103 | 0.003235 (0.003086, 0.003385) | 0.954 | 0.953 |
| 0.1200 | 7 | 0.0001812 | 0.007139 (0.006855, 0.00745) | 0.955 | 0.954 |
| 0.1200 | 30 | 0.0001407 | 0.00342 (0.003268, 0.00356) | 0.954 | 0.952 |
| 0.1200 | 90 | 0.0001361 | 0.001962 (0.001877, 0.002049) | 0.958 | 0.956 |

Nominal likelihood coverage is 95%. With 1,024 records, the binomial standard error near 95% is about 0.7 percentage points. Grid coverage uncertainty is recorded as Wilson intervals in the CSV. The signed K MLE is unconstrained; negative estimates are counted.

## What the likelihood analysis means

The conditional Euler likelihood is quadratic in (D,K), so its unconstrained MLE is an exact two-regressor least-squares fit. Known-sigma profiles are quadratic; with sigma profiled out, 2 delta log L = n log(1 + delta RSS/RSS_hat). Nominal intervals use chi-square thresholds. They are asymptotic because the regressors depend on the evolving stochastic phase. Estimating sigma changes interval widths but leaves these drift MLEs unchanged.

The stationary inverse FIM is an asymptotic covariance benchmark. Finite records can be biased and explore the phase distribution unevenly, especially near the locking boundary. A confidence interval for an ensemble-averaged FIM is not a parameter confidence interval.

## Small-noise conditioning is not zero absolute information

For fixed r=D/K strictly inside the tongue, let c=sqrt(1-r^2). Linearizing around the stable phase gives Var(psi) approximately sigma^2/(2 K c), hence Var(sin psi) approximately sigma^2 c/(2K). The resulting stationary-information asymptotics are:

$$\lambda_{\min}\simeq\frac{T c}{2K(1+r^2)},\qquad \lambda_{\max}\simeq\frac{T(1+r^2)}{\sigma^2}.$$

Thus the condition number diverges while the weakest information approaches a positive limit. In particular, det(F)=(T/sigma^2)^2 Var(sin psi) does not tend to zero in this limit: the sigma^-4 prefactor must be retained. These approximations are nonuniform at the edge (c=0) and do not apply to the exactly deterministic experiment. The rank loss occurs in the normalized moment matrix.

## Circular observations

Observed y = (psi + e) mod 2pi with independent Gaussian e of known SD tau. Fits use only observed phase differences and sin(y). All scenarios share the same latent paths. Naive fits treat observation error as process noise; corrected fits use the exact wrapped-Gaussian trigonometric moment identities and correct the correlation induced by differencing observation errors. The correction requires known tau and correctly recovered increments. It is a moment estimator, with no claimed individual confidence coverage. A non-positive corrected design variance is reported as a failure.

Observed winding ambiguities: 0. The experiment therefore does not assess low-frequency sampling with unresolved phase wraps. Unknown observation-noise variance would require a separate model or calibration.

| D | tau (rad) | Method | D bias | K bias | K RMSE | Failed fits |
|---:|---:|:---|---:|---:|---:|---:|
| 0.0000 | 0.00 | naive | -0.0001058 | 0.005374 | 0.01641 | 0 |
| 0.0000 | 0.00 | corrected | -0.0001058 | 0.005374 | 0.01641 | 0 |
| 0.0667 | 0.00 | naive | 0.0102 | 0.01212 | 0.02593 | 0 |
| 0.0667 | 0.00 | corrected | 0.0102 | 0.01212 | 0.02593 | 0 |
| 0.0800 | 0.00 | naive | 0.0007613 | 0.0007591 | 0.00808 | 0 |
| 0.0800 | 0.00 | corrected | 0.0007613 | 0.0007591 | 0.00808 | 0 |
| 0.1200 | 0.00 | naive | 0.0002291 | -7.712e-05 | 0.00339 | 0 |
| 0.1200 | 0.00 | corrected | 0.0002291 | -7.712e-05 | 0.00339 | 0 |
| 0.0000 | 0.01 | naive | -0.0001503 | 0.05199 | 0.05719 | 0 |
| 0.0000 | 0.01 | corrected | -0.0001069 | 0.005413 | 0.01658 | 0 |
| 0.0667 | 0.01 | naive | 0.05206 | 0.06233 | 0.072 | 0 |
| 0.0667 | 0.01 | corrected | 0.01024 | 0.01215 | 0.02602 | 0 |
| 0.0800 | 0.01 | naive | 0.002587 | 0.002821 | 0.01401 | 0 |
| 0.0800 | 0.01 | corrected | 0.0007769 | 0.0007768 | 0.008123 | 0 |
| 0.1200 | 0.01 | naive | 0.0002298 | -6.958e-05 | 0.003389 | 0 |
| 0.1200 | 0.01 | corrected | 0.0002283 | -7.79e-05 | 0.003389 | 0 |
| 0.0000 | 0.03 | naive | -0.000471 | 0.4099 | 0.4188 | 0 |
| 0.0000 | 0.03 | corrected | -0.0001064 | 0.005982 | 0.01998 | 0 |
| 0.0667 | 0.03 | naive | 0.3766 | 0.4518 | 0.4728 | 0 |
| 0.0667 | 0.03 | corrected | 0.009205 | 0.01093 | 0.02686 | 0 |
| 0.0800 | 0.03 | naive | 0.01692 | 0.01902 | 0.07696 | 0 |
| 0.0800 | 0.03 | corrected | 0.0008452 | 0.000849 | 0.009376 | 0 |
| 0.1200 | 0.03 | naive | 0.0002408 | -9.483e-06 | 0.003446 | 0 |
| 0.1200 | 0.03 | corrected | 0.000227 | -8.414e-05 | 0.003414 | 0 |
| 0.0000 | 0.06 | naive | -0.00139 | 1.444 | 1.464 | 0 |
| 0.0000 | 0.06 | corrected | -8.145e-05 | 0.004908 | 0.04109 | 0 |
| 0.0667 | 0.06 | naive | 1.369 | 1.645 | 1.701 | 0 |
| 0.0667 | 0.06 | corrected | 0.009195 | 0.01093 | 0.04947 | 0 |
| 0.0800 | 0.06 | naive | 0.06176 | 0.06999 | 0.2714 | 0 |
| 0.0800 | 0.06 | corrected | 0.0005263 | 0.0004763 | 0.0153 | 0 |
| 0.1200 | 0.06 | naive | 0.0002541 | 0.0001247 | 0.004047 | 0 |
| 0.1200 | 0.06 | corrected | 0.0001986 | -0.0001749 | 0.003616 | 0 |

## Scope of a manuscript claim

The defensible claim is about joint recovery of detuning and coupling under a specified phase observation model, including the gap between stationary information and finite-record recovery. Measurement-error sensitivity is a separate, practically relevant failure mode. The noise-free equilibrium rank loss does not imply structural nonidentifiability at positive process noise: the stationary density has full support, so Var(sin psi) is positive.

The Gaussian Euler model and stationary diffusion approximation are still idealizations. These results do not fit real data, validate phase extraction from a circadian signal, establish a new theorem, or certify a novel result relative to the literature.
