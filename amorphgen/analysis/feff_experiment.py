"""Measured EXAFS chi(k) against a FEFF ensemble, on one grid and one transform.

Follows :mod:`amorphgen.analysis.experiment`: measured data keep their own k
points, every structure is interpolated there before averaging, nothing is
extrapolated, and excluded points are counted. Residuals are calculated minus
observed. Bands describe sampling of independent structures only, never
measurement error or the uncertainty of fitted adjustments.

Both chi(R) curves are computed here from chi(k), with one window on the same
k points, so they are like-for-like. A measured chi(R) file cannot be used
instead: its window and k-weight are not recorded in it.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ._serialization import numpy_json_default as _json_default
from .experiment import _interpolate_segments, _measurement, _read_table
from .feff import _chir_bootstrap, chi_fourier_transform
from .uncertainty import summarize_structures

#: 2 m_e / hbar^2 in eV^-1 A^-2 (CODATA 2018), so that k^2 = C_K * (E - E0).
C_K = 0.26246842

#: Adjustments that can be fitted, and the keys holding their values.
FIT_PARAMETERS = ("amplitude", "e0", "sigma2", "offset")
_KEYS = {"amplitude": "amplitude", "e0": "delta_e0", "sigma2": "delta_sigma2",
         "offset": "offset"}
# Bounds and starts of the a-Ga2O3 fit in the AmorphGen paper.
_BOUNDS = {"amplitude": (0.0, np.inf), "e0": (-15.0, 15.0),
           "sigma2": (-0.01, 0.03), "offset": (-np.inf, np.inf)}
_STARTS = {"e0": (-10.0, 0.0, 10.0), "sigma2": (0.0, 0.003)}


def _kweight(value, name):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or not 0 <= value <= 3):
        raise ValueError(f"{name} must be an integer from 0 to 3")
    return int(value)


def _units(kweight):
    return {"x": "1/angstrom",
            "observed": "dimensionless" if kweight == 0 else f"angstrom^-{kweight}"}


def load_exafs_experiment(path, kweight=0, *, columns=None, delimiter=None,
                          skiprows=0, comments="#") -> dict:
    """Load a measured chi(k) from a numeric text or CSV file.

    Columns are k (1/A) and chi, or k^w chi with ``kweight=w`` declaring the
    weight already applied (2 for an Athena ``chik2`` column), plus an
    optional positive one-sigma error of that stored column. File parsing,
    ``columns`` selection and rejection rules are those of
    :func:`~amorphgen.analysis.experiment.load_experiment`. The stored column
    is kept as read; :func:`compare_feff_experiment` divides out the weight.
    """
    kweight = _kweight(kweight, "kweight")
    data, columns, delimiter = _read_table(path, columns, delimiter, skiprows, comments)
    k, observed, sigma = _measurement({"x": data[:, 0], "observed": data[:, 1],
                                      "sigma": data[:, 2] if data.shape[1] == 3 else None})
    return {"kind": "chik", "x": k.tolist(), "observed": observed.tolist(),
            "sigma": sigma.tolist() if sigma is not None else None,
            "kweight": kweight, "source": str(path), "units": _units(kweight),
            "columns": list(columns) if columns is not None else list(range(data.shape[1])),
            "delimiter": delimiter, "skiprows": int(skiprows), "comments": comments}


def _range(value, name, lower=None):
    bounds = np.asarray(value, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]
            or (lower is not None and bounds[0] < lower)):
        raise ValueError(f"{name} must be two finite values, lower < upper"
                         + (f", from {lower:g}" if lower is not None else ""))
    return [float(bounds[0]), float(bounds[1])]


def _fit_names(fit):
    if not fit:
        return []
    names = [n.strip() for n in fit.split(",") if n.strip()] if isinstance(fit, str) else list(fit)
    if any(n not in FIT_PARAMETERS for n in names) or len(set(names)) != len(names):
        raise ValueError(f"fit must name distinct adjustments from {', '.join(FIT_PARAMETERS)}")
    return [n for n in FIT_PARAMETERS if n in names]


def _fit_adjustments(k, target, weight, model_k, mean, w_e, kw, fixed, names):
    """Fit the named adjustments of the mean model to measured k^w chi.

    The recipe of the paper's a-Ga2O3 fit: a linear amplitude/offset start,
    then starts at dE0 in {-10, 0, 10} eV and dsigma2 in {0, 0.003} A^2 for
    whichever are free, keeping the lowest cost.
    """
    from scipy.optimize import least_squares

    lo_e0, hi_e0 = _BOUNDS["e0"] if "e0" in names else (fixed["delta_e0"],) * 2
    if (k.min() ** 2 - C_K * hi_e0 < model_k[0] ** 2
            or k.max() ** 2 - C_K * lo_e0 > model_k[-1] ** 2):
        raise ValueError(
            f"the model (k = {model_k[0]:g}-{model_k[-1]:g} 1/A) does not cover the fit "
            f"range {k.min():g}-{k.max():g} 1/A for every trial dE0 "
            f"({lo_e0:g} to {hi_e0:g} eV); narrow the k range")
    weighting = k ** kw

    def evaluate(values):
        shifted = np.sqrt(k ** 2 - C_K * values["delta_e0"])
        chi = (values["amplitude"] * np.interp(shifted, model_k, mean)
               * np.exp(-2.0 * values["delta_sigma2"] * k ** 2))
        return weighting * (chi + values["offset"] / k ** w_e)

    def solve(start, free, **options):
        keys = [_KEYS[n] for n in free]

        def residual(p):
            return (evaluate({**start, **dict(zip(keys, p))}) - target) * weight

        bounds = ([_BOUNDS[n][0] for n in free], [_BOUNDS[n][1] for n in free])
        return least_squares(residual, [start[key] for key in keys], bounds=bounds, **options)

    start = dict(fixed)
    for name in ("e0", "sigma2"):
        if name in names:
            start[_KEYS[name]] = 0.0
    linear = [n for n in ("amplitude", "offset") if n in names]
    if linear:
        start.update(zip((_KEYS[n] for n in linear), solve(start, linear).x))
    if "amplitude" in names:
        start["amplitude"] = max(start["amplitude"], 0.01)
    best = None
    for e0 in _STARTS["e0"] if "e0" in names else (start["delta_e0"],):
        for sigma2 in _STARTS["sigma2"] if "sigma2" in names else (start["delta_sigma2"],):
            trial = solve({**start, "delta_e0": e0, "delta_sigma2": sigma2}, names,
                          max_nfev=3000, x_scale="jac")
            if best is None or trial.cost < best.cost:
                best = trial
    if not best.success:
        raise ValueError(f"the fit did not converge: {best.message}")
    values = dict(zip((_KEYS[n] for n in names), map(float, best.x)))
    return values, {"fitted": names, "n_parameters": len(names),
                    "at_bound": {n: bool(a) for n, a in zip(names, best.active_mask)},
                    "cost": float(best.cost), "n_points": int(len(k)),
                    "bounds": {n: [b if np.isfinite(b) else None for b in _BOUNDS[n]]
                               for n in names},
                    "message": str(best.message)}


def compare_feff_experiment(result, experiment, *, amplitude=1.0, delta_e0=0.0,
                            delta_sigma2=0.0, offset=0.0, fit=(), k_range=None,
                            kmin=None, kmax=None, r_range=(1.0, 4.0),
                            confidence=0.95, n_bootstrap=1000, seed=0) -> dict:
    """Compare a collected FEFF ensemble with a measured chi(k).

    ``result`` comes from :func:`~amorphgen.analysis.feff.collect_feff_results`
    and ``experiment`` from :func:`load_exafs_experiment`. Every structure's
    chi_s(k) is adjusted as::

        A * chi_s(sqrt(k^2 - C_K*dE0)) * exp(-2*dsigma2*k^2) + offset / k^w_e

    (w_e: the weight of the measured column), interpolated onto the measured
    k points, then averaged with equal structure weights. The dE0 sign is that
    of IFEFFIT, Artemis and Larch (theory evaluated at k'), so a dE0 from a
    shell fit applies directly. ``offset`` is a constant in the measured
    column's weighting, for digitised curves with a display offset; it is part
    of the model in both k and R space.

    ``fit`` names adjustments (``amplitude``, ``e0``, ``sigma2``, ``offset``)
    to fit by least squares to k^w chi of the ensemble mean over ``k_range``,
    with the bounds and starts of the paper's a-Ga2O3 fit; the others keep the
    values given. Measured sigma, when present, weights the fit. The bands use
    the fitted values held fixed, so they describe structure sampling only.

    Both chi(k) are transformed on the measured k points the model supports,
    with the result's k-weight, taper and kstep. ``kmin`` / ``kmax`` default
    to the collected window and are moved inside that support if needed (and
    reported); values passed explicitly must fit, or the error names both
    ranges. ``k_range`` (default: the requested window within the measured
    data) sets the fit and the k-space metrics, ``r_range`` the R-space ones.

    Metrics, k space (k^w chi over ``k_range``): ``r_factor`` =
    sum(r^2)/sum(y^2) as in Artemis/Larch; ``normalized_squared_residual`` =
    sum(r^2)/sum((y - mean y)^2), the paper's statistic; ``rmse``; chi-square
    only with measured sigma (dof = N - fitted). R space (over ``r_range``,
    real and imaginary parts): ``r_factor`` = sum(|dchi|^2)/sum(|chi_obs|^2),
    ``d_r`` = sqrt(r_factor), and ``n_independent`` = 2*dk*dR/pi. All are
    descriptive: neighbouring points are correlated. The return value is
    JSON-native.
    """
    if not isinstance(experiment, Mapping) or experiment.get("kind") != "chik":
        raise ValueError("experiment must be a mapping returned by load_exafs_experiment")
    w_e = _kweight(experiment.get("kweight"), "experiment kweight")
    x, stored, sigma = _measurement(experiment)
    if not isinstance(result, Mapping):
        raise ValueError("result must be a mapping returned by collect_feff_results")
    for key in ("k", "chi", "kweight", "ft_kmin", "ft_kmax", "ft_dk"):
        if key not in result:
            raise ValueError(f"result lacks {key!r}; use collect_feff_results")
    model_k = np.asarray(result["k"], dtype=float)
    rows = result.get("chi_per_structure")
    rows = np.atleast_2d(np.asarray(result["chi"] if rows is None else rows, dtype=float))
    if (model_k.ndim != 1 or len(model_k) < 2 or rows.ndim != 2
            or rows.shape[1] != len(model_k) or not np.isfinite(rows).all()):
        raise ValueError("result must hold finite chi curves on its k grid")
    kw = _kweight(result["kweight"], "result kweight")
    dk = float(result["ft_dk"])
    kstep = float(np.round(model_k[1] - model_k[0], 10))
    fixed = {"amplitude": amplitude, "delta_e0": delta_e0,
             "delta_sigma2": delta_sigma2, "offset": offset}
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
           or not np.isfinite(v) for v in fixed.values()):
        raise ValueError("amplitude, delta_e0, delta_sigma2 and offset must be finite numbers")
    fixed = {key: float(value) for key, value in fixed.items()}
    names = _fit_names(fit)
    requested = [float(result["ft_kmin"]) if kmin is None else float(kmin),
                 float(result["ft_kmax"]) if kmax is None else float(kmax)]
    _range(requested, "kmin and kmax", lower=0.0)
    r_range = _range(r_range, "r_range", lower=0.0)

    positive = x > 0
    scale = np.where(positive, np.abs(x) ** w_e, np.nan)
    observed_chi = stored / scale
    sigma_chi = sigma / scale if sigma is not None else None
    if not positive.any():
        raise ValueError("no measured points at k > 0")
    k_range = (_range(k_range, "k_range", lower=0.0) if k_range is not None else
               _range([max(requested[0], x[positive][0]), min(requested[1], x[-1])],
                      "the requested window within the measured data"))
    in_k = positive & (x >= k_range[0]) & (x <= k_range[1])
    if in_k.sum() < 2:
        raise ValueError(f"fewer than two measured points in k_range {k_range}")

    fitting = {"fitted": [], "n_parameters": 0}
    adjustments = dict(fixed)
    if names:
        sigma_fit = x[in_k] ** kw * sigma_chi[in_k] if sigma_chi is not None else None
        weight = (np.ones(in_k.sum()) if sigma_fit is None else sigma_fit.min() / sigma_fit)
        values, fitting = _fit_adjustments(
            x[in_k], x[in_k] ** kw * observed_chi[in_k], weight, model_k,
            rows.mean(axis=0), w_e, kw, fixed, names)
        fitting["weighting"] = ("inverse measured variance" if sigma_fit is not None
                                else "unit weights")
        adjustments.update(values)

    shifted_sq = x ** 2 - C_K * adjustments["delta_e0"]
    shifted = np.sqrt(np.where(positive & (shifted_sq >= 0), shifted_sq, np.nan))
    aligned = np.asarray([_interpolate_segments(model_k, row, shifted) for row in rows])
    aligned = (adjustments["amplitude"] * aligned
               * np.exp(-2.0 * adjustments["delta_sigma2"] * x ** 2)
               + adjustments["offset"] / scale)
    supported = np.isfinite(aligned).all(axis=0)
    idx = np.flatnonzero(supported)
    if len(idx) < 2:
        raise ValueError("fewer than two measured points lie inside the model's k range")
    if np.any(np.diff(idx) != 1):
        raise ValueError("model support of the measured points is not contiguous")
    below_edge = positive & (shifted_sq < 0)
    counts = {"n_input_points": int(len(x)), "n_points": int(len(idx)),
              "n_excluded_nonpositive_k": int((~positive).sum()),
              "n_excluded_below_shifted_edge": int(below_edge.sum()),
              "n_excluded_outside_model": int((positive & ~below_edge & ~supported).sum()),
              "k_min": float(x[idx[0]]), "k_max": float(x[idx[-1]]),
              "model_k_min": float(model_k[0]), "model_k_max": float(model_k[-1])}

    # One window for both transforms, inside the k range they share.
    lo, hi, tol = x[idx[0]], x[idx[-1]], 1e-9
    window = {"kweight": kw, "kmin": requested[0], "kmax": requested[1], "dk": dk,
              "kstep": kstep}
    moved = {"kmin": False, "kmax": False}
    for name, explicit, outside, edge in (
            ("kmin", kmin is not None, requested[0] - dk / 2 < lo - tol, lo + dk / 2),
            ("kmax", kmax is not None, requested[1] + dk / 2 > hi + tol, hi - dk / 2)):
        if outside:
            if explicit:
                raise ValueError(
                    f"FT window {requested[0]:g}-{requested[1]:g} 1/A (dk = {dk:g}) does not "
                    f"fit inside the k range shared by the measured data and the model, "
                    f"{lo:.3f}-{hi:.3f} 1/A")
            window[name], moved[name] = float(edge), True
    if window["kmax"] <= window["kmin"]:
        raise ValueError(f"FT window is empty inside the shared k range {lo:.3f}-{hi:.3f} 1/A")

    k_used = x[idx]
    rows_used = aligned[:, idx]
    summary = summarize_structures(rows_used * k_used ** kw, confidence=confidence,
                                   n_bootstrap=n_bootstrap, seed=seed)
    observed_ft = chi_fourier_transform(k_used, observed_chi[idx], **window)
    calculated_ft = chi_fourier_transform(k_used, rows_used.mean(axis=0), **window)
    rmax = float(np.max(result["r"])) if len(result.get("r", [])) else 6.0
    keep = observed_ft["r"] <= rmax + tol
    r = observed_ft["r"][keep]
    chir_band = (_chir_bootstrap(k_used, rows_used, window, int(keep.sum()), confidence,
                                 n_bootstrap, seed)
                 if len(rows) > 1 and n_bootstrap > 0 else None)

    observed_kw = k_used ** kw * observed_chi[idx]
    calculated_kw = np.asarray(summary["mean"], dtype=float)
    residual_kw = calculated_kw - observed_kw
    sigma_kw = k_used ** kw * sigma_chi[idx] if sigma_chi is not None else None
    in_k_used = in_k[idx]
    y, res = observed_kw[in_k_used], residual_kw[in_k_used]
    total, spread = float(np.sum(y ** 2)), float(np.sum((y - y.mean()) ** 2))
    chi_square = (float(np.sum((res / sigma_kw[in_k_used]) ** 2))
                  if sigma_kw is not None else None)
    dof = int(in_k_used.sum()) - fitting["n_parameters"]
    k_metrics = {"n_points": int(in_k_used.sum()),
                 "r_factor": float(np.sum(res ** 2)) / total if total > 0 else None,
                 "normalized_squared_residual": (float(np.sum(res ** 2)) / spread
                                                 if spread > 0 else None),
                 "rmse": float(np.sqrt(np.mean(res ** 2))),
                 "chi_square": chi_square,
                 "degrees_of_freedom": dof if chi_square is not None else None,
                 "reduced_chi_square": (chi_square / dof if chi_square is not None and dof > 0
                                        else None)}

    parts = {}
    for label, ft in (("observed", observed_ft), ("calculated", calculated_ft)):
        parts[label] = {"re": ft["chir_re"][keep], "im": ft["chir_im"][keep],
                        "mag": ft["chir_mag"][keep]}
    d_re = parts["calculated"]["re"] - parts["observed"]["re"]
    d_im = parts["calculated"]["im"] - parts["observed"]["im"]
    in_r = (r >= r_range[0]) & (r <= r_range[1])
    denominator = float(np.sum(parts["observed"]["re"][in_r] ** 2
                               + parts["observed"]["im"][in_r] ** 2))
    r_factor = (float(np.sum(d_re[in_r] ** 2 + d_im[in_r] ** 2)) / denominator
                if denominator > 0 else None)
    r_metrics = {"n_points": int(in_r.sum()), "r_factor": r_factor,
                 "d_r": float(np.sqrt(r_factor)) if r_factor is not None else None,
                 "n_independent": float(2.0 * (observed_ft["kmax"] - observed_ft["kmin"])
                                        * (r_range[1] - r_range[0]) / np.pi)}

    window.update(kmin=observed_ft["kmin"], kmax=observed_ft["kmax"],
                  requested_kmin=requested[0], requested_kmax=requested[1],
                  kmin_moved=moved["kmin"], kmax_moved=moved["kmax"],
                  shape="hanning")
    return {
        "kind": "chik", "absorber": result.get("absorber"), "edge": result.get("edge"),
        "n_sites": result.get("n_sites"), "n_structures": int(len(rows)),
        "k": k_used.tolist(), "observed_chi": observed_chi[idx].tolist(),
        "observed_kwchi": observed_kw.tolist(),
        "sigma_kwchi": sigma_kw.tolist() if sigma_kw is not None else None,
        "calculated_chi": rows_used.mean(axis=0).tolist(),
        "calculated_kwchi": calculated_kw.tolist(), "residual_kwchi": residual_kw.tolist(),
        "window_k": np.interp(k_used, observed_ft["k"], observed_ft["window"]).tolist(),
        "in_k_range": in_k_used.tolist(), "uncertainty": summary,
        "included_indices": idx.tolist(),
        "excluded_indices": np.flatnonzero(~supported).tolist(),
        "r": r.tolist(),
        "observed_chir": {key: value.tolist() for key, value in parts["observed"].items()},
        "calculated_chir": {key: value.tolist() for key, value in parts["calculated"].items()},
        "residual_chir": {"re": d_re.tolist(), "im": d_im.tolist(),
                          "mag": np.hypot(d_re, d_im).tolist()},
        "chir_band": chir_band, "in_r_range": in_r.tolist(),
        "adjustments": adjustments, "fitting": fitting, "window": window,
        "k_range": k_range, "r_range": r_range,
        "metrics": {"k": k_metrics, "r": r_metrics}, "overlap": counts,
        "source": experiment.get("source"),
        "experiment_metadata": {key: experiment[key] for key in (
            "kind", "kweight", "source", "units", "columns", "delimiter", "skiprows",
            "comments") if key in experiment},
        "units": {"k": "1/angstrom", "kwchi": f"angstrom^-{kw}" if kw else "dimensionless",
                  "r": "angstrom, phase-uncorrected", "chir": f"angstrom^-{kw + 1}"},
        "constants": {"C_K_eV-1_A-2": C_K},
        "definitions": {
            "model": "A*chi_s(sqrt(k^2 - C_K*dE0))*exp(-2*dsigma2*k^2) + offset/k^w_e "
                     "per structure, then the equal-weight mean",
            "delta_e0_sign": "theory evaluated at k' = sqrt(k^2 - C_K*dE0) (IFEFFIT, Artemis, Larch)",
            "residual": "calculated minus observed",
            "k_r_factor": "sum(residual**2)/sum(observed**2) on k^w chi over k_range",
            "normalized_squared_residual": "sum(residual**2)/sum((observed-mean(observed))**2)",
            "chi_square": "sum((residual/sigma)**2); measurement sigma only; dof = N - fitted",
            "r_r_factor": "sum(dRe**2 + dIm**2)/sum(Re_obs**2 + Im_obs**2) over r_range",
            "d_r": "sqrt(r_r_factor)",
            "n_independent": "2*(kmax - kmin)*(rmax - rmin)/pi (Stern)",
            "ensemble_bands": "structure sampling only; adjustments held fixed; pointwise",
            "transform": "both chi(k) taken on the same measured k points and transformed "
                         "with one Hanning window",
        }}


def _number(value):
    return f"{value:.4g}" if value is not None else "undefined"


def format_feff_experiment_report(result) -> str:
    """Return a readable summary of :func:`compare_feff_experiment`."""
    number = _number
    counts, window, adj = result["overlap"], result["window"], result["adjustments"]
    fitted = set(result["fitting"]["fitted"])
    state = lambda name: "fitted" if name in fitted else "fixed"
    stored = result["experiment_metadata"].get("kweight", 0)
    column = "chi" if stored == 0 else f"k^{stored} chi"
    k_m, r_m = result["metrics"]["k"], result["metrics"]["r"]
    lines = [f"EXAFS comparison with measured chi(k) "
             f"({result.get('absorber')} {result.get('edge')} edge)",
             f"  Measured: {result.get('source')} (stored column {column})",
             f"  Points: {counts['n_points']} / {counts['n_input_points']}; "
             f"k = {counts['k_min']:.3g}-{counts['k_max']:.3g} 1/A",
             f"  Excluded: {counts['n_excluded_nonpositive_k']} at k <= 0; "
             f"{counts['n_excluded_below_shifted_edge']} below the shifted edge; "
             f"{counts['n_excluded_outside_model']} outside the model k range",
             f"  Adjustments: amplitude {adj['amplitude']:.4g} ({state('amplitude')}); "
             f"dE0 {adj['delta_e0']:.3f} eV ({state('e0')}); "
             f"dsigma2 {adj['delta_sigma2']:.5f} A^2 ({state('sigma2')}); "
             f"offset {adj['offset']:.4g} ({state('offset')})"]
    if fitted:
        bound = [n for n, hit in result["fitting"]["at_bound"].items() if hit]
        lines.append(f"  Fit: {result['fitting']['n_points']} points over k = "
                     f"{result['k_range'][0]:g}-{result['k_range'][1]:g} 1/A; "
                     f"{result['fitting']['weighting']}"
                     + (f"; at a bound: {', '.join(bound)}" if bound else ""))
    moved = [n for n in ("kmin", "kmax") if window[f"{n}_moved"]]
    lines.append(f"  FT window: k = {window['kmin']:.3g}-{window['kmax']:.3g} 1/A, Hanning, "
                 f"dk = {window['dk']:g}, k^{window['kweight']}, identical for both curves"
                 + (f"; {' and '.join(moved)} moved inside the shared k range" if moved else ""))
    lines.append(f"  k space, k = {result['k_range'][0]:g}-{result['k_range'][1]:g} 1/A: "
                 f"R-factor {number(k_m['r_factor'])}; normalised squared residual "
                 f"{number(k_m['normalized_squared_residual'])}; RMSE {number(k_m['rmse'])}")
    if k_m["chi_square"] is not None:
        lines.append(f"  Chi-square {number(k_m['chi_square'])}; reduced "
                     f"{number(k_m['reduced_chi_square'])} (dof {k_m['degrees_of_freedom']}; "
                     "measured sigma, correlated points)")
    lines.append(f"  R space, R = {result['r_range'][0]:g}-{result['r_range'][1]:g} A "
                 f"(phase-uncorrected): R-factor {number(r_m['r_factor'])}; "
                 f"D_R {number(r_m['d_r'])}; about {r_m['n_independent']:.0f} independent points")
    lines.append(f"  Ensemble: {result['n_structures']} structures; bands are structure "
                 "sampling only, adjustments held fixed. Residual = calculated - observed.")
    return "\n".join(lines)


def _csv_header(result):
    adj, window = result["adjustments"], result["window"]
    fitted = "+".join(result["fitting"]["fitted"]) or "none"
    return (f"# absorber={result.get('absorber')} edge={result.get('edge')} "
            f"n_sites={result.get('n_sites')} n_structures={result['n_structures']} "
            f"kweight={window['kweight']} window=hanning ft_kmin={window['kmin']:.6g} "
            f"ft_kmax={window['kmax']:.6g} ft_dk={window['dk']:.6g} R=phase-uncorrected "
            f"experiment_kweight={result['experiment_metadata'].get('kweight')} "
            f"amplitude={adj['amplitude']:.6g} delta_e0_eV={adj['delta_e0']:.6g} "
            f"delta_sigma2_A2={adj['delta_sigma2']:.6g} offset={adj['offset']:.6g} "
            f"fitted={fitted} residual=calculated-observed")


def _cell(value):
    """Missing statistics as empty cells, matching JSON null."""
    return "" if value is None or not np.isfinite(value) else f"{value:.8e}"


def save_feff_experiment_comparison(result, output_dir=".", prefix="analysis",
                                    dpi=300, save_pdf=False):
    """Write a :func:`compare_feff_experiment` result as CSV, JSON, text and figures.

    Files, with ``base = {prefix}_feff_experiment``: ``{base}_chik.csv`` (one
    row per measured k used), ``{base}_chir.csv`` (one row per R),
    ``{base}.json``, ``{base}_report.txt`` and ``{base}_chik.png`` /
    ``{base}_chir.png`` (plus ``.pdf`` with ``save_pdf``). Both CSVs start
    with a ``#`` line of the settings. Returns the paths by key.
    """
    # Reject nonstandard JSON numbers before creating any partial exports.
    serialized = json.dumps(result, indent=2, allow_nan=False, default=_json_default)
    from .scattering_output import _SUMMARY_FIELDS

    base = Path(output_dir) / (f"{prefix}_feff_experiment" if prefix else "feff_experiment")
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"json": f"{base}.json", "report": f"{base}_report.txt",
             "chik_csv": f"{base}_chik.csv", "chir_csv": f"{base}_chir.csv",
             "chik_png": f"{base}_chik.png", "chir_png": f"{base}_chir.png"}
    if save_pdf:
        paths.update(chik_pdf=f"{base}_chik.pdf", chir_pdf=f"{base}_chir.pdf")
    Path(paths["json"]).write_text(serialized + "\n")
    Path(paths["report"]).write_text(format_feff_experiment_report(result) + "\n")
    header = _csv_header(result)
    kw = result["window"]["kweight"]
    size = len(result["k"])
    summary = result["uncertainty"]
    sigma = result["sigma_kwchi"] or [None] * size
    stats = [[None if v is None else float(v) for v in summary[field]]
             for field in _SUMMARY_FIELDS]
    with open(paths["chik_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([header])
        w.writerow(["input_index", "k_invA", "observed_chi", f"observed_k{kw}chi",
                    f"sigma_k{kw}chi", "calculated_chi", f"calculated_k{kw}chi",
                    f"residual_k{kw}chi", "n_structures", *_SUMMARY_FIELDS, "confidence",
                    "window", "in_k_range"])
        for i in range(size):
            w.writerow([result["included_indices"][i], f"{result['k'][i]:.6f}",
                        *(_cell(result[key][i]) for key in (
                            "observed_chi", "observed_kwchi")),
                        _cell(sigma[i]),
                        *(_cell(result[key][i]) for key in (
                            "calculated_chi", "calculated_kwchi", "residual_kwchi")),
                        summary["n_per_point"][i], *(_cell(column[i]) for column in stats),
                        summary["confidence"], f"{result['window_k'][i]:.6f}",
                        bool(result["in_k_range"][i])])
    band = result["chir_band"] or {"bootstrap_low": [None] * len(result["r"]),
                                   "bootstrap_high": [None] * len(result["r"])}
    unit = f"invA{kw + 1}"
    obs, calc, res = result["observed_chir"], result["calculated_chir"], result["residual_chir"]
    with open(paths["chir_csv"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([header])
        w.writerow(["R_A", *(f"observed_{p}_{unit}" for p in ("mag", "re", "im")),
                    *(f"calculated_{p}_{unit}" for p in ("mag", "re", "im")),
                    *(f"residual_{p}_{unit}" for p in ("re", "im", "mag")),
                    "calculated_mag_bootstrap_low", "calculated_mag_bootstrap_high",
                    "in_r_range"])
        for i, r in enumerate(result["r"]):
            w.writerow([f"{r:.4f}", *(_cell(obs[p][i]) for p in ("mag", "re", "im")),
                        *(_cell(calc[p][i]) for p in ("mag", "re", "im")),
                        *(_cell(res[p][i]) for p in ("re", "im", "mag")),
                        _cell(band["bootstrap_low"][i]), _cell(band["bootstrap_high"][i]),
                        bool(result["in_r_range"][i])])
    _plot_chik(result, str(base) + "_chik", dpi, save_pdf)
    _plot_chir(result, str(base) + "_chir", dpi, save_pdf)
    return paths


def _two_panels():
    from .plotting import _figure
    return _figure(2, 1, figsize=(7, 5.7), sharex=True,
                   gridspec_kw={"height_ratios": [3, 1]})


def _finish(fig, axes, span, note, base, dpi, save_pdf):
    from .plotting import _apply_pub_style, _save_fig
    for axis in axes:
        axis.axvspan(*span, color="0.93", zorder=0, lw=0)
        _apply_pub_style(axis)
    axes[1].axhline(0.0, color=".5", ls=":", lw=.8)
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    fig.text(.12, .015, note, fontsize=7.5, color=".3")
    fig.tight_layout(rect=(0, .05, 1, 1), h_pad=.4)
    _save_fig(fig, base, dpi=dpi, save_pdf=save_pdf)


def _plot_chik(result, base, dpi, save_pdf):
    from .plotting import _EXP_COLOR, _PALETTE
    from .scattering_output import _band

    kw = result["window"]["kweight"]
    k = np.asarray(result["k"], dtype=float)
    summary = result["uncertainty"]
    as_array = lambda values: np.asarray([np.nan if v is None else v for v in values], float)
    fig, (ax, rax) = _two_panels()
    ax.plot(k, result["observed_kwchi"], color=_EXP_COLOR, lw=1.3, label="Measured")
    ax.plot(k, result["calculated_kwchi"], color=_PALETTE[0], lw=1.3, label="Ensemble mean")
    _band(ax, k, as_array(summary["ci_low"]), as_array(summary["ci_high"]),
          summary["confidence"], _PALETTE[0])
    rax.plot(k, result["residual_kwchi"], color=_PALETTE[0], lw=1.0)
    label = rf"$k^{{{kw}}}\chi(k)$ ($\mathrm{{\AA}}^{{-{kw}}}$)" if kw else r"$\chi(k)$"
    ax.set_ylabel(label)
    rax.set_ylabel("Calc. − obs.")
    rax.set_xlabel(r"$k$ ($\mathrm{\AA}^{-1}$)")
    m = result["metrics"]["k"]
    ax.set_title(f"R-factor = {_number(m['r_factor'])}   N = {m['n_points']}", fontsize=10)
    _finish(fig, (ax, rax), result["k_range"],
            "Shaded: k range of the metrics (and fit). Bands: structure sampling only.",
            base, dpi, save_pdf)


def _plot_chir(result, base, dpi, save_pdf):
    from .plotting import _EXP_COLOR, _PALETTE

    kw = result["window"]["kweight"]
    r = np.asarray(result["r"], dtype=float)
    fig, (ax, rax) = _two_panels()
    ax.plot(r, result["observed_chir"]["mag"], color=_EXP_COLOR, lw=1.3, label="Measured")
    ax.plot(r, result["calculated_chir"]["mag"], color=_PALETTE[0], lw=1.3,
            label="Ensemble mean")
    band = result["chir_band"]
    if band:
        ax.fill_between(r, band["bootstrap_low"], band["bootstrap_high"], color=_PALETTE[0],
                        alpha=.2, lw=0,
                        label=f"{100 * band['confidence']:g}% structure bootstrap")
    rax.plot(r, result["residual_chir"]["mag"], color=_PALETTE[0], lw=1.0)
    ax.set_ylabel(rf"$|\chi(R)|$ ($\mathrm{{\AA}}^{{-{kw + 1}}}$)")
    ax.set_ylim(bottom=0.0)
    rax.set_ylabel(r"$|\Delta\chi(R)|$")
    rax.set_xlabel(r"$R$ ($\mathrm{\AA}$), phase-uncorrected")
    ax.set_xlim(r.min(), r.max())
    w, m = result["window"], result["metrics"]["r"]
    ax.set_title(f"R-factor = {_number(m['r_factor'])}   k = {w['kmin']:.3g}–{w['kmax']:.3g} "
                 r"$\mathrm{\AA}^{-1}$", fontsize=10)
    _finish(fig, (ax, rax), result["r_range"],
            "Shaded: R range of the metrics. Both curves use one window on the same k points.",
            base, dpi, save_pdf)
