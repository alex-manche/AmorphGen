"""Measured chi(k) against a FEFF ensemble: grid, transform, adjustments, files."""
import csv
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from amorphgen.analysis.feff import chi_fourier_transform, collect_feff_results, feff_pipeline
from amorphgen.analysis.feff_experiment import (
    C_K, compare_feff_experiment, format_feff_experiment_report,
    load_exafs_experiment, save_feff_experiment_comparison,
)

K = np.arange(0.0, 20.0001, 0.05)


def shell(k, r, s2=0.004):
    return (4 / (np.maximum(k, .1) * r ** 2) * np.sin(2 * k * r + .3)
            * np.exp(-2 * s2 * k ** 2 - k / 12))


def ensemble_result(n=6, seed=1):
    """A collect_feff_results-like result with n structures on the 0.05 grid."""
    rng = np.random.default_rng(seed)
    per = np.array([shell(K, 1.95 + rng.normal(0, .01)) + .3 * shell(K, 3.3 + rng.normal(0, .03))
                    for _ in range(n)])
    ft = chi_fourier_transform(K, per.mean(0), kmin=3, kmax=19.5, dk=1)
    return {"k": K, "chi": per.mean(0), "chi_per_structure": per if n > 1 else None,
            "kweight": 2, "ft_kmin": 3.0, "ft_kmax": 19.5, "ft_dk": 1.0,
            "r": ft["r"][ft["r"] <= 6], "absorber": "Ga", "edge": "K", "n_sites": 10 * n}


def measured(k, chi, kweight=0, sigma=None):
    k = np.asarray(k, dtype=float)
    return {"kind": "chik", "x": k.tolist(), "observed": (k ** kweight * chi).tolist(),
            "sigma": sigma, "kweight": kweight, "source": "synthetic"}


def adjusted(result, k, amplitude=1.0, delta_e0=0.0, delta_sigma2=0.0):
    return (amplitude * np.interp(np.sqrt(k ** 2 - C_K * delta_e0), K, result["chi"])
            * np.exp(-2 * delta_sigma2 * k ** 2))


KX = np.arange(1.5, 13.0, 0.07)
TRUE = {"amplitude": .85, "delta_e0": 5.0, "delta_sigma2": .002}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


# ─── loader ────────────────────────────────────────────────────────────────

def test_loader_keeps_the_stored_weighted_column(tmp_path):
    path = tmp_path / "ga.chik"
    path.write_text("# k chik2 chik3\n3 4.5 13.5\n2 2 4\n")
    expt = load_exafs_experiment(path, kweight=2, columns=(0, 1))
    assert expt["kind"] == "chik" and expt["kweight"] == 2
    assert expt["x"] == [2, 3] and expt["observed"] == [2, 4.5]
    assert expt["units"] == {"x": "1/angstrom", "observed": "angstrom^-2"}
    assert expt["columns"] == [0, 1] and expt["sigma"] is None
    json.dumps(expt, allow_nan=False)
    for bad in (4, -1, True, 1.0):
        with pytest.raises(ValueError, match="kweight"):
            load_exafs_experiment(path, kweight=bad, columns=(0, 1))
    wide = tmp_path / "athena.chik"
    wide.write_text("# k chi chik chik2\n2 .5 1 2\n3 .5 1.5 4.5\n")
    with pytest.raises(ValueError, match="two or three columns"):
        load_exafs_experiment(wide)
    assert load_exafs_experiment(wide, columns=(0, 3), kweight=2)["observed"] == [2, 4.5]


def test_loader_rejects_rows_like_the_scattering_loader(tmp_path):
    path = tmp_path / "bad.dat"
    path.write_text("1 2\n2 wrong\n")
    with pytest.raises(ValueError, match="cannot read"):
        load_exafs_experiment(path)


# ─── comparison ────────────────────────────────────────────────────────────

def test_identical_spectra_give_identical_transforms():
    result = ensemble_result()
    kx = np.arange(1.03, 13.0, 0.07)
    cmp = compare_feff_experiment(result, measured(kx, np.interp(kx, K, result["chi"])))
    assert cmp["metrics"]["k"]["r_factor"] < 1e-20
    assert cmp["metrics"]["r"]["r_factor"] < 1e-20
    np.testing.assert_allclose(cmp["calculated_chir"]["mag"], cmp["observed_chir"]["mag"],
                               atol=1e-12)
    assert cmp["fitting"] == {"fitted": [], "n_parameters": 0}
    assert cmp["overlap"]["n_points"] == len(kx)
    json.dumps(cmp, allow_nan=False)


def test_fit_recovers_known_adjustments():
    result = ensemble_result()
    expt = measured(KX, adjusted(result, KX, **TRUE), kweight=2)
    cmp = compare_feff_experiment(result, expt, fit=("amplitude", "e0", "sigma2"),
                                  k_range=(3, 12))
    for key, value in TRUE.items():
        assert cmp["adjustments"][key] == pytest.approx(value, rel=1e-6)
    assert cmp["fitting"]["n_parameters"] == 3
    assert not any(cmp["fitting"]["at_bound"].values())
    assert cmp["metrics"]["k"]["r_factor"] < 1e-12
    assert cmp["metrics"]["r"]["r_factor"] < 1e-12
    assert "fitted" in format_feff_experiment_report(cmp)


def test_fixed_adjustments_are_applied_without_fitting():
    result = ensemble_result()
    expt = measured(KX, adjusted(result, KX, **TRUE))
    cmp = compare_feff_experiment(result, expt, **TRUE)
    assert cmp["fitting"]["n_parameters"] == 0
    assert cmp["adjustments"] == {**TRUE, "offset": 0.0}
    assert cmp["metrics"]["k"]["r_factor"] < 1e-20
    raw = compare_feff_experiment(result, expt)
    assert raw["metrics"]["k"]["r_factor"] > .01


def test_offset_is_a_constant_in_the_stored_weighting():
    result = ensemble_result()
    expt = measured(KX, adjusted(result, KX), kweight=2)
    expt["observed"] = (np.asarray(expt["observed"]) + .3).tolist()
    cmp = compare_feff_experiment(result, expt, fit="amplitude,offset", k_range=(3, 12))
    assert cmp["adjustments"]["offset"] == pytest.approx(.3, abs=1e-9)
    assert cmp["adjustments"]["amplitude"] == pytest.approx(1, abs=1e-9)
    assert cmp["metrics"]["k"]["r_factor"] < 1e-12


def test_chi_and_k2chi_inputs_agree():
    result = ensemble_result()
    chi = adjusted(result, KX, **TRUE)
    a = compare_feff_experiment(result, measured(KX, chi, 0), fit=("e0",))
    b = compare_feff_experiment(result, measured(KX, chi, 2), fit=("e0",))
    assert a["adjustments"]["delta_e0"] == pytest.approx(b["adjustments"]["delta_e0"])
    np.testing.assert_allclose(a["calculated_kwchi"], b["calculated_kwchi"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a["observed_kwchi"], b["observed_kwchi"], rtol=1e-12)
    assert a["metrics"]["r"]["r_factor"] == pytest.approx(b["metrics"]["r"]["r_factor"],
                                                          rel=1e-4)


def test_explicit_window_must_fit_and_default_window_moves():
    result = ensemble_result()
    expt = measured(KX, adjusted(result, KX))
    with pytest.raises(ValueError, match="does not fit inside"):
        compare_feff_experiment(result, expt, kmax=15.0)
    with pytest.raises(ValueError, match="does not fit inside"):
        compare_feff_experiment(result, expt, kmin=1.5)
    cmp = compare_feff_experiment(result, expt)
    window = cmp["window"]
    assert window["kmax_moved"] and not window["kmin_moved"]
    assert window["kmax"] == pytest.approx(KX[-1] - .5)
    assert window["requested_kmax"] == 19.5
    inside = compare_feff_experiment(result, expt, kmin=3.0, kmax=11.5)
    assert inside["window"]["kmax"] == 11.5 and not inside["window"]["kmax_moved"]
    assert "moved inside" in format_feff_experiment_report(cmp)


def test_excluded_points_are_counted():
    result = ensemble_result()
    kx = np.arange(0.0, 21.0, 0.1)
    cmp = compare_feff_experiment(result, measured(kx, np.sin(kx), 2), delta_e0=10.0,
                                  k_range=(3, 12))
    counts = cmp["overlap"]
    assert counts["n_excluded_nonpositive_k"] == 1
    assert counts["n_excluded_below_shifted_edge"] == int(np.sum((kx > 0) & (kx ** 2 < C_K * 10)))
    assert counts["n_excluded_outside_model"] > 0
    assert counts["k_max"] <= np.sqrt(20 ** 2 + C_K * 10)
    assert (counts["n_points"] + counts["n_excluded_nonpositive_k"]
            + counts["n_excluded_below_shifted_edge"]
            + counts["n_excluded_outside_model"]) == len(kx)
    assert len(cmp["excluded_indices"]) == len(kx) - counts["n_points"]


def test_measured_sigma_weights_the_fit_and_gives_chi_square():
    result = ensemble_result()
    chi = adjusted(result, KX, **TRUE)
    sigma = (0.01 * KX ** 2).tolist()
    cmp = compare_feff_experiment(result, measured(KX, chi, 2, sigma), fit=("amplitude",))
    metrics = cmp["metrics"]["k"]
    assert metrics["chi_square"] is not None
    assert metrics["degrees_of_freedom"] == metrics["n_points"] - 1
    assert cmp["fitting"]["weighting"] == "inverse measured variance"
    assert cmp["sigma_kwchi"][0] == pytest.approx(.01 * KX[0] ** 2)


def test_fit_needs_model_coverage_for_every_trial_shift():
    result = ensemble_result()
    kx = np.arange(1.5, 20.0, 0.05)
    with pytest.raises(ValueError, match="does not cover"):
        compare_feff_experiment(result, measured(kx, adjusted(result, kx)), fit=("e0",),
                                k_range=(3, 20))


@pytest.mark.parametrize("kwargs,match", [
    ({"fit": ("scale",)}, "fit must name"),
    ({"fit": ("e0", "e0")}, "fit must name"),
    ({"r_range": (4, 1)}, "r_range"),
    ({"k_range": (12, 3)}, "k_range"),
    ({"amplitude": float("nan")}, "finite"),
    ({"delta_e0": True}, "finite"),
    ({"k_range": (13.5, 14)}, "fewer than two"),
])
def test_bad_arguments_are_rejected(kwargs, match):
    result = ensemble_result()
    with pytest.raises(ValueError, match=match):
        compare_feff_experiment(result, measured(KX, adjusted(result, KX)), **kwargs)


def test_inputs_must_come_from_the_loaders():
    result = ensemble_result()
    expt = measured(KX, adjusted(result, KX))
    with pytest.raises(ValueError, match="load_exafs_experiment"):
        compare_feff_experiment(result, {**expt, "kind": "sq"})
    with pytest.raises(ValueError, match="lacks 'ft_kmin'"):
        compare_feff_experiment({k: v for k, v in result.items() if k != "ft_kmin"}, expt)


# ─── files ─────────────────────────────────────────────────────────────────

def test_files_have_settings_header_and_matching_rows(tmp_path):
    result = ensemble_result()
    cmp = compare_feff_experiment(result, measured(KX, adjusted(result, KX, **TRUE), 2),
                                  fit=("amplitude", "e0", "sigma2"), k_range=(3, 12),
                                  n_bootstrap=50)
    paths = save_feff_experiment_comparison(cmp, tmp_path, prefix="analysis_Ga", dpi=40)
    chik = rows(paths["chik_csv"])
    assert chik[0][0].startswith("# absorber=Ga edge=K n_sites=60 n_structures=6 kweight=2")
    assert "fitted=amplitude+e0+sigma2" in chik[0][0] and "R=phase-uncorrected" in chik[0][0]
    assert chik[1][:8] == ["input_index", "k_invA", "observed_chi", "observed_k2chi",
                           "sigma_k2chi", "calculated_chi", "calculated_k2chi", "residual_k2chi"]
    assert chik[1][-3:] == ["confidence", "window", "in_k_range"]
    assert len(chik) == 2 + cmp["overlap"]["n_points"]
    assert all(row[4] == "" for row in chik[2:])
    chir = rows(paths["chir_csv"])
    assert chir[1][0] == "R_A" and chir[1][-1] == "in_r_range"
    assert len(chir) == 2 + len(cmp["r"])
    assert all(row[-3] != "" for row in chir[2:])
    saved = json.loads(open(paths["json"]).read())
    assert saved["adjustments"] == cmp["adjustments"]
    for key in ("chik_png", "chir_png", "report"):
        assert open(paths[key], "rb").read()
    assert "R-factor" in open(paths["report"]).read()


def test_single_structure_result_has_no_bands(tmp_path):
    result = ensemble_result(n=1)
    cmp = compare_feff_experiment(result, measured(KX, adjusted(result, KX)))
    assert cmp["chir_band"] is None and cmp["n_structures"] == 1
    assert set(cmp["uncertainty"]["ci_low"]) == {None}
    paths = save_feff_experiment_comparison(cmp, tmp_path, dpi=40)
    assert all(row[-2] == "" and row[-3] == "" for row in rows(paths["chir_csv"])[2:])
    assert paths["chik_csv"].endswith("analysis_feff_experiment_chik.csv")


# ─── CLI and YAML ──────────────────────────────────────────────────────────

FAKE_FEFF = """
import math
r0 = 1.55 + (len(open('feff.inp').read()) % 13) * 0.01
with open('chi.dat', 'w') as fh:
    fh.write('# fake feff\\n#  k  chi  mag  phase\\n')
    for i in range(401):
        k = 0.05 * i
        c = math.exp(-0.01 * k * k) * math.sin(2 * k * r0) / max(k, 1e-9) ** 2
        fh.write(f'{k:11.4f}{c:14.6e}{abs(c):14.6e}{0.0:14.6e}\\n')
"""


def cell(shift=0.0):
    return Atoms(symbols=["Si", "Si", "O", "O", "O", "O"],
                 positions=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                            [1.6 + shift, 0.0, 0.0], [0.0, 1.6, 0.0],
                            [0.0, 0.0, 1.6 - shift], [5.6, 0.0, 0.0]],
                 cell=[8.0, 8.0, 8.0], pbc=True)


class TestCLI:
    @pytest.fixture
    def setup(self, tmp_path):
        atoms = [cell(0.0), cell(0.04), cell(-0.03)]
        structures = tmp_path / "structs"
        structures.mkdir()
        for n, a in enumerate(atoms):
            write(str(structures / f"s_{n}.xyz"), a, format="extxyz")
        exe = tmp_path / "fake-feff"
        exe.write_text(f"#!{sys.executable}\n{FAKE_FEFF}\n")
        exe.chmod(0o755)
        feff_pipeline(atoms, "Si,O", str(tmp_path / "feff"), executable=str(exe), radius=4.0)
        ff = collect_feff_results(str(tmp_path / "feff" / "Si"))
        kx = np.arange(1.0, 13.0, 0.05)
        spectrum = tmp_path / "si_k2chi.dat"
        np.savetxt(spectrum, np.c_[kx, kx ** 2 * np.interp(kx, ff["k"], ff["chi"])])
        return structures, tmp_path / "feff", spectrum

    def run(self, argv, capsys):
        from amorphgen.cli import main
        with patch("sys.argv", ["amorphgen", *argv]):
            main()
        return capsys.readouterr().out

    def test_named_spectrum_writes_comparison_files(self, setup, tmp_path, capsys):
        structures, feff, spectrum = setup
        plots = tmp_path / "plots"
        out = self.run(["--analyse", "--input-dir", str(structures), "--feff-collect", str(feff),
                        "--feff-experiment", f"Si={spectrum}", "--feff-experiment-kweight", "2",
                        "--feff-fit", "amplitude", "--feff-fit-rrange", "0.5", "4",
                        "--save-plot", str(plots)], capsys)
        assert out.count("EXAFS comparison with measured chi(k)") == 1
        assert "Si K edge" in out and "amplitude 1 (fitted)" in out
        for suffix in ("_chik.csv", "_chir.csv", ".json", "_report.txt", "_chik.png"):
            assert (plots / f"analysis_Si_feff_experiment{suffix}").exists()
        assert not (plots / "analysis_O_feff_experiment.json").exists()
        saved = json.loads((plots / "analysis_Si_feff_experiment.json").read_text())
        assert saved["r_range"] == [.5, 4.0]
        assert saved["metrics"]["k"]["r_factor"] < 1e-12

    @pytest.mark.parametrize("value,message", [
        ("{spectrum}", "name each file as EL=FILE"),
        ("Zn={spectrum}", "names Zn"),
    ])
    def test_spectra_must_match_trees(self, setup, capsys, value, message):
        structures, feff, spectrum = setup
        with pytest.raises(SystemExit):
            self.run(["--analyse", "--input-dir", str(structures), "--feff-collect", str(feff),
                      "--feff-experiment", value.format(spectrum=spectrum)], capsys)
        assert message in capsys.readouterr().out

    def test_spectrum_needs_a_collected_tree(self, setup, capsys):
        structures, _, spectrum = setup
        with pytest.raises(SystemExit):
            self.run(["--analyse", "--input-dir", str(structures),
                      "--feff-experiment", str(spectrum)], capsys)
        assert "needs --feff-collect or --feff-run" in capsys.readouterr().out

    def test_explicit_kmax_outside_the_data_is_an_error(self, setup, capsys):
        structures, feff, spectrum = setup
        with pytest.raises(SystemExit):
            self.run(["--analyse", "--input-dir", str(structures),
                      "--feff-collect", str(feff / "Si"), "--feff-kmax", "15",
                      "--feff-experiment", str(spectrum)], capsys)
        assert "does not fit inside" in capsys.readouterr().out


@pytest.mark.parametrize("analysis, ok", [
    ({"feff_experiment": "ga.dat", "feff_experiment_kweight": 2,
      "feff_experiment_columns": [0, 2], "feff_experiment_skiprows": 1,
      "feff_amplitude": .9, "feff_delta_e0": 4.5, "feff_delta_sigma2": .001,
      "feff_offset": 0, "feff_fit": "amplitude,e0", "feff_fit_krange": [3, 12],
      "feff_fit_rrange": [1, 4]}, True),
    ({"feff_experiment": {"Ga": "ga.dat", "In": {"file": "in.dat", "kweight": 3}},
      "feff_fit": ["sigma2"]}, True),
    ({"feff_experiment": {"Ga": {"path": "ga.dat"}}}, False),
    ({"feff_experiment": ["ga.dat"]}, False),
    ({"feff_experiment_kweight": 4}, False),
    ({"feff_experiment_columns": [0, 0]}, False),
    ({"feff_experiment_skiprows": -1}, False),
    ({"feff_fit": "scale"}, False),
    ({"feff_fit_krange": [12, 3]}, False),
    ({"feff_delta_e0": float("inf")}, False),
])
def test_yaml_keys_are_validated(analysis, ok):
    from amorphgen.configs.yaml_config import _validate_config
    _, errors = _validate_config({"analysis": analysis}, "test.yaml")
    assert (errors == []) is ok, errors
