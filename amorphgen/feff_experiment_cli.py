"""CLI options and reporting for measured EXAFS chi(k)."""

from __future__ import annotations

_FILE_KEYS = ("file", "kweight", "columns", "skiprows")


def add_feff_experiment_arguments(group):
    """Register the measured chi(k) options on the FEFF group."""
    group.add_argument("--feff-experiment", action="append", metavar="[EL=]FILE",
                       help="Measured chi(k): k [1/A], chi (or k^w chi, see "
                            "--feff-experiment-kweight), optional sigma. Compared with "
                            "the collected spectrum on the measured k points and through "
                            "one transform; writes analysis_<El>_feff_experiment_"
                            "{chik,chir}.csv, JSON, report and PNG under --save-plot. "
                            "Repeatable; EL= is required with several absorber trees.")
    group.add_argument("--feff-experiment-kweight", type=int, default=0, metavar="W",
                       help="k-weight already applied to the measured column "
                            "(default 0; 2 for an Athena chik2 column).")
    group.add_argument("--feff-experiment-columns", type=int, nargs="+", metavar="COL",
                       help="Two or three zero-based columns: k, value, optional sigma. "
                            "Default: exactly two or three columns.")
    group.add_argument("--feff-experiment-skiprows", type=int, default=0, metavar="N",
                       help="Header lines to skip in the measured file (default 0).")
    group.add_argument("--feff-amplitude", type=float, default=1.0, metavar="A",
                       help="Amplitude factor (S0^2) applied to the model (default 1).")
    group.add_argument("--feff-delta-e0", type=float, default=0.0, metavar="EV",
                       help="Energy shift of the model in eV, with the IFEFFIT/Artemis/"
                            "Larch sign (default 0).")
    group.add_argument("--feff-delta-sigma2", type=float, default=0.0, metavar="A2",
                       help="Extra Debye-Waller factor of the model in A^2 (default 0).")
    group.add_argument("--feff-offset", type=float, default=0.0, metavar="Y",
                       help="Constant added to the model in the measured column's "
                            "weighting, for digitised curves (default 0).")
    group.add_argument("--feff-fit", default=None, metavar="P[,P]",
                       help="Fit these adjustments to k^w chi(k): amplitude, e0, sigma2, "
                            "offset (default: none fitted).")
    group.add_argument("--feff-fit-krange", type=float, nargs=2, metavar=("KMIN", "KMAX"),
                       help="k range of the fit and the k-space metrics in 1/A "
                            "(default: the FT window within the measured data).")
    group.add_argument("--feff-fit-rrange", type=float, nargs=2, metavar=("RMIN", "RMAX"),
                       help="R range of the R-space metrics in A (default 1 4).")


def _option(args, config, typed, name):
    return (getattr(args, name) if typed("--" + name.replace("_", "-"))
            else config.get(name, getattr(args, name)))


def feff_experiment_specs(args, config, typed, n_trees):
    """Measured files by absorber symbol; None keys an unlabelled file.

    CLI values are ``FILE`` or ``EL=FILE``; YAML ``feff_experiment`` is a
    path, or a mapping from element to a path or to ``{file, kweight,
    columns, skiprows}``. An unlabelled file needs exactly one tree.
    """
    value = args.feff_experiment if typed("--feff-experiment") else config.get("feff_experiment")
    if not value:
        return {}
    from ase.data import chemical_symbols
    if isinstance(value, str):
        value = [value]
    specs = {}
    if isinstance(value, dict):
        for element, spec in value.items():
            spec = {"file": spec} if isinstance(spec, str) else dict(spec)
            specs[element] = spec
    else:
        for item in value:
            element, sep, path = item.partition("=")
            if sep and element in chemical_symbols[1:]:
                key, spec = element, {"file": path}
            else:
                key, spec = None, {"file": item}
            if key in specs:
                raise ValueError(f"two measured files for {key or 'the unlabelled tree'}")
            specs[key] = spec
    for element, spec in specs.items():
        if element is not None and element not in chemical_symbols[1:]:
            raise ValueError(f"{element!r} is not an element symbol")
        if not spec.get("file") or set(spec) - set(_FILE_KEYS):
            raise ValueError(f"each measured spectrum needs a file and only {_FILE_KEYS}")
    if None in specs and (len(specs) > 1 or n_trees > 1):
        raise ValueError("with several absorber trees, name each file as EL=FILE")
    return specs


def run_feff_experiment(ff, spec, args, config, typed, *, plot_dir=None,
                        report_path=None, dpi=300, save_pdf=False):
    """Compare one collected tree with its measured chi(k); report and save."""
    from .analysis.feff_experiment import (compare_feff_experiment,
                                           format_feff_experiment_report,
                                           load_exafs_experiment,
                                           save_feff_experiment_comparison)

    option = lambda name: _option(args, config, typed, name)
    experiment = load_exafs_experiment(
        spec["file"], kweight=spec.get("kweight", option("feff_experiment_kweight")),
        columns=spec.get("columns", option("feff_experiment_columns")),
        skiprows=spec.get("skiprows", option("feff_experiment_skiprows")))
    # An explicit window must fit the shared k range; the default one may move.
    explicit = lambda name: typed("--" + name.replace("_", "-")) or name in config
    result = compare_feff_experiment(
        ff, experiment,
        amplitude=option("feff_amplitude"), delta_e0=option("feff_delta_e0"),
        delta_sigma2=option("feff_delta_sigma2"), offset=option("feff_offset"),
        fit=option("feff_fit") or (),
        k_range=option("feff_fit_krange"),
        kmin=ff["ft_kmin"] if explicit("feff_kmin") else None,
        kmax=ff["ft_kmax"] if explicit("feff_kmax") else None,
        r_range=option("feff_fit_rrange") or (1.0, 4.0))
    text = format_feff_experiment_report(result)
    print(text)
    if report_path:
        with open(report_path, "a") as handle:
            handle.write("\n" + text + "\n")
    if plot_dir:
        save_feff_experiment_comparison(result, output_dir=plot_dir,
                                        prefix=f"analysis_{ff['absorber']}",
                                        dpi=dpi, save_pdf=save_pdf)
    else:
        print("  (pass --save-plot DIR to write the comparison CSV, JSON and PNG)")
    return result
