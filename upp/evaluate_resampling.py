import logging as log
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ftag import Flavours
from puma import Histogram, HistogramPlot

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.utils import path_append


def load_jets(paths, variable):
    variables = ["flavour_label", variable]
    df = pd.DataFrame(columns=variables)
    for path in paths:
        with h5py.File(path) as f:
            df = pd.concat([df, pd.DataFrame(f["jets"].fields(variables)[: int(1e6)])])
    return df


def make_hist(
    flavours,
    variable,
    in_paths,
    bins_range=None,
    suffix="",
    prefix="",
    out_dir=Path("plots/"),
):
    df = load_jets(in_paths, variable)

    plot = HistogramPlot(
        ylabel="Normalised Number of jets",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=50,
        y_scale=1.5,
        logy=True,
        norm=False,
        bins_range=bins_range,
        n_ratio_panels=1,
    )

    for label_value, label_string in enumerate([f.name for f in flavours]):
        puma_flavour = f"{label_string}jets" if len(label_string) == 1 else label_string
        if puma_flavour == "qcd":
            puma_flavour = "dijets"
        plot.add(
            Histogram(
                df[df["flavour_label"] == label_value][variable],
                label=Flavours[label_string].label,
                colour=Flavours[label_string].colour,
            ),
            key=label_value,
        )

    plot.set_reference(0)
    plot.plot()
    plot.plot_ratios()
    plot.draw()
    out_dir.mkdir(exist_ok=True)
    fname = f"{prefix}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.png"
    plot.savefig(out_path)
    log.info(f"Saved plot {out_path}")


def make_hist_my(
    flavours,
    variable,
    in_paths,
    bins_range=None,
    suffix="",
    prefix="",
    out_dir=Path("plots/"),
):
    df = load_jets(in_paths, variable)
    if bins_range is None:
        bins_range = (np.min(df[variable]), np.max(df[variable]))
    plt.figure()
    hists = []
    for label in range(3):
        hist = plt.hist(
            df[variable][df["flavour_label"] == label],
            bins=20,
            range=bins_range,
            histtype="step",
            density=False,
            label=flavours[label].name,
        )
        hists.append(hist[0])
    plt.yscale("log")
    plt.xlabel(variable)
    out_dir.mkdir(exist_ok=True)
    fname = f"{prefix}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.png"
    plt.savefig(out_path)
    # plot the difference to hist[0]
    plt.figure()
    for label in range(3):
        plt.step(
            hist[1][:-1],
            (hists[label] - hists[0]) / np.sqrt(hists[label] + hists[0]),
            where="post",
            label=f"{flavours[label].name} MSE "
            + str(
                np.nanmean(
                    ((hists[label] - hists[0]) / np.sqrt(hists[label] + hists[0])) ** 2
                )
            ),
        )
        plt.xlabel(variable)
    plt.legend()
    out_path = out_dir / f"{fname}{suffix}_diff.png"
    plt.savefig(out_path)


paths_orig_pdf = "/home/users/o/oleksiyu/WORK/umami/userTrx_full_pdf/preprocessing/preprocessed/PFlow-hybrid-resampled.h5"

config = PreprocessingConfig.from_file(
    Path("/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate.yaml"),
    "train",
)
paths_upp_pdf = [config.out_fname]


def make_single_method_plots(paths, config, prefix=""):
    for var in config.sampl_cfg.vars:
        make_hist_my(
            config.components.flavours, var, paths, prefix=prefix, bins_range=None
        )
        if "pt" in var:
            make_hist_my(
                config.components.flavours,
                var,
                paths,
                bins_range=(0, 500e3),
                suffix="low",
                prefix=prefix,
            )


make_single_method_plots(paths_upp_pdf, config, prefix="upp")
make_single_method_plots(paths_orig_pdf, config, prefix="orig")


## Single resampling method
# Plot resampling variables

# Plot difference bwtween the target and the resampled spectra

# Find RMSE

# Plot other variables

## Two resampling methods compare
