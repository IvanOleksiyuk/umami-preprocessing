import logging as log
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ftag import Flavours
from puma import Histogram, HistogramPlot

from upp.classes.preprocessing_config import PreprocessingConfig
from upp.logger import setup_logger
from upp.utils import path_append

setup_logger()


def RMSE(res, target):
    return np.sqrt(np.nanmean((res - target) ** 2 / (res + target)))


def load_jets(paths, variable):
    if isinstance(variable, str):
        variables = ["flavour_label", variable]
    else:
        variables = ["flavour_label"] + variable
    df = pd.DataFrame(columns=variables)
    for path in paths:
        with h5py.File(path) as f:
            df = pd.concat([df, pd.DataFrame(f["jets"].fields(variables)[: int(3e6)])])
    log.info(f"[bold green]jets loaded: {len(df)}")
    return df


def make_hist(
    flavours,
    variable,
    in_paths,
    bins_range=None,
    suffix="",
    prefix="",
    out_dir=Path("plots/"),
    bins=50,
):
    df = load_jets(in_paths, variable)

    plot = HistogramPlot(
        ylabel="Normalised Number of jets",
        atlas_second_tag="$\\sqrt{s}=13$ TeV",
        xlabel=variable,
        bins=bins,
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
    plot.savefig(out_path, bbox_inches="tight")
    log.info(f"Saved plot {out_path}")


def make_hist_my(
    flavours,
    variable,
    in_paths,
    bins=50,
    bins_range=None,
    suffix="",
    prefix="",
    out_dir=Path("plots/"),
    target=0,
):
    df = load_jets(in_paths, variable)
    if bins_range is None:
        bins_range = (np.min(df[variable]), np.max(df[variable]))
    plt.figure()
    hists = []
    for label in range(3):
        if flavours[label] == "b":
            c = "blue"
        elif flavours[label] == "c":
            c = "orange"
        elif flavours[label] == "u":
            c = "green"
        else:
            c = None

        hist = plt.hist(
            df[variable][df["flavour_label"] == label],
            bins=bins,
            range=bins_range,
            histtype="step",
            density=False,
            label=flavours[label],
            color=c,
        )
        hists.append(hist[0])
    plt.yscale("log")
    plt.xlabel(variable)
    out_dir.mkdir(exist_ok=True)
    fname = f"{prefix}_{variable}"
    out_path = out_dir / f"{fname}{suffix}.png"
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    # plot the difference to hist[0]
    plt.figure()
    target = hists[target]
    for label in range(3):
        plt.step(
            hist[1][:-1],
            (hists[label] - target) / np.sqrt(hists[label] + target),
            where="post",
            label=f"{flavours[label]} RMSE " + str(RMSE(hists[label], target)),
        )
        plt.xlabel(variable)
    plt.legend()
    out_path = out_dir / f"{fname}{suffix}_diff.png"
    plt.savefig(out_path, bbox_inches="tight")


def count_jet_upsamples(
    data,
    suffix="",
    prefix="",
    out_dir=Path("plots/"),
    figure=None,
):
    data_tuples = [tuple(row) for row in data]
    row_counts = Counter(data_tuples)
    vals = row_counts.values()
    plt.figure(figure)
    hist = plt.hist(
        vals,
        bins=np.arange(max(vals) + 1) - 0.5,
        log=True,
        histtype="step",
        label=f"total upsampling: {len(data)/len(row_counts):.2f}",
    )
    print(np.sum(hist[0] * np.arange(max(vals)) + 1))
    plt.xlabel("Number of occurrences")
    plt.ylabel("Number of jets")
    out_path = out_dir / f"{prefix}_occurances_{suffix}.png"
    plt.legend()
    plt.savefig(out_path, bbox_inches="tight")
    # Initialize counters for different occurrences
    occurrences = {}

    # Count rows with different occurrences
    for row, count in row_counts.items():
        occurrences[count] = occurrences.get(count, 0) + 1
    return occurrences


paths_orig_pdf = [
    "/home/users/o/oleksiyu/WORK/umami/userTrx_full_pdf_1M/preprocessing/preprocessed/PFlow-hybrid-resampled.h5"
]

config = PreprocessingConfig.from_file(
    Path(
        "/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate_pdf_sf1.yaml"
    ),
    "train",
)
paths_upp_pdf = [config.out_fname]

# config = PreprocessingConfig.from_file(
#     Path(
#         "/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate_pdfs_sf1.yaml"
#     ),
#     "train",
# )
# paths_upp2_pdf = [config.out_fname]

# config = PreprocessingConfig.from_file(
#     Path(
#         "/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate2_pdf_sfauto.yaml"
#     ),
#     "train",
# )
# paths_upp3_pdf = [config.out_fname]

# config = PreprocessingConfig.from_file(
#     Path(
#         "/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate3_pdf_sf1.yaml"
#     ),
#     "train",
# )
# paths_upp4_pdf = [config.out_fname]

config = PreprocessingConfig.from_file(
    Path(
        "/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate3_pdfu_sfauto.yaml"
    ),
    "train",
)
paths_upp_pdfu = [config.out_fname]

config = PreprocessingConfig.from_file(
    Path(
        "/home/users/o/oleksiyu/WORK/umami-preprocessing/user/user3/replicate3_pdf_sfauto.yaml"
    ),
    "train",
)
paths_upp_pdf = [config.out_fname]


def make_single_method_plots(paths, flavours, prefix="", target=0):
    for var in config.sampl_cfg.vars:
        make_hist_my(
            flavours, var, paths, prefix=prefix, bins_range=None, target=target
        )
        if "pt" in var:
            make_hist_my(
                flavours,
                var,
                paths,
                bins_range=(0, 500e3),
                suffix="low",
                prefix=prefix,
                target=target,
            )
    print("occurances:")
    df = load_jets(paths, config.variables.jets["inputs"])
    print("example row:", df.iloc[0].values)
    print("example row:", df.iloc[1].values)
    print("example row:", df.iloc[2].values)
    print(
        count_jet_upsamples(df[config.variables.jets["inputs"]].values, prefix=prefix)
    )


# make_single_method_plots(paths_upp_pdf, config, prefix="upp")
# make_single_method_plots(paths_upp2_pdf, config, prefix="upp2")
# make_single_method_plots(paths_upp3_pdf, config, prefix="upp3")


# make_single_method_plots(paths_orig_pdf, ["u", "c", "b"], prefix="orig", target=2)
# make_single_method_plots(
#     paths_upp_pdfu, ["b", "c", "u"], prefix="upp_pdfu_auto", target=0
# )
make_single_method_plots(
    paths_upp_pdf, ["b", "c", "u"], prefix="upp_pdf_auto", target=0
)

# Ideal RMSE distribution
# x = np.random.normal(0, 1, (1000, 50))
# y = np.random.normal(0, 1, (1000, 50))

# RMSEs = []
# for i in range(1000):
#     RMSEs.append(np.mean((x[i] - y[i]) ** 2 / 2) ** 0.5)

# plt.figure()
# plt.hist(RMSEs, bins=50, histtype="step")
# plt.xlabel("RMSE")
# plt.savefig("plots/ideal_RMSE.png", bbox_inches="tight")
# log.info(f"Mean: {np.mean(RMSEs)}")
# log.info(f"Median: {np.median(RMSEs)}")
# log.info(f"Std: {np.std(RMSEs)}")

## Single resampling method
# Plot resampling variables

# Plot difference bwtween the target and the resampled spectra

# Find RMSE

# Plot other variables

## Two resampling methods compare
