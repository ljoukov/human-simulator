"""Render the study results figure from published JSON. Requires Matplotlib and NumPy."""
import argparse
import json
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
INK, MUTED = "#172b30", "#465d60"
TEAL, CORAL, BLUE = "#067a73", "#ac4c32", "#376389"
PAPER, GRID, BAND = "#fafaf7", "#d4ded9", "#d7e8df"


def read(name):
    return json.loads((STUDY / "data" / (name + ".json")).read_text())


def spread(values, ax):
    """Pack overlapping dots in symmetric rows without changing their x values."""
    plot_width, plot_height = ax.get_window_extent().size / ax.figure.dpi * 72
    points_per_x = plot_width / np.ptp(ax.get_xlim())
    points_per_y = plot_height / np.ptp(ax.get_ylim())
    distance = 7.3
    lanes = [0] + [sign * n for n in range(1, 30) for sign in (1, -1)]
    recent = {}
    result = np.zeros(len(values))
    for i in np.argsort(values):
        for lane in lanes:
            if (values[i] - recent.get(lane, -np.inf)) * points_per_x > distance:
                result[i] = lane * distance / points_per_y
                recent[lane] = values[i]
                break
    return result


def plot(output):
    gait = read("gtps")
    video = read("video-audit")
    sweep = read("bootstrap-sensitivity")
    metal = read("benchmark-metal-sensitivity")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.right": False, "axes.spines.top": False, "axes.edgecolor": GRID,
        "axes.labelsize": 13, "axes.labelpad": 5, "axes.facecolor": PAPER,
        "figure.facecolor": PAPER, "legend.frameon": False,
    })
    fig = plt.figure(figsize=(17.8, 10.2))
    fig.text(.045, .935, "Hip mechanics from gait, video and instrumented implants",
             fontsize=27, fontfamily="DejaVu Serif", fontweight="bold")
    fig.add_artist(Line2D([.045, .966], [.907, .907], transform=fig.transFigure, color=INK, lw=1))
    fig.add_artist(Line2D([.045, .966], [.486, .486], transform=fig.transFigure, color=GRID, lw=1))

    def panel(x, y, letter, title):
        fig.text(x, y, letter, color=TEAL, weight="bold", fontsize=17)
        fig.text(x + .024, y, title, weight="bold", fontsize=16)

    def axis(bounds, grid_on):
        ax = fig.add_axes(bounds)
        ax.set_axisbelow(True)
        ax.grid(axis=grid_on, color=GRID, lw=.8)
        ax.tick_params(length=0, pad=7, labelsize=12)
        return ax

    def note(x, y, text):
        fig.text(x, y, text, va="top", fontsize=12.0, color=MUTED, linespacing=1.40)

    left, right = .045, .535
    panel(left, .858, "A", "Hip excursion overlaps between cohorts")
    panel(right, .858, "B", "Video overestimates thigh excursion in one stride")
    panel(left, .435, "C", "Lever-arm shifts leave a hip force error")
    panel(right, .435, "D", "Time to refit the same models on three devices")

    ax = axis([.132, .656, .327, .154], "x")
    ax.set(xlim=(-.15, 10.8), ylim=(-.55, 1.55), xticks=range(0, 11, 2),
           xlabel="Hip adduction–abduction range, 20–80% of stance (°)")
    overlap = gait["comparison"]["overlap"]
    low, high = overlap["sharedP10p90Deg"]
    ax.axvspan(low, high, color=BAND, zorder=0)
    for group, y, color in [("GTPS", 1, TEAL), ("HOA", 0, CORAL)]:
        x = np.array([p["excursionDeg"] for p in gait["people"] if p["group"] == group])
        ax.scatter(x, y + spread(x, ax), s=50, color=color, edgecolors=PAPER, linewidths=.7)
    ax.set_yticks([1, 0], [f'GTPS\n{gait["gtps"]["participants"]} records', f'Hip OA\n{gait["oa"]["participants"]} women'])
    ax.spines["left"].set_visible(False)
    note(left, .586,
         f"Shared 10th–90th percentile interval {low:.2f}–{high:.2f}° (shaded).\n"
         f'It contains {overlap["gtpsInside"]}/{gait["gtps"]["participants"]} GTPS records and '
         f'{overlap["oaInside"]}/{gait["oa"]["participants"]} OA participants.\n'
         "Diagnosis coincides with laboratory. The OA group was older.")

    ax = axis([.583, .656, .377, .154], "y")
    s = video["stride"]
    phase = np.array([f["phase"] for f in s["timeline"]]) * 100
    ax.plot(phase, [f["videoDeg"] for f in s["timeline"]], color=TEAL, lw=2.5, label="Video landmarks")
    ax.plot(phase, [f["referenceDeg"] for f in s["timeline"]], color=CORAL, lw=2.3, ls="--", label="Projected optical markers")
    ax.set(xlim=(0, 100), xticks=range(0, 101, 25), ylabel="Image-plane thigh tilt (°)", xlabel="Test stride (%)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.21), ncol=2, fontsize=11.4, columnspacing=1.4, handlelength=2.1)
    note(right, .586,
         f'One held-out OA stride, {s["nValid"]}/{s["nFrames"]} frames. Correlation r = {s["pearsonR"]:.3f}.\n'
         f'Video excursion {s["peakToPeakVideoDeg"]:.2f}° vs optical {s["peakToPeakReferenceDeg"]:.2f}°. Mean absolute error {s["maeDeg"]:.2f}°.\n'
         f'Holding the registration-period tilt constant gave {s["constantVideoBaselineMaeDeg"]:.2f}° error.')

    ax = axis([.103, .236, .357, .150], "y")
    offsets = [c["offsetMm"] for c in sweep["cases"]]
    q05, q50, q95 = np.transpose([c["jointForceBwQuantiles"] for c in sweep["cases"]])
    interval = ax.fill_between(offsets, q05, q95, color=BAND, label="5th–95th percentiles")
    line, = ax.plot(offsets, q50, "-o", color=TEAL, lw=2, ms=3.8, label="Model median")
    measured = sweep["measuredBw"]
    observed = ax.axhline(measured, color=CORAL, lw=2.2, ls="--", label="Implant")
    ax.legend(handles=[line, interval, observed], loc="upper right", bbox_to_anchor=(1.00, 1.23),
              ncol=3, fontsize=11, columnspacing=1.05, handlelength=1.6)
    ax.set(xlim=(min(offsets), max(offsets)), xticks=range(-15, 16, 5),
           ylim=(1.96, 3.59), yticks=[2, 2.5, 3, 3.5], ylabel="Hip force (body weights)",
           xlabel="Assumed frontal lever-arm shift (mm)")
    held_out = sweep["heldOutTrialId"].split("_")[0]
    note(left, .166,
         f'Held-out implant {held_out}, {sweep["heldOutPhase"]*100:.0f}% of walking stance. Across {len(offsets)} offsets,\n'
         f'the lowest model 5th percentile was {min(q05):.2f} body weights.\n'
         f'The implant measured {measured:.2f} body weights at this phase.')

    ax = axis([.639, .236, .320, .150], "x")
    timing = sweep["timing"]
    seconds = [median(timing["cpuSamplesSeconds"]), metal["exhaustive"]["warmEndToEndSeconds"], median(timing["warmCloudSamplesSeconds"])]
    ys = [2, 1, 0]
    ax.barh(ys, seconds, height=.39, color=[MUTED, TEAL, BLUE], zorder=2)
    for y, value in zip(ys, seconds):
        ax.text(value + .30, y, f"{value:.2f} s", va="center", weight="bold", fontsize=12.5)
    ax.set(yticks=ys, yticklabels=[f'M4 CPU · {timing["cpuThreads"]} cores', "M4 GPU · Metal", "H100 · Modal"],
           xlim=(0, 21), xticks=range(0, 21, 5), ylim=(-.55, 2.55),
           xlabel="Warm end-to-end time (s, median of 3 runs)")
    ax.spines["left"].set_visible(False)
    note(right, .166,
         f'{sweep["candidateResampleOffsetCombinations"]/1e9:.2f} billion candidate × bootstrap × offset combinations.\n'
         f'All {sweep["winnerAgreement"]:,} winning model IDs agreed. Cloud time includes the round trip.\n'
         'Timing starts with prepared features and ends with returned winners.')

    fig.add_artist(Line2D([.045, .966], [.072, .072], transform=fig.transFigure, color=GRID, lw=1))
    fig.text(left, .045, "GTPS = greater trochanteric pain syndrome. OA = osteoarthritis.  "
             "Gait and video data from Robinson et al. and Bertaux et al.  •  "
             "Implant data from OrthoLoad Hip Joint III, Bergmann & Damm (eds.).",
             fontsize=10, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=165, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "media" / "hip-results.png")
    plot(parser.parse_args().output)
