from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

project_dir = Path(__file__).resolve().parent.parent
results = project_dir / "results/experiments.parquet"

BUCKET_COLORS = {
    "match":     "#00a878",
    "untouched": "#b8b8b8",
    "wrong":     "#d1495b",
}


# fixed slot order (validated for colorblind separation), also the legend order
# mean_fill / knn_impute / mean_fill_indicator get the first three slots because they share scatter plots
STRATEGY_COLORS = {
    "mean_fill":            "#2a78d6",
    "knn_impute":           "#eb6834",
    "mean_fill_indicator":  "#1baf7a",
    "do_nothing":           "#eda100",
    "drop_rows":            "#e87ba4",
    "fuzzy_match":          "#008300",
}

STRATEGY_LINESTYLES = {
    "do_nothing":   "-",      # solid, it's the baseline
    "drop_rows":    "-",      # solid, it's the one that separates
    "mean_fill":    "--",
    "knn_impute":   ":",
    "fuzzy_match":  "-.",
    "mean_fill_indicator":  "-",
}

MECHANISM_COLORS = {
    "random_missing_values": "#2a78d6",
    "label_correlated_missing": "#eb6834",
    "self_masking_missing": "#1baf7a",
}

METRIC_LABELS = {"mean_auc": "ROC-AUC", "mean_f1": "F1"}

MECHANISM_MARKER_SHAPES = {
    "random_missing_values": 'o',
    "label_correlated_missing": '^',
    "typo_corruption": 's',
    "self_masking_missing": 'D',
}

MECHANISM_LABELS = {
    "random_missing_values": "Random missing (MCAR)",
    "label_correlated_missing": "Label-correlated missing (MAR on label)",
    "self_masking_missing": "Self-masking missing (MNAR)",
    "typo_corruption": "Typos",
}

HOLDOUT_LABELS = {
    "clean": "deployed on clean data",
    "corrupted": "deployed on equally corrupted data",
}

# strategies that actually do something for each corruption family
# (e.g. fuzzy_match on missing values is the same as do_nothing, so it is not drawn)
APPLICABLE_STRATEGIES = {
    "random_missing_values": ["do_nothing", "drop_rows", "mean_fill", "mean_fill_indicator", "knn_impute"],
    "label_correlated_missing": ["do_nothing", "drop_rows", "mean_fill", "mean_fill_indicator", "knn_impute"],
    "self_masking_missing": ["do_nothing", "drop_rows", "mean_fill", "mean_fill_indicator", "knn_impute"],
    "typo_corruption": ["do_nothing", "fuzzy_match"],
}

# "clean": train on dirty, deploy on clean (the original setup)
# "corrupted": the holdout gets the same corruption
def holdout(df, condition):
    return df[df["holdout_condition"] == condition]

df = pd.read_parquet(results)

def degradation_grid(df, metric, ylabel, outname):
    group = df.groupby(["model", "corruption_type", "strategy", "corruption_rate"]).agg(
        mean_val=(metric, "mean"),
        std_val=(metric, "std"),
        n=(metric, "count"),
    ).reset_index()

    group["se"] = group["std_val"] / np.sqrt(group["n"])

    models = df["model"].unique()
    corruption_types = df["corruption_type"].unique()

    fig, axes = plt.subplots(
        len(models), len(corruption_types),
        figsize=(5.3 * len(corruption_types), 9), sharey="row", sharex=True, squeeze=False,
    )

    for row, model in enumerate(models):
        for col, corruption_type in enumerate(corruption_types):

            plot_df = group[(group["model"] == model) & (group["corruption_type"] == corruption_type)]
            plot_df = plot_df.dropna(subset=["mean_val"])
            strategies = [s for s in STRATEGY_COLORS
                          if s in APPLICABLE_STRATEGIES[corruption_type] and s in set(plot_df["strategy"])]
            ax = axes[row, col]

            for strategy in strategies:
                strategy_df = plot_df[plot_df["strategy"] == strategy]
                ax.plot(
                    strategy_df["corruption_rate"],
                    strategy_df["mean_val"],
                    label=strategy,
                    color=STRATEGY_COLORS[strategy],
                    linestyle=STRATEGY_LINESTYLES[strategy],
                    marker="o",
                    markersize=4,
                    linewidth=1.6,
                    alpha=0.9,
                )

                ax.fill_between(
                    strategy_df["corruption_rate"],
                    strategy_df["mean_val"] - strategy_df["se"],
                    strategy_df["mean_val"] + strategy_df["se"],
                    alpha=0.15,
                    color=STRATEGY_COLORS[strategy]
                )

            # axis labels on the outer panels only
            if row == len(models) - 1:
                ax.set_xlabel("Corruption rate")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"{model} — {MECHANISM_LABELS[corruption_type]}", fontsize=10)

    # legend from every panel, so a strategy that only appears in one panel is still listed
    legend = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            legend.setdefault(label, handle)
    labels = [s for s in STRATEGY_COLORS if s in legend]
    fig.legend([legend[label] for label in labels], labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])


    fig.savefig(project_dir / f"figures/{outname}_degradation.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / f"figures/{outname}_degradation.svg", bbox_inches="tight")

    return fig

def divergence(df):
    df = df[(df["model"] == "hgb") & (df["corruption_type"] == "label_correlated_missing")]
    df = df[df["strategy"].isin(["do_nothing", "knn_impute"])]

    group = df.groupby(["strategy", "corruption_rate"]).agg(
        mean_auc=("roc_auc", "mean"),
        mean_f1=("f1", "mean")
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, sharey=True)

    strategies = group['strategy'].unique()
    lines = ["mean_auc", "mean_f1"]

    for row, metrics in enumerate(lines):
        ax = axes[row]
        ax.set_ylim(0, 1)
        ax.set_ylabel(METRIC_LABELS[metrics])

        for strategy in strategies:
            strategy_df = group[group["strategy"] == strategy]

            ax.plot(
                strategy_df["corruption_rate"],
                strategy_df[metrics],
                label=strategy,
                color=STRATEGY_COLORS[strategy],
                linestyle=STRATEGY_LINESTYLES[strategy],
                marker="o",
                markersize=4,
                linewidth=1.6,
                alpha=0.9,
            )

    fig.suptitle(
        "Label-correlated missingness, gradient boosting:\n"
        "AUC holds while F1 collapses",
        fontsize=13,
    )

    baseline = group[group["strategy"] == "do_nothing"].set_index("corruption_rate")
    changes = []
    for metrics in lines:
        change = baseline.loc[0.1, metrics] / baseline.loc[0.0, metrics] - 1
        changes.append(f"{METRIC_LABELS[metrics]} {change:+.1%}".replace(" -", " −"))

    for ax, pct in zip(axes, changes, strict=True):
        ax.axvline(0.1, color="#cccccc", linewidth=0.8, zorder=0)
        ax.annotate(
            f"{pct} at 10% vs uncorrupted (do_nothing)",
            xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
            fontsize=9,
            color="#555555",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fig.savefig(project_dir / "figures/fig3_divergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / "figures/fig3_divergence.svg", bbox_inches="tight")

    return fig

def section_bias(df, suffix=""):
    df = df[(df["model"] == "hgb") & (df["strategy"] == "drop_rows")]
    df = df[df["corruption_type"].isin(["random_missing_values", "label_correlated_missing"])]

    group = df.groupby(["corruption_type", "corruption_rate"]).agg(
            mean_auc=("roc_auc", "mean"),
            n_rows_trained=("n_rows_trained", "mean")
        ).reset_index()

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 8))

    axes[0].set_yscale("log")

    mechanisms = ["random_missing_values", "label_correlated_missing"]

    for mechanism in mechanisms:
        axes[0].set_ylabel("Training rows retained")
        axes[0].set_title("Rows retained by corruption mechanism")

        axes[1].set_ylabel("ROC-AUC")
        axes[1].set_title("Predictive performance by corruption mechanism")

        axes[1].set_xlabel("Corruption rate")
        mechanism_df = group[group["corruption_type"] == mechanism]

        axes[0].plot(
            mechanism_df["corruption_rate"],
            mechanism_df["n_rows_trained"],
            label=mechanism,
            color=MECHANISM_COLORS[mechanism],
            marker="o",
            markersize=4,
            linewidth=1.6,
            alpha=0.9,
        )

        axes[1].plot(
            mechanism_df["corruption_rate"],
            mechanism_df["mean_auc"],
            label=mechanism,
            color=MECHANISM_COLORS[mechanism],
            marker="o",
            markersize=4,
            linewidth=1.6,
            alpha=0.9,
        )

    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(project_dir / f"figures/fig4_selection_bias{suffix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / f"figures/fig4_selection_bias{suffix}.svg", bbox_inches="tight")

    return fig


def recovery_disconnect(df):
    df = df[df["recovery_applicable"].eq(True) & (df["model"] == "hgb")]
    df = df[df["strategy"].isin(["mean_fill", "knn_impute"])]

    group = df.groupby(["strategy", "corruption_type", "corruption_rate"]).agg(
            mean_rmse=("rmse", "mean"),
            mean_wasserstein=("wasserstein", "mean"),
            mean_variance_ratio=("variance_ratio", "mean")
        ).reset_index()

    strategies = [s for s in STRATEGY_COLORS if s in set(group["strategy"])]
    mechanisms = [m for m in MECHANISM_MARKER_SHAPES if m in set(group["corruption_type"])]

    fig, ax = plt.subplots(figsize=(8, 6))

    for strategy in strategies:
        for mechanism in mechanisms:
            plot_df = group[(group['corruption_type'] == mechanism) & (group['strategy'] == strategy)]

            plot_df = plot_df.sort_values("corruption_rate")

            ax.plot(
                plot_df["mean_rmse"],
                plot_df["mean_wasserstein"],
                color=STRATEGY_COLORS[strategy],
                linewidth=0.9,
                alpha=0.35,
                zorder=1,
            )

            ax.scatter(
                x=plot_df["mean_rmse"],
                y=plot_df["mean_wasserstein"],
                s=90,
                edgecolors="white",
                linewidths=1.5,
                c=STRATEGY_COLORS[strategy],
                marker=MECHANISM_MARKER_SHAPES[mechanism],
                label=f"{strategy}-{mechanism}"
            )

            for point, offset in ((plot_df.iloc[0], (-22, -12)), (plot_df.iloc[-1], (6, 4))):
                ax.annotate(
                    f"{point['corruption_rate']:.1f}",
                    xy=(point["mean_rmse"], point["mean_wasserstein"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                    color="#888888",
                )


    color_keys = [
        Line2D([], [], marker="o", linestyle="none", markersize=8,
            color=STRATEGY_COLORS[s], label=s)
        for s in strategies
    ]
    shape_keys = [
        Line2D([], [], marker=MECHANISM_MARKER_SHAPES[m], linestyle="none",
            markersize=8, color="#999999", label=MECHANISM_LABELS[m])
        for m in mechanisms
    ]

    leg1 = ax.legend(handles=color_keys, title="Strategy", loc="lower right",
            bbox_to_anchor=(1, 0.22), frameon=False, fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=shape_keys, title="Mechanism", loc="lower right",
            bbox_to_anchor=(1, 0), frameon=False, fontsize=9, title_fontsize=9)
    ax.grid(True, linewidth=0.5, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(colors="#666666", length=0)

    ax.set_title(
        "Mean fill wins on squared error and loses on distribution;\n"
        "under self-masking both imputers fail on both",
        loc="left", fontsize=12, pad=12,
    )

    ax.set_xlabel("RMSE, raw units pooled over columns (lower is better)")
    ax.set_ylabel("Wasserstein, raw units (lower is better)")

    fig.savefig(project_dir / "figures/fig5_recovery_disconnect.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / "figures/fig5_recovery_disconnect.svg", bbox_inches="tight")

    return fig


def fuzzy_bucket_composition(df):
    df = df[(df["corruption_type"] == "typo_corruption") & (df["model"] == "hgb") & (df["strategy"] == "fuzzy_match")]

    group = df.groupby(["corruption_rate"]).agg(
            mean_cat_match_rate=("cat_match_rate", "mean"),
            mean_cat_untouched_rate=("cat_untouched_rate", "mean"),
            mean_cat_wrong_label_rate=("cat_wrong_label_rate", "mean")
        ).reset_index()

    group = group[group["corruption_rate"] != 0.0]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    positions = range(len(group))

    bottoms = 0
    for key, col, label in [
        ("match",     "mean_cat_match_rate",       "Matched correctly"),
        ("untouched", "mean_cat_untouched_rate",   "Typo left in place"),
        ("wrong",     "mean_cat_wrong_label_rate", "Mapped to wrong label"),
    ]:
        ax.bar(positions, group[col], bottom=bottoms, width=0.6,
            color=BUCKET_COLORS[key], edgecolor="white", linewidth=2,
            label=label, zorder=2)
        bottoms = bottoms + group[col]

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{r:.0%}" for r in group["corruption_rate"]])
    ax.set_xlabel("Corruption rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", linewidth=0.5, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#666666", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=9)
    ax.set_title("Fuzzy matching gets more confident and more wrong",
                loc="left", fontsize=12, pad=38)

    for i, v in enumerate(group["mean_cat_wrong_label_rate"]):
        ax.text(i, 1.01, f"{v:.1%}", ha="center", fontsize=9, color="#52514e")

    fig.savefig(project_dir / "figures/fig6_fuzzy_bucket.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / "figures/fig6_fuzzy_bucket.svg", bbox_inches="tight")

    return fig


# a missing-value flag only helps when the deployment data is missing in the same way
def indicator_vs_deployment(df, model="logreg"):
    mechanisms = ["random_missing_values", "label_correlated_missing", "self_masking_missing"]
    strategies = ["mean_fill", "mean_fill_indicator"]
    conditions = ["clean", "corrupted"]

    df = df[(df["model"] == model) & df["strategy"].isin(strategies) & df["corruption_type"].isin(mechanisms)]

    group = df.groupby(["holdout_condition", "corruption_type", "strategy", "corruption_rate"]).agg(
        mean_auc=("roc_auc", "mean"),
        std_auc=("roc_auc", "std"),
        n=("roc_auc", "count"),
    ).reset_index()
    group["se"] = group["std_auc"] / np.sqrt(group["n"])

    fig, axes = plt.subplots(len(conditions), len(mechanisms), figsize=(13, 7.5), sharex=True, sharey=True)

    for row, condition in enumerate(conditions):
        for col, mechanism in enumerate(mechanisms):
            ax = axes[row, col]
            plot_df = group[(group["holdout_condition"] == condition) & (group["corruption_type"] == mechanism)]

            # uncorrupted reference
            reference = plot_df[plot_df["corruption_rate"] == 0.0]["mean_auc"].mean()
            ax.axhline(reference, color="#b8b8b8", linewidth=0.8, linestyle=":", zorder=0)

            for strategy in strategies:
                strategy_df = plot_df[plot_df["strategy"] == strategy].sort_values("corruption_rate")
                ax.plot(
                    strategy_df["corruption_rate"],
                    strategy_df["mean_auc"],
                    label=strategy,
                    color=STRATEGY_COLORS[strategy],
                    linestyle=STRATEGY_LINESTYLES[strategy],
                    marker="o",
                    markersize=4,
                    linewidth=2,
                )
                ax.fill_between(
                    strategy_df["corruption_rate"],
                    strategy_df["mean_auc"] - strategy_df["se"],
                    strategy_df["mean_auc"] + strategy_df["se"],
                    alpha=0.15,
                    color=STRATEGY_COLORS[strategy],
                    linewidth=0,
                )

                # direct label at the end of the line (first panel only, the legend covers the rest)
                if row == 0 and col == 0:
                    last = strategy_df.iloc[-1]
                    offset = 8 if strategy == "mean_fill_indicator" else -12
                    ax.annotate(strategy, xy=(last["corruption_rate"], last["mean_auc"]),
                                xytext=(-4, offset), textcoords="offset points",
                                ha="right", fontsize=8, color="#52514e")

            if row == 0:
                ax.set_title(MECHANISM_LABELS[mechanism], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"ROC-AUC\n{HOLDOUT_LABELS[condition]}")
            if row == len(conditions) - 1:
                ax.set_xlabel("Corruption rate")

            ax.grid(True, linewidth=0.5, color="#e5e5e5", zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#cccccc")
            ax.tick_params(colors="#666666", length=0)

    # the holdout missingness is generated from the holdout labels here, so the flag reads the label
    axes[1, 1].annotate(
        "missingness is generated from the label,\nso the flag leaks it (upper bound)",
        xy=(0.02, 0.97), xycoords="axes fraction", va="top",
        fontsize=8, color="#52514e",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(
        f"A missing-value flag helps only when deployment data is missing the same way ({model})",
        x=0.01, ha="left", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])

    fig.savefig(project_dir / f"figures/fig7_indicator_{model}.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / f"figures/fig7_indicator_{model}.svg", bbox_inches="tight")

    return fig


if __name__ == "__main__":
    for condition in ["clean", "corrupted"]:
        # the original figures keep their names, the corrupted holdout versions get a suffix
        suffix = "" if condition == "clean" else "_corrupted_holdout"
        condition_df = holdout(df, condition)

        degradation_grid(condition_df, "roc_auc", "ROC-AUC", f"fig1_auc{suffix}")
        degradation_grid(condition_df, "f1", "F1", f"fig2_f1{suffix}")
        section_bias(condition_df, suffix)
        plt.close("all")

    # the train/deploy mismatch story, only meaningful with a clean holdout
    divergence(holdout(df, "clean"))

    # recovery metrics only depend on train, so one holdout condition is enough (they are duplicated)
    recovery_disconnect(holdout(df, "clean"))
    fuzzy_bucket_composition(holdout(df, "clean"))

    indicator_vs_deployment(df, "logreg")
    indicator_vs_deployment(df, "hgb")
    plt.close("all")
