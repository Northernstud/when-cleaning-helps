import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

project_dir = Path(__file__).resolve().parent.parent
results = project_dir / "results/experiments.parquet"

BUCKET_COLORS = {
    "match":     "#00a878",
    "untouched": "#b8b8b8",
    "wrong":     "#d1495b",
}


STRATEGY_COLORS = {
    "do_nothing":   "#888888",
    "drop_rows":    "#d1495b",
    "mean_fill":    "#2a7de1",
    "knn_impute":   "#00a878",
    "fuzzy_match":  "#f5a623",
}

STRATEGY_LINESTYLES = {
    "do_nothing":   "-",      # solid, it's the baseline
    "drop_rows":    "-",      # solid, it's the one that separates
    "mean_fill":    "--",
    "knn_impute":   ":",
    "fuzzy_match":  "-.",
}

MECHANISM_COLORS = {
    "random_missing_values": "#d1495b",
    "label_correlated_missing": "#2a7de1",
}

METRIC_LABELS = {"mean_auc": "ROC-AUC", "mean_f1": "F1"}

MECHANISM_MARKER_SHAPES = {
    "random_missing_values": 'o',
    "label_correlated_missing": '^',
    "typo_corruption": 's',
}

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

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row", sharex=True)

    for row, model in enumerate(models):
        for col, corruption_type in enumerate(corruption_types):

            plot_df = group[(group["model"] == model) & (group["corruption_type"] == corruption_type)]
            plot_df = plot_df.dropna(subset=["mean_val"])
            strategies = plot_df['strategy'].unique()   
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

            ax.set_xlabel("Corruption rate")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{model} — {corruption_type}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])


    fig.savefig(project_dir / f"figures/{outname}_degradation.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / f"figures/{outname}_degradation.svg", bbox_inches="tight")

    plt.show()

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

    for ax, pct in zip(axes, ["AUC −0.6%", "F1 −58%"]):
        ax.axvline(0.1, color="#cccccc", linewidth=0.8, zorder=0)
        ax.annotate(
            f"{pct} vs clean",
            xy=(0.115, 0.5),
            fontsize=9,
            color="#555555",
        )
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fig.savefig(project_dir / "figures/fig3_divergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / "figures/fig3_divergence.svg", bbox_inches="tight")

    plt.show()

def section_bias(df):
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

    fig.savefig(project_dir / f"figures/fig4_selection_bias.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / f"figures/fig4_selection_bias.svg", bbox_inches="tight")
    
    plt.tight_layout()
    plt.show()  


def recovery_disconnect(df):
    df = df[(df["recovery_applicable"] == True) & (df["model"] == "hgb")]
    df = df[df["strategy"].isin(["mean_fill", "knn_impute"])]

    group = df.groupby(["strategy", "corruption_type", "corruption_rate"]).agg(
            mean_rmse=("rmse", "mean"),
            mean_wasserstein=("wasserstein", "mean"),
            mean_variance_ratio=("variance_ratio", "mean")
        ).reset_index()

    strategies = group['strategy'].unique() 
    mechanisms = group['corruption_type'].unique()

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
            markersize=8, color="#999999", label=m)
        for m in mechanisms
    ]

    leg1 = ax.legend(handles=color_keys, title="Strategy", loc="upper right", frameon=False, fontsize=9, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=shape_keys, title="Mechanism", loc="upper right",
            bbox_to_anchor=(1, 0.80), frameon=False, fontsize=9, title_fontsize=9)
    ax.grid(True, linewidth=0.5, color="#e5e5e5", zorder=0)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(colors="#666666", length=0)

    first, last = plot_df.iloc[0], plot_df.iloc[-1]
    for point, text in ((first, "0.0"), (last, "0.5")):
        ax.annotate(
            text,
            xy=(point["mean_rmse"], point["mean_wasserstein"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color="#888888",
        )

    ax.set_title(
        "Mean fill wins on squared error and loses on distribution",
        loc="left", fontsize=12, pad=12,
    )

    ax.set_xlabel("RMSE (Lower is better)")
    ax.set_ylabel("Wasserstein (lower is better)")

    fig.savefig(project_dir / "figures/fig5_recovery_disconnect.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / "figures/fig5_recovery_disconnect.svg", bbox_inches="tight")
    
    plt.show()


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
        ax.text(i, 1.01, f"{v:.1%}", ha="center", fontsize=9, color="#d1495b")

    fig.savefig(project_dir / f"figures/fig6_fuzzy_bucket.png", dpi=300, bbox_inches="tight")
    fig.savefig(project_dir / f"figures/fig6_fuzzy_bucket.svg", bbox_inches="tight")
    
    plt.show()




fuzzy_bucket_composition(df)
#recovery_disconnect(df)
#section_bias(df)
#divergence(df)
# degradation_grid(df, "roc_auc", "ROC-AUC", "fig1_auc")
# degradation_grid(df, "f1", "F1", "fig2_f1")

