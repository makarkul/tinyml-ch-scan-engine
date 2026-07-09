"""Generate comparison plots: CFAR baseline vs Attn vs TCN.

Uses the actual numbers already produced by eval_breakdown.py / compare_checkpoints.py
/ fft_cfar_baseline.py runs. Edit the RESULTS dict below if you re-run and get
updated numbers -- no need to re-run training to regenerate these plots.
"""
import matplotlib.pyplot as plt
import numpy as np

# ── Data from your actual runs ──────────────────────────────────────────
SNR_BINS = ["[-10,0)", "[0,10)", "[10,20)", "[20,30)", "[30,41)"]
SNR_N    = [2449, 2254, 768, 852, 930]

CFO_BINS = ["<-10kHz", "[-10,-3)", "[-3,3)", "[3,10)", ">10kHz"]
CFO_N    = [352, 496, 5737, 509, 406]

RESULTS = {
    "CFAR (baseline)": {
        "color": "#888888",
        "overall": {"P_d": 0.2320, "P_fa": 0.0497, "F1": 0.2766},
        "snr": {
            "P_d":  [0.0663, 0.1035, 0.3423, 0.5499, 0.6506],
            "P_fa": [0.0553, 0.0513, 0.0443, 0.0414, 0.0420],
            "F1":   [0.0847, 0.1336, 0.3899, 0.5707, 0.6430],
        },
        "blocker": {
            "no_blocker":  {"P_d": 0.2565, "P_fa": 0.0480, "F1": 0.3034},
            "blocker":     {"P_d": 0.2071, "P_fa": 0.0515, "F1": 0.2489},
        },
        "cfo": {
            "F1": [0.2276, 0.2064, 0.2857, 0.2902, 0.2525],
        },
        "params": 0,       # no learned parameters
        "macs": 400_000,
    },
    "Attn": {
        "color": "#4C72B0",
        "overall": {"P_d": 0.2991, "P_fa": 0.0257, "F1": 0.3912},
        "snr": {
            "P_d":  [0.0256, 0.0875, 0.5565, 0.8447, 0.9111],
            "P_fa": [0.0179, 0.0217, 0.0361, 0.0389, 0.0369],
            "F1":   [0.0432, 0.1371, 0.5878, 0.7681, 0.8138],
        },
        "blocker": {
            "no_blocker":  {"P_d": 0.3086, "P_fa": 0.0203, "F1": 0.4138},
            "blocker":     {"P_d": 0.2894, "P_fa": 0.0313, "F1": 0.3694},
        },
        "cfo": {
            "F1": [0.3084, 0.3078, 0.4039, 0.4145, 0.3443],
        },
        "params": 38_085,
        "macs": 514_400,
    },
    "TCN": {
        "color": "#DD8452",
        "overall": {"P_d": 0.3500, "P_fa": 0.0489, "F1": 0.3915},
        "snr": {
            "P_d":  [0.1385, 0.3655, 0.5560, 0.5461, 0.5770],
            "P_fa": [0.0611, 0.0495, 0.0370, 0.0369, 0.0333],
            "F1":   [0.1638, 0.4072, 0.5848, 0.5802, 0.6162],
        },
        "blocker": {
            "no_blocker":  {"P_d": 0.3514, "P_fa": 0.0490, "F1": 0.3917},
            "blocker":     {"P_d": 0.3486, "P_fa": 0.0488, "F1": 0.3913},
        },
        "cfo": {
            "F1": [0.1092, 0.1513, 0.4625, 0.1440, 0.1329],
        },
        "params": 211_943,
        "macs": 1_023_950,
    },
}

# NOTE: all numbers above are from a SINGLE training run / SINGLE checkpoint
# per model, at ONE matched operating point (target P_fa=0.05). No repeat-seed
# variance estimate exists yet. Treat differences smaller than a few points,
# especially in small-n bins (CFO tails, n=352-509), with caution.

TARGET_SNR_REGIME = "[0,10)"  # confirmed by OTA capture — the real deployment zone


def plot_snr_trend(metric: str, ylabel: str, filename: str, target_line: float | None = None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SNR_BINS))

    for name, r in RESULTS.items():
        ax.plot(x, r["snr"][metric], marker="o", linewidth=2.2, markersize=7,
                label=name, color=r["color"])

    # Highlight the real-world operating zone
    target_idx = SNR_BINS.index(TARGET_SNR_REGIME)
    ax.axvspan(target_idx - 0.5, target_idx + 0.5, alpha=0.12, color="red",
               label="Real OTA operating zone" if metric == "F1" else None)

    if target_line is not None:
        ax.axhline(target_line, linestyle="--", color="gray", linewidth=1, alpha=0.7)
        ax.text(len(SNR_BINS) - 1, target_line, f" target={target_line}",
                va="bottom", ha="right", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\ndB\n(n={n})" for b, n in zip(SNR_BINS, SNR_N)], fontsize=9)
    ax.set_xlabel("SNR bin")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs SNR — CFAR baseline vs ML models")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_overall_bars(filename: str):
    metrics = ["P_d", "P_fa", "F1"]
    names   = list(RESULTS.keys())
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        vals = [RESULTS[name]["overall"][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=RESULTS[name]["color"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(["P_d (higher better)", "P_fa (lower better)", "F1 (higher better)"])
    ax.set_ylabel("Score")
    ax.set_title("Overall test-set performance — CFAR vs Attn vs TCN")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_complexity(filename: str):
    names  = list(RESULTS.keys())
    params = [RESULTS[n]["params"] for n in names]
    macs   = [RESULTS[n]["macs"] for n in names]
    colors = [RESULTS[n]["color"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].bar(names, params, color=colors)
    axes[0].set_title("Model parameters")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(params):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(names, macs, color=colors)
    axes[1].set_title("MACs per inference")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(macs):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Compute / memory footprint (R6 — embedded feasibility)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_blocker_comparison(filename: str):
    metrics = ["P_d", "P_fa", "F1"]
    names   = list(RESULTS.keys())
    x       = np.arange(len(names))
    width   = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, metrics):
        no_b = [RESULTS[n]["blocker"]["no_blocker"][metric] for n in names]
        b    = [RESULTS[n]["blocker"]["blocker"][metric] for n in names]

        bars1 = ax.bar(x - width/2, no_b, width, label="No blocker", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x + width/2, b,    width, label="Blocker present", color="#C44E52", alpha=0.85)

        for bars in (bars1, bars2):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(max(no_b), max(b)) * 1.25)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Wideband blocker robustness — CFAR vs Attn vs TCN")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_cfo_robustness(filename: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(CFO_BINS))

    for name, r in RESULTS.items():
        ax.plot(x, r["cfo"]["F1"], marker="o", linewidth=2.2, markersize=7,
                label=name, color=r["color"])

    center_idx = CFO_BINS.index("[-3,3)")
    ax.axvspan(center_idx - 0.5, center_idx + 0.5, alpha=0.12, color="green",
               label="Well-disciplined oscillator zone")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(CFO_BINS, CFO_N)], fontsize=9)
    ax.set_xlabel("Carrier frequency offset (CFO)")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs frequency offset — iso-FAR (P_fa≈0.05)\nTCN's matched-filter frontend is sharply CFO-sensitive")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


if __name__ == "__main__":
    plot_snr_trend("F1",   "F1 score",              "plot_f1_vs_snr.png")
    plot_snr_trend("P_d",  "Probability of Detection", "plot_pd_vs_snr.png", target_line=0.9)
    plot_snr_trend("P_fa", "False Alarm Rate",       "plot_pfa_vs_snr.png")
    plot_overall_bars("plot_overall_comparison.png")
    plot_complexity("plot_complexity.png")
    plot_blocker_comparison("plot_blocker_comparison.png")
    plot_cfo_robustness("plot_cfo_robustness.png")
    print("\nAll plots saved to current directory.")