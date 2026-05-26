from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HISTORY_CSV = Path("results/training_history_final.csv")
OUTPUT_PDF = Path("results/training_curves.pdf")
OUTPUT_PNG = Path("results/training_curves.png")

# Estilo serif consistente con un paper LNCS.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def main() -> None:
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(
            f"No se encuentra {HISTORY_CSV}. "
            f"Ejecutar primero src/train_final.py para generarlo."
        )

    history = pd.read_csv(HISTORY_CSV)

    best_idx = history["f1_macro"].idxmax()
    best_epoch = int(history.loc[best_idx, "epoch"])
    best_f1 = float(history.loc[best_idx, "f1_macro"])

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Curvas de pérdida train vs val.
    ax_loss.plot(
        history["epoch"], history["train_loss"],
        marker="o", linewidth=1.8, label="Entrenamiento",
    )
    ax_loss.plot(
        history["epoch"], history["val_loss"],
        marker="s", linewidth=1.8, label="Validación",
    )
    ax_loss.axvline(
        best_epoch, color="gray", linestyle=":", linewidth=1.2,
        label=f"Mejor epoch ({best_epoch})",
    )
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Pérdida (BCE)")
    ax_loss.set_title("(a) Curvas de pérdida")
    ax_loss.legend(loc="best", framealpha=0.9)

    # (b) F1 macro de validación a lo largo de las epochs.
    ax_f1.plot(
        history["epoch"], history["f1_macro"],
        marker="o", linewidth=1.8, color="tab:green",
        label="F1 macro (validación)",
    )
    ax_f1.axvline(
        best_epoch, color="gray", linestyle=":", linewidth=1.2,
        label=f"Mejor epoch ({best_epoch})",
    )
    ax_f1.scatter(
        [best_epoch], [best_f1], color="red", s=60, zorder=5,
        label=f"F1 = {best_f1:.4f}",
    )
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1-score macro")
    ax_f1.set_title("(b) F1-score macro de validación")
    ax_f1.legend(loc="best", framealpha=0.9)

    fig.tight_layout()

    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, format="png", dpi=300, bbox_inches="tight")

    print(f"Figura guardada en: {OUTPUT_PDF}")
    print(f"Figura guardada en: {OUTPUT_PNG}")
    print(f"Mejor epoch: {best_epoch}, F1 macro = {best_f1:.4f}")


if __name__ == "__main__":
    main()
