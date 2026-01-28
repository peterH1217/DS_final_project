# import sys
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# # CONFIGURATION
# dataset_name = "Schirrmeister2017"

# # Setup Directories
# RESULTS_DIR = Path("results")
# SAVE_DIR = RESULTS_DIR / "grand_average"
# SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Creates folder if missing

# print(f"--- Generating Grand Average Plot for {dataset_name} ---")

# # 1. HARDCODED RESULTS (Derived from the manual run)
# # S1 to S14
# accuracies = [
#     92.2, 87.2, 96.2, 98.1, 97.5,   # S1-S5
#     83.1, 88.8, 83.1, 88.8, 80.9,   # S6-S10
#     84.5, 90.6, 81.0, 66.9          # S11-S14
# ]

# subjects = [f'S{i}' for i in range(1, 15)]

# # 2. Calculate Statistics
# grand_avg = np.mean(accuracies)
# print(f"Data loaded for {len(subjects)} subjects.")
# print(f" {dataset_name} GRAND AVERAGE: {grand_avg:.2f}%")

# # 3. PLOTTING SECTION
# plt.figure(figsize=(12, 6))

# # Color logic: Green if > 90%, Blue if > 80%, Orange if lower
# colors = []
# for acc in accuracies:
#     if acc >= 90: colors.append('#2E7D32') # Dark Green
#     elif acc >= 80: colors.append('#1565C0') # Blue
#     else: colors.append('#D84315') # Orange

# bars = plt.bar(subjects, accuracies, color=colors)

# # Grand Average Line
# plt.axhline(y=grand_avg, color='red', linestyle='--', linewidth=2, label=f'Grand Average ({grand_avg:.2f}%)')
# plt.axhline(y=25, color='gray', linestyle=':', label='Chance (25%)')

# # Labels and Titles
# plt.ylabel('Accuracy (%)')
# plt.title(f'{dataset_name} (High Gamma): Accuracy per Subject')
# plt.ylim(0, 105)
# plt.legend(loc='lower right')

# # Add numbers on bars
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#              f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# plt.tight_layout()

# # 4. SAVE THE FIGURE
# save_path = SAVE_DIR / f"grand_average_{dataset_name}.png"
# plt.savefig(save_path, dpi=300)
# print(f"Plot saved to: {save_path}")



import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Logging configuration
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)


# -------------------- CONFIG --------------------
DATASET_NAME = "Schirrmeister2017"

RESULTS_DIR = Path("results")
SAVE_DIR = RESULTS_DIR / "grand_average"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- FUNCTIONS --------------------
def load_results():
    accuracies = [
        92.2, 87.2, 96.2, 98.1, 97.5,
        83.1, 88.8, 83.1, 88.8, 80.9,
        84.5, 90.6, 81.0, 66.9,
    ]
    subjects = [f"S{i}" for i in range(1, 15)]
    return subjects, accuracies


def compute_grand_average(accuracies):
    return float(np.mean(accuracies))


def get_bar_colors(accuracies):
    colors = []
    for acc in accuracies:
        if acc >= 90:
            colors.append("#2E7D32")
        elif acc >= 80:
            colors.append("#1565C0")
        else:
            colors.append("#D84315")
    return colors


def plot_results(subjects, accuracies, grand_avg):
    plt.figure(figsize=(12, 6))

    colors = get_bar_colors(accuracies)
    bars = plt.bar(subjects, accuracies, color=colors)

    plt.axhline(
        y=grand_avg,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Grand Average ({grand_avg:.2f}%)",
    )
    plt.axhline(
        y=25,
        color="gray",
        linestyle=":",
        label="Chance (25%)",
    )

    plt.ylabel("Accuracy (%)")
    plt.title(f"{DATASET_NAME} (High Gamma): Accuracy per Subject")
    plt.ylim(0, 105)
    plt.legend(loc="lower right")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()

    save_path = SAVE_DIR / f"grand_average_{DATASET_NAME}.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"Plot saved to: {save_path}")


def main():
    logger.info(f"Generating Grand Average Plot for {DATASET_NAME}")

    subjects, accuracies = load_results()
    logger.info(f"Data loaded for {len(subjects)} subjects.")

    grand_avg = compute_grand_average(accuracies)
    logger.info(f"{DATASET_NAME} GRAND AVERAGE: {grand_avg:.2f}%")

    plot_results(subjects, accuracies, grand_avg)


if __name__ == "__main__":
    main()