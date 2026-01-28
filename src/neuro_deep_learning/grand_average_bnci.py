# import torch
# import numpy as np
# from neuro_deep_learning import fetch, dataset, cnn, train
# from pathlib import Path
# import sys
# import os
# import matplotlib.pyplot as plt
# # Fix path to look inside 'src'
# sys.path.append('src')  


# # CONFIGURATION
# dataset_name = "BNCI2014_001"      
# subject_ids = list(range(1, 10))  
# STRIDE = 100  
# CROP_SIZE = 500


# # Setup Directories
# RESULTS_DIR = Path("results")
# SAVE_DIR = RESULTS_DIR / "grand_average"
# SAVE_DIR.mkdir(parents=True, exist_ok=True)  # NEW: Creating folder if missing

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# accuracies = []
# found_subjects = [] # Track which subjects we actually found models for

# print(f"Calculating Grand Average for {dataset_name}")

# for subject_id in subject_ids:
#     model_path = RESULTS_DIR / "models" / f"best_model_{dataset_name}_S{subject_id}.pth"
    
#     # Check if model exists
#     if not model_path.exists():
#         print(f"Warning: Model for S{subject_id} not found. Skipping.")
#         continue
#     # 1. Load Data
#     _, raw_test = fetch.get_dataset(subject_id, dataset_name)
#     raw_test = dataset.preprocess_data(raw_test)
#     X_test, y_test = dataset.make_epochs(raw_test, tmin=-0.5, tmax=4.0)
#     X_test, y_test = dataset.remove_artifact_trials(X_test, y_test)
    
#     # 2. Prepare Loader
#     test_ds = dataset.CropsDataset(X_test, y_test, crop_size=CROP_SIZE, stride=STRIDE)
#     test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)
#     # 3. Load Model
#     n_chans = X_test.shape[1] 
#     model = cnn.DeepConvNet(n_channels=n_chans, n_classes=4, input_window_samples=CROP_SIZE).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
    
#     # 4. Predict
#     y_true, y_pred = train.predict_trials_by_mean_logits(model, test_loader, device)
#     acc = (y_true == y_pred).mean() * 100
    
#     # Store results dynamically
#     accuracies.append(acc)
#     found_subjects.append(f'S{subject_id}')
#     print(f"Subject {subject_id}: {acc:.2f}%")


# if len(accuracies) > 0:
#     grand_avg = np.mean(accuracies)
#     print(f" {dataset_name} GRAND AVERAGE: {grand_avg:.2f}%")
    
#     # PLOTTING SECTION (Updated to use real data)
#     plt.figure(figsize=(10, 6))
    
#     # Use the dynamic lists 'found_subjects' and 'accuracies'
#     bars = plt.bar(found_subjects, accuracies, color=['#4CAF50' if x > 60 else '#FF5722' for x in accuracies])
#     # Add Grand Average Line
#     plt.axhline(y=grand_avg, color='blue', linestyle='--', linewidth=2, label=f'Grand Average ({grand_avg:.1f}%)')
#     plt.axhline(y=25, color='gray', linestyle=':', label='Chance Level (25%)')
#     # Labels
#     plt.ylabel('Accuracy (%)')
#     plt.title(f'{dataset_name}: Accuracy per Subject')
#     plt.ylim(0, 100)
#     plt.legend()
#     # Add numbers on top of bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height + 1,
#                  f'{height:.1f}%', ha='center', va='bottom')
#     plt.tight_layout()
    
#     # SAVING THE FIGURE
#     save_path = SAVE_DIR / f"grand_average_{dataset_name}.png"
#     plt.savefig(save_path)
#     print(f"Plot saved to: {save_path}")
    
# else:
#     print("No models found! Check your 'results/models' path.")



import sys
import os
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from neuro_deep_learning import fetch, dataset, cnn, train


# Logging configuration
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

sys.path.append("src")


# -------------------- CONFIG --------------------
DATASET_NAME = "BNCI2014_001"
SUBJECT_IDS = list(range(1, 10))
STRIDE = 100
CROP_SIZE = 500

RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
SAVE_DIR = RESULTS_DIR / "grand_average"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- FUNCTIONS --------------------
def load_test_data(subject_id):
    _, raw_test = fetch.get_dataset(subject_id, DATASET_NAME)
    raw_test = dataset.preprocess_data(raw_test)
    X, y = dataset.make_epochs(raw_test, tmin=-0.5, tmax=4.0)
    return dataset.remove_artifact_trials(X, y)


def build_test_loader(X, y):
    test_ds = dataset.CropsDataset(
        X, y, crop_size=CROP_SIZE, stride=STRIDE
    )
    return torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False
    )


def load_model(n_chans, model_path):
    model = cnn.DeepConvNet(
        n_channels=n_chans,
        n_classes=4,
        input_window_samples=CROP_SIZE,
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )
    return model


def evaluate_subject(subject_id):
    model_path = MODELS_DIR / f"best_model_{DATASET_NAME}_S{subject_id}.pth"
    if not model_path.exists():
        logger.warning(f"Model for S{subject_id} not found. Skipping.")
        return None

    X_test, y_test = load_test_data(subject_id)
    test_loader = build_test_loader(X_test, y_test)

    model = load_model(X_test.shape[1], model_path)
    y_true, y_pred = train.predict_trials_by_mean_logits(
        model, test_loader, DEVICE
    )

    acc = (y_true == y_pred).mean() * 100
    logger.info(f"Subject {subject_id}: {acc:.2f}%")
    return acc


def plot_results(subjects, accuracies):
    grand_avg = np.mean(accuracies)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        subjects,
        accuracies,
        color=["#4CAF50" if x > 60 else "#FF5722" for x in accuracies],
    )

    plt.axhline(
        y=grand_avg,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Grand Average ({grand_avg:.1f}%)",
    )
    plt.axhline(
        y=25,
        color="gray",
        linestyle=":",
        label="Chance Level (25%)",
    )

    plt.ylabel("Accuracy (%)")
    plt.title(f"{DATASET_NAME}: Accuracy per Subject")
    plt.ylim(0, 100)
    plt.legend()

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    save_path = SAVE_DIR / f"grand_average_{DATASET_NAME}.png"
    plt.savefig(save_path)
    logger.info(f"Plot saved to: {save_path}")


def main():
    logger.info(f"Calculating Grand Average for {DATASET_NAME}")

    accuracies = []
    subjects = []

    for subject_id in SUBJECT_IDS:
        acc = evaluate_subject(subject_id)
        if acc is not None:
            accuracies.append(acc)
            subjects.append(f"S{subject_id}")

    if not accuracies:
        logger.error("No models found! Check your results/models path.")
        return

    logger.info(
        f"{DATASET_NAME} GRAND AVERAGE: {np.mean(accuracies):.2f}%"
    )
    plot_results(subjects, accuracies)


if __name__ == "__main__":
    main()