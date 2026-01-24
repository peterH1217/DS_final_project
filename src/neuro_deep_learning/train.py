import logging
import sys
import argparse
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from neuro_deep_learning import fetch, dataset, visualization, cnn

# ---------------- Logging configuration ----------------
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------------- Project paths ----------------
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_SIZE = 0.8
N_EPOCHS = 300
STRIDE = 1

PATIENCE = 50
CROP_SIZE = 500
BATCH_SIZE = 64


def run_epoch(model, loader, criterion, optimizer, device, is_train = True):
    if is_train:
        model.train() # activate the train mode: dropout, batch norm and so on
    else: 
        model.eval()

    correct, total, running_loss = 0, 0, 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb = xb.unsqueeze(1).to(device) # change data from  (Batch, Channels, Time) to (Batch, 1, Channels, Time)
            yb = yb.to(device)

            if is_train:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            else:
                out = model(xb)
                loss = criterion(out, yb)
            
            # metrics
            _, pred = torch.max(out.data, 1) # _ is for values (confidence), pred is for index (class)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
            running_loss += loss.item()
    acc = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return acc, avg_loss


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_acc is None:
            self.best_acc = val_acc
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_acc = val_acc
            self.counter = 0


def run_epoch(model, loader, criterion, optimizer, device, is_train=True):
    """
    Crop-level training loop (OK for training).
    loader must yield: (xb, yb, tb) but we ignore tb here.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    correct, total, running_loss = 0, 0, 0.0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in loader:
            # support both (xb,yb) and (xb,yb,tb)
            if len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch

            xb = xb.unsqueeze(1).to(device)
            yb = yb.to(device)

            if is_train:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            else:
                out = model(xb)
                loss = criterion(out, yb)

            _, pred = torch.max(out.data, 1)
            total += yb.size(0)
            correct += (pred == yb).sum().item()
            running_loss += loss.item()

    acc = 100 * correct / total
    avg_loss = running_loss / max(1, len(loader))
    return acc, avg_loss


def predict_trials_by_mean_logits(model, loader, device):
    """
    Paper-style evaluation:
    aggregate all crop logits per trial -> mean -> 1 prediction per trial.
    loader must yield: (xb, yb, trial_idx)
    """
    model.eval()
    trial_logits = {}  # trial_idx -> list[logits]
    trial_label  = {}  # trial_idx -> label

    with torch.no_grad():
        for xb, yb, tb in loader:
            xb = xb.unsqueeze(1).to(device)
            logits = model(xb).detach().cpu()  # (B, n_classes)

            for logit, y, t in zip(logits, yb, tb):
                t = int(t.item())
                y = int(y.item())
                trial_logits.setdefault(t, []).append(logit)
                trial_label[t] = y

    trial_ids = sorted(trial_logits.keys())
    y_true, y_pred = [], []

    for t in trial_ids:
        mean_logit = torch.stack(trial_logits[t]).mean(dim=0)
        pred = int(torch.argmax(mean_logit).item())
        y_true.append(trial_label[t])
        y_pred.append(pred)

    return np.array(y_true), np.array(y_pred)


def process_dataset(dataset_name: str) -> None:
    print(f"--- PROCESSING DATASET (POOLED SUBJECTS): {dataset_name} ---")

    # ============================================================
    # 1) COLLECT DATA FROM ALL SUBJECTS
    # ============================================================
    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []

    subject_ids = fetch.get_participants(dataset_name)

    for subject_id in subject_ids:
        print(f"Loading subject {subject_id}")

        raw_train, raw_test = fetch.get_dataset(subject_id=subject_id, dataset_name=dataset_name)

        raw_train = dataset.preprocess_data(raw_train)
        raw_test  = dataset.preprocess_data(raw_test)

        # paper: ConvNets best with -0.5s for trial-wise; for your cropped pipeline we keep it too
        X_s1, y_s1 = dataset.make_epochs(raw_train, tmin=-0.5, tmax=4.0)
        X_s2, y_s2 = dataset.make_epochs(raw_test,  tmin=-0.5, tmax=4.0)

        X_s1, y_s1 = dataset.remove_artifact_trials(X_s1, y_s1, threshold_std=20)
        X_s2, y_s2 = dataset.remove_artifact_trials(X_s2, y_s2, threshold_std=20)

        X_train_all.append(X_s1); y_train_all.append(y_s1)
        X_test_all.append(X_s2);  y_test_all.append(y_s2)

    X_train_all = np.concatenate(X_train_all, axis=0)
    y_train_all = np.concatenate(y_train_all, axis=0)
    X_test_all  = np.concatenate(X_test_all, axis=0)
    y_test_all  = np.concatenate(y_test_all, axis=0)

    print(f"Total pooled train trials (Session 1): {len(X_train_all)}")
    print(f"Total pooled test trials  (Session 2): {len(X_test_all)}")

    # ============================================================
    # 2) TRAIN / VAL SPLIT (POOLED SESSION 1)
    # ============================================================
    split_idx = int(len(X_train_all) * TRAIN_SIZE)

    train_ds = dataset.CropsDataset(
        X_train_all[:split_idx],
        y_train_all[:split_idx],
        crop_size=CROP_SIZE,
        stride=STRIDE
    )

    val_ds = dataset.CropsDataset(
        X_train_all[split_idx:],
        y_train_all[split_idx:],
        crop_size=CROP_SIZE,
        stride=STRIDE
    )

    test_ds = dataset.CropsDataset(
        X_test_all,
        y_test_all,
        crop_size=CROP_SIZE,
        stride=STRIDE
    )

    print(f"Train crops: {len(train_ds)} | Val crops: {len(val_ds)} | Test crops: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ============================================================
    # 3) MODEL, OPTIMIZER
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = cnn.DeepConvNet(
        n_channels=X_train_all.shape[1],
        n_classes=4,
        input_window_samples=CROP_SIZE
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    early_stopper = EarlyStopping(patience=PATIENCE)

    # ============================================================
    # 4) TRAINING LOOP
    # ============================================================
    history = {'train_acc': [], 'val_acc': [], 'test_acc': [],
               'train_loss': [], 'val_loss': [], 'test_loss': []}

    best_val_acc = -1.0

    for epoch in range(N_EPOCHS):
        # (A) crop-level training acc/loss
        train_acc, train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)

        # (B) trial-level val/test acc (paper-style)
        y_true_val, y_pred_val = predict_trials_by_mean_logits(model, val_loader, device)
        val_acc = (y_true_val == y_pred_val).mean() * 100

        y_true_test, y_pred_test = predict_trials_by_mean_logits(model, test_loader, device)
        test_acc = (y_true_test == y_pred_test).mean() * 100

        # (optional) we don't have a meaningful "trial-loss" here, keep as None
        val_loss = None
        test_loss = None

        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                RESULTS_DIR / f"best_model_{dataset_name}_epochs_{N_EPOCHS}_stride_{STRIDE}.pth"
            )
            print(f"Epoch {epoch}: New Best Val Acc: {val_acc:.1f}%")

        print(f"Epoch {epoch} | Train(crop): {train_acc:.1f}% | Val(trial): {val_acc:.1f}% | Test(trial): {test_acc:.1f}%")

        early_stopper(val_acc)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # ============================================================
    # 5) FINAL EVAL + CONFUSION MATRIX (TRIAL-WISE!)
    # ============================================================
    # Load best model
    best_path = RESULTS_DIR / f"best_model_{dataset_name}_epochs_{N_EPOCHS}_stride_{STRIDE}.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    # Final trial-wise predictions
    y_true_test, y_pred_test = predict_trials_by_mean_logits(model, test_loader, device)
    final_test_acc = (y_true_test == y_pred_test).mean() * 100
    print(f"Final Test Accuracy (TRIAL-wise, Session 2, pooled): {final_test_acc:.2f}%")

    # Save history + plots
    history_path = RESULTS_DIR / f"history_{dataset_name}_epochs_{N_EPOCHS}_stride_{STRIDE}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    visualization.plot_training_history(
        history,
        dataset_name,
        results_path=f"accuracy_{dataset_name}_epochs_{N_EPOCHS}_stride_{STRIDE}.png"
    )
    del history

    # Confusion matrix (trial-wise)
    class_mapping = fetch.get_dataset_class_mapping(dataset_name)
    cm_path = RESULTS_DIR / f"confusion_matrix_{dataset_name}_epochs_{N_EPOCHS}_stride_{STRIDE}.png"

    visualization.plot_test_confusion_matrix(
        y_true_test, y_pred_test,
        class_mapping=class_mapping,
        title=f"{dataset_name} Test Confusion Matrix (TRIAL-wise)",
        save_path=cm_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neuro Deep Learning Model")
    parser.add_argument("--dataset", type=str, default="BNCI2014_001", help="Dataset to process")
    args, unknown = parser.parse_known_args()

    try:
        process_dataset(args.dataset)
    except Exception as e:
        logger.exception(f"Failed to process {args.dataset}: {e}")

    logger.info(">>> Pipeline complete.")


if __name__ == "__main__":
    main()