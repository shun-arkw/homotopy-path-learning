# -----------------------------------------------------------------------------------------------
# Task 1.1:Implement a one-time step polynomial root-finding algorithm based on machine learning.
# -----------------------------------------------------------------------------------------------

import argparse
import os
import random
import logging
import time
from datetime import datetime
import pytz
import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    DataLoader,
    DataProcessor,
    RootFindingMLP,
    mse_sorted,
    evaluate_accuracy_within_threshold,
)


def setup_logging(save_path: str, timezone: str = 'Europe/Paris') -> tuple[logging.Logger, datetime]:
    """Initialize logging configuration"""
    # Configurable time zone
    tz = pytz.timezone(timezone)
    now_local = datetime.now(tz)
    timestamp = now_local.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_path, f"training_log_{timestamp}.log")
    
    # Create custom formatter for cleaner logs
    class CleanFormatter(logging.Formatter):
        def format(self, record):
            # For all messages, just show the message without timestamp
            return record.getMessage()
    
    # Setup file handler with clean formatter
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(CleanFormatter())
    
    # Setup console handler with timestamp
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )
    return logging.getLogger(__name__), now_local


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, model: nn.Module, device: str = "cpu", logger: logging.Logger | None = None, epoch_log_interval: int = 1):
        self.model = model
        self.device = device
        self.logger = logger
        self.epoch_log_interval = epoch_log_interval
        self.model.to(device)
    
    def train(self, train_dataset: DataProcessor, test_dataset: DataProcessor | None = None, epochs: int = 50, lr: float = 1e-3, batch_size: int = 512, lr_scheduler: str = "cosine") -> None:
        """Train the model with optional validation."""
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        # Select learning rate scheduler
        if lr_scheduler == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        elif lr_scheduler == "linear":
            # Linearly decay multiplicative factor from 1.0 to 0.0 across epochs
            sch = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        else:
            sch = None

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = None
        if test_dataset is not None:
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for ep in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            for features, target_roots in train_loader:
                features = features.to(self.device)
                target_roots = target_roots.to(self.device)

                pred = self.model(features)  # (B,2)
                loss = mse_sorted(pred, target_roots)

                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * features.size(0)

            if sch is not None:
                sch.step()
            train_loss = total/len(train_dataset)
            msg = f"[{ep:03d}] train_loss={train_loss:.6f}"

            if test_loader is not None:
                self.model.eval()
                test_loss = 0.0
                test_rmse = 0.0
                test_count = 0
                with torch.no_grad():
                    for features, target_roots in test_loader:
                        features = features.to(self.device)
                        target_roots = target_roots.to(self.device)
                        pred = self.model(features)
                        test_loss += mse_sorted(pred, target_roots).item() * features.size(0)
                        test_rmse += torch.sqrt(mse_sorted(pred, target_roots)).item() * features.size(0)
                        test_count += features.size(0)
                test_loss = test_loss/len(test_dataset)
                test_rmse = test_rmse/test_count
                msg += f"  test_loss={test_loss:.6f}  test_rmse={test_rmse:.6f}"
                
                # Log output (controlled by epoch_log_interval)
                if self.logger and self.epoch_log_interval > 0 and ep % self.epoch_log_interval == 0:
                    self.logger.info(f"Epoch {ep:03d}: train_loss={train_loss:.8f}, test_loss={test_loss:.8f}, test_rmse={test_rmse:.8f}")
            else:
                if self.logger and self.epoch_log_interval > 0 and ep % self.epoch_log_interval == 0:
                    self.logger.info(f"Epoch {ep:03d}: train_loss={train_loss:.8f}")
            # print(msg)

    @torch.no_grad()
    def predict(self, input_features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Predict the two roots at the next step and return them.

        Args:
            input_features: Tensor of shape (N, 6) in original scale.
                Per-sample layout:
                    [a_curr, b_curr, λ1_curr, λ2_curr, a_next, b_next]
                Element descriptions:
                    - a_curr, b_curr: Coefficients of the current quadratic x^2 + a x + b
                    - λ1_curr, λ2_curr: Current-step roots (descending: λ1_curr ≥ λ2_curr)
                    - a_next, b_next: Coefficients of the next quadratic x^2 + a x + b
            mean: Mean tensor for normalization (from training data)
            std: Standard-deviation tensor for normalization (from training data)

        Returns:
            Tensor of shape (N, 2).
                Per-sample layout:
                    [λ1_next, λ2_next]: Predicted next-step roots (descending: λ1_next ≥ λ2_next)

        Notes:
            - For readability, representation is unified as descending (λ1 ≥ λ2).
            - Loss/metrics are order-invariant: both predictions and targets are
              sorted ascending before comparison (see utils.mse_sorted).
        """
        self.model.eval()
        x = torch.as_tensor(input_features, dtype=torch.float32)
        x_std = (x - mean) / std
        x_std = x_std.to(self.device)
        pred = self.model(x_std).cpu()  # (N,2)
        pred, _ = torch.sort(pred, dim=1, descending=True)
        return pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-path", type=str, help="Path to train dataset")
    parser.add_argument("--test-dataset-path", type=str, help="Path to test dataset")
    parser.add_argument("--save-path", type=str, help="Path to save model and scaler")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-filter-disc", action="store_true",
                        help="If set, do NOT filter by discriminant>0 (not recommended for real-root training).")
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.1],
                        help="RMSE accuracy threshold(s) for evaluation (can specify multiple). Default: [0.1]")
    parser.add_argument("--epoch-log-interval", type=int, default=1,
                        help="Log epoch results every N epochs (default: 1, set to 0 to disable epoch logging)")
    parser.add_argument("--timezone", type=str, default="Europe/Paris",
                        help="Timezone for logging (default: Europe/Paris)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "linear"],
                        help="Learning rate scheduler type (cosine or linear). Default: cosine")
    args = parser.parse_args()
    
    # Initialize logging configuration
    os.makedirs(args.save_path, exist_ok=True)
    logger, now_local = setup_logging(args.save_path, args.timezone)
    
    # Log experiment configuration
    logger.info("=== Experiment Configuration ===")
    logger.info(f"Location: {args.timezone}")
    logger.info(f"Start time: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Train dataset: {args.train_dataset_path}")
    logger.info(f"Test dataset: {args.test_dataset_path}")
    logger.info(f"Save path: {args.save_path}")
    logger.info(f"Train size: {args.train_size}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Hidden size: {args.hidden}")
    logger.info(f"Hidden depth: {args.depth}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"LR scheduler: {args.lr_scheduler}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Discriminant filter: {not args.no_filter_disc}")
    logger.info(f"Accuracy thresholds: {args.threshold}")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize data loader and model
    data_loader = DataLoader()
    model = RootFindingMLP(in_dim=6, hidden=args.hidden, depth=args.depth, out_dim=2)
    trainer = ModelTrainer(model, device=args.device, logger=logger, epoch_log_interval=args.epoch_log_interval)
    
    # Load datasets
    logger.info("\n=== Dataset Loading ===")
    logger.info(f"Loading train dataset: {args.train_dataset_path}")
    train_input_features, train_target_roots = data_loader.load_data(
        args.train_dataset_path, 
        args.train_size, 
        filter_real_discriminant=not args.no_filter_disc
    )
    logger.info(f"Train dataset size: {len(train_input_features)} samples")
    
    logger.info(f"Loading test dataset: {args.test_dataset_path}")
    test_input_features, test_target_roots = data_loader.load_data(
        args.test_dataset_path, 
        args.test_size, 
        filter_real_discriminant=not args.no_filter_disc
    )
    logger.info(f"Test dataset size: {len(test_input_features)} samples")

    # Use test set for validation during training
    train_dataset = DataProcessor(train_input_features, train_target_roots)
    test_dataset = DataProcessor(
        test_input_features, 
        test_target_roots, 
        mean=train_dataset.mean.clone(), 
        std=train_dataset.std.clone()
    )
    
    # Train model
    logger.info("\n=== Model Training Started ===")
    logger.info(f"Train dataset size: {len(train_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    logger.info(f"Model architecture: RootFindingMLP(hidden={args.hidden}, depth={args.depth})")
    logger.info(f"Training configuration: {args.epochs} epochs, lr={args.lr}, batch_size={args.batch_size}")
    
    # Measure training time
    training_start_time = time.time()
    trainer.train(
        train_dataset, 
        test_dataset, 
        epochs=args.epochs, 
        lr=args.lr, 
        batch_size=args.batch_size,
        lr_scheduler=args.lr_scheduler
    )
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    
    logger.info("\n=== Model Training Completed ===")
    logger.info(f"Training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    
    # Save model and scaler
    logger.info("\n=== Model Saving ===")
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, "model.pt"))
    torch.save(
        {"mean": train_dataset.mean, "std": train_dataset.std}, 
        os.path.join(args.save_path, "scaler.pt")
    )
    logger.info(f"Model and scaler saved to: {args.save_path}")
    print(f"Saved model and scaler to {args.save_path}")
    
    # Evaluate model on test set
    logger.info("\n=== Final Evaluation ===")
    model.to(args.device)
    
    # Measure inference time
    inference_start_time = time.time()
    with torch.no_grad():
        # Test set evaluation
        test_input_tensor = torch.as_tensor(test_input_features, dtype=torch.float32)
        preds_test = trainer.predict(test_input_tensor, train_dataset.mean, train_dataset.std)
        test_target_tensor = torch.as_tensor(test_target_roots, dtype=torch.float32)
        rmse_test = torch.sqrt(mse_sorted(preds_test, test_target_tensor)).item()
        # Compute accuracy for each threshold
        rmse_accuracies: dict[float, float] = {}
        for thr in args.threshold:
            rmse_accuracies[thr] = evaluate_accuracy_within_threshold(preds_test, test_target_tensor, thr)
        
        # Detailed evaluation results logging
        logger.info(f"Test set evaluation results:")
        logger.info(f"  - Test samples: {len(test_input_features)}")
        logger.info(f"  - RMSE (sorted): {rmse_test:.8f}")
        for thr in sorted(args.threshold, reverse=True):
            logger.info(f"  - RMSE accuracy (threshold={thr}): {rmse_accuracies[thr]:.2f}%")
        
        # Additional statistics
        mse_test = mse_sorted(preds_test, test_target_tensor).item()
        logger.info(f"  - MSE (sorted): {mse_test:.8f}")
        
        # Prediction and target statistics
        pred_sorted, _ = torch.sort(preds_test, dim=1)
        target_sorted, _ = torch.sort(test_target_tensor, dim=1)
        pred_mean = pred_sorted.mean(dim=0)
        target_mean = target_sorted.mean(dim=0)
        pred_std = pred_sorted.std(dim=0)
        target_std = target_sorted.std(dim=0)
        
        logger.info(f"  - Prediction mean: [{pred_mean[0]:.8f}, {pred_mean[1]:.8f}]")
        logger.info(f"  - Target mean: [{target_mean[0]:.8f}, {target_mean[1]:.8f}]")
        logger.info(f"  - Prediction std: [{pred_std[0]:.8f}, {pred_std[1]:.8f}]")
        logger.info(f"  - Target std: [{target_std[0]:.8f}, {target_std[1]:.8f}]")
        
        print(f"[TEST] RMSE(sorted)={rmse_test:.8f}")
        for thr in sorted(args.threshold, reverse=True):
            print(f"[TEST] RMSE_Accuracy(threshold={thr})={rmse_accuracies[thr]:.2f}%")
    
    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time
    logger.info(f"Inference time: {inference_duration:.4f} seconds")
    logger.info(f"Inference time per sample: {inference_duration/len(test_input_features)*1000:.4f} ms")
    
    logger.info("\n=== Experiment Completed ===")
    logger.info(f"Total training time: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    logger.info(f"Total inference time: {inference_duration:.4f} seconds")

if __name__ == "__main__":
    main()
