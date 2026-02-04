# Common components shared by one_step_root_finding.py and multi_steps_root_finding.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Data Loading
# =========================


class DataLoader:
    """Parse dataset lines and build tensors/lists.

    Line format:
    "a_curr b_curr λ1_curr λ2_curr a_next b_next # λ1_next λ2_next"
    
    Where:
    - (a_curr, b_curr): current step coefficients
    - (λ1_curr, λ2_curr): current step roots (λ1 ≥ λ2)
    - (a_next, b_next): next step coefficients
    - (λ1_next, λ2_next): next step roots (λ1 ≥ λ2)
    """

    @staticmethod
    def _parse_dataset_line(line: str) -> tuple[list[float], list[float]]:
        if "#" not in line:
            raise ValueError("Missing '#' separator in line")
        left, right = line.split("#", 1)

        left_nums = [float(x) for x in left.strip().split()]
        if len(left_nums) != 6:
            raise ValueError(
                f"Expected 6 numbers on the left; got {len(left_nums)}: {left_nums}"
            )

        right_nums = [float(x) for x in right.strip().split()]
        if len(right_nums) != 2:
            raise ValueError(
                f"Expected 2 numbers on the right; got {len(right_nums)}: {right_nums}"
            )

        return left_nums, right_nums

    @staticmethod
    def load_data(path: str, dataset_size: int | None, filter_real_discriminant: bool | None = None) -> tuple[list[list[float]], list[list[float]]]:
        inputs: list[list[float]] = []
        targets: list[list[float]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if dataset_size is not None and len(inputs) >= dataset_size:
                    break
                line = line.strip()
                if not line or line.startswith("//") or line.startswith("#"):
                    continue
                inp, targ = DataLoader._parse_dataset_line(line)
                if filter_real_discriminant:
                    # keep only end polynomial with real roots: c^2 - 4 d > 0
                    c, d = inp[4], inp[5]
                    if c * c - 4.0 * d <= 0:
                        continue
                inputs.append(inp)
                targets.append(targ)
        if not inputs:
            raise RuntimeError("No valid samples loaded.")
        return inputs, targets


class DataProcessor(torch.utils.data.Dataset):
    def __init__(self, input_features, target_roots, mean=None, std=None):
        self.input_features = torch.as_tensor(input_features, dtype=torch.float32)
        self.target_roots = torch.as_tensor(target_roots, dtype=torch.float32)
        if mean is None or std is None:
            mean = self.input_features.mean(0, keepdim=True)
            std = self.input_features.std(0, unbiased=False, keepdim=True) + 1e-8
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.input_features.shape[0]

    def __getitem__(self, idx):
        features = (self.input_features[idx] - self.mean[0]) / self.std[0]
        return features, self.target_roots[idx]


# =========================
# Model
# =========================


class RootFindingMLP(nn.Module):
    def __init__(self, in_dim: int = 6, hidden: int = 128, depth: int = 4, out_dim: int = 2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(depth)
            ]
        )
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.inp(x))
        for blk in self.blocks:
            h = h + blk(h)
        return self.proj(h)


def mse_sorted(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate MSE in an order-invariant way by sorting both sides ASC along dim=1.

    Policy:
        - Representation: model outputs and internal states are handled in DESC (λ1 ≥ λ2).
        - Metrics/Loss: to be order-invariant, both pred and target are sorted ASC
          and then compared. ASC or DESC would be equivalent as long as both sides
          share the same direction; we choose ASC consistently for metrics here.
    """
    ps, _ = torch.sort(pred, dim=1)
    ts, _ = torch.sort(target, dim=1)
    return F.mse_loss(ps, ts, reduction="mean")


def evaluate_accuracy_within_threshold(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> float:
    """Order-invariant RMSE accuracy: percentage of samples with RMSE <= threshold.

    Both pred and target are sorted ascending along dim=1 before comparison.
    """
    ps, _ = torch.sort(pred, dim=1)
    ts, _ = torch.sort(target, dim=1)
    rmse_per = torch.sqrt(F.mse_loss(ps, ts, reduction="none").mean(dim=1))
    return (rmse_per <= threshold).float().mean().item() * 100.0


# =========================
# Model/scaler I/O helpers
# =========================


def infer_arch_from_state_dict(state_dict: dict) -> tuple[int, int]:
    if "inp.weight" not in state_dict:
        return 128, 4
    hidden = state_dict["inp.weight"].shape[0]
    depth = 0
    while f"blocks.{depth}.2.weight" in state_dict or f"blocks.{depth}.0.weight" in state_dict:
        depth += 1
    if depth == 0:
        keys = [k for k in state_dict.keys() if k.startswith("blocks.")]
        depth = max(int(k.split(".")[1]) for k in keys) + 1 if keys else 4
    return hidden, depth


def load_model_and_scaler(model_path: str, scaler_path: str, device: str):
    state = torch.load(model_path, map_location=device)
    hidden, depth = infer_arch_from_state_dict(state)
    model = RootFindingMLP(in_dim=6, hidden=hidden, depth=depth, out_dim=2)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    scaler = torch.load(scaler_path, map_location=device)
    mean = scaler["mean"]
    std = scaler["std"]
    return model, mean, std
