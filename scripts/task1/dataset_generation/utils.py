import warnings
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D",
    category=UserWarning,
    module="matplotlib.projections"
)
import os
import json
from dataclasses import asdict
from typing import TYPE_CHECKING
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .dataset_generator import OneStepPair, PathRecord


class DatasetWriter:
    """
    Utility to persist Task 1.1 one-step pairs (`OneStepPair`) and homotopy paths (`PathRecord`).

    Features:
    - Save OneStepPair objects in .txt and .jsonl formats
    - Save PathRecord objects (full paths or endpoints only) in .txt and .jsonl formats
    - Extract start/end pairs from full datasets
    - Human-readable .txt format for inspection
    - Machine-friendly .jsonl format for processing

    Example usage:
        writer = DatasetWriter("output_dir")
        # Save all pairs
        writer.save_pairs(pairs, "train", "dataset")
        # Save only endpoints
        writer.save_hc_endpoints(hc_paths, "endpoints")
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_pairs_txt(self, pairs: list["OneStepPair"], filename: str = "hc_pairs.txt") -> str:
        """
        Save OneStepPair objects in human-readable .txt format.
        
        Format per line:
        "a_curr b_curr λ1_curr λ2_curr a_next b_next # λ1_next λ2_next"
        
        Where:
        - (a_curr, b_curr): current step coefficients
        - (λ1_curr, λ2_curr): current step roots (λ1 ≥ λ2)
        - (a_next, b_next): next step coefficients  
        - (λ1_next, λ2_next): next step roots (λ1 ≥ λ2)
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                # Direct attribute access - more readable and efficient
                f.write(
                    f"{pair.coeffs_curr[0]:.15f} {pair.coeffs_curr[1]:.15f} "
                    f"{pair.roots_curr[0]:.15f} {pair.roots_curr[1]:.15f} "
                    f"{pair.coeffs_next[0]:.15f} {pair.coeffs_next[1]:.15f} "
                    f"# {pair.roots_next[0]:.15f} {pair.roots_next[1]:.15f}\n"
                )
        return path

    def save_pairs_jsonl(self, pairs: list["OneStepPair"], filename: str = "hc_pairs.jsonl") -> str:
        """
        Save OneStepPair objects in JSON Lines format.
        
        Each line contains a JSON object with keys:
        - coeffs_curr: [a_curr, b_curr] - current step coefficients
        - roots_curr: [λ1_curr, λ2_curr] - current step roots (λ1 ≥ λ2)
        - coeffs_next: [a_next, b_next] - next step coefficients
        - roots_next: [λ1_next, λ2_next] - next step roots (λ1 ≥ λ2)
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for i, pair in enumerate(pairs):
                # Use asdict() for cleaner, more maintainable code
                obj = asdict(pair)
                # Convert tuples to lists for JSON serialization
                for key in obj:
                    obj[key] = list(obj[key])
                
                f.write(json.dumps(obj, ensure_ascii=False))
                if i < len(pairs) - 1:
                    f.write("\n")
        return path

    def save_pairs(self, pairs: list["OneStepPair"], base_filename: str = "hc_pairs") -> tuple[str, str]:
        """
        Save the given `pairs` in both .txt and .jsonl formats.
        """
        txt_path = self.save_pairs_txt(pairs, filename=f"{base_filename}.txt")
        jsonl_path = self.save_pairs_jsonl(pairs, filename=f"{base_filename}.jsonl")
        return txt_path, jsonl_path

    def save_hc_paths(self, hc_paths: list["PathRecord"], filename: str = "hc_paths.jsonl") -> str:
        """
        Save PathRecord objects as JSON Lines format.
        
        Each line contains a JSON object with keys:
        - coeffs_along_path: [[a_0, b_0], [a_1, b_1], ...] - coefficients along homotopy path
        - roots_along_path: [[λ1_0, λ2_0], [λ1_1, λ2_1], ...] - roots along homotopy path
        - coeffs_start: [a_start, b_start] - starting coefficients
        - coeffs_end: [a_end, b_end] - ending coefficients  
        - roots_start: [λ1_start, λ2_start] - starting roots (λ1 ≥ λ2)
        - roots_end: [λ1_end, λ2_end] - ending roots (λ1 ≥ λ2)
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for i, path_record in enumerate(hc_paths):
                # Use asdict() for cleaner code, then convert numpy arrays and tuples
                obj = asdict(path_record)
                obj["coeffs_along_path"] = path_record.coeffs_along_path.tolist()
                obj["roots_along_path"] = path_record.roots_along_path.tolist()
                # Convert tuples to lists for JSON serialization
                for key in ["coeffs_start", "coeffs_end", "roots_start", "roots_end"]:
                    obj[key] = list(obj[key])
                
                f.write(json.dumps(obj, ensure_ascii=False))
                if i < len(hc_paths) - 1:
                    f.write("\n")
        return path

    def save_hc_endpoints_txt(self, hc_paths: list["PathRecord"], filename: str = "hc_endpoints.txt") -> str:
        """
        Save only endpoints per homotopy path in human-readable .txt format.

        Format per line:
        "a_start b_start λ1_start λ2_start a_end b_end # λ1_end λ2_end"
        
        Where:
        - (a_start, b_start): starting coefficients
        - (λ1_start, λ2_start): starting roots (λ1 ≥ λ2)
        - (a_end, b_end): ending coefficients
        - (λ1_end, λ2_end): ending roots (λ1 ≥ λ2)
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for path_record in hc_paths:
                # Direct attribute access - cleaner and more efficient
                f.write(
                    f"{path_record.coeffs_start[0]:.15f} {path_record.coeffs_start[1]:.15f} "
                    f"{path_record.roots_start[0]:.15f} {path_record.roots_start[1]:.15f} "
                    f"{path_record.coeffs_end[0]:.15f} {path_record.coeffs_end[1]:.15f} "
                    f"# {path_record.roots_end[0]:.15f} {path_record.roots_end[1]:.15f}\n"
                )
        return path

    def save_hc_endpoints_jsonl(self, hc_paths: list["PathRecord"], filename: str = "hc_endpoints.jsonl") -> str:
        """
        Save only endpoints per homotopy path in JSON Lines format.
        
        Each line contains a JSON object with keys:
        - coeffs_start: [a_start, b_start] - starting coefficients
        - roots_start: [λ1_start, λ2_start] - starting roots (λ1 ≥ λ2)
        - coeffs_end: [a_end, b_end] - ending coefficients
        - roots_end: [λ1_end, λ2_end] - ending roots (λ1 ≥ λ2)
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for i, path_record in enumerate(hc_paths):
                # Create a subset dict with only endpoint fields
                obj = {
                    "coeffs_start": list(path_record.coeffs_start),
                    "roots_start": list(path_record.roots_start),
                    "coeffs_end": list(path_record.coeffs_end),
                    "roots_end": list(path_record.roots_end),
                }
                f.write(json.dumps(obj, ensure_ascii=False))
                if i < len(hc_paths) - 1:
                    f.write("\n")
        return path

    def save_hc_endpoints(self, hc_paths: list["PathRecord"], base_filename: str = "hc_endpoints") -> tuple[str, str]:
        """
        Save endpoints per path to both .txt and .jsonl. Returns (txt_path, jsonl_path).
        """
        txt_path = self.save_hc_endpoints_txt(hc_paths, filename=f"{base_filename}.txt")
        jsonl_path = self.save_hc_endpoints_jsonl(hc_paths, filename=f"{base_filename}.jsonl")
        return txt_path, jsonl_path


class DatasetVisualizer:
    """
    Visualization utilities for one-step pairs and homotopy paths.

    What this class provides
    ------------------------
    - Start vs end root distributions (side-by-side scatter plots)
    - Start vs end coefficient distributions (side-by-side scatter plots)
    - All-steps distributions for roots and coefficients along paths
    - Optional discriminant curves for coefficients: D=0 (b = a^2/4) and
      D=τ (b = a^2/4 - τ/4)

    Parameters
    ----------
    max_coeff_abs:
        If provided, use this symmetric bound B for both axes in coefficient
        plots and to generate discriminant curves over a ∈ [-B, B].
    discriminant_margin:
        If provided together with `max_coeff_abs`, draw the offset discriminant
        curve D=τ as b = a^2/4 - τ/4.

    Notes
    -----
    - The class configures matplotlib to use mathtext (no external TeX).
    - Public methods return the saved figure path, or an empty string if the
      given data is empty.
    """

    def __init__(self, max_coeff_abs: float | None = None, discriminant_margin: float | None = None) -> None:
        # Configure matplotlib for TeX-like math rendering (mathtext)
        plt.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.serif": ["CMU Serif", "DejaVu Serif"],
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        })
        self.max_coeff_abs = max_coeff_abs
        self.discriminant_margin = discriminant_margin

        self.suptitle_fontsize = 20
        self.title_fontsize = 20
        self.label_fontsize = 18
        self.legend_fontsize = 18

    # ------------------------------------------------------------------
    # Helpers (data detection, filesystem, and small plot primitives)
    # ------------------------------------------------------------------

    def _ensure_output_dir(self, output_dir: str) -> None:
        """Create the output directory if it does not already exist."""
        os.makedirs(output_dir, exist_ok=True)

    def _draw_axes_crosshair(self, ax) -> None:
        """Draw thin axis crosshair (x=0 and y=0 lines)."""
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=0.8)
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.8, linewidth=0.8)

    def _compute_y_eq_x_segment(self, x_vals: list[float], y_vals: list[float]) -> tuple[float, float] | None:
        """Return [mn, mx] that spans y=x across the data range.

        Returns None if inputs are empty.
        """
        if not x_vals or not y_vals:
            return None
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        return min(x_min, y_min), max(x_max, y_max)

    def _compute_discriminant_curves(
        self,
        a_vals: list[float] | None,
        num_points: int = 400,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Compute discriminant curves for coefficient plots.

        If `max_coeff_abs` is set, generate a over [-B, B]. Otherwise, if
        `a_vals` is provided, generate a over [min(a_vals)-1, max(a_vals)+1].
        Returns (a_curve, b_D0, b_Dtau). Each element can be None if not
        applicable.
        """
        B = self.max_coeff_abs
        tau = self.discriminant_margin
        a_curve: np.ndarray | None = None
        b_D0: np.ndarray | None = None
        b_Dtau: np.ndarray | None = None
        if B is not None:
            a_curve = np.linspace(-B, B, num_points)
            b_D0 = 0.25 * (a_curve ** 2)
            if tau is not None:
                b_Dtau = 0.25 * (a_curve ** 2) - (tau / 4.0)
            return a_curve, b_D0, b_Dtau
        if a_vals:
            a_min, a_max = min(a_vals), max(a_vals)
            a_curve = np.linspace(a_min - 1.0, a_max + 1.0, max(50, num_points // 2))
            b_D0 = 0.25 * (a_curve ** 2)
            # When B is None, only D=0 is well-defined without tau scaling
            if tau is not None:
                b_Dtau = 0.25 * (a_curve ** 2) - (tau / 4.0)
        return a_curve, b_D0, b_Dtau

    def _extract_endpoints_arrays(
        self,
        data: list["PathRecord"],
    ) -> tuple[
        list[float], list[float], list[float], list[float],
        list[tuple[float, float]], list[tuple[float, float]],
    ]:
        """Extract start/end arrays for roots and coefficients from PathRecord data."""
        start_roots_x: list[float] = []
        start_roots_y: list[float] = []
        end_roots_x: list[float] = []
        end_roots_y: list[float] = []
        start_coeffs: list[tuple[float, float]] = []
        end_coeffs: list[tuple[float, float]] = []
        
        for path_record in data:
            start_roots_x.append(float(path_record.roots_start[0]))
            start_roots_y.append(float(path_record.roots_start[1]))
            end_roots_x.append(float(path_record.roots_end[0]))
            end_roots_y.append(float(path_record.roots_end[1]))
            start_coeffs.append((float(path_record.coeffs_start[0]), float(path_record.coeffs_start[1])))
            end_coeffs.append((float(path_record.coeffs_end[0]), float(path_record.coeffs_end[1])))
        
        return start_roots_x, start_roots_y, end_roots_x, end_roots_y, start_coeffs, end_coeffs

    def plot_roots_endpoints(self, data: list["PathRecord"], output_dir: str, filename: str) -> str:
        """Plot start vs end root distributions.

        Parameters
        ----------
        data:
            List of PathRecord objects.
        output_dir:
            Directory where the figure will be saved.
        filename:
            Output filename (e.g., "hc_roots_endpoints.png").

        Returns
        -------
        str
            Absolute or relative path to the saved figure, or empty string if
            there is no data to plot.
        """
        (start_x, start_y, end_x, end_y, _, _) = self._extract_endpoints_arrays(data)
        if not start_x and not end_x:
            return ""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("Distribution of Roots $(\lambda_1, \lambda_2)$", fontsize=self.suptitle_fontsize)
        if start_x:
            ax1.set_aspect("equal", adjustable="box")
            ax1.scatter(start_x, start_y, c="blue", alpha=0.6, s=10, marker="o", label=r"Roots $(\lambda_1^Q,\lambda_2^Q)$")
            self._draw_axes_crosshair(ax1)
            seg = self._compute_y_eq_x_segment(start_x, start_y)
            if seg is not None:
                mn, mx = seg
                ax1.plot([mn, mx], [mn, mx], "k--", alpha=0.8, linewidth=0.8, label=r"$\lambda_1=\lambda_2$")
            ax1.set_title("$Q(x)=x^2 + a_Q x + b_Q$", fontsize=self.title_fontsize)
            ax1.set_xlabel(r"Root $\lambda_1^Q$", fontsize=self.label_fontsize)
            ax1.set_ylabel(r"Root $\lambda_2^Q$", fontsize=self.label_fontsize)
            ax1.legend(fontsize=self.legend_fontsize, loc="upper right")
            ax1.grid(True, alpha=0.3)
        if end_x:
            ax2.set_aspect("equal", adjustable="box")
            ax2.scatter(end_x, end_y, c="red", alpha=0.6, s=10, marker="x", label=r"Roots $(\lambda_1^P,\lambda_2^P)$")
            self._draw_axes_crosshair(ax2)
            seg = self._compute_y_eq_x_segment(end_x, end_y)
            if seg is not None:
                mn, mx = seg
                ax2.plot([mn, mx], [mn, mx], "k--", alpha=0.8, linewidth=0.8, label=r"$\lambda_1=\lambda_2$")
            ax2.set_title("$P(x)=x^2 + a_P x + b_P$", fontsize=self.title_fontsize)
            ax2.set_xlabel(r"Root $\lambda_1^P$", fontsize=self.label_fontsize)
            ax2.set_ylabel(r"Root $\lambda_2^P$", fontsize=self.label_fontsize)
            ax2.legend(fontsize=self.legend_fontsize, loc="upper right")
            ax2.grid(True, alpha=0.3)
        self._ensure_output_dir(output_dir)
        path = os.path.join(output_dir, filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def plot_coeffs_endpoints(self, data: list["PathRecord"], output_dir: str, filename: str, show_discriminant: bool = True) -> str:
        """Plot start vs end coefficient distributions with optional discriminant curves.

        Parameters
        ----------
        data:
            List of PathRecord objects.
        output_dir:
            Directory where the figure will be saved.
        filename:
            Output filename (e.g., "hc_coefficients_endpoints.png").
        show_discriminant:
            If True, draw discriminant curves D=0 and optionally D=τ.

        Returns
        -------
        str
            Absolute or relative path to the saved figure, or empty string if
            there is no data to plot.
        """
        (_, _, _, _, start_coeffs, end_coeffs) = self._extract_endpoints_arrays(data)
        if not start_coeffs and not end_coeffs:
            return ""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle("Distribution of Coefficients $(a, b)$", fontsize=self.suptitle_fontsize)

        def scatter_coeffs(ax, coeff_pairs, color, marker, label, discr_lw: float) -> None:
            if not coeff_pairs:
                return
            a_vals, b_vals = zip(*coeff_pairs)
            ax.scatter(a_vals, b_vals, c=color, alpha=0.6, s=10, marker=marker, label=label)
            self._draw_axes_crosshair(ax)
            if show_discriminant:
                a_curve, b_D0, b_Dtau = self._compute_discriminant_curves(list(a_vals))
                if a_curve is not None and b_D0 is not None:
                    ax.plot(a_curve, b_D0, "k-", alpha=0.9, linewidth=discr_lw,
                            label=r"$D=0:\ b=frac{1}{4}a^2$")
                if b_Dtau is not None:
                    ax.plot(a_curve, b_Dtau, color="red", linestyle="-", alpha=0.9, linewidth=discr_lw,
                            label=r"$D=\tau:\ b=\frac{1}{4}a^2-\frac{\tau}{4}$")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=self.legend_fontsize, loc="upper right")
            if self.max_coeff_abs is not None:
                B = self.max_coeff_abs
                ax.set_xlim(-B, B)
                ax.set_ylim(-B, B)

        if start_coeffs:
            scatter_coeffs(ax1, start_coeffs, "blue", "o", r"Coefficients $(a_Q, b_Q)$", 1.0)
            ax1.set_title("$Q(x)=x^2 + a_Q x + b_Q$", fontsize=self.title_fontsize)
            ax1.set_xlabel(r"Coefficient $a_Q$", fontsize=self.label_fontsize)
            ax1.set_ylabel(r"Coefficient $b_Q$", fontsize=self.label_fontsize)
        if end_coeffs:
            scatter_coeffs(ax2, end_coeffs, "red", "x", r"Coefficients $(a_P, b_P)$", 1.0)
            ax2.set_title("$P(x)=x^2 + a_P x + b_P$", fontsize=self.title_fontsize)
            ax2.set_xlabel(r"Coefficient $a_P$", fontsize=self.label_fontsize)
            ax2.set_ylabel(r"Coefficient $b_P$", fontsize=self.label_fontsize)

        self._ensure_output_dir(output_dir)
        path = os.path.join(output_dir, filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def plot_roots_paths(self, hc_paths: list["PathRecord"], output_dir: str, filename: str) -> str:
        """Plot all (λ1, λ2) points along homotopy paths in a single scatter plot.

        Returns the saved figure path, or an empty string if `hc_paths` is empty.
        """
        if not hc_paths:
            return ""
        # Flatten all root pairs across all steps and all paths
        x_vals: list[float] = []
        y_vals: list[float] = []
        for pr in hc_paths:
            roots = pr.roots_along_path  # shape (num_steps+1, 2)
            x_vals.extend(roots[:, 0].astype(float).tolist())
            y_vals.extend(roots[:, 1].astype(float).tolist())
        if not x_vals:
            return ""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle("Roots along homotopy paths $(\lambda_1, \lambda_2)$", fontsize=self.suptitle_fontsize)
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(x_vals, y_vals, c="purple", alpha=0.5, s=10, marker=".", label=r"All steps")
        self._draw_axes_crosshair(ax)
        seg = self._compute_y_eq_x_segment(x_vals, y_vals)
        if seg is not None:
            mn, mx = seg
            ax.plot([mn, mx], [mn, mx], "k--", alpha=0.8, linewidth=0.8, label=r"$\lambda_1=\lambda_2$")
        ax.set_title("Homotopy paths (roots)", fontsize=self.title_fontsize)
        ax.set_xlabel(r"Root $\lambda_1$", fontsize=self.label_fontsize)
        ax.set_ylabel(r"Root $\lambda_2$", fontsize=self.label_fontsize)
        ax.legend(fontsize=self.legend_fontsize, loc="upper right")
        ax.grid(True, alpha=0.3)
        self._ensure_output_dir(output_dir)
        path = os.path.join(output_dir, filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def plot_coeffs_paths(self, hc_paths: list["PathRecord"], output_dir: str, filename: str, show_discriminant: bool = True) -> str:
        """Plot all (a, b) points along homotopy paths in a single scatter plot.

        Returns the saved figure path, or an empty string if `hc_paths` is empty.
        """
        if not hc_paths:
            return ""
        a_vals: list[float] = []
        b_vals: list[float] = []
        for pr in hc_paths:
            coeffs = pr.coeffs_along_path  # shape (num_steps+1, 2)
            a_vals.extend(coeffs[:, 0].astype(float).tolist())
            b_vals.extend(coeffs[:, 1].astype(float).tolist())
        if not a_vals:
            return ""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.suptitle("Coefficients along homotopy paths $(a, b)$", fontsize=self.suptitle_fontsize)
        ax.scatter(a_vals, b_vals, c="teal", alpha=0.5, s=10, marker=".", label="All steps")
        self._draw_axes_crosshair(ax)
        # Discriminant curves if requested
        if show_discriminant:
            a_curve, b_D0, b_Dtau = self._compute_discriminant_curves(a_vals)
            if a_curve is not None and b_D0 is not None:
                ax.plot(a_curve, b_D0, "k-", alpha=0.9, linewidth=1,
                        label=r"$D=0:\ b=\frac{1}{4}a^2$")
            if b_Dtau is not None:
                ax.plot(a_curve, b_Dtau, color="red", linestyle="-", alpha=0.9, linewidth=1,
                        label=r"$D=\tau:\ b=\frac{1}{4}a^2-\frac{\tau}{4}$")
        if self.max_coeff_abs is not None:
            B = self.max_coeff_abs
            ax.set_xlim(-B, B)
            ax.set_ylim(-B, B)
        ax.set_title("Homotopy paths (coefficients)", fontsize=self.title_fontsize)
        ax.set_xlabel(r"Coefficient $a$", fontsize=self.label_fontsize)
        ax.set_ylabel(r"Coefficient $b$", fontsize=self.label_fontsize)
        ax.legend(fontsize=self.legend_fontsize, loc="upper right")
        ax.grid(True, alpha=0.25)
        self._ensure_output_dir(output_dir)
        path = os.path.join(output_dir, filename)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path


