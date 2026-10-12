"""Per-instance direct action search used as an oracle baseline.

The oracle does not learn a policy. For one fixed target system it searches
the latent action box ``[-action_limit, action_limit]^dim`` directly and
returns the best action found. Because the search is finite, the result is an
estimate of (a lower bound on) the attainable improvement over the linear
path, not the true optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from homotopy_path_learning.config import Phase6Config, build_env, build_system_spec
from homotopy_path_learning.evaluation.datasets import fixed_evaluation_seeds


@dataclass(frozen=True)
class OracleSearchResult:
    """Result of :func:`search_best_action`.

    ``best_action`` has shape ``(dim,)`` and dtype ``float64``.
    ``costs`` has shape ``(n_evaluations,)`` and dtype ``float64`` and lists
    every objective value in evaluation order; ``costs[0]`` is the zero action.
    """

    best_action: npt.NDArray[np.float64]
    best_cost: float
    zero_cost: float
    costs: npt.NDArray[np.float64]

    @property
    def n_evaluations(self) -> int:
        """Number of objective evaluations performed."""

        return int(self.costs.size)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive, got {number}.")
    return number


def _objective_value(objective: Callable[[npt.NDArray[np.float64]], float], action: np.ndarray) -> float:
    value = float(objective(np.array(action, dtype=np.float64, copy=True)))
    if not np.isfinite(value):
        raise ValueError(f"objective returned a non-finite value {value!r}.")
    return value


def search_best_action(
    objective: Callable[[npt.NDArray[np.float64]], float],
    *,
    dim: int,
    action_limit: float,
    rng: np.random.Generator,
    budget: int,
    n_random: int,
    initial_sigma: float = 0.3,
    min_sigma: float = 1.0e-3,
) -> OracleSearchResult:
    """Minimize ``objective`` over the box ``[-action_limit, action_limit]^dim``.

    The search uses ``budget`` objective evaluations in total:

    1. the zero action (the linear path in the Bezier environment);
    2. ``n_random - 1`` uniform samples from the box;
    3. the remaining evaluations as a (1+1) evolution strategy started from
       the best point so far. Candidates are ``clip(best + sigma * z)`` with
       ``z ~ N(0, I)``. A candidate is accepted when its cost is not larger
       than the incumbent (so the search can move along plateaus of the
       integer-valued step counts). ``sigma`` follows the 1/5 success rule:
       it is multiplied by ``1.5`` on a strict improvement and by
       ``1.5 ** -0.25`` otherwise, and kept in
       ``[min_sigma, 2] * action_limit``.

    ``initial_sigma`` and ``min_sigma`` are relative to ``action_limit``.
    The objective receives a fresh ``float64`` array of shape ``(dim,)`` and
    must return a finite cost (smaller is better). It is assumed to be
    deterministic. Because the zero action is always evaluated first, the
    returned ``best_cost`` never exceeds ``zero_cost``. Only ``rng`` is used
    for sampling, so the result is reproducible for a fixed generator state.
    """

    if not callable(objective):
        raise TypeError("objective must be callable.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator.")
    dim_value = _positive_int("dim", dim)
    budget_value = _positive_int("budget", budget)
    n_random_value = _positive_int("n_random", n_random)
    if n_random_value > budget_value:
        raise ValueError(f"n_random={n_random_value} must not exceed budget={budget_value}.")
    limit = float(action_limit)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"action_limit must be positive and finite, got {action_limit!r}.")
    sigma0 = float(initial_sigma)
    sigma_floor = float(min_sigma)
    if not (np.isfinite(sigma0) and sigma0 > 0.0):
        raise ValueError(f"initial_sigma must be positive and finite, got {initial_sigma!r}.")
    if not (np.isfinite(sigma_floor) and 0.0 < sigma_floor <= sigma0):
        raise ValueError("min_sigma must satisfy 0 < min_sigma <= initial_sigma.")

    costs: list[float] = []

    zero = np.zeros(dim_value, dtype=np.float64)
    zero_cost = _objective_value(objective, zero)
    costs.append(zero_cost)
    best_action = zero
    best_cost = zero_cost

    for _ in range(n_random_value - 1):
        candidate = rng.uniform(-limit, limit, size=dim_value)
        cost = _objective_value(objective, candidate)
        costs.append(cost)
        if cost < best_cost:
            best_action, best_cost = candidate, cost

    sigma = sigma0 * limit
    sigma_min = sigma_floor * limit
    sigma_max = 2.0 * limit
    incumbent, incumbent_cost = best_action, best_cost
    for _ in range(budget_value - n_random_value):
        candidate = np.clip(incumbent + sigma * rng.standard_normal(dim_value), -limit, limit)
        cost = _objective_value(objective, candidate)
        costs.append(cost)
        if cost < incumbent_cost:
            sigma = min(sigma * 1.5, sigma_max)
        else:
            sigma = max(sigma * 1.5**-0.25, sigma_min)
        if cost <= incumbent_cost:
            incumbent, incumbent_cost = candidate, cost
        if cost < best_cost:
            best_action, best_cost = candidate, cost

    return OracleSearchResult(
        best_action=np.array(best_action, dtype=np.float64, copy=True),
        best_cost=float(best_cost),
        zero_cost=float(zero_cost),
        costs=np.asarray(costs, dtype=np.float64),
    )


ORACLE_FIELDS = [
    "instance_id",
    "target_seed",
    "latent_dim",
    "action_dim",
    "linear_cost",
    "linear_success",
    "zero_action_cost",
    "oracle_cost",
    "oracle_success",
    "oracle_n_failed",
    "oracle_accepted_steps",
    "oracle_rejected_steps",
    "delta_j_percent",
    "n_evaluations",
    "best_evaluation_index",
    "best_action_l2_norm",
    "best_action_max_abs",
    "elapsed_seconds",
]


ORACLE_SUMMARY_FIELDS = [
    "method",
    "n_instances",
    "problem_success_rate",
    "mean_cost",
    "median_cost",
    "delta_j_percent",
    "positive_rate",
    "captured_fraction",
]


@dataclass(frozen=True)
class OracleRun:
    """Per-instance oracle results for one config.

    ``rows`` follow :data:`ORACLE_FIELDS`. ``best_actions`` has shape
    ``(num_instances, action_dim)`` and ``cost_traces`` has shape
    ``(num_instances, budget)``; both are ``float64`` and ordered like
    ``rows`` (``target_seed = evaluation.seed + instance_id``). ``basis`` is the
    latent basis ``U`` with shape ``(2 * n_free_coeffs, latent_dim)``.
    """

    rows: list[dict[str, Any]]
    best_actions: npt.NDArray[np.float64]
    cost_traces: npt.NDArray[np.float64]
    basis: npt.NDArray[np.float64]


def resolve_latent_dim(config: Phase6Config, latent_dim: int | str | None) -> int:
    """Return the latent dimension for an oracle search.

    ``None`` keeps ``config.path.latent_dim``. ``"full"`` selects
    ``2 * n_free_coeffs``, i.e. a square orthonormal basis that spans every
    real direction of the free coefficients. An integer must lie in
    ``[1, 2 * n_free_coeffs]``.
    """

    real_dim = 2 * int(np.count_nonzero(build_system_spec(config).free_coefficient_mask))
    if latent_dim is None:
        return int(config.path.latent_dim)
    if isinstance(latent_dim, str):
        if latent_dim == "full":
            return real_dim
        try:
            latent_dim = int(latent_dim)
        except ValueError as exc:
            raise ValueError(f"latent_dim must be an integer or 'full', got {latent_dim!r}.") from exc
    value = _positive_int("latent_dim", latent_dim)
    if value > real_dim:
        raise ValueError(f"latent_dim={value} exceeds 2 * n_free_coeffs={real_dim}.")
    return value


def with_latent_dim(config: Phase6Config, latent_dim: int) -> Phase6Config:
    """Return a copy of ``config`` with ``path.latent_dim`` replaced."""

    return replace(config, path=replace(config.path, latent_dim=_positive_int("latent_dim", latent_dim)))


def run_oracle(
    config: Phase6Config,
    *,
    budget: int,
    n_random: int,
    search_seed: int,
    num_instances: int | None = None,
    backend: object | None = None,
) -> OracleRun:
    """Search the best latent action separately for each evaluation instance.

    Instances are the fixed evaluation targets ``evaluation.seed + i`` used by
    the PPO evaluation, so oracle and policy costs can be compared per
    instance. The objective is the mean Bezier tracking cost
    ``J = N_acc + rho * N_rej`` (with the failure penalty) against the cached
    linear baseline of the same target. The search generator for instance
    ``i`` is ``default_rng([search_seed, target_seed])``. ``backend`` is passed
    to :func:`build_env` and is intended for tests.
    """

    seeds = fixed_evaluation_seeds(
        seed=config.evaluation.seed,
        num_instances=config.evaluation.num_instances if num_instances is None else num_instances,
    )
    env = build_env(config, backend=backend)
    rows: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    traces: list[np.ndarray] = []
    try:
        action_dim = int(env.action_space.shape[0])
        basis = np.array(env.parameterization.basis, dtype=np.float64, copy=True)
        for instance_id, target_seed in enumerate(seeds):
            start = time.time()
            env.reset(seed=target_seed)
            context = env.current_episode_context()

            def objective(action: npt.NDArray[np.float64]) -> float:
                return float(env.evaluate_action_against_context(context, action).tracking_cost.mean)

            result = search_best_action(
                objective,
                dim=action_dim,
                action_limit=config.path.action_limit,
                rng=np.random.default_rng([int(search_seed), int(target_seed)]),
                budget=budget,
                n_random=n_random,
            )
            best = env.evaluate_action_against_context(context, result.best_action)
            if float(best.tracking_cost.mean) != result.best_cost:
                raise RuntimeError(
                    "re-evaluating the best action gave a different cost; "
                    "the tracking objective is not deterministic."
                )
            linear_cost = float(context.linear_cost.mean)
            rows.append(
                {
                    "instance_id": instance_id,
                    "target_seed": int(target_seed),
                    "latent_dim": int(config.path.latent_dim),
                    "action_dim": action_dim,
                    "linear_cost": linear_cost,
                    "linear_success": int(bool(context.linear_result.success)),
                    "zero_action_cost": result.zero_cost,
                    "oracle_cost": result.best_cost,
                    "oracle_success": int(bool(best.tracking_result.success)),
                    "oracle_n_failed": int(best.tracking_result.n_failed),
                    "oracle_accepted_steps": int(best.tracking_result.accepted_steps),
                    "oracle_rejected_steps": int(best.tracking_result.rejected_steps),
                    "delta_j_percent": _delta_j_percent(result.best_cost, linear_cost),
                    "n_evaluations": result.n_evaluations,
                    "best_evaluation_index": int(np.argmin(result.costs)),
                    "best_action_l2_norm": float(np.linalg.norm(result.best_action)),
                    "best_action_max_abs": float(np.max(np.abs(result.best_action))),
                    "elapsed_seconds": time.time() - start,
                }
            )
            actions.append(result.best_action)
            traces.append(result.costs)
    finally:
        env.close()
    return OracleRun(
        rows=rows,
        best_actions=np.stack(actions).astype(np.float64),
        cost_traces=np.stack(traces).astype(np.float64),
        basis=basis,
    )


def _delta_j_percent(cost: float, linear_cost: float) -> float:
    if linear_cost <= 0.0:
        raise ValueError(f"linear cost must be positive, got {linear_cost!r}.")
    return (float(cost) - float(linear_cost)) / float(linear_cost) * 100.0


def learned_costs_from_evaluation(
    evaluation_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[int]]:
    """Return ``LearnedBezier`` mean costs and success flags aligned with ``oracle_rows``.

    ``evaluation_rows`` are rows of a PPO ``evaluation/evaluation.csv``. Every
    oracle ``target_seed`` must appear, and the ``Linear`` cost recorded there
    must equal the oracle's linear cost; otherwise the two runs did not use
    the same instances or tracker settings and a ``ValueError`` is raised.
    """

    learned: dict[int, float] = {}
    learned_success: dict[int, int] = {}
    linear: dict[int, float] = {}
    for row in evaluation_rows:
        seed = int(row["target_seed"])
        if row["method"] == "LearnedBezier":
            learned[seed] = float(row["mean_cost"])
            learned_success[seed] = int(row["problem_success"])
        elif row["method"] == "Linear":
            linear[seed] = float(row["mean_cost"])
    costs: list[float] = []
    successes: list[int] = []
    for row in oracle_rows:
        seed = int(row["target_seed"])
        if seed not in learned or seed not in linear:
            raise ValueError(f"evaluation rows do not contain target_seed={seed}.")
        if not np.isclose(linear[seed], float(row["linear_cost"]), rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"Linear cost mismatch for target_seed={seed}: "
                f"evaluation={linear[seed]}, oracle={row['linear_cost']}."
            )
        costs.append(learned[seed])
        successes.append(learned_success[seed])
    return costs, successes


def summarize_oracle(
    oracle_rows: Sequence[Mapping[str, Any]],
    learned: Mapping[str, tuple[Sequence[float], Sequence[int]]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate oracle rows and optional aligned policy costs.

    ``delta_j_percent`` is computed from instance means,
    ``(mean J_method - mean J_lin) / mean J_lin * 100``; ``positive_rate`` is
    the fraction of instances with ``J_method < J_lin``. ``captured_fraction``
    is ``(mean J_lin - mean J_method) / (mean J_lin - mean J_oracle)``, the share
    of the oracle's improvement obtained by a method (empty when the oracle
    does not improve on average). ``learned`` maps a label (for example a PPO
    run id) to ``(costs, success_flags)`` aligned with ``oracle_rows``, as
    returned by :func:`learned_costs_from_evaluation`.
    """

    if not oracle_rows:
        raise ValueError("oracle_rows must not be empty.")
    linear = np.array([float(row["linear_cost"]) for row in oracle_rows], dtype=np.float64)
    oracle = np.array([float(row["oracle_cost"]) for row in oracle_rows], dtype=np.float64)
    linear_mean = float(np.mean(linear))
    headroom = linear_mean - float(np.mean(oracle))

    def summary(method: str, costs: np.ndarray, success_rate: float) -> dict[str, Any]:
        mean_cost = float(np.mean(costs))
        return {
            "method": method,
            "n_instances": int(costs.size),
            "problem_success_rate": float(success_rate),
            "mean_cost": mean_cost,
            "median_cost": float(np.median(costs)),
            "delta_j_percent": _delta_j_percent(mean_cost, linear_mean),
            "positive_rate": float(np.mean(costs < linear)),
            "captured_fraction": (linear_mean - mean_cost) / headroom if headroom > 0.0 else "",
        }

    summaries = [
        summary("Linear", linear, float(np.mean([float(row["linear_success"]) for row in oracle_rows]))),
        summary("Oracle", oracle, float(np.mean([float(row["oracle_success"]) for row in oracle_rows]))),
    ]
    for label, (values, success_flags) in (learned or {}).items():
        costs = np.asarray(values, dtype=np.float64)
        flags = np.asarray(success_flags, dtype=np.float64)
        if costs.shape != linear.shape or flags.shape != linear.shape:
            raise ValueError(
                f"learned costs and success flags for {label!r} must have shape {linear.shape}."
            )
        summaries.append(summary(f"LearnedBezier[{label}]", costs, float(np.mean(flags))))
    return summaries


__all__ = [
    "ORACLE_FIELDS",
    "ORACLE_SUMMARY_FIELDS",
    "OracleRun",
    "OracleSearchResult",
    "learned_costs_from_evaluation",
    "resolve_latent_dim",
    "run_oracle",
    "search_best_action",
    "summarize_oracle",
    "with_latent_dim",
]
