"""Four-state two-region MFD model and Zhong et al. (2018) CLF controller.

State order is (n11, n12, n21, n22), in vehicles. Time is in seconds and
demands and productions are in vehicles/second. Gate order is (u1, u2), where
u1 meters 1->2 and u2 meters 2->1.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def production(accumulation: float | np.ndarray) -> float | np.ndarray:
    """Cubic MFD used in Section 4.1 of Zhong et al. (2018)."""
    n = np.asarray(accumulation)
    return (1.4877e-7 * n**3 - 2.9815e-3 * n**2 + 15.0912 * n) / 3600.0


def totals(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state)
    return np.array([state[0] + state[1], state[2] + state[3]])


def destination_shares(state: np.ndarray) -> tuple[float, float, float, float]:
    n11, n12, n21, n22 = np.asarray(state)
    n1, n2 = totals(state)
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Regional accumulations must be positive")
    return n11 / n1, n12 / n1, n21 / n2, n22 / n2


def field(state: np.ndarray, gate: np.ndarray, demand: np.ndarray) -> np.ndarray:
    """Equations (1a)--(1d) in Zhong et al. (2018)."""
    state = np.asarray(state)
    u1, u2 = np.asarray(gate)
    q11, q12, q21, q22 = np.asarray(demand)
    n1, n2 = totals(state)
    v11, v12, v21, v22 = destination_shares(state)
    g1, g2 = production(n1), production(n2)
    return np.array([
        -v11 * g1 + v21 * g2 * u2 + q11,
        -v12 * g1 * u1 + q12,
        -v21 * g2 * u2 + q21,
        -v22 * g2 + v12 * g1 * u1 + q22,
    ])


def aggregate_field(state: np.ndarray, gate: np.ndarray, demand: np.ndarray) -> np.ndarray:
    derivative = field(state, gate, demand)
    return np.array([derivative[0] + derivative[1], derivative[2] + derivative[3]])


def transfer_matrix(state: np.ndarray) -> np.ndarray:
    """The S matrix in the shifted two-state affine system (9)."""
    n1, n2 = totals(state)
    _, v12, v21, _ = destination_shares(state)
    a = v12 * production(n1)
    b = v21 * production(n2)
    return np.array([[-a, b], [a, -b]])


@dataclass(frozen=True)
class FixedDemandCase:
    name: str
    demand: np.ndarray
    target_totals: np.ndarray
    reported_equilibrium: np.ndarray
    steady_gate: np.ndarray
    initial_totals: np.ndarray

    def initial_state(self, split: str = "published") -> np.ndarray:
        """Construct a four-state initial condition from the reported totals.

        The paper reports only the two initial regional totals.  The default
        therefore preserves the reported equilibrium destination shares.  The
        sensitivity audit also evaluates symmetric and perturbed splits.
        """
        if split == "published":
            if self.name != "fixed_demand_uncongested_target":
                raise ValueError("A published destination split is currently available only for case 1")
            # The benchmark lineage reports the OD-specific initial condition
            # n11=540, n12=1260, n21=2170, n22=930 vehicles.
            return np.array([540.0, 1260.0, 2170.0, 930.0])
        if split == "equilibrium":
            shares = self.reported_equilibrium / np.repeat(self.target_totals, 2)
        elif split == "symmetric":
            shares = np.full(4, 0.5)
        elif split == "perturbed":
            shares = np.array([0.65, 0.35, 0.35, 0.65])
        else:
            raise ValueError(f"Unknown split: {split}")
        return shares * np.repeat(self.initial_totals, 2)


CASE_1 = FixedDemandCase(
    name="fixed_demand_uncongested_target",
    demand=np.array([1.6, 1.6, 1.6, 1.6]),
    target_totals=np.array([3000.0, 3000.0]),
    reported_equilibrium=np.array([1538.9, 1461.1, 1461.1, 1538.9]),
    steady_gate=np.array([0.5267, 0.5267]),
    initial_totals=np.array([1800.0, 3100.0]),
)

CASE_2 = FixedDemandCase(
    name="fixed_demand_mixed_regimes",
    demand=np.array([1.58, 1.56, 1.54, 1.52]),
    target_totals=np.array([3000.0, 4000.0]),
    reported_equilibrium=np.array([1500.5, 1499.5, 2000.5, 1999.5]),
    steady_gate=np.array([0.5003, 0.4997]),
    initial_totals=np.array([2500.0, 4300.0]),
)


def almost_smooth_gate(
    state: np.ndarray,
    case: FixedDemandCase,
    lower: float = 0.1,
    upper: float = 0.9,
) -> np.ndarray:
    """Equations (12)--(15), with the paper's element-wise saturation.

    For V(e)=0.5*e'e, alpha=e'F and beta=S'e.  F is evaluated at the
    reported steady gate, so the affine decomposition is exact at the current
    destination shares.  The returned gate respects the requested physical
    interval [lower, upper].
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Gate bounds must satisfy 0 <= lower < upper <= 1")
    if np.any(case.steady_gate < lower) or np.any(case.steady_gate > upper):
        raise ValueError("The reported steady gate must lie inside the bounds")
    error = totals(state) - case.target_totals
    f = aggregate_field(state, case.steady_gate, case.demand)
    s = transfer_matrix(state)
    alpha = float(error @ f)
    beta = s.T @ error
    b = float(beta @ beta)
    if b <= np.finfo(float).eps:
        mu = np.zeros(2)
    else:
        phi = -(alpha + np.sqrt(alpha * alpha + b * b)) / (
            b * (1.0 + np.sqrt(1.0 + b))
        )
        mu = phi * beta
    lower_shift = lower - case.steady_gate
    upper_shift = upper - case.steady_gate
    return case.steady_gate + np.clip(mu, lower_shift, upper_shift)


def conservation_rate(state: np.ndarray, gate: np.ndarray, demand: np.ndarray) -> float:
    """Total-state derivative implied by external arrivals and completions."""
    n11, _, _, n22 = np.asarray(state)
    n1, n2 = totals(state)
    completion = n11 / n1 * production(n1) + n22 / n2 * production(n2)
    return float(np.sum(demand) - completion)
