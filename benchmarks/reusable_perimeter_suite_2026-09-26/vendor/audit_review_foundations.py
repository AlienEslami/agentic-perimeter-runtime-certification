"""Independent checks of the transport balances and scalar certificate identities.

Run from the repository root. Uses the installed plant and archived matrices;
no simulation record or live-call result is overwritten.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import mfd_cubic_pilot as mfd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(20260905)
    A = mfd.complex_step_jacobian(
        lambda x: mfd.mfd_rhs(x, mfd.U_STAR), mfd.X_STAR
    )
    B = mfd.complex_step_jacobian(
        lambda u: mfd.mfd_rhs(mfd.X_STAR, u), mfd.U_STAR
    )
    C = np.eye(6)[:2]
    eig = np.linalg.eigvals(A)
    ctrl = [int(np.linalg.matrix_rank(np.column_stack((v * np.eye(6) - A, B)))) for v in eig]
    obs = [int(np.linalg.matrix_rank(np.vstack((v * np.eye(6) - A, C)))) for v in eig]
    assert ctrl.count(5) == obs.count(5) == 1
    assert all(rank == 6 or value.real < 0 for rank, value in zip(ctrl, eig))
    assert all(rank == 6 or value.real < 0 for rank, value in zip(obs, eig))
    stable_mode = float(eig[ctrl.index(5)].real)
    assert abs(stable_mode + 3.742815) < 1e-10

    hrow = np.array([1., 0., -1., 0., -3., 0.])
    assert np.max(np.abs(hrow @ B)) < 1e-12
    assert np.max(np.abs(hrow @ A - stable_mode * hrow)) < 1e-12
    balance_error = 0.
    class_error = 0.
    sensitivity_error = 0.
    for _ in range(1000):
        totals = rng.uniform(10, 33, 2)
        shares = rng.dirichlet([4, 3, 2], size=2) * totals[:, None]
        x = np.array([*totals, shares[0, 0], shares[1, 0], shares[0, 2], shares[1, 2]])
        u = rng.uniform(0.1, 0.9, 4)
        demand = mfd.DEMAND * rng.uniform(0.9, 1.1, 6)
        rhs = mfd.mfd_rhs(x, u, demand)
        outflow = mfd.transportation_rates(x, u)['completed_trips_units_per_hour']
        residual = rhs[0] + rhs[1] - (demand.sum() + (mfd.Q01 + mfd.Q02) * u[2] - outflow)
        balance_error = max(balance_error, abs(float(residual)))
        hdot = hrow @ mfd.mfd_rhs(x, u)
        expected = -u[0] * mfd.mfd_outflow(x[0]) / x[0] * (hrow @ x)
        class_error = max(class_error, abs(float(hdot - expected)))
        slope = 3*mfd.MFD_A*x[0]**2 + 2*mfd.MFD_B*x[0] + mfd.MFD_C
        analytic = x[2] * (x[0]*slope - mfd.mfd_outflow(x[0])) / x[0]**2
        numerical = mfd.class_flow(x[2], x[0] + 1e-25j).imag / 1e-25
        sensitivity_error = max(sensitivity_error, abs(float(analytic - numerical)))
    assert max(balance_error, class_error, sensitivity_error) < 1e-10

    # Integrate the transport balance with an independent cumulative-outflow
    # state, using a fixed admissible command and matched demand/initial state.
    residuals = []
    for offset in (0.0, 0.005):
        u = mfd.U_STAR.copy()
        u[2] += offset
        initial = np.r_[mfd.X_STAR + [0.01, -0.01, 0., 0., 0., 0.], 0.]
        def rhs(t, z):
            return np.r_[mfd.mfd_rhs(z[:6], u),
                         100*mfd.transportation_rates(z[:6], u)['completed_trips_units_per_hour']]
        run = solve_ivp(rhs, (0., 0.5), initial, method='DOP853', rtol=1e-11, atol=1e-12)
        assert run.success
        final = run.y[:, -1]
        predicted = 100*((mfd.DEMAND.sum() + 4*u[2])*0.5 + initial[:2].sum() - final[:2].sum())
        residuals.append(abs(float(final[-1] - predicted)))
    assert max(residuals) < 1e-7

    scalar_identity_error = 0.
    for _ in range(1000):
        nu = rng.uniform(1, 5)
        beta = rng.uniform(.01, .95) / nu
        c, kappa = rng.uniform(0, 2, 2)
        pre = c + beta*(kappa + (nu-1)*c)/(1-nu*beta)
        sharp = (kappa + nu*(1-beta)*c)/(1-nu*beta)
        scalar_identity_error = max(scalar_identity_error, abs(nu*pre + kappa - sharp))
        ell = nu*c + rng.uniform(.1, 2)
        r = rng.uniform(.1, 3)
        dwell = np.log(nu*(ell-c)/(ell-nu*c))/r
        successor = nu*(c + np.exp(-r*dwell)*(ell-c))
        assert abs(successor-ell) < 1e-11
    assert scalar_identity_error < 1e-10

    report = {
        'passed': True,
        'random_balance_checks': 1000,
        'network_balance_residual_units_per_hour': balance_error,
        'class_composition_residual_units_per_hour': class_error,
        'class_flow_jacobian_residual': sensitivity_error,
        'stable_uncontrollable_and_unobservable_eigenvalue_per_hour': stable_mode,
        'independent_integrated_balance_residual_vehicles': max(residuals),
        'sharp_robust_radius_identity_residual': scalar_identity_error,
    }
    target = ROOT / 'reproducibility/results/mfd_review_foundations_audit.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
