# Stronger finite-scenario comparator: additional sweep

Added after the first development cases showed that the five-scenario planner's checks left mixed corners uncovered. Preserve that comparator and add an all-corner planner using 128 initial corners x five residual profiles, plus the nominal center objective rollout (641 scenarios). No numerical result from this stronger comparator existed when these settings were specified.

Use the same two command blocks, one-minute RK4 prediction, actuator constraints, service deadline, three starts, analytic gradients and SLSQP tolerances. Impose one vehicle of fixed planning clearance on state and service constraints. This clearance is a numerical planning choice; only the separate interval checker supplies the uncertainty-family certificate. Optimize nominal weighted cost subject to these finite-scenario constraints.

Evaluate every completed deadline contract, development and held-out, using the same 640-realization physical replay. Then independently certify the chosen plan; when needed, apply the same ascending 0.05 repair grid toward the verified continuation. This compares tools within the certification architecture and separates finite-scenario feasibility, formal admission and realized traffic performance. Store these results separately so both comparators remain available.
