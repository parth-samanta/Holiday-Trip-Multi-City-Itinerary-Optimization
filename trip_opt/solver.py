from dataclasses import dataclass
import numpy as np
import cvxpy as cp
from .data import ItineraryData
from .model import build_model

@dataclass
class ItinerarySolution:
    status: str
    objective_value: float | None
    selected_indices: list
    route_indices: list   # order of cities visited including home at start & end
    stay_days: np.ndarray | None
    total_time: float | None
    total_cost: float | None
    stay_cost: float | None
    travel_cost: float | None


def _extract_route(y_val: np.ndarray) -> list:
    # Constructing route 0 -> ... -> 0.

    n = y_val.shape[0]
    route = [0]
    current = 0
    visited_steps = 0

    while True:
        next_candidates = np.where(y_val[current, :] > 0.5)[0]
        if len(next_candidates) != 1:
            # infeasible or ambiguous; break
            break
        nxt = int(next_candidates[0])
        route.append(nxt)
        current = nxt
        visited_steps += 1

        if current == 0:
            break
        if visited_steps > n + 5:  # safety
            break
    return route


def solve_itinerary(
    data: ItineraryData,
    solver_name: str = "ECOS_BB",
    verbose: bool = False,
    **solver_opts,
) -> ItinerarySolution:

    components = build_model(data)
    prob = components.problem

    try:
        prob.solve(solver=solver_name, verbose=verbose, **solver_opts)
    except cp.SolverError:
        # Fallback to any available solver
        prob.solve(verbose=verbose, **solver_opts)

    status = prob.status
    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return ItinerarySolution(
            status=status,
            objective_value=None,
            selected_indices=[],
            route_indices=[],
            stay_days=None,
            total_time=None,
            total_cost=None,
            stay_cost=None,
            travel_cost=None,
        )

    x_val = np.round(components.x.value).astype(int)
    y_val = np.round(components.y.value).astype(int)
    t_val = components.t.value

    selected = [i for i in range(1, data.n_cities) if x_val[i] > 0.5]
    route = _extract_route(y_val)

    travel_time_term = np.sum(data.travel_time * y_val)
    stay_cost_term = np.sum(data.stay_cost_per_day * t_val)
    travel_cost_term = np.sum(data.travel_cost * y_val)
    total_time = float(np.sum(t_val) + travel_time_term)
    total_cost = float(stay_cost_term + travel_cost_term)

    return ItinerarySolution(
        status=status,
        objective_value=float(prob.value),
        selected_indices=selected,
        route_indices=route,
        stay_days=t_val,
        total_time=total_time,
        total_cost=total_cost,
        stay_cost=float(stay_cost_term),
        travel_cost=float(travel_cost_term),
    )
