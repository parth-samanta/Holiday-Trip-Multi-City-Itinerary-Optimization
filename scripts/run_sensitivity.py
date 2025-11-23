
#Vary the budget parameter and analyse how the optimal objective changes.


import numpy as np
from copy import deepcopy
from trip_opt.data import generate_synthetic_data, ItineraryData
from trip_opt.solver import solve_itinerary
from trip_opt.viz import plot_budget_sensitivity


def run_budget_sensitivity(
    base_data: ItineraryData,
    n_points: int = 6,
    factor_low: float = 0.6,
    factor_high: float = 1.4,
):
    budgets = np.linspace(base_data.budget * factor_low,
                          base_data.budget * factor_high,
                          n_points)
    objectives = []

    for B in budgets:
        data = deepcopy(base_data)
        data.budget = float(B)
        sol = solve_itinerary(data, solver_name="ECOS_BB", verbose=False)
        if sol.objective_value is None:
            objectives.append(np.nan)
        else:
            objectives.append(sol.objective_value)

    return budgets, objectives


def main():
    base_data = generate_synthetic_data(seed=42)
    budgets, objectives = run_budget_sensitivity(base_data)
    plot_budget_sensitivity(budgets, objectives)


if __name__ == "__main__":
    main()
