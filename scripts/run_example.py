import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from trip_opt.data import generate_synthetic_data
from trip_opt.solver import solve_itinerary
from trip_opt.analysis import print_solution
from trip_opt.viz import plot_stay_durations, plot_budget_allocation


def main():
    data = generate_synthetic_data(seed=42)
    sol = solve_itinerary(data, solver_name="ECOS_BB", verbose=False)
    print_solution(data, sol)
    # Graph 1: stay per city
    plot_stay_durations(data, sol)
    # Graph 2: budget allocation (stay vs travel vs unused)
    plot_budget_allocation(data, sol)


if __name__ == "__main__":
    main()
