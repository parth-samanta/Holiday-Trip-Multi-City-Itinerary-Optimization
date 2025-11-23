import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from .data import ItineraryData

@dataclass
class ModelComponents:
    data: ItineraryData
    x: cp.Variable         # city selection (boolean)
    y: cp.Variable         # route arcs (boolean)
    t: cp.Variable         # stay length in each city (continuous)
    u: cp.Variable         # MTZ ordering vars
    constraints: list
    objective: cp.Expression
    problem: cp.Problem


def build_model(data: ItineraryData) -> ModelComponents:
    """
    Building a mixed-integer model:
      - x_i ∈ {0,1}: city i is included
      - y_ij ∈ {0,1}: route goes directly from i to j
      - t_i ≥ 0: days spent in city i
      - MTZ constraints to avoid subtours on cities 1..n-1

    no stay allowed for home (City 0), always included in the tour.
    """
    n = data.n_cities

    # Decision variables
    x = cp.Variable(n, boolean=True)
    y = cp.Variable((n, n), boolean=True)
    t = cp.Variable(n, integer=True)
    # MTZ order variables for cities 1..n-1 (u[0]=0)
    u = cp.Variable(n)

    constraints = []
    # Home city must be included, but stay time is zero
    constraints += [x[0] == 1, t[0] == 0]
    constraints.append(t >= 0)

    # No self loops
    for i in range(n):
        constraints.append(y[i, i] == 0)

    # Flow constraints for non-home cities: in-degree = out-degree = x_i
    for k in range(1, n):
        constraints.append(cp.sum(y[k, :]) == x[k])  # outbound
        constraints.append(cp.sum(y[:, k]) == x[k])  # inbound

    # Home city: exactly one outbound and one inbound edge if any city is visited
    # (this ensures a single tour starting and ending at home)
    constraints.append(cp.sum(y[0, :]) == 1)
    constraints.append(cp.sum(y[:, 0]) == 1)

    # Stay-time linking constraints (only if city selected)
    min_stay = data.min_stay
    max_stay = data.max_stay

    for i in range(1, n):
        constraints.append(t[i] >= min_stay[i] * x[i])
        constraints.append(t[i] <= max_stay[i] * x[i])
    # For home: no stay
    constraints.append(t[0] == 0)

    # Total trip time (travel + stay) constraint
    travel_time_term = cp.sum(cp.multiply(data.travel_time, y))
    total_time = cp.sum(t) + travel_time_term
    constraints.append(total_time <= data.total_trip_days)

    # Budget constraint (stay + travel)
    stay_cost_term = cp.sum(cp.multiply(data.stay_cost_per_day, t))
    travel_cost_term = cp.sum(cp.multiply(data.travel_cost, y))
    total_cost = stay_cost_term + travel_cost_term
    constraints.append(total_cost <= data.budget)

    # MTZ subtour elimination
    # 1 <= u[i] <= n-1 for cities 1..n-1, and u[0] = 0
    constraints.append(u[0] == 0)
    for i in range(1, n):
        constraints.append(u[i] >= 1)
        constraints.append(u[i] <= n - 1)

    M = n - 1
    # Only apply MTZ on non-home cities
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            constraints.append(u[i] - u[j] + M * y[i, j] <= M - 1)

    # Objective: maximise enjoyment - penalty * travel time
    enjoyment_term = cp.sum(cp.multiply(data.enjoyment_per_day, t))
    travel_penalty_term = data.travel_time_penalty * travel_time_term
    objective_expr = enjoyment_term - travel_penalty_term

    objective = cp.Maximize(objective_expr)
    problem = cp.Problem(objective, constraints)

    return ModelComponents(
        data=data,
        x=x,
        y=y,
        t=t,
        u=u,
        constraints=constraints,
        objective=objective_expr,
        problem=problem,
    )
