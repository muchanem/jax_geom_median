import jax
import jax.numpy as jnp
from types import SimpleNamespace

def geometric_median(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``numpy.ndarray``.
        Each inner list has the same "shape".
    :param weights: ``numpy.ndarray`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
    	Equivalently, this is a smoothing parameter. Default 1e-6.
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``numpy.ndarray`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    # initialize median estimate at mean
    median = weighted_average(points, weights)
    objective_value = geometric_median_objective(median, points, weights)
    logs = [objective_value]

    # Weiszfeld iterations
    early_termination = False
    for _ in range(maxiter):
        prev_obj_value = objective_value
        new_weights = weights / jax.lax.clamp(eps, jnp.linalg.norm(points-median, axis=1), jnp.inf)
        median = weighted_average(points, new_weights)

        objective_value = geometric_median_objective(median, points, weights)
        logs.append(objective_value)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            early_termination = True
            break

    return SimpleNamespace(
        median=median,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def weighted_average(points, weights):
    return jnp.average(points, axis=0, weights=weights, keepdims=True)
def geometric_median_objective(median, points, weights):
    return jnp.average(jnp.linalg.norm(points-median, axis=1), weights=weights, axis=0)
