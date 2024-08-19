import jax.numpy as jnp
import jax

from .weiszfeld_array import geometric_median_array, geometric_median_per_component
from .weiszfeld_list_of_array import geometric_median
from . import utils

def compute_geometric_median(
	points, weights=None, per_component=False, skip_typechecks=False,
	eps=1e-6, maxiter=100, ftol=1e-20
):
	""" Compute the geometric median of points `points` with weights given by `weights`.
	"""
	if weights is None:
		n = len(points)
		weights = jnp.ones(n)
	if type(points) != jax.Array:
		raise ValueError(
			f"We expect `points` as a 2d array. Got {type(points)}"
		)
	if per_component:
		raise NotImplementedError(
			f"No per component implementation"
		)
	else:
		to_return = geometric_median(points, weights, eps, maxiter, ftol)
	return to_return
