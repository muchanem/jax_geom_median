from itertools import zip_longest
import jax
import jax.numpy as jnp

def check_list_of_array_format(points):
	check_shapes_compatibility(points, -1)

def check_list_of_list_of_array_format(points):
	# each element of `points` is a list of arrays of compatible shapes
	components = zip_longest(*points, fillvalue=jnp.array(0))
	for i, component in enumerate(components):
		check_shapes_compatibility(component, i)

def check_shapes_compatibility(list_of_arrays, i):
	arr0 = list_of_arrays[0]
	if not isinstance(arr0, jax.Array):
		raise ValueError(
			"Expected points of format list of `jax.Array`s.",
			f"Got {type(arr0)} for component {i} of point 0."
		)
	shape = arr0.shape
	for j, arr in enumerate(list_of_arrays[1:]):
		if not isinstance(arr, jax.Array):
			raise ValueError(
				f"Expected points of format list of `jax.Array`s. Got {type(arr)}",
				f"for component {i} of point {j+1}."
			)
		if arr.shape != shape:
			raise ValueError(
				f"Expected shape {shape} for component {i} of point {j+1}.",
				f"Got shape {arr.shape} instead."
			)
