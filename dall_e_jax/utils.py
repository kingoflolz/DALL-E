import jax.numpy as jnp

logit_laplace_eps: float = 0.1


def map_pixels(x: jnp.ndarray) -> jnp.ndarray:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != jnp.float32:
		raise ValueError('expected input to have type float')

	return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x: jnp.ndarray) -> jnp.ndarray:
	if len(x.shape) != 4:
		raise ValueError('expected input to be 4d')
	if x.dtype != jnp.float32:
		raise ValueError('expected input to have type float')

	return jnp.clip((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)
