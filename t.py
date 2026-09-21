import jax
import jax.numpy as jnp

x = jnp.zeros((32, ), dtype=jnp.float32)
y = jax.lax.bitcast_convert_type(x, jnp.float16)
print(y.shape)  # (32, 2)
