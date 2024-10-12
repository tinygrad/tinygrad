#from typeguard import install_import_hook
#install_import_hook(__name__)

from jaxtyping import Array, Float, PyTree, jaxtyped
import jax.numpy as jnp
from typeguard import typechecked as typechecker

@jaxtyped(typechecker=typechecker)
def matrix_multiply(x: Float[Array, "dim1 dim2"],
                    y: Float[Array, "dim2 dim3"]
                  ) -> Float[Array, "dim1 dim3"]:
                  return x@y

if __name__ == "__main__":
  a: Float[Array, "10 10"] = jnp.zeros((2, 2))
  b: Float[Array, "10 10"] = jnp.zeros((2, 2))
  matrix_multiply(a, b)

