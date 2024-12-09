import jax
import jax.numpy as jnp
import numpy as np
from jax import jit,grad
from jax.experimental.ode import odeint
from jax.scipy.special import factorial
import matplotlib.pyplot as plt
import optax
import imageio
import os

# Define the simulation parameters
# t_span = jnp.linspace(0, 20, 10000)
# C_values = jnp.linspace(0.01, 2, 100)
jax.config.update("jax_enable_x64", True)

def objective(f, t, C, v_e):
    dfm_dt = jnp.zeros_like(f, dtype=jnp.complex128)
    indices = jnp.arange(len(f))
    dfm_dt = (-1j * jnp.roll(f, 1) * jnp.sqrt((indices + 1) / 2)
              -1j * jnp.sqrt(indices / 2) * jnp.roll(f, -1) -1j*v_e*f)
    dfm_dt = dfm_dt.at[1].add(1j*C*f[0])
    dfm_dt = dfm_dt.at[0].set(-1j * f[1] * jnp.sqrt((1) / 2)-1j*v_e*f[0])
    dfm_dt = dfm_dt.at[-1].set(-1j * jnp.sqrt(indices[-1] / 2) * f[-2] -1j*v_e*f[-1])
    return dfm_dt

def solve_ode(C, t, m_max,v_e):
    y0 = jnp.zeros(m_max, dtype=jnp.complex128)
    y0 = y0.at[0].set(1)
    solution = odeint(objective, y0, t, C,v_e)
    return solution
    # return jnp.log(jnp.abs(solution)**2)

