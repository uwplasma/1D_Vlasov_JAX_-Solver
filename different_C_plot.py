import jax
import jax.numpy as jnp
from jax import jit,grad
from jax.experimental.ode import odeint
from jax.scipy.special import factorial
import matplotlib.pyplot as plt
import optax
import imageio
import os
from function_solver import objective
from function_solver import solve_ode
jax.config.update("jax_enable_x64", True)

m_max = 80
t_span = jnp.linspace(0, 10, 200)  # Time span for the simulation
Cs = jnp.linspace(0, 16, 16) # Constant C value

# List of different v_e values to test
v_e = 0

# Create a figure for subplots
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
axes = axes.flatten()

for i, C in enumerate(Cs):
    solution = solve_ode(C, t_span, m_max, v_e)
    data = jnp.transpose(solution)
    ax = axes[i]
    im = ax.imshow(data, aspect='auto', cmap='viridis', origin='lower',
                   extent=[t_span[0], t_span[-1], 0, m_max-1])
    ax.plot(t_span, solution[:, 0])
    ax.set_title(f'Evolution of f_0 with $C={C}$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Mode number')

# fig.colorbar(im, ax=axes.ravel().tolist(), label='log(|fn|^2)')
plt.tight_layout()
plt.show()