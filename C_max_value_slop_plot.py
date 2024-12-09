import numpy as np
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
# from function_solver import objective
import jax.numpy as jnp
import jax
# from function_solver import solve_ode
import matplotlib.pyplot as plt
from jax.experimental.ode import odeint
import pandas as pd
from scipy.stats import linregress
jax.config.update("jax_enable_x64", True)

m_max = 40
t_span = jnp.linspace(0, 10, 100)
k_values = jnp.linspace(0.01, 1, 100)
m = 9.1093837e-31
q = 1.602e-19
e = 8.8541878188e-12
w = jnp.sqrt(jnp.sqrt(jnp.pi)*q**2/m/e)
# C_values = w/k_values/1j
# print(C_values)
# print(w)

def objective(f, t, k, v_e):
    indices = jnp.arange(len(f))
    dfm_dt = (-1j*jnp.sqrt(2)*k * (jnp.roll(f, 1)) * jnp.sqrt((jnp.roll(indices, 1) / 2)+
              jnp.sqrt(indices / 2) * jnp.roll(f, -1) + v_e*f))
    dfm_dt = (dfm_dt.at[1].add(1/(k*1j)*f[0]))
    dfm_dt = dfm_dt.at[0].set(-1j*jnp.sqrt(2)*k*(f[1] * jnp.sqrt(1 / 2)+v_e*f[0]))
    dfm_dt = dfm_dt.at[-1].set((-1j*jnp.sqrt(2)*k*(jnp.sqrt(indices[-1] / 2) * f[-2]+v_e*f[-1])))
    return dfm_dt

def solve_ode(k, t, m_max,v_e):
    y0 = jnp.zeros(m_max, dtype=jnp.complex128)
    y0 = y0.at[0].set(1)
    solution = odeint(objective, y0, t, k,v_e)
    return jnp.log(jnp.abs(solution))


v_e = 0

slopes = []

mathematica_data = pd.read_csv('damping_rate.csv')

# for k in k_values:
#     solution = solve_ode(k, t_span, m_max, v_e)
#     solution = solution[:, 0]
#     plt.plot(t_span, solution)
#     # plt.legend()
#     plt.show()


for k in k_values:
    solution = solve_ode(k, t_span, m_max, v_e)
    solution = solution[:, 0]

    # looking for local maxima
    local_maxima_indices, _ = find_peaks(solution)
    local_maxima_values = solution[local_maxima_indices]
    local_maxima_times = t_span[local_maxima_indices]

    # slops
    if len(local_maxima_times) > 1:  # fit for graph have 2+ mix local max
        slope, intercept, r_value, p_value, std_err = linregress(local_maxima_times, local_maxima_values)
        slopes.append(slope)

        # plot fit for each C
        fit_line = slope * local_maxima_times + intercept
        # plt.plot(local_maxima_times, fit_line, 'r--', label=f'Fit Line: slope={slope:.2f}')
        # plt.plot(t_span, solution, label=f'K = {k:.2f}')
        # plt.legend()
        # plt.show()

# plot c value and slope
plt.figure()
plt.plot(k_values, slopes, '-o', label='Slope vs. k')
plt.plot(mathematica_data.values[:80, 0], mathematica_data.values[:80, 1], '-x', label='Mathematica Slope vs. k')
plt.xlabel('C/k value')
plt.ylabel('Slope at Local Maxima')
plt.title('Slope vs. k Value')
plt.legend()
plt.xlim(0,0.6)
plt.ylim(-0.2,0.1)
plt.show()


