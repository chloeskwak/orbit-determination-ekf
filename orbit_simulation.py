import numpy as np
import matplotlib.pyplot as plt

# Earth's gravitational parameter (km^3/s^2)
mu = 398600.4418

def two_body_dynamics(state):
    """
    Computes acceleration from two-body gravity
    """
    r = state[:3]
    v = state[3:]

    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3

    return np.concatenate((v, a))


def rk4_step(state, dt):
    """
    Runge-Kutta 4 integrator
    """
    k1 = two_body_dynamics(state)
    k2 = two_body_dynamics(state + 0.5 * dt * k1)
    k3 = two_body_dynamics(state + 0.5 * dt * k2)
    k4 = two_body_dynamics(state + dt * k3)

    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def propagate_orbit(state0, dt, steps):
    """
    Propagate spacecraft orbit
    """
    states = [state0]

    state = state0
    for _ in range(steps):
        state = rk4_step(state, dt)
        states.append(state)

    return np.array(states)


# Initial circular orbit
r0 = np.array([7000, 0, 0])  # km
v0 = np.array([0, 7.5, 0])   # km/s

state0 = np.concatenate((r0, v0))

dt = 10        # seconds
steps = 1000

trajectory = propagate_orbit(state0, dt, steps)

# Plot orbit
plt.figure()
plt.plot(trajectory[:,0], trajectory[:,1])
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.title("Simulated Orbit")
plt.axis("equal")
plt.grid()

plt.savefig("orbit_trajectory.png")
plt.show()
