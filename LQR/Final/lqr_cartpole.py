import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import matrix_rank
import control
from scipy.linalg import solve_continuous_are

# Params

M = 1      # kg
m = 0.1    # kg
l = 0.5    # m
g = 9.81   # kg.m/s^2

# Initialize
p = 1 # m
theta = 0.05 #rad
p_dot = 0
theta_dot = 0

def state_space_model(A, B, x_t_minus_1, u_t_minus_1):
  state_estimate_x_t = (A @ x_t_minus_1) + (B @ u_t_minus_1)
  return state_estimate_x_t

# mechanical variables

alpha = (M+m)*g/(M*l)
beta = -m*g/M
gamma = -1/(M*l)
delta = 1/M

# Matrices

# State transition matrix A (4x4)
A = np.array([[0, 1, 0, 0],
              [alpha, 0, 0, 0],
              [0, 0, 0, 1],
              [beta, 0, 0, 0]])

# B matrix (4x1)
B = np.array([[0],
              [gamma],
              [0],
              [delta]])

# state x (4x1)
x = np.array([[theta],
              [theta_dot],
              [p],
              [p_dot]])

x_f = np.array([[0],
                [0],
                [0],
                [0]]) # desired state

# Cost matrices

Q = [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]]

R = [[1]]

# Check_controllability

C = np.hstack([B, A@B, A@A@B, A@A@A@B])
print("controllability rank =", matrix_rank(C))

def CARE_Solver(A, B, Q, R, dt, T, P_terminal=None):
  N = int(T/dt)
  n = A.shape[0]
  if P_terminal is None:
    P = np.zeros((n,n))
  else:
    P = P_terminal.copy()

  for t in range(N):
    dP = -(A.T @ P + P @ A - P @ B @ np.linalg.pinv(R) @ B.T @ P + Q)
    P = P - dP*dt

  return P

P = CARE_Solver(A, B, Q, R, 0.01, 10)
K = -np.linalg.pinv(R) @ B.T @ P

P_check = P - solve_continuous_are(A, B, Q, R)

#print("P = ",P)
#print("P_check = ", P_check)


print("K = ", K)

# K_Check

K_check, S, E = control.lqr(A, B, Q, R)

print("K_check = ", -K_check)
print("S = ", S)
print("Poles= ", E)

T = 10
dt = 0.01
N = int(T/dt)

states = [x.copy()]
u_log = []

for t in range(N):
  u_opt = K @ x
  x += state_space_model(A, B, x, u_opt)*dt

  if t%100 == 0:
    print(f"t = {t*dt:.1f}s, x = {x.ravel()}")

  states.append(x.copy())
  u_log.append(u_opt[0,0])

states = np.hstack(states)
u_log = np.array(u_log)

# Pause before motion: 2 seconds of rest
pause_frames = int(2 / dt)
initial_state = states[:, 0:1]
pause_states = np.repeat(initial_state, pause_frames, axis=1)
states_with_pause = np.hstack([pause_states, states])
N_total = states_with_pause.shape[1]

# Plotting
time = np.linspace(0, T, N+1)
# Adjust u_log time to match states (append last u or ignore last state)
# states has N+1 points, u_log has N points. 
# We'll plot u_log vs time[:-1]

fig, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
labels = ['theta', 'theta_dot', 'position', 'position_dot']
ylabels = ['Angle [rad]', 'Ang. Vel [rad/s]', 'Position [m]', 'Velocity [m/s]']

for i in range(4):
    axs[i].plot(time, states[i], label=labels[i])
    axs[i].set_ylabel(ylabels[i])
    axs[i].grid(True)
    axs[i].legend(loc='upper right')

axs[4].plot(time[:-1], u_log, label='Control (u)', color='r')
axs[4].set_ylabel('Force [N]')
axs[4].set_xlabel('Time (s)')
axs[4].grid(True)
axs[4].legend(loc='upper right')

fig.suptitle('LQR Control Results')
plt.tight_layout()
plt.savefig('states_plot.png')

# Animation
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_ylim(-1.2, 1.2)
cart_width = 0.3
cart_height = 0.2
cart = plt.Rectangle((0, 0), cart_width, cart_height, fc='k')
ax.add_patch(cart)
line, = ax.plot([], [], 'o-', lw=3, color='orange')
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def init():
    cart.set_xy((-cart_width / 2, -cart_height / 2))
    line.set_data([], [])
    time_text.set_text('')
    return cart, line, time_text

def animate(i):
    p = states_with_pause[2, i]
    theta = states_with_pause[0, i]
    pend_x = p - l * np.sin(theta)
    pend_y = l * np.cos(theta)
    view_half_width = 2.0
    ax.set_xlim(p - view_half_width, p + view_half_width)
    cart.set_xy((p - cart_width / 2, -cart_height / 2))
    line.set_data([p, pend_x], [0, pend_y])
    time_text.set_text(f"t = {i*dt - 2.0:.2f} s") # correct time shift for pause
    return cart, line, time_text

ani = FuncAnimation(fig, animate, frames=N_total, init_func=init,
                    blit=True, interval=dt * 1000)

print("Saving animation to simulation.gif...")
ani.save('simulation.gif', writer='pillow', fps=30)
print("Done.")

# plt.show()
