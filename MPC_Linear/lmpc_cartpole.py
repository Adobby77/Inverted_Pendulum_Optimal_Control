import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize

# Params

M = 1      # kg
m = 0.1    # kg
l = 0.5    # m
g = 9.81   # kg.m/s^2

# Initialize
p = 0 # m
theta = 0.03 #rad
p_dot = 0
theta_dot = 0

# mechanical variables

alpha = (M+m)*g/(M*l)
beta = -m*g/M
gamma = -1/(M*l)
delta = 1/M

# Model Matrices

# Am (4x4)
Am = np.array([[0, 1, 0, 0],
               [alpha, 0, 0, 0],
               [0, 0, 0, 1],
               [beta, 0, 0, 0]])

# Bm (4x1)
Bm = np.array([[0],
               [gamma],
               [0],
               [delta]])

# Cm (4x4)
Cm = np.eye(4)

# Discretized Model Matrices

from scipy.signal import cont2discrete

Dm = np.zeros((2, 1))
dt = 0.02 # sec

Ad, Bd, Cd, Dd, _ = cont2discrete((Am, Bm, Cm, Dm), dt)

# Augmented Model Matrices

# A matrix ((4+1)x(4+1))

Od = np.zeros((4,4))
I  =np.eye((4))

A = np.block([[Ad, Od.T],
              [Cd@Ad, I]])

# B matrix ((4+1)x1)

B = np.block([[Bd],
              [Cd@Bd]])

# C matrix (2x(2+4))

C = np.block([Od, I])

# MPC Background (Horizon, Matrices, Cost function)

# Horizon
Np = 150
Nc = 100

# Matrices

# F

def construct_F(A, C, Np):
  F = []

  for i in range(Np):
    F.append(C @ np.linalg.matrix_power(A, i+1))

  F = np.vstack(F)

  return F

# Phi matrix

def construct_Phi(A, B, C, Np, Nc):
  ny = C.shape[0]
  nu = B.shape[1]

  Phi = []

  for i in range(Np):
      row = []
      for j in range(Nc):
          if j <= i:
              term = C @ np.linalg.matrix_power(A, i - j) @ B
          else:
              term = np.zeros((ny, nu))
          row.append(term)

      Phi.append(np.hstack(row))

  Phi = np.vstack(Phi)
  return Phi

# Reference Data Matrix

def construct_ref(ref, Np):
  Rs = np.kron(np.ones((Np,1)), ref)
  return Rs

# Penalizer Rbar

def construct_penalize(rw, Nc):
  Ru = rw * np.eye(Nc)
  return Ru

# Cost function

def cost(Delta_U, Rs, Ru, Phi, F, x):
  Delta_U = Delta_U.reshape((-1,1))
  Y = F @ x + Phi @ Delta_U
  J = (Rs-Y).T@(Rs-Y) + (Delta_U.T@Ru@Delta_U)[0,0]
  return J

# Simulation

T = 15 #sec
steps = int(T/dt)

# Initialize

# state xm (4x1)
xm = np.array([[theta],
               [theta_dot],
               [p],
               [p_dot]])

# Increment Delta_xm
xm_prev = np.zeros((4,1))
Delta_xm = xm - xm_prev

# output y (2x1)
y_k = Cd @ xm

# Augmented State [Delta_xm, y]
x = np.block([[Delta_xm],
              [y_k]])

# Initialize Log
Delta_xm_log = []
xm_log = []
y_log = []
u_log = []
u_prev = 0

#MPC matrices
F = construct_F(A, C, Np)
Phi = construct_Phi(A, B, C, Np, Nc)
Ru = construct_penalize(10, Nc)

for k in range(steps):

  #1. reference signal / Reference Data vector
  ref = np.array([[0],
                  [0],
                  [0],
                  [0]])

  Rs = construct_ref(ref, Np)


  #2. Optimization
  # Delta_U_0 = np.zeros(Nc) # Initial Guess
  # result = minimize(cost,
  #                   Delta_U_0,
  #                   args=(Rs, Ru, Phi, F, x),
  #                   method='BFGS')

  # Delta_U_opt = result.x
  # Delta_u0 = float(Delta_U_opt[0])

  Delta_U_opt = np.linalg.pinv(Phi.T@Phi+Ru)@Phi.T@(Rs-F@x)
  Delta_u0 = float(Delta_U_opt[0,0])

  #3. Update
  x = A@x + B*Delta_u0

  #4 Save
  Delta_xm = x[:4]
  y_k = x[4:]

  xm = xm + Delta_xm

  Delta_xm_log.append(Delta_xm.flatten())
  xm_log.append(xm.flatten())
  y_log.append(y_k.flatten())

  u_prev = u_prev + Delta_u0
  u_log.append(u_prev)


# Check Eigenvalues

# 전체 이득 행렬
K_full = np.linalg.pinv(Phi.T @ Phi + Ru) @ Phi.T

# 선형 맵핑 형태로 바꾸기
Ky = K_full @ Rs  # → Rs = reference vector
Kmpc = K_full @ F  # → F @ x = prediction from current state

# Δu[0]만 추출
L = np.zeros((1, Nc))  # [1 0 ... 0]
L[0, 0] = 1

Ky_feedback = L @ Ky  # scalar gain * r(k)
Kmpc_feedback = L @ Kmpc  # state feedback


# print("Ky_feedback:", Ky_feedback)
# print("Kmpc_feedback:", Kmpc_feedback)

SYS = A-B@Kmpc_feedback
print(np.linalg.eigvals(SYS))


# ========================  Visualize (UNIFIED)  ========================
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Logs to arrays
time = np.arange(steps) * dt
x_log = np.array(xm_log)        # (steps, 4) => [theta, theta_dot, x, x_dot]
u_log = np.array(u_log)

# ---------- Unified state plots ----------
fig, axs = plt.subplots(5, 1, figsize=(11, 8), sharex=True)
labels = ["theta", "theta_dot", "x", "x_dot"]
for i in range(4):
    axs[i].plot(time, x_log[:, i])
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)
axs[4].plot(time, u_log)
axs[4].set_ylabel("u")
axs[4].grid(True)
axs[-1].set_xlabel("Time [s]")
fig.suptitle("LMPC — States & Control (Unified)")
fig.tight_layout(rect=[0, 0, 1, 0.96])

# ---------- Unified animation ----------
# Extract data
theta_log = x_log[:, 0]   # theta
pos_log   = x_log[:, 2]   # x (cart position)

# Initial pause for smooth start (2s)
pause_frames = max(0, int(round(2.0 / dt)))
if pause_frames > 0:
    theta_anim = np.concatenate([np.full(pause_frames, theta_log[0]), theta_log])
    pos_anim   = np.concatenate([np.full(pause_frames, pos_log[0]),   pos_log])
else:
    theta_anim, pos_anim = theta_log, pos_log

# Style config (same as LQR/NMPC 통일)
cart_width  = 0.40
cart_height = 0.18
pole_length = l
pole_color  = "#ff7f0e"   # unified orange
view_half   = 2.5
ylim = (-0.4, 1.4)

fig2, ax = plt.subplots(figsize=(10, 4))
ax.set_ylim(*ylim)
ax.set_xlim(pos_anim[0] - view_half, pos_anim[0] + view_half)
ax.set_title("LMPC - Inverted Pendulum (Unified)")
ax.grid(True, alpha=0.4)

cart = Rectangle((0, 0), cart_width, cart_height, fc='k', ec='k')
ax.add_patch(cart)
(pole_line,) = ax.plot([], [], lw=3, c=pole_color)
time_text = ax.text(0.02, 0.92, '', transform=ax.transAxes)

def _set_cart_center(xc):
    cart.set_xy((xc - cart_width/2.0, 0.0))

def init():
    _set_cart_center(pos_anim[0])
    pole_line.set_data([], [])
    time_text.set_text('')
    return cart, pole_line, time_text

def animate(i):
    p = float(pos_anim[i])
    th = float(theta_anim[i])

    # cart
    _set_cart_center(p)

    # pole (pivot at top center of cart)
    base = np.array([p, cart_height])
    tip  = base + pole_length * np.array([np.sin(th), np.cos(th)])
    pole_line.set_data([base[0], tip[0]], [base[1], tip[1]])

    # follow cart on x
    ax.set_xlim(p - view_half, p + view_half)

    # time text
    time_text.set_text(f"t = {i*dt:.2f} s")
    return cart, pole_line, time_text

ani = FuncAnimation(fig2, animate, frames=len(pos_anim),
                    init_func=init, interval=dt*1000, blit=False)

plt.show()
# ======================================================================