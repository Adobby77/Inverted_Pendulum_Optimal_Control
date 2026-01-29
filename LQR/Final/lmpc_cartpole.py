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


# Visualize

time = np.arange(steps) * dt
x_log = np.array(xm_log)
y_log = np.array(y_log)


plt.figure(figsize=(12, 8))

labels = ['theta', 'theta_dot', 'position', 'position_dot']
for i in range(4):
    plt.subplot(5, 1, i+1)
    plt.plot(time, x_log[:, i])
    plt.ylabel(labels[i])
    plt.grid()

plt.subplot(5, 1, 5)
plt.plot(time, u_log)
plt.xlabel('Time [s]')
plt.ylabel('Control Input u')
plt.grid()


# # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Animation (Cart-Pole)
from matplotlib.patches import Rectangle

# Extract data
theta_log = y_log[:, 0]   # angle
pos_log = np.array(xm_log)[:, 2]     # cart position

# System drawing parameters
cart_width = 0.3
cart_height = 0.1
pole_length = l  # m

fig, ax = plt.subplots(figsize=(8, 4))

import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax.set_ylim(-0.2, 1.2)

ax.set_xlim(-2, 2)
ax.set_ylim(-0.2, 1.2)

cart = Rectangle((0, 0), cart_width, cart_height, fc='k')
pole, = ax.plot([], [], lw=4, color='r', alpha=1)

ax.add_patch(cart)
time_template = 'Time = %.1fs'
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def init():
    cart.set_xy((pos_log[0] - cart_width/2, 0))  # 🔧 Data 좌표계 기준
    cart.set_transform(ax.transData)             # 🔧 Data 좌표계로 설정
    pole.set_data([], [])
    time_text.set_text('')
    return cart, pole, time_text


def animate(i):
    p = pos_log[i]
    theta = theta_log[i]

    # cart 위치 (Data 좌표계)
    cart.set_xy((p - cart_width/2, 0))   # 🔧 cart 중심 위치
    cart.set_transform(ax.transData)     # 🔧 Data 좌표계 고정

    # pole 위치
    x_pole = [p, p + pole_length * np.sin(theta)]
    y_pole = [cart_height, cart_height + pole_length * np.cos(theta)]
    pole.set_data(x_pole, y_pole)

    # x축 뷰
    view_half_range = 2.5
    ax.set_xlim(p - view_half_range, p + view_half_range)

    # y축 고정
    ax.set_ylim(-0.2, 1.2)

    # 시간 텍스트
    time_text.set_text(time_template % (i * dt))
    return cart, pole, time_text






ani = FuncAnimation(fig, animate, frames=len(pos_log),
                    interval=dt*1000, blit=False, init_func=init)

plt.show()
