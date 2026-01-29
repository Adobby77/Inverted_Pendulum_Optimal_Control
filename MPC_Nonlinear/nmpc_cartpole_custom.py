import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Params
M = 1.0       # kg
m = 0.1       # kg
l = 0.5         # m
g = 9.81    # kg.m/s^2

dt = 0.1        # Time step for discretization
Np = 20         # Prediction horizon
Nc = Np         # Control horizon

# Symbolic state variables
x = ca.SX.sym('x')
x_dot = ca.SX.sym('x_dot')
theta = ca.SX.sym('theta')
theta_dot = ca.SX.sym('theta_dot')
states = ca.vertcat(x, x_dot, theta, theta_dot)
n_states = states.size1()  # should be 4

# Reference
xr = 0.0
thetar = 0.0
ref = ca.SX([xr, 0, thetar, 0])

# Control variable
u = ca.SX.sym('u')
n_controls = u.size1()     # should be 1


# Decision variables for the entire horizon
# X will be of size (n_states x (N+1)) and U of size (n_controls x N)
X = ca.SX.sym('X', n_states, Np+1)
U = ca.SX.sym('U', n_controls, Np)

# Dynamics
def cartpole_dynamics(state, u):
    x, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
    sin_theta = ca.sin(theta)
    cos_theta = ca.cos(theta)
    
    total_mass = M + m
    temp = (u + m * l * theta_dot**2 * sin_theta) / total_mass

    theta_ddot = (g * sin_theta - cos_theta * temp) / (l * (4/3 - m * cos_theta**2 / total_mass))
    x_ddot = temp - (m * l * theta_ddot * cos_theta) / total_mass
    
    return ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)

# Discretize

def RK4(f_dyn, dt):
    def integrator(x, u):
        k1 = f_dyn(x, u)
        k2 = f_dyn(x + dt/2 * k1, u)
        k3 = f_dyn(x + dt/2 * k2, u)
        k4 = f_dyn(x + dt * k3, u)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return integrator

f = RK4(cartpole_dynamics, dt)

# Disturbance

def disturbance(i):
    if i%20 == 0:
        d = ca.DM([0, 
                   0, 
                   np.random.uniform(-0.1, 0.1), 
                   np.random.uniform(-0.5, 0.5)])
        return d
    else:
        return ca.DM.zeros(4)

# Weights

Q = ca.SX([[1,   0,    0,    0],
           [0,   0,    0,    0],
           [0,   0,   100,   0],
           [0,   0,    0,    0]])

R = 1

# Initialize cost function and constraints
cost = 0                     # Accumulated cost
eq_cosntraint = []           # Equality constraints list

# Parameter for the initial state:
X0 = ca.SX.sym('X0', n_states)

# Enforce initial condition constraint: the first state in X equals the current state X0.
eq_cosntraint.append(X[:, 0] - X0)

# Loop over each control interval to build the cost and dynamics constraints
for k in range(Np):
    # Cost function: Penalize errors in cart position and pole angle plus control effort.
    cost += (X[:,k]-ref).T@Q@(X[:,k]-ref) + + R * U[0, k]**2
    # Dynamics constraint: next state equals the discrete dynamics from current state and control.
    x_next = f(X[:, k], U[:, k])
    eq_cosntraint.append(X[:, k+1] - x_next)

# Terminal cost
cost += (X[:, Np] - ref).T @ Q @ (X[:, Np] - ref)

# Define limits for the horizontal position and control input
# Horizontal position constraint (for cart's x-position): x_min <= x <= x_max
x_min = -2.0
x_max = 2.0

# Control constraint: u_min <= u <= u_max
u_min = -12.0
u_max =  12.0

# The decision vector consists of:
#  - First: states, shaped as (n_states*(N+1),)
#  - Second: control inputs, shaped as (n_controls*N,)
# # Reshape decision variables into a single column vector.
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
n_decision = int(opt_vars.numel()) # the number of opt_vars (=104)
lbx = -np.inf * np.ones(n_decision) # initialize Lower bound to -infty
ubx =  np.inf * np.ones(n_decision) # initialize Upper bound to infty

# Set box constraints for the "horizontal position" (state variable x)
# Each time step: the x-component is the first element in the state block.
# Other state variables (x_dot, theta, theta_dot) stll have no constraint. (-infty, infty)
for i in range(Np+1):
    idx = i * n_states  # index for x at time step i
    lbx[idx] = x_min
    ubx[idx] = x_max

# Set box constraints for the control variable (u)
# Controls come after the state variables in the decision vector.
start_u = n_states * (Np+1)
for i in range(Np):
    idx = start_u + i * n_controls  # index for control at time step i
    lbx[idx] = u_min
    ubx[idx] = u_max

# Concatenate all constraints into one vector.
eq_cosntraint_concat = ca.vertcat(*eq_cosntraint)

# Define bounds for the constraints (all equality constraints are set to 0)
g_bound = np.zeros(int(eq_cosntraint_concat.numel()))
lbg = g_bound
ubg = g_bound
# lbg <= g <= ubg
# Here, lbg = ubg = 0
# Thus, g = 0, i.e. X[:, k+1] = f(X[:,k], U[:,k])


# Define the NLP problem
nlp_problem = {
    'f': cost,       # Objective function (cost)
    'x': opt_vars,  # Decision variables ([X,U])
    'g': eq_cosntraint_concat,  # Constraints (g)
    'p': X0         # Parameter: initial state
}

# Create an NLP solver instance using IPOPT.
opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp_problem, opts)

# Simulation variables
sim_time = 15.0     # total simulation time (seconds)
N = int(sim_time/dt)
t_history = [0]
state_history = []  # To store state trajectories
control_history = []  # To store control inputs

# Initialize state
init_guess = np.zeros(n_decision)
x_current = np.array([0.3, 0.0, 3.14, 0.0])
state_history.append(x_current)

# ------------------------------
# Simulation Loop
# ------------------------------
for i in range(N):
    # Solve the NMPC for the current state x_current
    sol = solver(x0=init_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_current)
    sol_opt = sol['x'].full().flatten()
    init_guess = sol_opt
    
    # Extract the control sequence from the solution. The state trajectory is first,
    # so the first control is located at index = n_states*(N+1)
    u_opt = sol_opt[n_states*(Np+1): n_states*(Np+1)+n_controls]
    u0_opt = u_opt[0]
    control_history.append(u0_opt)
    
    # Propagate the system using the discrete dynamics f (Euler integration)
    # Note: We use the same function f defined earlier with CasADi operators
    x_next = f(x_current, u0_opt) + disturbance(i)*0
    x_current = np.array(x_next.full()).flatten()
    
    # Store the new state and time
    state_history.append(x_current)
    t_history.append((i+1)*dt)

    #print("Step", i, "| status:", solver.stats()['return_status'], "| theta:", x_current[2])
    if i % 20 == 0:
        print(f"[Impulse] Step {i}, time = {i*dt:.1f}s, disturbance = {disturbance(i)}")
    
    # Optionally, update initial guess to warm-start the next iteration (shift the sequence)
    # For simplicity, we keep using the zero initial guess here.
    
state_history = np.array(state_history)
control_history = np.array(control_history)

# -----------------
#   Visualization
# -----------------

# Plot

# --- Plot State Trajectories and Control Input ---
fig2, axs = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

labels = ['Cart Position (x)', 'Cart Velocity (v)', 'Pole Angle (θ)', 'Pole Angular Velocity (ω)', 'Control Input (u)']
units = ['m', 'm/s', 'rad', 'rad/s', 'N']
data = [state_history[:, 0], state_history[:, 1], state_history[:, 2], state_history[:, 3], control_history]

# Time axis for control is one step shorter
t_for_u = t_history[:-1] if len(control_history) < len(t_history) else t_history

for i in range(5):
    axs[i].plot(t_for_u if i == 4 else t_history, data[i], label=labels[i])
    axs[i].set_ylabel(f'{labels[i]}\n[{units[i]}]')
    axs[i].grid(True)
    axs[i].legend(loc='upper right')

axs[-1].set_xlabel('Time [s]')
fig2.suptitle('NMPC Cart-Pole States and Control law', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])


# Animation

# Assume these variables exist:
# state_history: numpy array of shape (num_steps, 4) [columns: x, x_dot, theta, theta_dot]
# control_history: numpy array of shape (num_steps-1,) containing control inputs at each simulation step

# For illustration purposes, here are dummy histories (remove these lines when using your actual data)
num_steps = N  # Example: 51 simulation steps
dt = 0.1
t_history = np.arange(0, num_steps*dt, dt)
# --------------------------------------------------

# Visualization parameters
cart_width = 0.4
cart_height = 0.2
pole_length = 0.5  # Ensure this matches the length used in your simulation

# Create the figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(-5, 5)
ax.set_ylim(-1.0, 2.0)
ax.set_xlabel('position (m)')
# ax.set_ylabel('y (m)')
ax.set_title('NMPC_Cartpole')

# Create plot elements: rectangle for the cart and a line for the pole
cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fc='k', ec='k')
ax.add_patch(cart_patch)
pole_line, = ax.plot([], [], lw=3, c='r')

# Text annotations for time and control input
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
control_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=12)

# Initialization function for the animation
def init():
    cart_patch.set_xy((-cart_width/2, 0))
    pole_line.set_data([], [])
    time_text.set_text('')
    control_text.set_text('')
    return cart_patch, pole_line, time_text, control_text

# Animation update function
def animate(i):
    # Extract state: [x, x_dot, theta, theta_dot]
    x_val = state_history[i, 0]
    theta_val = state_history[i, 2]
    
    # Update cart: center the cart at x_val, with the bottom at y = 0
    cart_x = x_val - cart_width/2
    cart_patch.set_xy((cart_x, 0))
    
    # The pivot point (where the pole is attached) is at the top-center of the cart.
    cart_center = np.array([x_val, cart_height])
    
    # Compute the pole tip using basic trigonometry
    pole_tip = cart_center + pole_length * np.array([np.sin(theta_val), np.cos(theta_val)])
    pole_line.set_data([cart_center[0], pole_tip[0]], [cart_center[1], pole_tip[1]])
    
    # Update annotations: simulation time and control input.
    time_text.set_text(f'Time = {i*dt:.2f} s')
    # Use the last available control for the last frame.
    u_val = control_history[i] if i < len(control_history) else control_history[-1]
    control_text.set_text(f'Control u = {u_val:.2f}')
    
    return cart_patch, pole_line, time_text, control_text

# Create the animation object. The 'interval' parameter is set in milliseconds.
ani = animation.FuncAnimation(fig, animate, frames=len(state_history),
                              init_func=init, interval=dt*1000, blit=True)

plt.show()