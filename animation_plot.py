import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from aquarel import load_theme
import mplcyberpunk

from model import solvingAI

theme = load_theme("minimal_dark") #.set_font('Monospace')
theme.apply()
mplcyberpunk.make_lines_glow()
mplcyberpunk.add_underglow()

# Run the simulation with the plasticity learning
ei = solvingAI()
# ei.lr = 0 # for no plasticity condition
soln, w_0, w_traj, w_final = ei.run_sim()

# Get the mean and standard deviation of the exc and inh firing rates
e_fr_avg = np.mean(soln[:ei.N-ei.D, :], axis=0)
e_fr_std = np.std(soln[:ei.N-ei.D, :], axis=0) # maybe add this later
i_fr_avg = np.mean(soln[ei.N-ei.D:, :], axis=0)
i_fr_std = np.std(soln[ei.N-ei.D:, :], axis=0) # maybe add this later
# corr = (e_fr_avg)/np.mean(e_fr_avg) - i_fr_avg/np.mean(i_fr_avg)

rates = np.maximum(0, (e_fr_avg - i_fr_avg))

ei_widxs = ei.full_idx # indices of the weights that change
w_traj = np.insert(w_traj, 0, w_0[ei_widxs[:, 0], ei_widxs[:, 1]], axis=1)
time = np.arange(0, ei.T, ei.h) # simulation time


fig = plt.figure(tight_layout=True, figsize=(8, 3))
gs = gridspec.GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1:])

# Plot the weights in the first subplot
im = ax1.imshow(w_0, aspect='auto', cmap='coolwarm', vmin=-0.05, vmax=0.2)
#plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

# Plot the solution in the second subplot
ti, in_soln = [], []
te, ex_soln = [], []
lni, = ax2.plot([], [])
lne, = ax2.plot([], [])
y_lower = min(np.amin(e_fr_avg), np.amin(i_fr_avg)) - 2
y_upper = max(np.amax(e_fr_avg), np.amax(i_fr_avg)) + 2

def init():
    # Weight matrix plot formatting
    ax1.set_title('Weights')
    ax1.set_xlabel('Presynaptic Neurons')
    ax1.set_ylabel('Postsynaptic Neurons')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Solution formatting
    ax2.set_xlim(0, ei.T)
    ax2.set_ylim(y_lower, y_upper)
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Subthreshold Activity')
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.legend(['Inhibitory', 'Excitatory'], loc='upper right', 
               frameon=False)

    return im, lne, lni,

def update(frame):
    frame_idx = np.where(time == frame)[0][0]

    # Update the weight image
    new_weights = w_0.copy()
    new_weights[ei_widxs[:, 0], ei_widxs[:, 1]] = w_traj[:, frame_idx]
    im.set_data(new_weights)

    # Update the times
    ti.append(frame)
    te.append(frame)

    # Update the solution trajectories
    in_soln.append(i_fr_avg[frame_idx])
    ex_soln.append(e_fr_avg[frame_idx])

    # Do the setting for the solution trajectories
    lni.set_data(ti, in_soln)
    lne.set_data(te, ex_soln)

    return im, lne, lni,
 

ani = FuncAnimation(fig, update, frames=time, init_func=init, blit=True)

#plt.show()

ani.save('soln.gif', writer='Pillow', fps=20)
