import sys
import threading
from queue import Queue
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from matplotlib.animation import FuncAnimation

# =========================
# Parameters
# =========================
nx, ny   = 120, 120
alpha    = 0.20    # 4-neighbor explicit step; keep <= 0.25 for stability
steps    = 600     # number of diffusion steps on the LEFT panel only
interval = 30      # ms between animation frames
cmap     = "turbo"
mode     = "wrap"  # use SAME mode for sim and Gaussian: "wrap" or "reflect"

# =========================
# Initial condition (u0)
# =========================
u0 = np.zeros((nx, ny), dtype=np.float64)
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

def blob(cx, cy, amp, s):
    return amp * np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*s*s))

u0 += blob(nx*0.30, ny*0.35, amp=80.0, s=4.5)
u0 += blob(nx*0.70, ny*0.65, amp=60.0, s=6.0)
u0[nx//2-2:nx//2+2, ny//2-2:ny//2+2] += 90.0

# Working copy for live diffusion (left)
u = u0.copy()

# =========================
# Explicit Euler step kernel (4-neighbor)
# Sum(kernel) = 1 for mass conservation
# =========================
K = np.array([
    [0.0,    alpha, 0.0   ],
    [alpha,  1 - 4*alpha, alpha],
    [0.0,    alpha, 0.0   ]
], dtype=np.float64)

# =========================
# Input thread: read times 't' from stdin
# =========================
t_queue = Queue()

def stdin_reader():
    print(
        "\nType a time t (>=0) and press Enter.\n"
        "Right panel updates to analytic u(t) = G_t * u0.\n"
        "Enter 'q' to stop reading new times (animation keeps running).\n",
        file=sys.stderr, flush=True
    )
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        if s.lower() == 'q':
            break
        try:
            t = float(s)
            if t < 0:
                print("t must be >= 0", file=sys.stderr)
                continue
            t_queue.put(t)
        except ValueError:
            print("Please enter a number (or 'q' to quit).", file=sys.stderr)

threading.Thread(target=stdin_reader, daemon=True).start()

# =========================
# Figure with shared color scale & one colorbar
# =========================
vmin, vmax = u0.min(), u0.max()

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
imL = axL.imshow(u,  cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
imR = axR.imshow(u0, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
for ax in (axL, axR):
    ax.set_axis_off()

axL.set_title("Live diffusion (4-neighbor)")
axR.set_title("Analytic u(t) = G_t * u0   (t = 0, σ = 0)")
cbar = fig.colorbar(imL, ax=[axL, axR], fraction=0.046, pad=0.04)
cbar.set_label("Temperature")

def init():
    imL.set_data(u)
    imR.set_data(u0)
    return [imL, imR]

def animate(frame):
    global u

    # Step LEFT panel only while frame < steps
    if frame < steps:
        u = convolve(u, K, mode=mode)
        if u.min() < 0:
            u[u < 0] = 0.0
        imL.set_array(u)
        axL.set_title(f"Live diffusion (step {frame+1}/{steps})")
    else:
        # After finishing steps, keep polling stdin but don't update the left
        axL.set_title(f"Live diffusion (completed {steps} steps)")

    # Always poll for new times for RIGHT panel
    while not t_queue.empty():
        t = t_queue.get_nowait()
        sigma = float(np.sqrt(2.0 * alpha * t))  # σ = sqrt(2 D t), D=alpha (dt=dx=1)
        u_t = gaussian_filter(u0, sigma=sigma, mode=mode)
        imR.set_array(u_t)
        axR.set_title(f"Analytic u(t)   (t = {t:g}, σ = {sigma:.3f})")

    return [imL, imR]

# Infinite frame generator so we keep polling indefinitely
anim = FuncAnimation(
    fig, animate,
    frames=itertools.count(),
    init_func=init,
    interval=interval,
    blit=False,
    repeat=False
)

plt.show()
