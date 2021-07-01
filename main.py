import os
import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Schrodinger(object):
    def __init__(self, psi_0, V, tf, dt, dx, k):
        self.psi_0 = psi_0
        self.V = V
        self.tf = tf
        self.dt = dt
        self.dx = dx
        self.k = k

    def fourier_step(self, psi_0, V, dt, k):
        exp = np.exp(((1j)/2)*dt*(2*np.pi*(1j)*k)**2)
        fourier = np.fft.fft(np.exp(1j*dt*V)*psi_0)
        fourier_final = np.fft.ifft(exp*fourier)
        return fourier_final

    def bpm_method(self, psi_0, V, tf, dt, k):
        M = int(tf/dt)
        for i in range(1, M):
            psi_0 = s.fourier_step(psi_0, V, dt, k)
        return psi_0

    def psi(self, psi_0, V, tf, dt):
        return np.abs(s.bpm_method(psi_0, V, tf, dt, k))

def gaussian(x, mean, std, k0):
    return np.exp(-((x-mean)**2)/(4*std**2)+ 1j*x*k0)/(2*np.pi*std**2)**0.25

L = 400 # domain size
N = 2000 # number of discret points
dx = L/N # distance between points

tf = 1.0 # final time
dt = 0.01 # time steps size

k = 2*np.pi*np.fft.fftfreq(N, d=dx) # wavenumber

p0 = 2.0
d = np.sqrt(N*dt/2.)
x = np.arange(-L/2, L/2, dx)
psi_0 = gaussian(x, x.max() - 10*d, d, -p0)

V = -110/np.cosh(x)**2

s = Schrodinger(psi_0, V, tf, dt, dx, k)

fig, ax = plt.subplots(figsize=(10, 4))

plt.title("Wave packet colliding with a potential barrier", fontsize=14, c="#000000")

plt.axvline(0, 0, 1, c="#000000", linestyle='--')

bpm_plot, = ax.plot(x, s.bpm_method(psi_0, V, tf, dt, k), c="#6ED5FA")
psi_plot, = ax.plot(x, s.psi(psi_0, V, tf, dt), c="#6B7CFA")

def animate(i):
    bpm_plot.set_data(x, s.bpm_method(psi_0, V, tf + i/25, dt, k))
    psi_plot.set_data(x, s.psi(psi_0, V, tf + i/25, dt))
    return bpm_plot, psi_plot

ani = animation.FuncAnimation(fig, animate, interval=20, blit=True, save_count=66)

p = os.path.abspath(sys.argv[0])
archive_path, arq = os.path.split(p)
os.chdir(archive_path)

ani.save("tunnelvision.gif")