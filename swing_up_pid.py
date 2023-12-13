"""
Simulation of pendulum swing-up by energy control.

Equations:
	th'' = (g * sin(th) - u * cos(th)) / L,
	u = k * E * th' * cos(th),
	where E = m * (th' * L) ^ 2 / 2 + m * g * L * (cos(th) - 1), zero-energy is in upright position

System:
	th' = Y,
	Y' = (g * sin(th) - u * cos(th)) / L,
	x' = Z,
	Z' = u = k * E * Y * cos(th),

State: 
	[th, Y, x, Z]

References:
- Swinging up a pendulum by energy control - K.J. Astrom, K. Furuta
"""

import numpy as np

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as pp
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from math import pi
from numpy import sin, cos, sign
import control as ctl

# physical constants
g = 9.8
L = 1.0
m = 0.5
M = 3
m_eq = M+(m/4)
# simulation time
dt = 0.05
Tmax = 60
t = np.arange(0.0, Tmax, dt)

# initial conditions
Y = .0 		# pendulum angular velocity
th = pi - 0.1		# pendulum angle
x = .0		# cart position
x0 = 0		# desired cart position
Z = -0.05	# cart velocity
k = 0.05	# control gain coefficient
u_max = 3
#state = np.array([th, Y, x, Z])
state = np.array([x,th,Z, Y])

def energy(th, dth):
	return m * dth * L * dth * L / 6 + m * g * L * (cos(th) - 1)/2

def sign(x):
	if x> 0:
		return 1
	elif x<0:
		return -1
	else:
		return 1 

def sat(x):
	if x<4:
		return x
	else:
		return 4

def derivatives(state, t):
	ds = np.zeros_like(state)

	_th = state[1]
	_Y = state[3]	# th'
	_x = state[0]
	_Z = state[2]	# x'

	E = energy(_th, _Y)
	if _th >= 0.1 * pi and _th <= 1.9 * pi:
		#u = 0.75*sign(E* _Y * cos(_th)) 
		u = 0.4* E * _Y * cos(_th)
	else:
		u = sat(k*E *sign(_Y * cos(_th))) 
		#u =0

	ds[0] = state[2]
	ds[1] = state[3]
	ds[2] = u
	ds[3] = 1.5*(g * sin(_th) - u * cos(_th)) / L
	if _th<=0.1:
		ds[0] = 0
		ds[1] = 0
		ds[2] = 0
		ds[3] = 0

	return ds

print("Integrating...")
# integrate your ODE using scipy.integrate.
solution = integrate.odeint(derivatives, state, t)
print("Done")

ths = solution[:, 1]
Ys = solution[:, 3]
xs = solution[:, 0]
vs = solution[:, 2]
pxs = L * sin(ths) + xs
pys = L * cos(ths)
temp = 0
for i in ths:
	if i<=0.1:
		break
	else:
		temp+=1


x_eq = np.array([0,0,0,0])
A = np.array([[0,0,1,0],
	         [0,0,0,1],
			 [0,-3*m*g/m_eq,0,0],
			 [0,(3*g/(2*L)+(9*m*g)/(8*L*m_eq)),0,0]])
B = np.array([[0],
	          [0],
			  [1/m_eq],
			  [3/(2*L*m_eq)]])
C = np.identity(4)
D = np.array([[0],[0],[0],[0]])




fig = pp.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-6, 10), ylim=(-1.2, 1.2))
ax.set_aspect('equal')
ax.grid()

patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

energy_template = 'E = %.3f J'
energy_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

cart_width = 0.3
cart_height = 0.2

def init():
    line.set_data([], [])
    time_text.set_text('')
    energy_text.set_text('')

    patch.set_xy((-cart_width/2, -cart_height/2))
    patch.set_width(cart_width)
    patch.set_height(cart_height)
    return line, time_text, energy_text, patch


# def animate(i):
#     thisx = [xs[i], pxs[i]]
#     thisy = [0, pys[i]]

#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*dt))
    
#     E = energy(ths[i], Ys[i])
#     energy_text.set_text(energy_template % (E))

#     patch.set_x(xs[i] - cart_width/2)
#     return line, time_text, energy_text, patch

xs_final  = np.append(xs[:temp],sol[:,0])
ths_final = np.append(ths[:temp],sol[:,1])
vs_final = np.append(vs[:temp],sol[:,2])
Ys_final = np.append(Ys[:temp],sol[:,3])
pxs_final = L * sin(ths_final) + xs_final
pys_final = L * cos(ths_final)

def animate(i):
    thisx = [xs_final[i], pxs_final[i]]
    thisy = [0, pys_final[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    
    E = energy(ths_final[i], Ys_final[i])
    energy_text.set_text(energy_template % (E))

    patch.set_x(xs_final[i] - cart_width/2)
    return line, time_text, energy_text, patch


ani = animation.FuncAnimation(fig, animate, np.arange(1, len(xs_final)),
                              interval=25, blit=True, init_func=init)


pp.figure()
pp.subplot(211)

Es = np.vectorize(energy)(ths, Ys)
Us = k * Es * Ys * cos(ths)

# pp.plot(t, Us, label='U')
# pp.plot(t, vs, label='v')
pp.plot(t, xs, label='x')
# pp.plot(t, ths, label="th")
# pp.plot(t, Ys, label="th'")
pp.grid(True)
pp.legend()
pp.subplot(212)
pp.plot(t, Es, label='E')
pp.grid(True)
pp.legend()
pp.show()


# Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Sergey Royz'), bitrate=1800)
# ani.save('controlled-cart.mp4', writer=writer)