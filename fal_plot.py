
import numpy as np
import matplotlib.pyplot as plt
import ADRC

u = []
z = []
o = []
p = []

error = []



'''
for i in np.arange(-10,10,0.001):
    error.append(i)
    u.append(fal(i,0.5,0.01))
    z.append(fal(i,0.25,0.01))
    o.append(fal(i,1.5,0.01))
    p.append(fal(i,1,0.01))
'''
for i in np.arange(-10,10,0.001):
    error.append(i)
    z.append(ADRC.fal(i, 0.5, 0.01))
    u.append(ADRC.fal(i, 0.5, 0.05))
    p.append(ADRC.fal(i, 0.5, 0.1))
    o.append(ADRC.fal(i, 0.5, 0.2))

plt.figure('Fal')
p1 = plt.subplot(111)

p1.plot(error, z, 'r', linewidth=2, label='delta=0.01')
p1.plot(error, u, 'g', linewidth=2, label='delta=0.05')
p1.plot(error, p, 'y', linewidth=2, label='delta=0.1')
p1.plot(error, o, 'b', linewidth=2, label='delta=0.2')

p1.set_ylim(-1,1)
p1.set_xlim(-0.5, 0.5)
plt.legend(loc='lower right')
p1.grid(True)
p1.set_xlabel('e', fontsize=14)
p1.set_ylabel('y', fontsize=14)
p1.set_title('Fal', fontsize=14)


plt.show()
