
from math import *
from pylab import *
import numpy as np

def fal(e, alfa, delta):

    s = (np.sign(e+delta)-np.sign(e-delta))/2.0
    fal_out = e * s/(pow(delta, 1-alfa))+pow(abs(e), alfa)*np.sign(e)*(1-s)
    return fal_out

def fsg(x, d):

    fsg_out = (np.sign(x+d) - np.sign(x-d))/2.0
    return fsg_out


def fhan(x1, x2, r, h0):

    d = r * h0 * h0
    a0 =h0 * x2
    y =x1+a0
    a1 = sqrt(d * (d + 8 * abs(y)))
    a2 = a0 + np.sign(y) * (a1-d)/2.0
    a = (a0 + y) * fsg(y, d) + a2 * (1 - fsg(y, d))
    fhan_out = -1 * r * (a / d) * fsg(a, d) - r * np.sign(a) * (1 - fsg(a, d))
    return fhan_out

'''
h = 0.001
N = 6000
x1 = np.zeros(N)
x2 = np.zeros(N)
x3 = np.zeros(N)
x4 = np.zeros(N)
x5 = np.zeros(N)
x6 = np.zeros(N)
u =  []
error=[]

xx1 = np.zeros(N)
xx2 = np.zeros(N)
v11 = np.zeros(N)
v12 = np.zeros(N)
v21 = np.zeros(N)
v22 = np.zeros(N)
z1 = np.zeros(N)
z2 = np.zeros(N)
z3 = np.zeros(N)
z21 = np.zeros(N)
z22 = np.zeros(N)
z23 = np.zeros(N)
t = np.zeros(N)
u1 = np.zeros(N)
u2 = np.zeros(N)
uu1 = np.zeros(N)
uu2 = np.zeros(N)
y1 = np.zeros(N)
y2 = np.zeros(N)
f1 = np.zeros(N)
f2 = np.zeros(N)
y1_set = np.zeros(N)
y2_set = np.zeros(N)

BB = np.zeros([2, 2])

b11 = 3
b12 = 1
b21 = 3
b22 = 2

r = 100
r0 = 2

b01 = 0.1
b02 = 0.5
b03 = 1


h1 = 0.1
c = 1.0


y1_star = 2
y2_star = 1


def TD_fhan(x1,x2,r0,h,track_signal):

    # TD 跟踪微分器
    fh = fhan(x1[i] - track_signal, x2[i], r0, h)
    x1[i + 1] = x1[i] + h * x2[i]
    x2[i + 1] = x2[i] + h * fh



for i in range(N-1):
    t[i+1] = i * h    # 时间轴

    #y1_set[i] = cos(t[i])
    #y2_set[i] = 2 * sin(t[i])

    TD_fhan(x1, x2, 2, 0.001, track_signal=y1_star)
    TD_fhan(x3, x4, 4, 0.001, track_signal=y1_star)
    TD_fhan(x5, x6, 1, 0.001, track_signal=y1_star)

    e = z1[i] - x1[i]

    fe = fal(e, 0.5, h)
    fe1 = fal(e, 1, h)

    z1[i + 1] = z1[i] + h * (z2[i] - b01 * e)
    z2[i + 1] = z2[i] + h * (z3[i] - b02 * fe + u1[i])
    z3[i + 1] = z3[i] + h * (-b03 * fe1)

    e1 = x1[i] - z1[i]
    e2 = x2[i] - z2[i]

    u1[i + 1] = -fhan(e1, c * e2, r, h1) - z3[i]

plt.figure('Tarcking_Show')
p1 = plt.subplot(411)
p2 = plt.subplot(412)
p3 = plt.subplot(413)
p4 = plt.subplot(414)

p1.plot(t, x5, 'g', linewidth=2,label='r=1')
p1.plot(t, x6, 'g', linewidth=2)
p1.plot(t, x1, 'r', linewidth=2,label='r=2')
p1.plot(t, x2, 'r', linewidth=2)
p1.plot(t, x3, 'b', linewidth=2,label='r=4')
p1.plot(t, x4, 'b', linewidth=2)


p1.plot(t, x5, 'g', linewidth=2,label='h=0.0005')
p1.plot(t, x1, 'r', linewidth=2,label='h=0.001')
#p1.plot(t, x2, 'b', linewidth=1)
p1.plot(t, x3, 'b', linewidth=2,label='h=0.002')
#p1.plot(t, x4, 'b', linewidth=1)

p1.set_ylim(0, 3.25)
p1.set_xlim(0, 6)
plt.legend(loc='lower right')
p1.grid(True)
p1.set_xlabel('t(s)', fontsize = 14)
p1.set_ylabel('y', fontsize = 14)
p1.set_title('TD', fontsize = 14)


p2.plot(t, z1, 'm', t, z2, 'g',t,z3,'y', linewidth=0.5)
p2.grid(True)
p2.set_xlabel('t', fontsize = 14)
p2.set_ylabel('y', fontsize = 14)
p2.set_title('ESO', fontsize = 14)


p3.plot(t, u1, 'm', linewidth=1)
p3.grid(True)
p3.set_xlabel('t', fontsize = 14)
p3.set_ylabel('y', fontsize = 14)
p3.set_title('U', fontsize = 14)




p4.plot(t,f2,'m',t,z23,'g',linewidth=2)
p4.grid(True)
p4.set_xlabel('t',fontsize = 14)
p4.set_ylabel('y',fontsize = 14)
p4.set_title('ESO Tracking Performance f2',fontsize = 14)

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,hspace=0.35, wspace=0.3)


plt.show()
'''