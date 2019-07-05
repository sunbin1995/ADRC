
import numpy as np
import matplotlib.pyplot as plt
import ADRC


def TD_fhan(x1,x2,r0,h,track_signal):

    # TD 跟踪微分器
    fh = ADRC.fhan(x1[i] - track_signal, x2[i], r0, h)
    x1[i + 1] = x1[i] + h * x2[i]
    x2[i + 1] = x2[i] + h * fh


N = 6000
h = 0.001
y1=2

t = np.zeros(N)
x1 = np.zeros(N)
x2 = np.zeros(N)
x3 = np.zeros(N)
x4 = np.zeros(N)
x5 = np.zeros(N)
x6 = np.zeros(N)


for i in range(N-1):
    t[i+1] = i * h    # 时间轴

    TD_fhan(x1, x2, 2, 0.001, track_signal=y1)
    TD_fhan(x3, x4, 4, 0.001, track_signal=y1)
    TD_fhan(x5, x6, 1, 0.001, track_signal=y1)

plt.figure('Tarcking_Show')
p1 = plt.subplot(111)

p1.plot(t, x5, 'g', linewidth=2,label='r=1')
p1.plot(t, x6, 'g', linewidth=2)
p1.plot(t, x1, 'r', linewidth=2,label='r=2')
p1.plot(t, x2, 'r', linewidth=2)
p1.plot(t, x3, 'b', linewidth=2,label='r=4')
p1.plot(t, x4, 'b', linewidth=2)

p1.set_ylim(0, 3.25)
p1.set_xlim(0, 6)
plt.legend(loc='lower right')
p1.grid(True)
p1.set_xlabel('t(s)', fontsize = 14)
p1.set_ylabel('y', fontsize = 14)
p1.set_title('TD', fontsize = 14)
plt.show()
