from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
from math import log2

# K1 = [7.278, 8.610, 10.00, 11.48, 13.00, 14.52, 16.11, 17.66, 19.19]
# K1 = [k * 10**-7 for k in K1]
# K2 = [3.837, 4.864, 6.095, 7.516, 9.183, 11.12, 13.34, 15.89, 18.79]
# K2 = [k * 10**-10 for k in K2]

def k1(T):
    # observe that 0 <= T <= 40 must be fullfilled for this formula to be reasonable!
    return 10**-7 * (0.0006*T**2 + 0.2758*T + 7.2329)

def k2(T):
    # observe that 0 <= T <= 40 must be fullfilled for this formula to be reasonable!
    return 10**-10 * (0.00004*T**3 + 0.00308*T**2 + 0.19092*T + 3.83453)

def omega(T):
    # return hp / (hp**2 + k1(T) * hp + k1(T)*k2(T))
    return 10**14 / (1 + 1/k1(T) + 1/(k1(T) * k2(T)))

def dF(N, N0):
    return 3.6 * log2(N / N0) + 343 * 10**3

def diffeq(y, t, alpha, beta, gamma, delta, mu1, mu2, lmda, N0):
    N1, N2, N3, T1, T2 = y
    dydt = [(-alpha*(N1 - omega(T1)*N2)),
            alpha*(N1 - omega(T1)*N2) - beta*(N2-delta*N3),
            beta*(N2 - delta*N3),
            1/mu1 * (-lmda*T1 - gamma*(T1 - T2) + dF(N1, N0)),
            1/mu2*gamma*(T1 - T2)]
    return dydt

def plot(t, sol):
    plt.figure(1)
    plt.plot(t, sol[:, 0], 'r', label='N1(t)')
    plt.plot(t, sol[:, 1], 'g', label='N2(t)')
    plt.plot(t, sol[:, 2], 'b', label='N3(t)')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.grid()

    plt.figure(2)
    plt.plot(t, sol[:, 3], 'r', label='T1(t)')
    plt.plot(t, sol[:, 4], 'g', label='T2(t)')
    plt.xlabel('t')

    plt.legend(loc='best')
    plt.grid()

    print("N1 quota: ", [N1 / 409.8 for N1 in sol[:, 0]])
    print("omega(T1): ", omega(sol[:, 3][0]))

    print("T1 diff: ", sol[:, 3][0] - sol[:, 3][-1])
    print("T2 diff: ", sol[:, 4][0] - sol[:, 4][-1])

    plt.show()

def main():
    rho = 1
    cp = 4183
    H1 = 150
    H2 = 2400

    alpha = 1/5 # made up -> should come from Henry's law
    beta = 1/20 # made up
    gamma = 1.6 * 10**4 # from here: https://link.springer.com/article/10.1007/s003820000059, above eq. (7) (called k there)
    delta = H1/H2
    mu1 = rho * cp * H1
    mu2 = rho * cp * H2
    lmda = 0 # lmao just let it be zero for now
    N0 = 409.8 # mean value 2019 (wikipedia)

    N1_0 = N0*2
    N2_0 = N0/2
    N3_0 = N0/2
    T1_0 = 25
    T2_0 = 5
    t0 = 0
    tf = 100
    y0 = [N1_0, N2_0, N3_0, T1_0, T2_0]
    t = np.linspace(t0, tf, 1001)
    sol = odeint(diffeq, y0, t, args=(alpha, beta, gamma, delta, mu1, mu2, lmda, N0))
    plot(t, sol)

main()
