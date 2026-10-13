"""CRB with a flat background: lambda(x) = N [phi(x) + beta], x in units of the
line width sigma, window |x| <= 6.  Fisher information per unit N for the centre
(mu) and width (sigma) relative to the background-free value; the ratio
I(0)/I(beta) is the variance inflation factor (1 + c B/N).  Background total
B = beta * N * 12 over the window.  Also the number of pixels (48) does not
matter in the continuous limit.  ASCII output."""
import numpy as np
x = np.linspace(-6, 6, 24001); dx = x[1] - x[0]
phi = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
dmu = x * phi            # d phi / d mu  (sigma = 1)
dsig = (x**2 - 1) * phi  # d phi / d sigma
I0_mu = np.sum(dmu**2 / phi) * dx; I0_sig = np.sum(dsig**2 / phi) * dx
print("background-free: I_mu=%.4f (expect 1) I_sig=%.4f (expect 2)" % (I0_mu, I0_sig))
print("beta   B/N    infl_mu  infl_sig  c_mu=(infl_mu-1)/(B/N)  c_sig")
for beta in (0.001, 0.003, 0.01, 0.03, 0.1):
    Imu = np.sum(dmu**2 / (phi + beta)) * dx; Isig = np.sum(dsig**2 / (phi + beta)) * dx
    BN = beta * 12.0
    im, isg = I0_mu / Imu, I0_sig / Isig
    print("%.3f  %.3f  %.4f  %.4f   %.3f   %.3f" % (beta, BN, im, isg, (im - 1) / BN, (isg - 1) / BN))
