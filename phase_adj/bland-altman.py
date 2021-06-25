import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(ax, data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean  = np.mean([data1, data2], axis=0)
    diff  = data1 - data2
    md    = np.mean(diff)
    sd    = np.std(diff, axis=0)
    top   = md + 1.96*sd
    bot   = md - 1.96*sd

    ax.scatter(mean, diff, *args, **kwargs)
    ax.axhline(md,  color='gray', linestyle='--')
    ax.axhline(top, color='red',  linestyle='--')
    ax.axhline(bot, color='red',  linestyle='--')

    ax.set_title('$H_{CPU}$ vs. $H_{GPU}$ (Bland-Altman)')
    ax.set_xlabel('Entropy Mean')
    ax.set_ylabel('Entropy Difference')

    ax.annotate('$\mu = %f$' % md,             xy=(8.7,  md+0.001), horizontalalignment='right')
    ax.annotate('$\mu+1.96\sigma = %f$' % top, xy=(8.7, top+0.001), horizontalalignment='right')
    ax.annotate('$\mu-1.96\sigma = %f$' % bot, xy=(8.7, bot+0.001), horizontalalignment='right')

fig, ax = plt.subplots(1,1)
# put entropy results for all datasets in these arrays.
# cpu results
x1 = [8.566987,
      5.349163,
      7.329765,
      7.048244,
      7.046798,
      7.058389,
      7.064681,
      7.056219,
      4.813764,
      7.183409,
      6.267911,
      6.357502,
      7.314604,
      7.182136,
      6.731287,
      5.472880,
      6.267911]
# gpu results
x2 = [8.562888,
      5.316638,
      7.312196,
      7.040686,
      7.026342,
      7.051334,
      7.061121,
      7.055362,
      4.813372,
      7.179878,
      6.253969,
      6.355861,
      7.314502,
      7.169786,
      6.716239,
      5.471442,
      6.253969]
bland_altman_plot(ax, x1, x2)
plt.show()
