import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

gpt_params = np.array([1_414_848, 5_062_080, 20_316_912, 50_102_640, 107_905_824])
gpt_loss = np.array([0.6684, 0.5730, 0.5265, 0.5106, 0.5027])

rnn_params = np.array([1_102_433, 3_622_753, 10_747_105, 42_182_753])
rnn_loss = np.array([0.6534, 0.6029, 0.6018, 0.6308])

def power_law(n, a, alpha, c):
    return a * (n ** -alpha) + c

p0 = [10, 0.1, 0.4]
try:
    popt_gpt, _ = curve_fit(power_law, gpt_params, gpt_loss, p0=p0, maxfev=10000)
    fit_success = True
except:
    fit_success = False

plt.figure(figsize=(7, 5))
plt.loglog(gpt_params, gpt_loss, 'bo-', markersize=8, linewidth=2, label='GPT Data')

if fit_success:
    x_space = np.geomspace(min(gpt_params) * 0.8, max(gpt_params) * 1.2, 100)
    plt.loglog(
        x_space,
        power_law(x_space, *popt_gpt),
        'k--',
        alpha=0.6,
        label=f'Fit ($\\alpha={popt_gpt[1]:.3f}$)'
    )

plt.xlabel('Parameters (N)')
plt.ylabel('Validation Loss')
plt.title('Part 2: Transformer Scaling Laws')
plt.grid(True, which="major", ls="-", alpha=0.4)
plt.grid(True, which="minor", ls=":", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('gpt_scaling.png', dpi=300)
plt.show()

plt.figure(figsize=(7, 5))
plt.loglog(rnn_params, rnn_loss, 'rs--', markersize=8, linewidth=2, label='LSTM Data')

plt.xlabel('Parameters (N)')
plt.ylabel('Validation Loss')
plt.title('Part 3: RNN Scaling Limits')
plt.grid(True, which="major", ls="-", alpha=0.4)
plt.grid(True, which="minor", ls=":", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('rnn_scaling.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.loglog(gpt_params, gpt_loss, 'bo-', markersize=8, linewidth=2, label='Transformer (GPT)')

if fit_success:
    plt.loglog(x_space, power_law(x_space, *popt_gpt), 'b--', alpha=0.3)

plt.loglog(rnn_params, rnn_loss, 'rs--', markersize=8, linewidth=2, label='Recurrent NN (LSTM)')

plt.xlabel('Number of Parameters (N)', fontweight='bold')
plt.ylabel('Validation Loss (L)', fontweight='bold')
plt.title('Comparison: Transformer vs. RNN', fontsize=14)
plt.grid(True, which="major", ls="-", alpha=0.4)
plt.grid(True, which="minor", ls=":", alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig('comparison.png', dpi=300)
plt.show()
