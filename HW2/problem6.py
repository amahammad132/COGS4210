import numpy as np
import matplotlib.pyplot as plt

# Base parameters
PRIOR       = 0.01
SENSITIVITY = 0.95
SPECIFICITY = 0.90

def posterior(prior, sensitivity, specificity):
    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
    return (sensitivity * prior) / p_pos

BASE_POST = posterior(PRIOR, SENSITIVITY, SPECIFICITY)

x = np.linspace(0.001, 0.999, 500)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Bayesian Sensitivity Analysis — Diagnostic Test for Disease X", fontsize=14)

# Plot 1: Posterior vs Prior
axes[0].plot(x, posterior(x, SENSITIVITY, SPECIFICITY))
axes[0].axvline(PRIOR, color='red', linestyle='--', label=f'Base prior = {PRIOR:.0%}')
axes[0].axhline(BASE_POST, color='gray', linestyle=':', label=f'Base posterior = {BASE_POST:.1%}')
axes[0].set_xlabel("Prior Probability P(X)")
axes[0].set_ylabel("P(Disease | Positive Test)")
axes[0].set_title("Posterior vs. Prior")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Plot 2: Posterior vs Sensitivity
axes[1].plot(x, posterior(PRIOR, x, SPECIFICITY), color='orange')
axes[1].axvline(SENSITIVITY, color='red', linestyle='--', label=f'Base sensitivity = {SENSITIVITY:.0%}')
axes[1].axhline(BASE_POST, color='gray', linestyle=':', label=f'Base posterior = {BASE_POST:.1%}')
axes[1].set_xlabel("Sensitivity P(Positive Test|X)")
axes[1].set_ylabel("P(Disease | Positive Test)")
axes[1].set_title("Posterior vs. Sensitivity")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# Plot 3: Posterior vs Specificity
axes[2].plot(x, posterior(PRIOR, SENSITIVITY, x), color='green')
axes[2].axvline(SPECIFICITY, color='red', linestyle='--', label=f'Base specificity = {SPECIFICITY:.0%}')
axes[2].axhline(BASE_POST, color='gray', linestyle=':', label=f'Base posterior = {BASE_POST:.1%}')
axes[2].set_xlabel("Specificity P(Negative Test|not X)")
axes[2].set_ylabel("P(Disease | Positive Test)")
axes[2].set_title("Posterior vs. Specificity")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()