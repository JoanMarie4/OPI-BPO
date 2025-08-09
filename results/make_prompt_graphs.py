
import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['Raw', 'Formatted', 'Formatted-Alt-Op']
prompt_lengths = [358.6, 403.5, 401.9]
optimized_prompt_lengths = [722.8, 694.0, 433.6]
token_counts = [81.6, 91.3, 90.9]
optimized_token_counts = [143.7, 141.2, 89.8]

# X locations for groups
x = np.arange(len(datasets))
width = 0.35  # width of the bars

# === Plot 1: Prompt Lengths ===
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, prompt_lengths, width, label='Original Prompt Length')
plt.bar(x + width/2, optimized_prompt_lengths, width, label='Optimized Prompt Length')

plt.ylabel('Characters')
plt.title('Prompt Length Comparison')
plt.xticks(x, datasets)
plt.legend()
plt.tight_layout()
plt.savefig('prompt_lengths.png')

# === Plot 2: Token Counts ===
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, token_counts, width, label='Original Prompt Tokens')
plt.bar(x + width/2, optimized_token_counts, width, label='Optimized Prompt Tokens')

plt.ylabel('Tokens')
plt.title('Token Count Comparison')
plt.xticks(x, datasets)
plt.legend()
plt.tight_layout()
plt.savefig('token_counts.png')
