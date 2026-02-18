# Neural Architecture Evolution - Google Colab Notebook
# Copy this into a new Google Colab notebook!

"""
🧬 NEURAL ARCHITECTURE EVOLUTION
==================================
Automatically discover neural network architectures using evolution!

Instructions:
1. Run each cell in order
2. Watch evolution happen in real-time
3. Experiment with different tasks!
"""

# ============================================================
# CELL 1: Setup
# ============================================================

!pip install numpy matplotlib -q

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List

print("✅ Setup complete!")

# ============================================================
# CELL 2: Evolution Engine (copy entire evolve.py here)
# ============================================================

# [Paste the contents of evolve.py here]
# Or use: !wget <raw-github-url-to-evolve.py>

# ============================================================
# CELL 3: Example 1 - Simple Regression
# ============================================================

print("📊 EXAMPLE 1: Sum Numbers Task")
print("=" * 60)

# Create task: learn to sum 3 numbers
np.random.seed(42)
X_train = np.random.randn(200, 5)
y_train = np.sum(X_train[:, :3], axis=1, keepdims=True)

# Configure evolution
config = EvolutionConfig(
    population_size=30,
    generations=25,
    max_layers=4
)

# Run evolution!
evolution = ArchitectureEvolution(config)
best = evolution.evolve(X_train, y_train, input_size=5, output_size=1)

# ============================================================
# CELL 4: Visualize Evolution
# ============================================================

print("\n📈 Evolution Progress")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot fitness over generations
ax1.plot(evolution.history['best_fitness'], 'b-', linewidth=2, label='Best')
ax1.plot(evolution.history['avg_fitness'], 'r--', linewidth=2, label='Average')
ax1.set_xlabel('Generation', fontsize=12)
ax1.set_ylabel('Fitness', fontsize=12)
ax1.set_title('Fitness Over Generations', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot architecture complexity over generations
complexities = [len(g) for g in evolution.history['best_genomes']]
ax2.plot(complexities, 'g-', linewidth=2, marker='o')
ax2.set_xlabel('Generation', fontsize=12)
ax2.set_ylabel('Number of Layers', fontsize=12)
ax2.set_title('Architecture Depth Over Time', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n🏆 Best Architecture: {best}")
print(f"📊 Final Fitness: {best.fitness:.4f}")
print(f"📉 Final Loss: {best.loss:.4f}")

# ============================================================
# CELL 5: Example 2 - Time Series Prediction
# ============================================================

print("\n📊 EXAMPLE 2: Sine Wave Prediction")
print("=" * 60)

# Create time series task: predict next value in sine wave
np.random.seed(43)
t = np.linspace(0, 100, 500)
signal = np.sin(t) + np.random.randn(500) * 0.1

# Create sequences
sequence_length = 10
X_ts = []
y_ts = []
for i in range(len(signal) - sequence_length):
    X_ts.append(signal[i:i+sequence_length])
    y_ts.append(signal[i+sequence_length])

X_ts = np.array(X_ts)
y_ts = np.array(y_ts).reshape(-1, 1)

print(f"Training samples: {len(X_ts)}")
print(f"Sequence length: {sequence_length}")

# Evolve architecture
config = EvolutionConfig(
    population_size=25,
    generations=20,
    max_layers=4,
    layer_sizes=[8, 16, 32, 64]
)

evolution_ts = ArchitectureEvolution(config)
best_ts = evolution_ts.evolve(X_ts, y_ts, 
                               input_size=sequence_length, 
                               output_size=1)

print(f"\n🏆 Best Time Series Architecture: {best_ts}")

# ============================================================
# CELL 6: Example 3 - Classification Task
# ============================================================

print("\n📊 EXAMPLE 3: Binary Classification")
print("=" * 60)

# Create classification task: XOR-like problem
np.random.seed(44)
n_samples = 300

X_class = np.random.randn(n_samples, 2)
y_class = ((X_class[:, 0] > 0) != (X_class[:, 1] > 0)).astype(float).reshape(-1, 1)

# Add noise
y_class = y_class + np.random.randn(n_samples, 1) * 0.1

print(f"Training samples: {n_samples}")
print(f"Features: 2")

# Evolve architecture
config = EvolutionConfig(
    population_size=30,
    generations=20,
    max_layers=3,
    layer_sizes=[4, 8, 16, 32]
)

evolution_class = ArchitectureEvolution(config)
best_class = evolution_class.evolve(X_class, y_class, 
                                    input_size=2, 
                                    output_size=1)

print(f"\n🏆 Best Classification Architecture: {best_class}")

# ============================================================
# CELL 7: Compare All Results
# ============================================================

print("\n" + "=" * 60)
print("📊 FINAL COMPARISON")
print("=" * 60)

results = [
    ("Sum Task", best, evolution.history['best_fitness'][-1]),
    ("Time Series", best_ts, evolution_ts.history['best_fitness'][-1]),
    ("Classification", best_class, evolution_class.history['best_fitness'][-1])
]

for task, genome, fitness in results:
    print(f"\n{task}:")
    print(f"  Architecture: {genome.layers}")
    print(f"  Layers: {len(genome.layers)}")
    print(f"  Total params: ~{sum(genome.layers)}")
    print(f"  Final fitness: {fitness:.4f}")

# ============================================================
# CELL 8: Experiment - YOUR TURN!
# ============================================================

print("\n🎮 YOUR TURN TO EXPERIMENT!")
print("=" * 60)
print("""
Try modifying:
1. population_size - more = better exploration, slower
2. generations - more = better results, takes longer
3. max_layers - deeper networks possible
4. layer_sizes - available layer sizes

Create your own task:
- Financial data prediction
- Your own dataset
- Different problem types

Example:
    X_custom = ...  # your input data
    y_custom = ...  # your target data
    
    config = EvolutionConfig(
        population_size=40,
        generations=30
    )
    
    evolution = ArchitectureEvolution(config)
    best = evolution.evolve(X_custom, y_custom, 
                           input_size=X_custom.shape[1],
                           output_size=y_custom.shape[1])
""")

# YOUR CODE HERE
# X_custom = ...
# y_custom = ...

# ============================================================
# CELL 9: Save Results
# ============================================================

from google.colab import files

# Save evolution history
evolution.save_history('evolution_history.json')
print("✅ Saved evolution_history.json")

# Save best architecture
best_arch = {
    'architecture': best.layers,
    'fitness': best.fitness,
    'loss': best.loss
}

import json
with open('best_architecture.json', 'w') as f:
    json.dump(best_arch, f, indent=2)

print("✅ Saved best_architecture.json")

# Download files
files.download('evolution_history.json')
files.download('best_architecture.json')

print("\n🎉 Experiment complete! Files downloaded.")
print("Share your results! 🚀")
