# 🧬 Neural Architecture Evolution

**Evolve neural network architectures using genetic algorithms!**

Inspired by biological evolution, this project automatically discovers effective neural network architectures without manual design. Instead of hand-crafting networks, let evolution find the optimal structure!

## 🌟 Key Idea

Just like nature evolves organisms through mutation and selection:
- **Genomes** = Neural network architectures (layers, sizes)
- **Mutation** = Change layer sizes, add/remove layers
- **Crossover** = Combine successful architectures
- **Selection** = Keep architectures that perform best
- **Evolution** = Better architectures emerge over generations!

## 🎯 Why This Matters

Current AI development:
- Humans design architectures (transformer, CNN, etc.)
- Scale them bigger and bigger
- Limited by human creativity

This approach:
- **Automated architecture search**
- Discovers structures humans might not design
- Can find task-specific optimal architectures
- Computationally efficient (quick fitness evaluation)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd neural-evolution

# Install dependencies (just numpy!)
pip install numpy
```

### Run Demo

```bash
python evolve.py
```

This will:
1. Create a simple regression task (sum numbers)
2. Evolve neural architectures for 20 generations
3. Show the best architecture discovered
4. Compare to baseline

### Example Output

```
🧬 NEURAL ARCHITECTURE EVOLUTION
============================================================
Population: 30
Generations: 20
Input size: 5, Output size: 1
Training samples: 200

Gen   1 | Best: [32] | Fitness: 0.3845 | Loss: 1.6012 | Avg: 0.2456
Gen   2 | Best: [16, 32] | Fitness: 0.4123 | Loss: 1.4251 | Avg: 0.2789
Gen   3 | Best: [32, 64, 32] | Fitness: 0.4567 | Loss: 1.1895 | Avg: 0.3012
...
Gen  20 | Best: [32, 64, 32, 16] | Fitness: 0.5234 | Loss: 0.9103 | Avg: 0.4123

============================================================
EVOLUTION COMPLETE!

Best architecture: [32, 64, 32, 16]
✅ Improvement over baseline: 23.4%
```

## 📊 Use Cases

### 1. Time Series Prediction
```python
from evolve import ArchitectureEvolution, EvolutionConfig

# Your time series data
X_train = ...  # shape (samples, sequence_length)
y_train = ...  # shape (samples, output_dim)

config = EvolutionConfig(
    population_size=50,
    generations=30,
    max_layers=5
)

evolution = ArchitectureEvolution(config)
best = evolution.evolve(X_train, y_train, 
                       input_size=sequence_length,
                       output_size=output_dim)

print(f"Best architecture: {best}")
```

### 2. Financial Prediction
Automatically find the best architecture for stock price prediction, market forecasting, etc.

### 3. Custom Tasks
The evolution can adapt to any supervised learning task!

## 🧬 How It Works

### Genome Representation
```python
# A genome is just a list of hidden layer sizes
genome = [32, 64, 32]  # 3 hidden layers

# Becomes network: Input → 32 → 64 → 32 → Output
```

### Evolution Operations

**Mutation:**
- Change layer size: `[32, 64]` → `[32, 128]`
- Add layer: `[32, 64]` → `[32, 64, 32]`
- Remove layer: `[32, 64, 32]` → `[32, 32]`

**Crossover:**
```python
Parent A: [32, 64, 32]
Parent B: [16, 128, 64, 16]
         ↓ (split and combine)
Child:    [32, 64, 128, 16]
```

**Selection:**
- Tournament selection: Pick k random, keep best
- Elitism: Always keep top performers
- Fitness: 1 / (1 + loss)

### Algorithm
```
1. Initialize random population
2. For each generation:
   a. Evaluate fitness (train each network)
   b. Select best performers
   c. Create offspring (crossover + mutation)
   d. Replace population
3. Return best architecture
```

## 🔧 Configuration

```python
config = EvolutionConfig(
    population_size=50,      # Number of architectures per generation
    generations=30,          # Number of evolution cycles
    mutation_rate=0.4,       # Probability of mutation
    crossover_rate=0.6,      # Probability of crossover
    elite_size=5,            # Top N to keep unchanged
    tournament_size=3,       # Tournament selection size
    max_layers=5,            # Maximum network depth
    layer_sizes=[8,16,32,64,128]  # Possible layer sizes
)
```

## 📈 Results & Insights

From experiments, evolution typically discovers:

1. **Depth matters**: Often finds 3-4 layer networks vs single layer
2. **Optimal sizing**: Converges on specific layer sizes (often 32, 64)
3. **Task-specific**: Different tasks → different optimal architectures
4. **Efficiency**: Finds good architectures in 15-30 generations

## 🎓 Biological Inspiration

| Biology | Neural Evolution |
|---------|-----------------|
| DNA | Network architecture |
| Mutation | Layer changes |
| Crossover | Architecture mixing |
| Fitness | Network performance |
| Selection | Keep best performers |
| Speciation | Population diversity |

## 🔬 Advanced Usage

### Custom Fitness Function

```python
def custom_fitness(genome, X, y):
    # Create network
    network = SimpleNetwork([input_size] + genome.layers + [output_size])
    loss = network.train(X, y)
    
    # Custom fitness (e.g., balance accuracy and size)
    parameter_count = sum(genome.layers)
    efficiency = 1.0 / (loss * parameter_count)
    
    return efficiency

# Use in evolution...
```

### Save/Load Best Architectures

```python
# Save evolution history
evolution.save_history('evolution_history.json')

# Best architecture is in evolution.population[0]
best = evolution.population[0]
print(best.to_dict())
```

## 🚧 Limitations & Future Work

**Current Limitations:**
- Uses simple training (not full backprop)
- Only fully-connected layers
- CPU only (no GPU)
- Limited to supervised learning

**Potential Extensions:**
- Add CNN/RNN/Attention blocks
- Use proper gradient descent
- Multi-objective optimization (accuracy + speed + size)
- GPU acceleration
- Transfer learning integration
- AutoML integration

## 🤝 Contributing

Ideas for contributions:
1. Add more layer types (Conv, LSTM, Attention)
2. Implement proper backpropagation
3. Add GPU support
4. Create visualization tools
5. Benchmark on standard datasets
6. Add more evolution strategies

## 📚 References

- [NEAT: NeuroEvolution of Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
- [Genetic Algorithms in Search, Optimization and Machine Learning](https://www.goodreads.com/book/show/142613.Genetic_Algorithms_in_Search_Optimization_and_Machine_Learning)

## 📜 License

MIT License - feel free to use and modify!

## 🎉 Try It Now!

### Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Click above to run in Google Colab (no installation needed!)

### Local Setup
```bash
python evolve.py
```

---

**Made with ❤️ and evolution** 🧬
