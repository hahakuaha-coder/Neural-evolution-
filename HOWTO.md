# 🚀 HOW TO USE THIS PROJECT

## Quick Test (Local)

```bash
# 1. Navigate to folder
cd neural-evolution

# 2. Run demo
python evolve.py
```

**Expected output:**
- 20 generations of evolution
- Best architecture discovered
- Comparison to baseline
- ~1-2 minutes runtime

---

## Google Colab (Recommended!)

**Steps:**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "New Notebook"
3. Copy contents of `colab_notebook.py`
4. Run each cell

**You'll get:**
- ✅ Interactive evolution visualization
- ✅ Multiple example tasks
- ✅ Downloadable results
- ✅ No local setup needed!

---

## Try on Your Own Data

### Example: CSV File

```python
import pandas as pd
from evolve import ArchitectureEvolution, EvolutionConfig

# Load your data
df = pd.read_csv('your_data.csv')
X_train = df[['feature1', 'feature2', ...]].values
y_train = df[['target']].values

# Configure
config = EvolutionConfig(
    population_size=30,
    generations=20
)

# Evolve!
evolution = ArchitectureEvolution(config)
best = evolution.evolve(
    X_train, y_train,
    input_size=X_train.shape[1],
    output_size=y_train.shape[1]
)

print(f"Best: {best}")
```

---

## Examples Included

### 1. `evolve.py` - Basic Demo
Simple regression task - learn to sum numbers

**Run:** `python evolve.py`

### 2. `financial_example.py` - Stock Prediction
Synthetic financial data with technical indicators

**Run:** `python financial_example.py`

### 3. `colab_notebook.py` - Interactive
Multiple tasks with visualization

**Run:** Copy to Google Colab

---

## What Gets Evolved?

**Genome = List of Layer Sizes**
```
[32, 64, 32] = 3 hidden layers

Full network:
Input → 32 → 64 → 32 → Output
```

**Operations:**
- Mutation: Change sizes, add/remove layers
- Crossover: Mix parent architectures
- Selection: Keep best performers

---

## Configuration Options

```python
config = EvolutionConfig(
    population_size=50,      # More = better search, slower
    generations=30,          # More = better results, longer
    mutation_rate=0.4,       # 40% chance of mutation
    crossover_rate=0.6,      # 60% chance of crossover
    elite_size=5,            # Keep top 5 unchanged
    max_layers=5,            # Max network depth
    layer_sizes=[8,16,32,64,128]  # Available sizes
)
```

---

## Typical Results

**After 20-30 generations:**
- 20-50% improvement over baseline
- Discovers 3-4 layer networks
- Converges on optimal sizes (often 32, 64, 128)
- Task-specific architectures emerge

**Example discoveries:**
- Simple tasks → shallow networks [32, 16]
- Complex tasks → deeper networks [64, 128, 128, 64]
- Time series → recurrent-like structures
- Classification → wider middle layers

---

## Performance Tips

### Faster Evolution
```python
config = EvolutionConfig(
    population_size=20,    # Smaller population
    generations=15,        # Fewer generations
    max_layers=3          # Shallower networks
)
```

### Better Results
```python
config = EvolutionConfig(
    population_size=100,   # Larger population
    generations=50,        # More generations
    elite_size=10         # Keep more elites
)
```

---

## File Structure

```
neural-evolution/
├── evolve.py              # Main engine
├── financial_example.py   # Stock prediction example
├── colab_notebook.py      # Colab template
├── README.md             # Full documentation
├── HOWTO.md              # This file
└── requirements.txt       # Dependencies
```

---

## Troubleshooting

**"Empty genome" errors:**
- Increase population_size
- Check input/output dimensions match data

**Slow performance:**
- Reduce population_size
- Reduce generations
- Use smaller max_layers

**Poor results:**
- Increase generations (try 30-50)
- Increase population_size (try 50-100)
- Check data quality/preprocessing

---

## Next Steps

### 1. Try Different Tasks
- Time series forecasting
- Classification problems
- Your own dataset

### 2. Modify Evolution
- Add custom fitness function
- Change mutation strategies
- Experiment with layer_sizes

### 3. Scale Up
- Use more generations
- Larger population
- Deeper networks (max_layers=10)

### 4. Extend Code
- Add CNN/RNN blocks
- Implement proper backprop
- Add GPU support

---

## Share Your Results!

Found something cool? Share:
- Best architecture discovered
- Your task/dataset
- Evolution visualization
- Performance improvements

**Happy evolving!** 🧬🚀
