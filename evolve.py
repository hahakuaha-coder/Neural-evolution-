"""
Neural Architecture Evolution
==============================
Evolutionary algorithm to discover neural network architectures.

Inspired by biological evolution and genetic algorithms.
"""

import numpy as np
import random
from typing import List, Tuple, Callable
import json
from dataclasses import dataclass, asdict


@dataclass
class EvolutionConfig:
    """Configuration for evolution"""
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.4
    crossover_rate: float = 0.6
    elite_size: int = 5
    tournament_size: int = 3
    max_layers: int = 5
    layer_sizes: List[int] = None
    
    def __post_init__(self):
        if self.layer_sizes is None:
            self.layer_sizes = [8, 16, 32, 64, 128]


class NeuralGenome:
    """Represents a neural network architecture as a genome"""
    
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.fitness = 0.0
        self.loss = float('inf')
    
    def __str__(self):
        return f"[{' → '.join(map(str, self.layers))}]"
    
    def __repr__(self):
        return f"NeuralGenome({self.layers}, fitness={self.fitness:.4f})"
    
    def to_dict(self):
        return {
            'layers': self.layers,
            'fitness': self.fitness,
            'loss': self.loss
        }


class SimpleNetwork:
    """Simple neural network that can be trained quickly"""
    
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.weights = []
        self._initialize_weights()
    
    def _initialize_weights(self):
        for i in range(len(self.layers) - 1):
            W = np.random.randn(self.layers[i], self.layers[i+1]) * 0.1
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append((W, b))
    
    def forward(self, X):
        h = X
        for W, b in self.weights[:-1]:
            h = np.tanh(h @ W + b)
        # Linear output layer
        W, b = self.weights[-1]
        h = h @ W + b
        return h
    
    def train(self, X, y, iterations=100, lr=0.01):
        """Simple evolutionary training - perturb weights toward better loss"""
        best_loss = float('inf')
        best_weights = None
        patience = 0
        
        for _ in range(iterations):
            pred = self.forward(X)
            loss = np.mean((pred - y) ** 2)
            
            if loss < best_loss:
                best_loss = loss
                best_weights = [(W.copy(), b.copy()) for W, b in self.weights]
                patience = 0
            else:
                patience += 1
                if patience > 20:
                    break
            
            # Perturb weights
            for i in range(len(self.weights)):
                W, b = self.weights[i]
                self.weights[i] = (
                    W + np.random.randn(*W.shape) * lr,
                    b + np.random.randn(*b.shape) * lr
                )
        
        if best_weights:
            self.weights = best_weights
        
        return best_loss


class ArchitectureEvolution:
    """Main evolution engine"""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.population: List[NeuralGenome] = []
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_genomes': []
        }
    
    def _random_genome(self) -> NeuralGenome:
        """Create random genome"""
        n_layers = random.randint(1, self.config.max_layers)
        layers = [random.choice(self.config.layer_sizes) for _ in range(n_layers)]
        return NeuralGenome(layers)
    
    def _mutate(self, genome: NeuralGenome) -> NeuralGenome:
        """Mutate a genome"""
        layers = genome.layers.copy()
        
        mutation = random.choice(['change', 'add', 'remove'])
        
        if mutation == 'change' and layers:
            idx = random.randint(0, len(layers) - 1)
            layers[idx] = random.choice(self.config.layer_sizes)
        
        elif mutation == 'add' and len(layers) < self.config.max_layers:
            layers.append(random.choice(self.config.layer_sizes))
        
        elif mutation == 'remove' and len(layers) > 1:
            layers.pop(random.randint(0, len(layers) - 1))
        
        return NeuralGenome(layers)
    
    def _crossover(self, parent1: NeuralGenome, parent2: NeuralGenome) -> NeuralGenome:
        """Crossover two genomes"""
        if not parent1.layers or not parent2.layers:
            return parent1
        
        split1 = random.randint(0, len(parent1.layers))
        split2 = random.randint(0, len(parent2.layers))
        
        child_layers = parent1.layers[:split1] + parent2.layers[split2:]
        return NeuralGenome(child_layers)
    
    def _tournament_selection(self) -> NeuralGenome:
        """Select genome using tournament selection"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda g: g.fitness)
    
    def evaluate_fitness(self, genome: NeuralGenome, 
                        X_train, y_train, 
                        input_size: int, output_size: int) -> float:
        """Evaluate genome fitness"""
        try:
            layers = [input_size] + genome.layers + [output_size]
            network = SimpleNetwork(layers)
            loss = network.train(X_train, y_train, iterations=100)
            
            genome.loss = loss
            genome.fitness = 1.0 / (1.0 + loss)
            return genome.fitness
        
        except Exception as e:
            genome.fitness = 0.0
            genome.loss = float('inf')
            return 0.0
    
    def evolve(self, X_train, y_train, 
               input_size: int, output_size: int,
               verbose: bool = True):
        """Run evolution"""
        
        # Initialize population
        self.population = [self._random_genome() 
                          for _ in range(self.config.population_size)]
        
        if verbose:
            print("🧬 NEURAL ARCHITECTURE EVOLUTION")
            print("=" * 60)
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {self.config.generations}")
            print(f"Input size: {input_size}, Output size: {output_size}")
            print(f"Training samples: {len(X_train)}\n")
        
        for gen in range(self.config.generations):
            # Evaluate population
            for genome in self.population:
                self.evaluate_fitness(genome, X_train, y_train, 
                                    input_size, output_size)
            
            # Sort by fitness
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            
            # Track history
            best = self.population[0]
            avg_fitness = np.mean([g.fitness for g in self.population])
            
            self.history['best_fitness'].append(best.fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_genomes'].append(best.layers.copy())
            
            if verbose:
                print(f"Gen {gen+1:3d} | Best: {best} | "
                      f"Fitness: {best.fitness:.4f} | Loss: {best.loss:.4f} | "
                      f"Avg: {avg_fitness:.4f}")
            
            # Create next generation
            next_gen = []
            
            # Elitism - keep best
            next_gen.extend(self.population[:self.config.elite_size])
            
            # Generate offspring
            while len(next_gen) < self.config.population_size:
                if random.random() < self.config.crossover_rate:
                    parent1 = self._tournament_selection()
                    parent2 = self._tournament_selection()
                    child = self._crossover(parent1, parent2)
                else:
                    parent = self._tournament_selection()
                    child = self._mutate(parent)
                
                next_gen.append(child)
            
            self.population = next_gen
        
        # Final evaluation
        for genome in self.population:
            self.evaluate_fitness(genome, X_train, y_train, 
                                input_size, output_size)
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        if verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION COMPLETE!")
            print(f"\nBest architecture: {self.population[0]}")
        
        return self.population[0]
    
    def save_history(self, filename: str):
        """Save evolution history"""
        with open(filename, 'w') as f:
            json.dump(self.history, f, indent=2)


def demo():
    """Quick demo"""
    # Create simple regression task
    np.random.seed(42)
    random.seed(42)
    
    X_train = np.random.randn(200, 5)
    y_train = np.sum(X_train[:, :3], axis=1, keepdims=True)
    
    # Run evolution
    config = EvolutionConfig(
        population_size=30,
        generations=20,
        max_layers=4
    )
    
    evolution = ArchitectureEvolution(config)
    best_genome = evolution.evolve(X_train, y_train, input_size=5, output_size=1)
    
    # Test best architecture
    print(f"\nTesting best architecture...")
    layers = [5] + best_genome.layers + [1]
    network = SimpleNetwork(layers)
    loss = network.train(X_train, y_train, iterations=200)
    print(f"Final loss with more training: {loss:.4f}")
    
    # Compare to baseline
    baseline = SimpleNetwork([5, 16, 1])
    baseline_loss = baseline.train(X_train, y_train, iterations=200)
    print(f"\nBaseline [16] loss: {baseline_loss:.4f}")
    print(f"Evolved {best_genome.layers} loss: {loss:.4f}")
    if baseline_loss > loss:
        improvement = (baseline_loss - loss) / baseline_loss * 100
        print(f"✅ Improvement: {improvement:.1f}%")
    else:
        print(f"Baseline was better this time (evolution is stochastic!)")


if __name__ == "__main__":
    demo()
