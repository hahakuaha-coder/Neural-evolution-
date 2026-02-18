"""
Financial Data Example - Stock Price Prediction
================================================
Evolve architecture for predicting stock returns!
"""

import numpy as np
import random
from evolve import ArchitectureEvolution, EvolutionConfig

def generate_synthetic_stock_data(n_days=500):
    """Generate synthetic stock data with technical indicators"""
    # Price with trend + noise
    t = np.linspace(0, 10, n_days)
    price = 100 + 10 * t + 5 * np.sin(t) + np.random.randn(n_days) * 2
    
    # Technical indicators
    def moving_average(x, window):
        return np.convolve(x, np.ones(window)/window, mode='valid')
    
    ma_5 = moving_average(price, 5)
    ma_20 = moving_average(price, 20)
    
    # Align arrays
    min_len = min(len(price), len(ma_5), len(ma_20))
    price = price[-min_len:]
    ma_5 = ma_5[-min_len:]
    ma_20 = ma_20[-min_len:]
    
    # Volume (synthetic)
    volume = np.random.lognormal(10, 1, min_len)
    
    # RSI-like indicator
    returns = np.diff(price, prepend=price[0])
    
    return price, ma_5, ma_20, volume, returns

def create_sequences(price, ma_5, ma_20, volume, returns, lookback=20):
    """Create input sequences for prediction"""
    X = []
    y = []
    
    for i in range(lookback, len(price) - 1):
        # Features: recent prices, MAs, volume, returns
        features = []
        features.extend(price[i-lookback:i])
        features.extend(ma_5[i-lookback:i])
        features.extend(ma_20[i-lookback:i])
        features.extend(volume[i-lookback:i])
        features.extend(returns[i-lookback:i])
        
        X.append(features)
        # Target: next day return (up/down)
        y.append([returns[i+1]])
    
    return np.array(X), np.array(y)

def main():
    print("💰 FINANCIAL DATA - ARCHITECTURE EVOLUTION")
    print("=" * 60)
    
    # Set seeds
    np.random.seed(42)
    random.seed(42)
    
    # Generate data
    print("Generating synthetic stock data...")
    price, ma_5, ma_20, volume, returns = generate_synthetic_stock_data(n_days=500)
    
    # Create sequences
    lookback = 20
    X_train, y_train = create_sequences(price, ma_5, ma_20, volume, returns, lookback)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Features per sample: {X_train.shape[1]}")
    print(f"Lookback window: {lookback} days\n")
    
    # Evolve architecture
    config = EvolutionConfig(
        population_size=40,
        generations=25,
        max_layers=5,
        layer_sizes=[16, 32, 64, 128],
        elite_size=8
    )
    
    evolution = ArchitectureEvolution(config)
    best = evolution.evolve(X_train, y_train, 
                           input_size=X_train.shape[1],
                           output_size=1,
                           verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n🏆 Best Architecture: {best}")
    print(f"📊 Fitness: {best.fitness:.4f}")
    print(f"📉 Loss: {best.loss:.4f}")
    print(f"🧬 Depth: {len(best.layers)} layers")
    print(f"📦 Approximate parameters: {sum(best.layers)}")
    
    # Analyze evolution
    print(f"\n📈 Evolution Stats:")
    print(f"  Initial best fitness: {evolution.history['best_fitness'][0]:.4f}")
    print(f"  Final best fitness: {evolution.history['best_fitness'][-1]:.4f}")
    improvement = (evolution.history['best_fitness'][-1] - 
                  evolution.history['best_fitness'][0]) / evolution.history['best_fitness'][0] * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    # Common patterns discovered
    print(f"\n🔍 Architecture Patterns:")
    all_genomes = evolution.history['best_genomes']
    
    # Most common layer sizes
    from collections import Counter
    all_sizes = [size for genome in all_genomes for size in genome]
    common_sizes = Counter(all_sizes).most_common(3)
    print(f"  Most used layer sizes: {[s for s, _ in common_sizes]}")
    
    # Average depth over time
    depths = [len(g) for g in all_genomes]
    print(f"  Average depth: {np.mean(depths):.1f} layers")
    print(f"  Depth range: {min(depths)} to {max(depths)} layers")
    
    print("\n✅ Evolution complete!")
    print("\n💡 Try running with your own financial data:")
    print("   - Load CSV with pandas")
    print("   - Calculate technical indicators")
    print("   - Format as sequences")
    print("   - Run evolution!")

if __name__ == "__main__":
    main()
