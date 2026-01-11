import numpy as np
from stable_baselines3 import SAC
from portfolio_env import PortfolioEnv
import matplotlib.pyplot as plt


class BenchmarkStrategy:
    """Stratégie d'allocation fixe pour comparaison"""
    
    def __init__(self, weights, name):
        """
        Args:
            weights: [w_equity, w_bonds, w_cash]
            name: nom de la stratégie
        """
        self.weights = np.array(weights)
        self.name = name
    
    def predict(self, obs):
        """Retourne l'action (allocation fixe)"""
        return self.weights, None


def run_episode(strategy, env, render=False):
    """
    Exécute un épisode complet avec une stratégie donnée.
    
    Returns:
        dict: Historique de l'épisode
    """
    
    obs = env.reset()
    done = False
    
    history = {
        'wealth': [env.unwrapped.wealth],
        'weights': [env.unwrapped.weights.copy()],
        'inflation': [env.unwrapped.inflation],
        'returns': []
    }
    
    while not done:
        action, _ = strategy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        history['wealth'].append(info['wealth'])
        history['weights'].append(info['weights'].copy())
        history['inflation'].append(info['inflation'])
        history['returns'].append(info['portfolio_return'])
        
        if render:
            env.render()
    
    return history


def compute_metrics(history, initial_wealth=10000.0):
    """
    Calcule les métriques financières standard.
    
    Returns:
        dict: Métriques de performance
    """
    
    wealth_series = np.array(history['wealth'])
    returns = np.array(history['returns'])
    
    # Richesse finale
    final_wealth = wealth_series[-1]
    total_return = (final_wealth / initial_wealth - 1) * 100
    
    # Rendement annualisé
    n_years = len(returns) / 12
    annualized_return = ((final_wealth / initial_wealth) ** (1/n_years) - 1) * 100
    
    # Volatilité annualisée
    volatility = np.std(returns) * np.sqrt(12) * 100
    
    # Sharpe ratio (approximatif, sans taux sans risque)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cummax = np.maximum.accumulate(wealth_series)
    drawdown = (wealth_series - cummax) / cummax * 100
    max_drawdown = np.min(drawdown)
    
    metrics = {
        'final_wealth': final_wealth,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown
    }
    
    return metrics


def evaluate_all_strategies(n_episodes=20):
    """
    Compare l'agent DRL avec les stratégies de benchmark.
    
    Returns:
        dict: Résultats pour chaque stratégie
    """
    
    # Définir les stratégies de benchmark
    strategies = {
        'DRL Agent': None,  # Chargé après
        '100% Cash': BenchmarkStrategy([0, 0, 1], '100% Cash'),
        '60/40': BenchmarkStrategy([0.6, 0.4, 0], '60/40'),
        '100% Equity': BenchmarkStrategy([1, 0, 0], '100% Equity')
    }
    
    # Charger l'agent DRL entraîné
    try:
        model = SAC.load('./models/equilibre/sac_portfolio_final')
        strategies['DRL Agent'] = model
        print("Agent DRL chargé depuis ./models/equilibre/sac_portfolio_final")
    except:
        print("Impossible de charger l'agent DRL. Entraînez-le d'abord avec train.py")
        del strategies['DRL Agent']
    
    # Environnement d'évaluation
    env = PortfolioEnv(initial_wealth=10000.0, horizon_months=120, risk_aversion=0.2)
    
    # Résultats pour chaque stratégie
    results = {}
    
    print("\n" + "="*70)
    print("ÉVALUATION DES STRATÉGIES D'ALLOCATION")
    print("="*70 + "\n")
    
    for strategy_name, strategy in strategies.items():
        print(f"Évaluation: {strategy_name}...")
        
        all_metrics = []
        all_histories = []
        
        for episode in range(n_episodes):
            history = run_episode(strategy, env, render=False)
            metrics = compute_metrics(history)
            all_metrics.append(metrics)
            
            if episode == 0:  # Sauvegarder un historique exemple
                all_histories.append(history)
        
        # Calcul des statistiques moyennes
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        std_metrics = {
            key: np.std([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        results[strategy_name] = {
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'example_history': all_histories[0] if all_histories else None
        }
        
        print(f"  ✓ Richesse finale: {avg_metrics['final_wealth']:.0f}€ ± {std_metrics['final_wealth']:.0f}€")
        print(f"  ✓ Rendement annualisé: {avg_metrics['annualized_return']:.2f}% ± {std_metrics['annualized_return']:.2f}%")
        print(f"  ✓ Sharpe ratio: {avg_metrics['sharpe_ratio']:.2f}")
        print()
    
    return results


def print_comparison_table(results):
    """Affiche un tableau comparatif des stratégies"""
    
    print("\n" + "="*100)
    print("TABLEAU COMPARATIF DES STRATÉGIES (moyenne sur 20 épisodes)")
    print("="*100)
    
    header = f"{'Stratégie':<15} | {'Richesse Finale':>15} | {'Rdt Annualisé':>14} | {'Volatilité':>11} | {'Sharpe':>7} | {'Max DD':>8}"
    print(header)
    print("-"*100)
    
    for strategy_name, data in results.items():
        m = data['avg_metrics']
        row = (f"{strategy_name:<15} | "
               f"{m['final_wealth']:>12,.0f}€  | "
               f"{m['annualized_return']:>11.2f}%  | "
               f"{m['volatility']:>8.2f}%  | "
               f"{m['sharpe_ratio']:>7.2f} | "
               f"{m['max_drawdown']:>7.2f}%")
        print(row)
    
    print("="*100 + "\n")


if __name__ == "__main__":
    # Évaluation complète
    results = evaluate_all_strategies(n_episodes=20)
    
    # Affichage du tableau comparatif
    print_comparison_table(results)
    
    # Analyse qualitative
    print("ANALYSE:")
    print("  • L'agent DRL doit montrer un meilleur ratio rendement/risque que les benchmarks fixes")
    print("  • Le Sharpe ratio est la métrique clé pour comparer les stratégies")
    print("  • Un Sharpe > 1.0 est considéré comme bon, > 2.0 comme excellent")
    print("  • Le Max Drawdown mesure la perte maximale subie")
    print("\nÉvaluation terminée!")
    print("Pour visualiser graphiquement, exécutez:\n python visualize.py")