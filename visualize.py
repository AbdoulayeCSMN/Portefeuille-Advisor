import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from portfolio_env import PortfolioEnv
from evaluate import BenchmarkStrategy, run_episode, compute_metrics

# Configuration matplotlib pour de beaux graphiques
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_wealth_evolution(results, save_path='wealth_evolution.png'):
    """
    Graphique 1: Évolution de la richesse au cours du temps
    """
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {
        'DRL Agent': '#2E86AB',
        '100% Cash': '#A23B72',
        '60/40': '#F18F01',
        '100% Equity': '#C73E1D'
    }
    
    for strategy_name, data in results.items():
        if data['example_history']:
            history = data['example_history']
            wealth = np.array(history['wealth'])
            months = np.arange(len(wealth))
            years = months / 12
            
            ax.plot(years, wealth, 
                   label=strategy_name, 
                   linewidth=2.5,
                   color=colors.get(strategy_name, 'gray'),
                   alpha=0.9)
    
    ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Capital initial')
    ax.set_xlabel('Années', fontsize=12, fontweight='bold')
    ax.set_ylabel('Richesse (€)', fontsize=12, fontweight='bold')
    ax.set_title('Évolution de la Richesse sur 10 ans', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {save_path}")
    plt.close()


def plot_allocation_dynamics(results, save_path='allocation_dynamics.png'):
    """
    Graphique 2: Évolution des allocations de l'agent DRL
    """
    
    if 'DRL Agent' not in results or not results['DRL Agent']['example_history']:
        print("Pas de données DRL Agent pour l'allocation")
        return
    
    history = results['DRL Agent']['example_history']
    weights = np.array(history['weights'])
    months = np.arange(len(weights))
    years = months / 12
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.fill_between(years, 0, weights[:, 0], 
                     label='Actions', alpha=0.7, color='#2E86AB')
    ax.fill_between(years, weights[:, 0], weights[:, 0] + weights[:, 1], 
                     label='Obligations', alpha=0.7, color='#F18F01')
    ax.fill_between(years, weights[:, 0] + weights[:, 1], 1.0, 
                     label='Cash', alpha=0.7, color='#A23B72')
    
    ax.set_xlabel('Années', fontsize=12, fontweight='bold')
    ax.set_ylabel('Allocation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Allocation Dynamique de l\'Agent DRL', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {save_path}")
    plt.close()


def plot_metrics_comparison(results, save_path='metrics_comparison.png'):
    """
    Graphique 3: Comparaison des métriques clés
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = list(results.keys())
    colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Rendement annualisé
    ax1 = axes[0, 0]
    returns = [results[s]['avg_metrics']['annualized_return'] for s in strategies]
    errors = [results[s]['std_metrics']['annualized_return'] for s in strategies]
    bars1 = ax1.bar(strategies, returns, yerr=errors, capsize=5, 
                     color=colors_list[:len(strategies)], alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Rendement Annualisé (%)', fontweight='bold')
    ax1.set_title('Rendement Annualisé', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Volatilité
    ax2 = axes[0, 1]
    vols = [results[s]['avg_metrics']['volatility'] for s in strategies]
    errors_vol = [results[s]['std_metrics']['volatility'] for s in strategies]
    bars2 = ax2.bar(strategies, vols, yerr=errors_vol, capsize=5,
                     color=colors_list[:len(strategies)], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Volatilité (%)', fontweight='bold')
    ax2.set_title('Volatilité Annualisée', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Sharpe ratio
    ax3 = axes[1, 0]
    sharpes = [results[s]['avg_metrics']['sharpe_ratio'] for s in strategies]
    bars3 = ax3.bar(strategies, sharpes, 
                     color=colors_list[:len(strategies)], alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax3.set_title('Sharpe Ratio (Rendement/Risque)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Seuil "bon"')
    ax3.legend()
    
    # Maximum Drawdown
    ax4 = axes[1, 1]
    drawdowns = [results[s]['avg_metrics']['max_drawdown'] for s in strategies]
    bars4 = ax4.bar(strategies, drawdowns, 
                     color=colors_list[:len(strategies)], alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Max Drawdown (%)', fontweight='bold')
    ax4.set_title('Drawdown Maximum', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {save_path}")
    plt.close()


def plot_risk_return_scatter(results, save_path='risk_return_scatter.png'):
    """
    Graphique 4: Scatter plot Risque vs Rendement
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'DRL Agent': '#2E86AB',
        '100% Cash': '#A23B72',
        '60/40': '#F18F01',
        '100% Equity': '#C73E1D'
    }
    
    for strategy_name, data in results.items():
        m = data['avg_metrics']
        ax.scatter(m['volatility'], m['annualized_return'], 
                  s=300, alpha=0.7, 
                  color=colors.get(strategy_name, 'gray'),
                  edgecolors='black', linewidth=2,
                  label=strategy_name)
        
        # Annoter les points
        ax.annotate(strategy_name, 
                   (m['volatility'], m['annualized_return']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=colors.get(strategy_name, 'gray'), 
                            alpha=0.3))
    
    ax.set_xlabel('Volatilité Annualisée (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rendement Annualisé (%)', fontsize=12, fontweight='bold')
    ax.set_title('Frontière Risque-Rendement', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé: {save_path}")
    plt.close()


def generate_all_visualizations():
    """Génère tous les graphiques de visualisation"""
    
    print("\n" + "="*70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("="*70 + "\n")
    
    # Charger les stratégies
    strategies = {
        '100% Cash': BenchmarkStrategy([0, 0, 1], '100% Cash'),
        '60/40': BenchmarkStrategy([0.6, 0.4, 0], '60/40'),
        '100% Equity': BenchmarkStrategy([1, 0, 0], '100% Equity')
    }
    
    # Charger l'agent DRL
    try:
        model = SAC.load('./models/equilibre/sac_portfolio_final')
        strategies['DRL Agent'] = model
        print("Agent DRL chargé\n")
    except:
        print("Agent DRL non trouvé, visualisation sans DRL\n")
    
    # Environnement
    env = PortfolioEnv(initial_wealth=10000.0, horizon_months=120, risk_aversion=0.2)
    
    # Collecter les données (1 épisode par stratégie pour visualisation)
    results = {}
    for strategy_name, strategy in strategies.items():
        print(f"Collecte données: {strategy_name}...")
        history = run_episode(strategy, env, render=False)
        metrics = compute_metrics(history)
        
        results[strategy_name] = {
            'avg_metrics': metrics,
            'std_metrics': {k: 0 for k in metrics.keys()},  # Pas d'erreur pour 1 épisode
            'example_history': history
        }
    
    print("\n" + "-"*70 + "\n")
    
    # Générer les graphiques
    plot_wealth_evolution(results, 'wealth_evolution.png')
    plot_allocation_dynamics(results, 'allocation_dynamics.png')
    plot_metrics_comparison(results, 'metrics_comparison.png')
    plot_risk_return_scatter(results, 'risk_return_scatter.png')
    
    print("\n" + "="*70)
    print("VISUALISATIONS GÉNÉRÉES AVEC SUCCÈS!")
    print("="*70)
    print("\nFichiers créés:")
    print("  • wealth_evolution.png      - Évolution de la richesse")
    print("  • allocation_dynamics.png   - Allocation dynamique de l'agent DRL")
    print("  • metrics_comparison.png    - Comparaison des métriques")
    print("  • risk_return_scatter.png   - Frontière risque-rendement")
    print("\nOuvrez ces fichiers pour analyser les résultats visuellement!")


if __name__ == "__main__":
    generate_all_visualizations()