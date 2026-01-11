import numpy as np
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Importer l'environnement (doit être dans le même dossier)
from portfolio_env import PortfolioEnv


def make_env(risk_aversion=0.2):
    """Factory function pour créer l'environnement"""
    def _init():
        return PortfolioEnv(
            initial_wealth=10000.0,
            horizon_months=120,
            risk_aversion=risk_aversion
        )
    return _init


def train_agent(risk_profile='equilibre', total_timesteps=100000):
    """
    Entraîne un agent SAC avec un profil de risque donné.
    
    Args:
        risk_profile: 'conservateur', 'equilibre', ou 'agressif'
        total_timesteps: nombre d'étapes d'entraînement
    """
    
    # Définition du paramètre de risk aversion selon le profil
    risk_aversion_map = {
        'conservateur': 0.5,   # Plus risk-averse
        'equilibre': 0.2,      # Modéré
        'agressif': 0.05       # Moins risk-averse
    }
    
    risk_aversion = risk_aversion_map[risk_profile]
    
    print(f"\n{'='*60}")
    print(f"Entraînement Agent SAC - Profil {risk_profile.upper()}")
    print(f"Risk Aversion λ = {risk_aversion}")
    print(f"{'='*60}\n")
    
    # Création de l'environnement
    env = DummyVecEnv([make_env(risk_aversion)])
    eval_env = DummyVecEnv([make_env(risk_aversion)])
    
    # Callbacks pour évaluation et sauvegarde
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{risk_profile}/',
        log_path=f'./logs/{risk_profile}/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'./models/{risk_profile}/checkpoints/',
        name_prefix='sac_portfolio'
    )
    
    # Hyperparamètres SAC optimisés pour ce problème
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=f'./tensorboard/{risk_profile}/'
    )

    # Entraînement
    print("Début de l'entraînement...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10
    )

    # Sauvegarde du modèle final
    model.save(f'./models/{risk_profile}/sac_portfolio_final')
    print(f"\nModèle sauvegardé dans ./models/{risk_profile}/")
    
    return model, env


def evaluate_agent(model, env, n_episodes=20, render=False):
    """
    Évalue un agent entraîné sur plusieurs épisodes.
    
    Returns:
        dict: Statistiques de performance
    """
    
    final_wealths = []
    total_returns = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_return = 0
        initial_wealth = 10000.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            
            if render and episode == 0:
                # La méthode render doit être appelée sur l'environnement sous-jacent
                env.envs[0].render()
        
        final_wealth = info[0]['wealth']
        final_wealths.append(final_wealth)
        total_returns.append((final_wealth / initial_wealth - 1) * 100)
    
    stats = {
        'mean_final_wealth': np.mean(final_wealths),
        'std_final_wealth': np.std(final_wealths),
        'mean_return': np.mean(total_returns),
        'std_return': np.std(total_returns),
        'min_wealth': np.min(final_wealths),
        'max_wealth': np.max(final_wealths)
    }
    
    return stats


if __name__ == "__main__":
    # Créer les dossiers nécessaires
    import os
    for profile in ['conservateur', 'equilibre', 'agressif']:
        os.makedirs(f'./models/{profile}', exist_ok=True)
        os.makedirs(f'./logs/{profile}', exist_ok=True)
        os.makedirs(f'./tensorboard/{profile}', exist_ok=True)

    parser = argparse.ArgumentParser(description="Entraîner l'agent DRL pour allocation de portefeuille.")
    parser.add_argument(
        '--risk_profile',
        type=str,
        default='equilibre',
        choices=['conservateur', 'equilibre', 'agressif'],
        help="Profil de risque pour l'entraînement de l'agent."
    )
    args = parser.parse_args()
    
    # Entraîner l'agent avec profil équilibré
    print("ENTRAÎNEMENT DE L'AGENT DRL POUR ALLOCATION DE PORTEFEUILLE")
    print("="*60)
    
    model, env = train_agent(
        risk_profile=args.risk_profile,
        total_timesteps=100000  # Augmenter à 200k-500k pour meilleurs résultats
    )
    
    # Évaluation finale
    print("\n" + "="*60)
    print("ÉVALUATION DE L'AGENT ENTRAÎNÉ")
    print("="*60 + "\n")
    
    stats = evaluate_agent(model, env, n_episodes=20, render=True)
    
    print("\nRésultats sur 20 épisodes:")
    print(f"  Richesse finale moyenne: {stats['mean_final_wealth']:.0f}€ ± {stats['std_final_wealth']:.0f}€")
    print(f"  Rendement moyen: {stats['mean_return']:.2f}% ± {stats['std_return']:.2f}%")
    print(f"  Min/Max richesse: {stats['min_wealth']:.0f}€ / {stats['max_wealth']:.0f}€")
    
    print("\nEntraînement terminé!")
    print("Pour visualiser les résultats, exécutez: python visualize.py")
    print("Pour voir les logs TensorBoard: tensorboard --logdir=./tensorboard/")