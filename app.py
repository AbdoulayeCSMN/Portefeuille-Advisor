import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from portfolio_env import PortfolioEnv
import os

# Configuration de la page
st.set_page_config(page_title="Robo-Advisor IA", layout="wide")

# --- Fonctions de Simulation et Visualisation ---

@st.cache_resource
def load_model(risk_profile):
    """Charge un modèle SAC pré-entraîné."""
    model_path = f"./models/{risk_profile}/sac_portfolio_final.zip"
    if os.path.exists(model_path):
        return SAC.load(model_path)
    return None

def run_simulation(model, env, n_episodes=50):
    """Exécute la simulation et retourne les historiques de richesse et d'allocations."""
    all_wealth_histories = []
    all_allocations_histories = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        wealth_history = [env.envs[0].initial_wealth]
        allocations_history = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            
            # L'action déterminée par model.predict est l'allocation que nous devons enregistrer.
            # Elle est déjà dans la variable 'action'.
            normalized_action = np.abs(action[0])
            normalized_action /= np.sum(normalized_action)
            
            allocations_history.append(normalized_action)
            wealth_history.append(info[0]['wealth'])

        all_wealth_histories.append(wealth_history)
        all_allocations_histories.append(allocations_history)

    return all_wealth_histories, all_allocations_histories

def plot_wealth_evolution(histories, title, initial_wealth):
    """Génère le graphique de l'évolution de la richesse."""
    fig, ax = plt.subplots(figsize=(12, 7))
    df = pd.DataFrame(histories).T
    
    q25 = df.quantile(0.25, axis=1)
    q50 = df.quantile(0.50, axis=1)
    q75 = df.quantile(0.75, axis=1)
    
    ax.plot(q50.index, q50, lw=2, label='Richesse Médiane')
    ax.fill_between(q50.index, q25, q75, alpha=0.3, label='Intervalle Interquartile (25%-75%)')
    ax.axhline(initial_wealth, color='red', ls='--', label='Capital Initial')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Richesse du Portefeuille (€)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_allocation_dynamics(allocations_histories, title):
    """Génère le graphique de la dynamique d'allocation."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Calculer la médiane des allocations à travers les épisodes
    median_trajectory = np.median([h for h in allocations_histories], axis=0)
    df = pd.DataFrame(median_trajectory, columns=['Actions', 'Obligations', 'Cash'])
    
    ax.stackplot(df.index, df['Actions'], df['Obligations'], df['Cash'], 
                 labels=df.columns, alpha=0.8, colors=['#FF9999','#66B2FF','#99FF99'])
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Proportion du Portefeuille")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper center', ncol=3)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return fig

def plot_final_allocation(allocations_histories):
    """Génère un diagramme circulaire de l'allocation finale."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    final_allocations = [h[-1] for h in allocations_histories]
    median_final = np.median(final_allocations, axis=0)
    
    labels = ['Actions', 'Obligations', 'Cash']
    colors = ['#FF9999','#66B2FF','#99FF99']
    ax.pie(median_final, labels=labels, autopct='%1.1f%%', startangle=90, 
           colors=colors, wedgeprops=dict(width=0.4, edgecolor='w'))
    
    ax.set_title("Allocation Finale Suggérée (Médiane)", fontsize=16)
    centre_circle = plt.Circle((0,0),0.50,fc='white')
    fig.gca().add_artist(centre_circle)
    
    ax.axis('equal')
    plt.tight_layout()
    return fig

# --- Interface Streamlit ---

st.title("🤖 Robo-Advisor IA pour la Gestion de Portefeuille")
st.markdown("Une application de démonstration pour un agent d'apprentissage par renforcement (DRL) qui gère un portefeuille financier.")

# --- Barre Latérale de Configuration ---
with st.sidebar:
    st.header("⚙️ Paramètres de Simulation")

    risk_profile = st.selectbox(
        "1. Choisissez votre profil de risque :",
        options=['equilibre', 'agressif', 'conservateur'],
        format_func=lambda x: x.capitalize()
    )

    initial_wealth = st.slider(
        "2. Capital de départ (€) :",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )

    horizon_years = st.slider(
        "3. Horizon d'investissement (années) :",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )

    run_button = st.button("Lancer la Simulation", type="primary")

# --- Zone Principale d'Affichage ---
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False

if run_button:
    st.session_state.simulation_done = False
    with st.spinner(f"Chargement de l'agent IA avec le profil **{risk_profile}**..."):
        model = load_model(risk_profile)

    if model is None:
        st.error(f"Le modèle pour le profil '{risk_profile}' n'a pas été trouvé.")
    else:
        st.success(f"Agent IA **{risk_profile}** chargé.")

        risk_aversion_map = {'conservateur': 0.5, 'equilibre': 0.2, 'agressif': 0.05}
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: PortfolioEnv(
            initial_wealth=float(initial_wealth),
            horizon_months=horizon_years * 12,
            risk_aversion=risk_aversion_map[risk_profile]
        )])

        with st.spinner(f"Simulation en cours... ({50} scénarios)"):
            wealth_histories, allocations_histories = run_simulation(model, env, n_episodes=50)
            
            st.session_state.wealth_histories = wealth_histories
            st.session_state.allocations_histories = allocations_histories
            st.session_state.simulation_params = {
                'profile': risk_profile,
                'initial_wealth': initial_wealth
            }
            st.session_state.simulation_done = True
            st.rerun()

if st.session_state.simulation_done:
    st.header("📈 Résultats de la Simulation")
    
    params = st.session_state.simulation_params
    wealth_histories = st.session_state.wealth_histories
    allocations_histories = st.session_state.allocations_histories
    
    final_wealths = [h[-1] for h in wealth_histories]
    median_final_wealth = np.median(final_wealths)
    
    st.success(f"Simulation pour le profil **{params['profile'].capitalize()}** terminée.")

    tab1, tab2 = st.tabs(["Évolution du Portefeuille", "Stratégie de l'Agent"])

    with tab1:
        st.subheader("Performance Financière")
        col1, col2, col3 = st.columns(3)
        col1.metric("Capital Initial", f"{params['initial_wealth']:,.0f} €")
        col2.metric("Richesse Finale Médiane", f"{median_final_wealth:,.0f} €", f"{((median_final_wealth/params['initial_wealth'])-1)*100:.1f}%")
        
        fig_wealth = plot_wealth_evolution(
            wealth_histories, 
            f"Évolution du Portefeuille - Profil {params['profile'].capitalize()}",
            params['initial_wealth']
        )
        st.pyplot(fig_wealth)

    with tab2:
        st.subheader("Comment l'agent investit votre argent")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig_alloc_dyn = plot_allocation_dynamics(
                allocations_histories,
                "Stratégie d'Allocation Dynamique (Médiane)"
            )
            st.pyplot(fig_alloc_dyn)
            st.markdown("Ce graphique montre comment l'agent ajuste la répartition de votre portefeuille chaque mois pour s'adapter aux conditions du marché.")
            
        with col2:
            fig_final_alloc = plot_final_allocation(allocations_histories)
            st.pyplot(fig_final_alloc)
            st.markdown("Le diagramme ci-contre présente l'allocation finale typique que l'agent recommande à la fin de l'horizon d'investissement.")

else:
    st.info("Veuillez configurer votre simulation dans la barre latérale et cliquer sur **Lancer la Simulation**.")

st.markdown("---")
st.write("Projet de Deep Reinforcement Learning")
