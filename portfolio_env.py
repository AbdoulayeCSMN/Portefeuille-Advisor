import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Environnement d'allocation de portefeuille simplifié pour Deep RL.
    
    État: [richesse_normalisée, w_actions, w_bonds, inflation, volatilité, horizon_restant]
    Action: [w_actions, w_bonds, w_cash] (allocations en %)
    
    Hypothèses simplificatrices:
    - 3 actifs: actions, obligations, cash
    - Pas de coûts de transaction
    - Pas de fiscalité
    - Décisions mensuelles
    - Horizon fixe (10 ans = 120 mois)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 initial_wealth=10000.0,
                 horizon_months=120,  # 10 ans
                 risk_aversion=0.2):  # λ dans la reward function
        
        super().__init__()
        
        # Paramètres de l'environnement
        self.initial_wealth = initial_wealth
        self.horizon_months = horizon_months
        self.risk_aversion = risk_aversion
        
        # Paramètres des actifs (rendements mensuels)
        self.mu_equity = 0.07 / 12      # 7% annualisé
        self.sigma_equity = 0.15 / np.sqrt(12)  # 15% vol annuelle
        
        self.mu_bonds = 0.02 / 12       # 2% annualisé
        self.sigma_bonds = 0.03 / np.sqrt(12)   # 3% vol annuelle
        
        self.nominal_interest_rate = 0.005 / 12  # 0.5% annualisé (taux bas!)
        
        # Paramètres de l'inflation (processus AR(1))
        self.inflation_mean = 0.02 / 12   # 2% annualisé
        self.inflation_persistence = 0.8
        self.inflation_shock_std = 0.005 / 12
        
        # Espace d'état: [W_norm, w_eq, w_bonds, inflation, volatility, horizon_norm]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -0.01, 0.0, 0.0]),
            high=np.array([10.0, 1.0, 1.0, 0.1, 0.5, 1.0]),
            dtype=np.float32
        )
        
        # Espace d'action: [w_equity, w_bonds, w_cash]
        # Note: la contrainte sum=1 sera appliquée via softmax
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Variables d'état
        self.wealth = None
        self.weights = None  # [w_eq, w_bonds, w_cash]
        self.current_step = None
        self.inflation = None
        self.volatility = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialisation
        self.wealth = self.initial_wealth
        self.weights = np.array([0.33, 0.33, 0.34])  # Allocation initiale équilibrée
        self.current_step = 0
        self.inflation = self.inflation_mean
        self.volatility = self.sigma_equity  # Volatilité initiale
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        # Normalisation de l'action pour assurer sum=1 (projection sur simplex)
        action = np.abs(action)  # Assurer positivité
        action = action / (action.sum() + 1e-8)  # Normaliser
        
        w_equity, w_bonds, w_cash = action
        
        # Génération des rendements des actifs
        r_equity = np.random.normal(self.mu_equity, self.sigma_equity)
        r_bonds = np.random.normal(self.mu_bonds, self.sigma_bonds)
        r_cash = self.nominal_interest_rate - self.inflation  # Rendement réel
        
        # Rendement du portefeuille
        r_portfolio = (w_equity * r_equity + 
                      w_bonds * r_bonds + 
                      w_cash * r_cash)
        
        # Mise à jour de la richesse
        wealth_before = self.wealth
        self.wealth = self.wealth * (1 + r_portfolio)
        
        # Mise à jour de l'inflation (processus AR(1))
        shock = np.random.normal(0, self.inflation_shock_std)
        self.inflation = (self.inflation_persistence * self.inflation + 
                         (1 - self.inflation_persistence) * self.inflation_mean + 
                         shock)
        
        # Mise à jour de la volatilité (simplifiée)
        self.volatility = self.sigma_equity * (1 + 0.3 * np.random.randn())
        self.volatility = np.clip(self.volatility, 0.01, 0.5)
        
        # Mise à jour des poids
        self.weights = action
        
        # Calcul de la reward (log-return - pénalité de volatilité)
        log_return = np.log(self.wealth / wealth_before + 1e-8)
        reward = log_return - self.risk_aversion * self.volatility
        
        # Avancement du temps
        self.current_step += 1
        
        # Conditions de terminaison
        terminated = self.current_step >= self.horizon_months
        truncated = self.wealth <= 0.01 * self.initial_wealth  # Ruine
        
        obs = self._get_observation()
        info = {
            'wealth': self.wealth,
            'portfolio_return': r_portfolio,
            'inflation': self.inflation,
            'weights': self.weights.copy()
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Construit le vecteur d'observation"""
        obs = np.array([
            self.wealth / self.initial_wealth,  # Richesse normalisée
            self.weights[0],  # w_equity
            self.weights[1],  # w_bonds
            self.inflation * 12,  # Inflation annualisée
            self.volatility * np.sqrt(12),  # Volatilité annualisée
            1.0 - self.current_step / self.horizon_months  # Horizon restant
        ], dtype=np.float32)
        
        return obs
    
    def render(self):
        if self.current_step % 12 == 0:  # Affichage annuel
            print(f"Année {self.current_step//12} | "
                  f"Richesse: {self.wealth:.0f}€ | "
                  f"Allocation: Eq={self.weights[0]:.2f} "
                  f"Bonds={self.weights[1]:.2f} "
                  f"Cash={self.weights[2]:.2f}")
                  