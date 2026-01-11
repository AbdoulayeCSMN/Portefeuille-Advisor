# Projet de Reinforcement Learning pour la Gestion de Portefeuille

## 1. Informations Générales

- **Étudiant :** CHAIBOU SAIDOU ABDOULAYE
- **Professeur Encadrant :** TAWFIK MASROUR
- **

---

## 2. Description du Projet

Ce projet met en œuvre un agent d'apprentissage par renforcement profond (DRL) pour l'allocation dynamique d'actifs dans un portefeuille financier. L'agent, basé sur l'algorithme **Soft Actor-Critic (SAC)**, apprend une stratégie d'investissement pour maximiser le rendement ajusté au risque sur un horizon de 10 ans.

Veillez vous référez au [rapport](rapport/rapport_projet_rl.pdf) pour plus de détails.

---

L'environnement, développé avec **Gymnasium**, simule un marché financier mensuel avec trois classes d'actifs : **Actions, Obligations et Liquidités (Cash)**. La simulation intègre des rendements stochastiques pour les actifs et un modèle d'inflation auto-régressif.

Trois modèles distincts ont été entraînés pour correspondre à trois profils d'investisseurs, différenciés par leur aversion au risque (`risk_aversion`) dans la fonction de récompense :
- **Conservateur (`risk_aversion = 0.5`)**: Forte pénalité sur la volatilité, favorisant la préservation du capital.
- **Équilibré (`risk_aversion = 0.2`)**: Compromis entre la recherche de rendement et le contrôle du risque.
- **Agressif (`risk_aversion = 0.05`)**: Faible pénalité sur la volatilité, visant une croissance maximale du capital.

Le projet inclut une application interactive **Streamlit** pour simuler et visualiser les stratégies de chaque agent.

---

## 3. Structure du Projet

```
.
├── app.py                  # Application web interactive (Streamlit)
├── portfolio_env.py        # Classe de l'environnement de simulation (Gymnasium)
├── train.py                # Script pour l'entraînement des modèles
├── evaluate.py             # Script pour comparer l'agent "Équilibré" à des benchmarks
├── visualize.py            # Script pour générer des graphiques de comparaison
├── models/                 # Contient les modèles SAC entraînés (.zip)
├── logs/                   # Logs d'évaluation générés durant l'entraînement
├── tensorboard/            # Logs pour la visualisation avec TensorBoard
└── README.md               # Ce fichier
```

---

## 4. Guide d'Utilisation

### Étape 1: Installation des Dépendances

Assurez-vous d'avoir Python 3.8+ installé. Les dépendances sont déduites des imports dans les fichiers. Installez-les via pip :

```bash
pip install streamlit stable-baselines3[extra] gymnasium pandas numpy matplotlib seaborn
```
*Note: `stable-baselines3[extra]` inclut le support pour TensorBoard.*

### Étape 2: Entraînement des Agents

Le script `train.py` permet d'entraîner un agent pour un profil de risque donné. Il utilise l'algorithme SAC avec une architecture de réseau de neurones (MLP) de `[256, 256]`.

Pour entraîner le modèle pour le profil **équilibré** (par défaut) :
```bash
python train.py
```

Pour spécifier un autre profil (par exemple, **agressif**) :
```bash
python train.py --risk_profile agressif
```
-   **Choix de profil :** `conservateur`, `equilibre`, `agressif`.
-   Le script sauvegarde les checkpoints, le meilleur modèle (`best_model.zip`) et le modèle final (`sac_portfolio_final.zip`) dans le dossier `models/<nom_du_profil>/`.
-   Les logs d'entraînement pour TensorBoard sont sauvegardés dans `tensorboard/<nom_du_profil>/`. Pour les visualiser : `tensorboard --logdir=./tensorboard`.

### Étape 3: Utilisation de l'Application Interactive

C'est la méthode recommandée pour explorer les performances des agents. Le script `app.py` lance une application web avec Streamlit.

```bash
streamlit run app.py
```

Dans l'interface, vous pouvez :
1.  **Choisir le profil de risque** de l'agent à simuler.
2.  Définir le **capital de départ** et l'**horizon d'investissement**.
3.  Lancer la simulation, qui exécute 50 scénarios pour fournir des résultats statistiques robustes (médiane et intervalle interquartile).
4.  Visualiser les résultats sous forme de graphiques :
    -   Évolution de la valeur du portefeuille.
    -   Stratégie d'allocation dynamique (Actions / Obligations / Cash) de l'agent au fil du temps.
    -   Allocation finale suggérée.

### Étape 4 (Alternative): Scripts d'Évaluation et de Visualisation

Ces scripts sont moins flexibles que l'application Streamlit et sont principalement conçus pour analyser l'agent **équilibré**.

1.  **Évaluation comparative (`evaluate.py`)**: Ce script compare le modèle **équilibré** à des stratégies statiques (100% Actions, 60/40, 100% Cash). Il affiche un tableau récapitulatif des métriques (Rendement Annualisé, Volatilité, Sharpe Ratio, Max Drawdown) dans la console.

    ```bash
    python evaluate.py
    ```

2.  **Génération de graphiques (`visualize.py`)**: Ce script génère et sauvegarde quatre graphiques (`.png`) qui comparent également le modèle **équilibré** aux benchmarks :
    - `wealth_evolution.png`
    - `allocation_dynamics.png` (pour l'agent DRL uniquement)
    - `metrics_comparison.png`
    - `risk_return_scatter.png`

    ```bash
    python visualize.py
    ```

---

## 5. Détails Techniques

### L'Environnement (`portfolio_env.py`)

-   **Espace d'observation** : Un vecteur de 6 variables d'état : `[richesse_normalisée, %_actions, %_obligations, inflation_annualisée, volatilité_annualisée, horizon_restant]`.
-   **Espace d'action** : Un vecteur de 3 variables continues représentant l'allocation cible pour `[Actions, Obligations, Cash]`. Une normalisation est appliquée pour que la somme des poids soit égale à 1.
-   **Fonction de récompense** : `récompense = log_return - risk_aversion * volatilité`. Ce mécanisme pousse l'agent à trouver un équilibre entre le rendement et le risque, avec une intensité qui dépend du profil de l'investisseur.
-   **Fin d'épisode** : L'épisode se termine lorsque l'horizon de 120 mois (10 ans) est atteint, ou si la valeur du portefeuille tombe en dessous de 1% de sa valeur initiale (ruine).