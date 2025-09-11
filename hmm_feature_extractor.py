# -*- coding: utf-8 -*-
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from config import HMM_STATES, WINDOW_SIZE

def extract_hmm_features(data_dict, n_states=HMM_STATES, window_size=WINDOW_SIZE):
    hmm_features = []
    hidden_states_dict = {}

    for site, X in data_dict.items():
        
        if len(X) < 5:
            hmm_features.append(np.zeros(n_states * (X.shape[1] + 1)))
            continue

        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            tol=1e-6,
            random_state=42
        )
        model.fit(X_scaled)
        states = model.predict(X_scaled)

        trans_counts = np.zeros((n_states, n_states))
        for i in range(1, len(states)):
            trans_counts[states[i - 1], states[i]] += 1

        row_sums = trans_counts.sum(axis=1, keepdims=True)
        trans_probs = np.where(row_sums > 0, trans_counts / row_sums, 0)
        trans_feat = trans_probs.mean(axis=0)  

    
        window_states = states[-window_size:] if len(states) >= window_size else states
        state_means = model.means_[window_states].mean(axis=0)

        hmm_feat = np.concatenate([state_means, trans_feat])
        hmm_features.append(hmm_feat)


    return np.array(hmm_features), hidden_states_dict
