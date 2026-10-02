import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss

EPS = 1e-12

class ConfidenceLayer():
    def __init__(self, type):
        self.type = type

    def get_confidence_score(probab_dist):
        max_prob_class = 0
        confidence_score = 0

    def _to_pseudo_logits(probs):
        probs = np.clip(probs, EPS, 1 - EPS)
        return np.log(probs)

    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


    # ---------------- PART 1: TUNE T ----------------
    def tune_temperature(val_probs, val_labels):
        """
        val_probs  : (N, C) predict_proba output on a held-out validation set
        val_labels : (N,) true integer class labels
        Returns optimal scalar T.
        """
        logits = _to_pseudo_logits(val_probs)

        def nll(T):
            T = max(T, 1e-3)
            scaled = _softmax(logits / T)
            return log_loss(val_labels, scaled, labels=np.arange(val_probs.shape[1]))

        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        print(f"Optimal T = {result.x:.4f} | calibrated loss = {result.fun:.4f} "
            f"(uncalibrated: {log_loss(val_labels, val_probs):.4f})")
        return result.x


    # ---------------- PART 2: INFERENCE ----------------
    def apply_temperature(probs, T):
        """
        probs : (N, C) predict_proba output (test/new data)
        T     : temperature from tune_temperature()
        Returns calibrated (N, C) probabilities.
        """
        logits = _to_pseudo_logits(probs)
        return _softmax(logits / T)
