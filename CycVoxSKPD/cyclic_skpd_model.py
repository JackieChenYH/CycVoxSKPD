import numpy as np
from scipy.optimize import minimize
from .skpd_transformer import SKPD_Transformer

class CyclicShiftSKPD:
    """
    Implements the Cyclic-Shift Sparse Kronecker Product Decomposition (SKPD) model
    for binary classification, strictly following the user's manuscript.
    """
    def __init__(self, hparams: dict):
        self.hparams = hparams
        self.term = hparams['term']
        
        self.transformer = SKPD_Transformer(
            p1=hparams['p1'], d1=hparams['d1'],
            p2=hparams['p2'], d2=hparams['d2'],
            p3=hparams.get('p3', 1), d3=hparams.get('d3', 1)
        )
        self.p = self.transformer.p
        self.d = self.transformer.d

        # Initialize attributes for fitted parameters
        self.A_ = None
        self.B_ = None
        self.gamma_ = None
        self.intercept_ = 0.0
        self.optimal_shift_ = (0, 0)
        self.loss_history_ = []

    def _calculate_logits(self, X_tilde, A, B, gamma=None, intercept=0.0, Z=None):
        logits = np.zeros(X_tilde.shape[0])
        for r in range(self.term):
            logits += (X_tilde @ B[r, :].T) @ A[r, :].T
        if Z is not None and gamma is not None:
            logits += Z @ gamma.flatten()
        logits += intercept
        return logits

    def _logistic_loss(self, y_true, logits):
        y_pred_prob = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        y_pred_prob = np.clip(y_pred_prob, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob))

    def _regularization_penalty(self, A, B, gamma):
        pen_a = self.hparams['lmbda_a'] * np.sum(np.abs(A))
        alpha = self.hparams['alpha']
        pen_b = self.hparams['lmbda_b'] * (alpha * np.sum(np.abs(B)) + (1-alpha) * np.sum(B**2))
        pen_gamma = 0.0
        if gamma is not None:
            pen_gamma = self.hparams['lmbda_gamma'] * np.sum(np.abs(gamma))
        return pen_a + pen_b + pen_gamma

    def _objective_B(self, B_flat, X_tilde, Y, A, offset):
        B = B_flat.reshape(self.term, self.d)
        logits = offset + self._calculate_logits(X_tilde, A, B)
        loss = self._logistic_loss(Y, logits)
        reg = self.hparams['lmbda_b'] * (self.hparams['alpha'] * np.sum(np.abs(B)) + (1-self.hparams['alpha']) * np.sum(B**2))
        return loss + reg

    def _objective_A(self, A_flat, X_tilde, Y, B, offset):
        A = A_flat.reshape(self.term, self.p)
        logits = offset + self._calculate_logits(X_tilde, A, B)
        loss = self._logistic_loss(Y, logits)
        reg = self.hparams['lmbda_a'] * np.sum(np.abs(A))
        return loss + reg

    def _objective_gamma_intercept(self, params, Z, Y, offset):
        gamma = params[:-1].reshape(-1, 1)
        intercept = params[-1]
        logits = offset + Z @ gamma.flatten() + intercept
        loss = self._logistic_loss(Y, logits)
        reg = self.hparams['lmbda_gamma'] * np.sum(np.abs(gamma))
        return loss + reg

    def fit(self, X, Y, Z=None, max_iter=100, J=1, tol=1e-4, verbose=True):
        n_samples = X.shape[0]

        # --- Line 2: Initialize Parameters (Strictly following manuscript) ---
        X_tilde_init = self.transformer.forward(X)
        weighted_avg_X = np.mean(X_tilde_init * Y[:, np.newaxis, np.newaxis], axis=0)
        U, _, _ = np.linalg.svd(weighted_avg_X, full_matrices=False)
        self.A_ = U[:, :self.term].T
        
        self.B_ = np.ones((self.term, self.d))
        
        self.intercept_ = 0.0
        if Z is not None:
            self.gamma_ = np.zeros((Z.shape[1], 1))
        
        current_shift = (0, 0)

        # --- Outer Loop (t) ---
        for t in range(max_iter):
            A_old_t, B_old_t, shift_old_t = self.A_.copy(), self.B_.copy(), current_shift
            gamma_old_t = self.gamma_.copy() if Z is not None else None

            if verbose: print(f"\n--- Outer Iter {t+1}/{max_iter} ---")

            # --- Update Optimal Shift Q ---
            s_max_default = int(np.floor(max(self.hparams['d1'], self.hparams['d2']) / 2))
            s_max = self.hparams.get('s_max', s_max_default)  # Maximum shift range, no 'aligned' check

            # Hyperparameters for shift control (can be tuned via hparams)
            shift_penalty = self.hparams.get('shift_penalty', 0.01)  # Penalty coefficient for shift magnitude
            shift_threshold = self.hparams.get('shift_threshold', 0.01)  # Threshold for accepting no shift

            # Initialize variables for shift search
            best_loss = float('inf')
            best_shift = (0, 0)

            # Calculate loss at no-shift position (0, 0)
            X_tilde_no_shift = self.transformer.forward(X, shift=(0, 0))
            logits_no_shift = self._calculate_logits(X_tilde_no_shift, self.A_, self.B_, self.gamma_, self.intercept_, Z)
            loss_no_shift = self._logistic_loss(Y, logits_no_shift)  # Baseline loss without penalty

            # Grid search over possible shifts
            for q1 in range(-s_max, s_max + 1):
                for q2 in range(-s_max, s_max + 1):
                    shift_candidate = (q1, q2)
                    X_tilde_shifted = self.transformer.forward(X, shift=shift_candidate)
                    logits = self._calculate_logits(X_tilde_shifted, self.A_, self.B_, self.gamma_, self.intercept_, Z)
                    # Loss with penalty for shift magnitude
                    loss = self._logistic_loss(Y, logits) + shift_penalty * (q1**2 + q2**2)
                    if loss < best_loss:
                        best_loss = loss
                        best_shift = shift_candidate

            # Adjust best_loss to remove penalty for fair comparison
            best_shift_raw_loss = best_loss - shift_penalty * (best_shift[0]**2 + best_shift[1]**2)

            # Decide whether to shift: if no-shift loss is close to best shifted loss, prefer (0, 0)
            if loss_no_shift - best_shift_raw_loss < shift_threshold:
                current_shift = (0, 0)  # No shift if improvement is below threshold
            else:
                current_shift = best_shift  # Use best shift if it significantly improves loss

            if verbose:
                print(f"Optimal Shift Found = {current_shift}")
            
            X_tilde = self.transformer.forward(X, shift=current_shift)

            # --- Inner Loop (j) for Parameter Updates ---
            for j in range(J):
                # Update B
                offset_for_B = self._calculate_logits(X_tilde, self.A_, np.zeros_like(self.B_), self.gamma_, self.intercept_, Z)
                res_b = minimize(self._objective_B, x0=self.B_.flatten(), args=(X_tilde, Y, self.A_, offset_for_B), method='L-BFGS-B')
                self.B_ = res_b.x.reshape(self.term, self.d)
                
                # Update A
                offset_for_A = self._calculate_logits(X_tilde, np.zeros_like(self.A_), self.B_, self.gamma_, self.intercept_, Z)
                res_a = minimize(self._objective_A, x0=self.A_.flatten(), args=(X_tilde, Y, self.B_, offset_for_A), method='L-BFGS-B')
                self.A_ = res_a.x.reshape(self.term, self.p)

                # Update gamma and intercept
                if Z is not None:
                    offset_for_gamma = self._calculate_logits(X_tilde, self.A_, self.B_)
                    params_init = np.append(self.gamma_.flatten(), self.intercept_)
                    res_g = minimize(self._objective_gamma_intercept, x0=params_init, args=(Z, Y, offset_for_gamma), method='L-BFGS-B')
                    self.gamma_ = res_g.x[:-1].reshape(-1, 1)
                    self.intercept_ = res_g.x[-1]
                else: 
                    offset_for_intercept = self._calculate_logits(X_tilde, self.A_, self.B_)
                    res_i = minimize(lambda i: self._logistic_loss(Y, offset_for_intercept + i), x0=self.intercept_, method='L-BFGS-B')
                    self.intercept_ = res_i.x[0]

                if verbose and J > 1:
                    current_loss = self._logistic_loss(Y, self._calculate_logits(X_tilde, self.A_, self.B_, self.gamma_, self.intercept_, Z))
                    print(f"  Inner Iter {j+1}/{J}: Logistic Loss = {current_loss:.6f}")

            # --- Check for Convergence ---
            total_loss = self._logistic_loss(Y, self._calculate_logits(X_tilde, self.A_, self.B_, self.gamma_, self.intercept_, Z)) + self._regularization_penalty(self.A_, self.B_, self.gamma_)
            self.loss_history_.append(total_loss)
            if verbose: print(f"End of Outer Iter {t+1}: Total Penalized Loss={total_loss:.6f}")

            diff_A = np.linalg.norm(self.A_ - A_old_t)
            diff_B = np.linalg.norm(self.B_ - B_old_t)
            diff_gamma = 0.0
            if Z is not None:
                diff_gamma = np.linalg.norm(self.gamma_ - gamma_old_t)

            if t > 0 and diff_A < tol and diff_B < tol and diff_gamma < tol and current_shift == shift_old_t:
                if verbose: print(f"Convergence reached at iteration {t+1}.")
                break
        
        self.optimal_shift_ = current_shift
        return self

    def predict_proba(self, X, Z=None):
        X_tilde = self.transformer.forward(X, shift=self.optimal_shift_)
        logits = self._calculate_logits(X_tilde, self.A_, self.B_, self.gamma_, self.intercept_, Z)
        return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    def predict(self, X, Z=None, threshold=0.5):
        return (self.predict_proba(X, Z) >= threshold).astype(int)

    def get_coefficient_tensor(self):
        if self.A_ is None or self.B_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        p1,p2 = self.hparams['p1'], self.hparams['p2']
        d1,d2 = self.hparams['d1'], self.hparams['d2']
        p3,d3 = self.hparams.get('p3', 1), self.hparams.get('d3', 1)
        C = np.zeros((p1*d1, p2*d2, p3*d3))
        for r in range(self.term):
            Ar = self.A_[r, :].reshape(p1, p2, p3)
            Br = self.B_[r, :].reshape(d1, d2, d3)
            C += np.kron(Ar, Br)
        return C.squeeze()