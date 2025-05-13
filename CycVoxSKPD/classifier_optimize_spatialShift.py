import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from CycVoxSKPD.ten_mat_operator import Ten2MatOperator, CyclicShiftOperator, TranslationShiftOperator


class SKPD_LogisticRegressor_Cyclic:
    def __init__(self, hparams: dict, Ds: list = None, max_iter: int = 100, verbose: bool = True):
        self.p1 = hparams['p1']
        self.d1 = hparams['d1']
        self.p2 = hparams['p2']
        self.d2 = hparams['d2']
        self.p3 = hparams['p3']
        self.d3 = hparams['d3']
        self.term = hparams['term']
        self.lmbda_a = hparams.get('lmbda_a', 0.0)
        self.lmbda_b = hparams.get('lmbda_b', 0.0)
        self.lmbda_gamma = hparams.get('lmbda_gamma', 0.0)
        self.alpha = hparams.get('alpha', 0.25)
        self.normalization_A = hparams.get('normalization_A', False)
        self.normalization_B = hparams.get('normalization_B', False)
        self.use_cyclic = hparams.get('use_cyclic', True)
        self.max_iter = max_iter
        self.verbose = verbose

        self.s1 = self.p1 * self.p2 * self.p3 
        self.s2 = self.d1 * self.d2 * self.d3

        # Initialize parameters
        self.A1 = np.zeros((self.term, self.s1))
        self.A2 = np.zeros((self.term, self.s1))
        self.B1 = np.random.normal(0, 0.01, (self.term, self.s2))
        self.B2 = np.random.normal(0, 0.01, (self.term, self.s2))
        self.gamma = None

        self.Ds = Ds
        self.n = len(self.Ds) if Ds is not None else 0
        self.X1 = np.array([self.Ds[i][0] for i in range(self.n)]) if Ds is not None else None
        self.X2 = np.array([self.Ds[i][1] for i in range(self.n)]) if Ds is not None else None
        self.Y = np.array([self.Ds[i][2] for i in range(self.n)]) if Ds is not None else None
        self.Z = np.array([self.Ds[i][3] for i in range(self.n)]) if Ds is not None and len(self.Ds[0]) > 3 else None
        if self.Z is not None:
            self.gamma = np.zeros((self.Z.shape[1], 1))

        self.ten2mat = Ten2MatOperator(self.p1, self.d1, self.p2, self.d2, self.p3, self.d3)
        self.cyclic_shift = CyclicShiftOperator()
        self.shift_operator = TranslationShiftOperator(shift_amount=(1, 1, 1))
        self.X1_transformed = np.array([self.ten2mat.forward(x) for x in self.X1]) if self.X1 is not None else None

        if self.use_cyclic and self.X2 is not None:
            # Apply the forward shift in the original space
            shifted_X2 = np.array([self.cyclic_shift.cs(
                x if x.ndim == 3 else x[..., np.newaxis],
                forward=True,
                original_dims=(self.p1 * self.d1, self.p2 * self.d2, self.p3 * self.d3)
            ) for x in self.X2])

            # Log after shift (no plot here as per your request)
            sample_shifted_X2 = shifted_X2[0]
            sample_shifted_X2_2d = sample_shifted_X2.squeeze() if self.p3 * self.d3 == 1 else sample_shifted_X2[:, :, 0]
            shifted_X2_max_pos = np.unravel_index(np.argmax(sample_shifted_X2_2d), sample_shifted_X2_2d.shape)
            # print(f"Sample X2 largest value position after forward shift (original space 128x128): {shifted_X2_max_pos}")

            self.X2_transformed = np.array([self.ten2mat.forward(x) for x in shifted_X2])
        else:
            self.X2_transformed = self.X1_transformed
        # Scalers
        self.scaler_X1 = StandardScaler()
        self.scaler_X2 = StandardScaler()
        self.scaler_Z = StandardScaler() if self.Z is not None else None
        if self.X1_transformed is not None:
            self.X1_transformed = self.X1_transformed.reshape(self.n, self.s1, self.s2)
            X1_flat = self.X1_transformed.reshape(self.n, self.s1 * self.s2)  # (n, 16384)
            self.scaler_X1.fit(X1_flat)
            self.X1_transformed = self.scaler_X1.transform(X1_flat).reshape(self.n, self.s1, self.s2)
        if self.X2_transformed is not None:
            self.X2_transformed = self.X2_transformed.reshape(self.n, self.s1, self.s2)
            X2_flat = self.X2_transformed.reshape(self.n, self.s1 * self.s2)
            self.scaler_X2.fit(X2_flat)
            self.X2_transformed = self.scaler_X2.transform(X2_flat).reshape(self.n, self.s1, self.s2)
        if self.Z is not None:
            self.Z = self.scaler_Z.fit_transform(self.Z)

    def logistic_loss(self, y_true, probs):
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))

    def regularization_penalty(self):
        penalty_a = self.lmbda_a * np.sum([np.linalg.norm(self.A1[r, :], ord=1) + np.linalg.norm(self.A2[r, :], ord=1) for r in range(self.term)])
        penalty_b = self.lmbda_b * np.sum([
            self.alpha * (np.linalg.norm(self.B1[r, :], ord=1) + np.linalg.norm(self.B2[r, :], ord=1)) + 
            (1 - self.alpha) * (np.linalg.norm(self.B1[r, :], ord=2)**2 + np.linalg.norm(self.B2[r, :], ord=2)**2)
            for r in range(self.term)
        ])
        penalty_gamma = self.lmbda_gamma * np.sum(np.abs(self.gamma)) if self.gamma is not None else 0.0
        return penalty_a + penalty_b + penalty_gamma

    def _predict_proba_transformed(self, X1_transformed, X2_transformed, A1=None, B1=None, A2=None, B2=None, Z=None, gamma=None, exclude=None, return_logits=False):
        batch_size = X1_transformed.shape[0]
        logits = np.zeros(batch_size)
        A1 = self.A1 if A1 is None else A1
        B1 = self.B1 if B1 is None else B1
        A2 = self.A2 if A2 is None else A2
        B2 = self.B2 if B2 is None else B2
        for i in range(batch_size):
            logit = 0
            for r in range(self.term):
                if exclude != 'B1' and exclude != 'A1':
                    logit += np.dot(A1[r, :], X1_transformed[i] @ B1[r, :])
                if self.use_cyclic and X2_transformed is not None:
                    if exclude != 'B2' and exclude != 'A2':
                        logit += np.dot(A2[r, :], X2_transformed[i] @ B2[r, :])
            if Z is not None and gamma is not None and exclude != 'gamma':
                logit += np.dot(Z[i], gamma).squeeze()
            logits[i] = logit
        if return_logits:
            return logits
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        return probs
    
    def predict_proba(self, X1, X2=None):
        X1_transformed = np.array([self.ten2mat.forward(x) for x in X1])
        X1_transformed = X1_transformed.reshape(len(X1), self.s1, self.s2)
        X1_flat = X1_transformed.reshape(len(X1), self.s1 * self.s2)
        X1_transformed = self.scaler_X1.transform(X1_flat).reshape(len(X1), self.s1, self.s2)

        if self.use_cyclic and X2 is not None:
            X2_transformed = np.array([self.ten2mat.forward(self.cyclic_shift.cs(x, forward=True)) for x in X2])
            X2_transformed = X2_transformed.reshape(len(X2), self.s1, self.s2)
            X2_flat = X2_transformed.reshape(len(X2), self.s1 * self.s2)
            X2_transformed = self.scaler_X2.transform(X2_flat).reshape(len(X2), self.s1, self.s2)
        else:
            X2_transformed = X1_transformed

        return self._predict_proba_transformed(X1_transformed, X2_transformed, Z=self.Z, gamma=self.gamma)

    def _objective_B1(self, B1_flat, F1_scaled, Y, offset):
        B1 = B1_flat.reshape(self.term, self.s2)
        logits = F1_scaled @ B1_flat + offset
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        loss = -np.mean(Y * np.log(probs + 1e-15) + (1 - Y) * np.log(1 - probs + 1e-15))
        penalty_b = self.lmbda_b * np.sum([
            self.alpha * np.linalg.norm(B1[r, :], ord=1) + 
            (1 - self.alpha) * np.linalg.norm(B1[r, :], ord=2)**2 
            for r in range(self.term)
        ])
        return loss + penalty_b

    def _objective_B2(self, B2_flat, F2_scaled, Y, offset):
        B2 = B2_flat.reshape(self.term, self.s2)
        logits = F2_scaled @ B2_flat + offset
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        loss = -np.mean(Y * np.log(probs + 1e-15) + (1 - Y) * np.log(1 - probs + 1e-15))
        penalty_b = self.lmbda_b * np.sum([
            self.alpha * np.linalg.norm(B2[r, :], ord=1) + 
            (1 - self.alpha) * np.linalg.norm(B2[r, :], ord=2)**2 
            for r in range(self.term)
        ])
        return loss + penalty_b

    def _objective_A1(self, A1_flat, G1_scaled, Y, offset):
        A1 = A1_flat.reshape(self.term, self.s1)
        logits = G1_scaled @ A1_flat + offset
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        loss = -np.mean(Y * np.log(probs + 1e-15) + (1 - Y) * np.log(1 - probs + 1e-15))
        penalty_a = self.lmbda_a * np.sum([np.linalg.norm(A1[r, :], ord=1) for r in range(self.term)])
        return loss + penalty_a

    def _objective_A2(self, A2_flat, G2_scaled, Y, offset):
        A2 = A2_flat.reshape(self.term, self.s1)
        logits = G2_scaled @ A2_flat + offset
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        loss = -np.mean(Y * np.log(probs + 1e-15) + (1 - Y) * np.log(1 - probs + 1e-15))
        penalty_a = self.lmbda_a * np.sum([np.linalg.norm(A2[r, :], ord=1) for r in range(self.term)])
        return loss + penalty_a

    def _objective_gamma(self, gamma_flat, Z_scaled, Y, offset):
        gamma = gamma_flat.reshape(-1, 1)
        logits = Z_scaled @ gamma_flat + offset
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        loss = -np.mean(Y * np.log(probs + 1e-15) + (1 - Y) * np.log(1 - probs + 1e-15))
        penalty_gamma = self.lmbda_gamma * np.sum(np.abs(gamma))
        return loss + penalty_gamma

    def update_B1(self):
        F1 = np.zeros((self.n, self.s2 * self.term))
        for i in range(self.n):
            for r in range(self.term):
                start_idx = r * self.s2
                end_idx = (r + 1) * self.s2
                F1[i, start_idx:end_idx] = self.X1_transformed[i].T @ self.A1[r, :]
        scaler_F1 = StandardScaler()
        F1_scaled = scaler_F1.fit_transform(F1)
        # Offset excludes B1 contribution
        offset = np.zeros(self.n)
        if self.use_cyclic:
            offset += self._predict_proba_transformed(self.X1_transformed, self.X2_transformed, exclude='B1')
        if self.Z is not None:
            offset += np.dot(self.Z, self.gamma).flatten()
        offset = offset / np.max(np.abs(offset)) if np.max(np.abs(offset)) > 1e-10 else np.zeros_like(offset)

        B1_init = self.B1.ravel()
        result = minimize(
            fun=self._objective_B1,
            x0=B1_init,
            args=(F1_scaled, self.Y, offset),
            method='L-BFGS-B',
            options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-4, 'gtol': 1e-4}
        )
        # if not result.success:
        #     print(f"Optimization for B1 failed: {result.message}")
        self.B1 = result.x.reshape(self.term, self.s2)
        if self.normalization_B:
            for r in range(self.term):
                norm = np.linalg.norm(self.B1[r, :], ord=2)
                if norm > 0:
                    self.B1[r, :] /= norm
        # print(f"B1 after update - min: {self.B1.min()}, max: {self.B1.max()}")
    

    def update_B2(self):
        if not self.use_cyclic:
            self.B2 = np.zeros_like(self.B2)
            return
        F2 = np.zeros((self.n, self.s2 * self.term))
        for i in range(self.n):
            for r in range(self.term):
                start_idx = r * self.s2
                end_idx = (r + 1) * self.s2
                F2[i, start_idx:end_idx] = self.X2_transformed[i].T @ self.A2[r, :]
        scaler_F2 = StandardScaler()
        F2_scaled = scaler_F2.fit_transform(F2)
        # Offset excludes B2 contribution
        offset = np.zeros(self.n)
        offset += self._predict_proba_transformed(self.X1_transformed, self.X2_transformed, exclude='B2')
        if self.Z is not None:
            offset += np.dot(self.Z, self.gamma).flatten()
        offset = offset / np.max(np.abs(offset)) if np.max(np.abs(offset)) > 1e-10 else np.zeros_like(offset)

        B2_init = self.B2.ravel()
        result = minimize(
            fun=self._objective_B2,
            x0=B2_init,
            args=(F2_scaled, self.Y, offset),
            method='L-BFGS-B',
            options={'maxiter': 10000, 'maxfun': 50000, 'ftol': 1e-4, 'gtol': 1e-4}
        )
        # if not result.success:
        #     print(f"Optimization for B2 failed: {result.message}")
        self.B2 = result.x.reshape(self.term, self.s2)
        if self.normalization_B:
            for r in range(self.term):
                norm = np.linalg.norm(self.B2[r, :], ord=2)
                if norm > 0:
                    self.B2[r, :] /= norm


    def update_A1(self):
        G1 = np.zeros((self.n, self.s1 * self.term))
        for i in range(self.n):
            for r in range(self.term):
                start_idx = r * self.s1
                end_idx = (r + 1) * self.s1
                G1[i, start_idx:end_idx] = self.X1_transformed[i] @ self.B1[r, :]
        scaler_G1 = StandardScaler()
        G1_scaled = scaler_G1.fit_transform(G1)
        # Offset excludes A1 contribution
        offset = np.zeros(self.n)
        if self.use_cyclic:
            offset += self._predict_proba_transformed(self.X1_transformed, self.X2_transformed, exclude='A1')
        if self.Z is not None:
            offset += np.dot(self.Z, self.gamma).flatten()
        offset = offset / np.max(np.abs(offset)) if np.max(np.abs(offset)) > 1e-10 else np.zeros_like(offset)

        A1_init = self.A1.ravel()
        result = minimize(
            fun=self._objective_A1,
            x0=A1_init,
            args=(G1_scaled, self.Y, offset),
            method='L-BFGS-B',
            options={'maxiter': 25000, 'maxfun': 50000, 'ftol': 1e-4, 'gtol': 1e-4}
        )
        # if not result.success:
        #     print(f"Optimization for A1 failed: {result.message}")
        self.A1 = result.x.reshape(self.term, self.s1)
        if self.normalization_A:
            for r in range(self.term):
                norm = np.linalg.norm(self.A1[r, :], ord=2)
                if norm > 0:
                    self.A1[r, :] /= norm

    def update_A2(self):
        if not self.use_cyclic:
            self.A2 = np.zeros_like(self.A2)
            return
        G2 = np.zeros((self.n, self.s1 * self.term))
        for i in range(self.n):
            for r in range(self.term):
                start_idx = r * self.s1
                end_idx = (r + 1) * self.s1
                G2[i, start_idx:end_idx] = self.X2_transformed[i] @ self.B2[r, :]
        scaler_G2 = StandardScaler()
        G2_scaled = scaler_G2.fit_transform(G2)
        # Offset excludes A2 contribution
        offset = np.zeros(self.n)
        offset += self._predict_proba_transformed(self.X1_transformed, self.X2_transformed, exclude='A2')
        if self.Z is not None:
            offset += np.dot(self.Z, self.gamma).flatten()
        offset = offset / np.max(np.abs(offset)) if np.max(np.abs(offset)) > 1e-10 else np.zeros_like(offset)

        A2_init = self.A2.ravel()
        result = minimize(
            fun=self._objective_A2,
            x0=A2_init,
            args=(G2_scaled, self.Y, offset),
            method='L-BFGS-B',
            options={'maxiter': 25000, 'maxfun': 50000, 'ftol': 1e-4, 'gtol': 1e-4}
        )
        # if not result.success:
        #     print(f"Optimization for A2 failed: {result.message}")
        self.A2 = result.x.reshape(self.term, self.s1)
        if self.normalization_A:
            for r in range(self.term):
                norm = np.linalg.norm(self.A2[r, :], ord=2)
                if norm > 0:
                    self.A2[r, :] /= norm

    def update_gamma(self):
        if self.Z is None:
            return
        Z_scaled = self.scaler_Z.transform(self.Z)
        offset = self._predict_proba_transformed(self.X1_transformed, self.X2_transformed)
        offset = offset / np.max(np.abs(offset)) if np.max(np.abs(offset)) > 1e-10 else np.zeros_like(offset)

        gamma_init = self.gamma.ravel() if self.gamma is not None else np.random.normal(0, 0.1, self.Z.shape[1])
        result = minimize(
            fun=self._objective_gamma,
            x0=gamma_init,
            args=(Z_scaled, self.Y, offset),
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-4, 'gtol': 1e-4}
        )
        # if not result.success:
        #     print(f"Optimization for gamma failed: {result.message}")
        self.gamma = result.x.reshape(-1, 1)

    def fit(self):
        # Initialize A1 using SVD on weighted average of X1_transformed
        weighted_average_X1 = np.mean([self.Y[i] * self.X1_transformed[i] for i in range(self.n)], axis=0)
        U, _, _ = np.linalg.svd(weighted_average_X1)
        self.A1 = U[:, :self.term].T  # Shape: (term, s1)

        # Initialize A2 using SVD on weighted average of X2_transformed
        weighted_average_X2 = np.mean([self.Y[i] * self.X2_transformed[i] for i in range(self.n)], axis=0)
        U, _, _ = np.linalg.svd(weighted_average_X2)
        self.A2 = U[:, :self.term].T  # Shape: (term, s1)

        # If not using cyclic/translation shift, zero out A2 and B2
        if not self.use_cyclic:
            self.A2 = np.zeros((self.term, self.s1))
            self.B2 = np.zeros((self.term, self.s2))

        # Optimization loop
        prev_loss = float('inf')
        for iteration in range(self.max_iter):
            print()
            self.update_B1()
            self.update_A1()
            if self.use_cyclic:
                self.update_B2()
                self.update_A2()
            if self.Z is not None:
                self.update_gamma()
            probs = self._predict_proba_transformed(self.X1_transformed, self.X2_transformed)
            loss = self.logistic_loss(self.Y, probs) + self.regularization_penalty()

            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.6f}")

            # Convergence check (skip first iteration)
            if iteration > 0 and abs(prev_loss - loss) / max(prev_loss, 1e-10) < 1e-5 and iteration >= 5:
                if self.verbose:
                    print("Convergence detected.")
                break
            prev_loss = loss

        # Construct final coefficient tensor C
        C = np.zeros((self.p1 * self.d1, self.p2 * self.d2, self.p3 * self.d3))
        for r in range(self.term):
            # C1 contribution (unshifted)
            A1_r = self.A1[r, :].reshape(self.p1, self.p2, self.p3)
            B1_r = self.B1[r, :].reshape(self.d1, self.d2, self.d3)
            A1_r_2d = A1_r.squeeze() if self.p3 == 1 else A1_r[:, :, 0]
            A1_max_pos = np.unravel_index(np.argmax(A1_r_2d), A1_r_2d.shape)
            print(f"A1_r largest circle position (32x32 space): {A1_max_pos}")
            C += np.kron(A1_r, B1_r)

            # C2 contribution (shifted back)
            if self.use_cyclic:
                A2_r = self.A2[r, :].reshape(self.p1, self.p2, self.p3)
                B2_r = self.B2[r, :].reshape(self.d1, self.d2, self.d3)
                kron_A2_B2 = np.kron(A2_r, B2_r)  # Shape: (p1*d1, p2*d2, p3*d3)
                # Shift C2 backward to align with original positions
                C += self.shift_operator.cs(kron_A2_B2, forward=False)

        # Squeeze if 2D case
        if self.p3 * self.d3 == 1:
            C = C.squeeze(axis=2)

        return self.A1, self.A2, self.B1, self.B2, self.gamma, C, loss
    def predict(self, X1, X2=None, threshold=0.5):
        probs = self.predict_proba(X1, X2)
        return (probs >= threshold).astype(int)