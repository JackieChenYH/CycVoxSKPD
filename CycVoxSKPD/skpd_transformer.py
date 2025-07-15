# skpd_transformer.py

import numpy as np

class SKPD_Transformer:
    """
    Transforms and shifts tensor data for the Sparse Kronecker Product Decomposition model.
    Applies a global cyclic shift to the entire tensor *before* unfolding.
    """
    def __init__(self, p1: int, d1: int, p2: int, d2: int, p3: int = 1, d3: int = 1):
        self.p1, self.d1 = p1, d1
        self.p2, self.d2 = p2, d2
        self.p3, self.d3 = p3, d3
        self.p = self.p1 * self.p2 * self.p3
        self.d = self.d1 * self.d2 * self.d3

    def forward(self, X: np.ndarray, shift: tuple[int, int] = (0, 0)) -> np.ndarray:
        # Ensure input is a batch
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        elif X.ndim == 3 and self.p3 * self.d3 > 1:
             X = X[np.newaxis, ...]

        n_samples = X.shape[0]
        is_3d_tensor = (X.ndim == 4)

        # Apply the cyclic shift to the entire tensor(s) first.
        if shift != (0, 0):
            X = np.roll(X, shift=shift, axis=(1, 2))

        X_transformed = np.zeros((n_samples, self.p, self.d))

        for i in range(n_samples):
            sample = X[i]
            vectorized_patches = []
            
            for k1 in range(self.p1):
                for k2 in range(self.p2):
                    if is_3d_tensor:
                        for k3 in range(self.p3):
                            patch = sample[k1*self.d1:(k1+1)*self.d1,
                                           k2*self.d2:(k2+1)*self.d2,
                                           k3*self.d3:(k3+1)*self.d3]
                            vectorized_patches.append(patch.flatten())
                    else:
                        patch = sample[k1*self.d1:(k1+1)*self.d1,
                                       k2*self.d2:(k2+1)*self.d2]
                        vectorized_patches.append(patch.flatten())
            
            X_transformed[i, :, :] = np.array(vectorized_patches)

        return X_transformed