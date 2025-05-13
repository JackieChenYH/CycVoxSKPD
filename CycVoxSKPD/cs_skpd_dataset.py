from CycVoxSKPD.ten_mat_operator import Ten2MatOperator, CyclicShiftOperator
import numpy as np

class CS_SKPDDataSet:
    def __init__(self, hparams, X, Y, Z=None):
        assert len(X) == len(Y), "X and Y must have the same length."
        self.p1 = hparams['p1']
        self.d1 = hparams['d1']
        self.p2 = hparams['p2']
        self.d2 = hparams['d2']
        self.p3 = hparams['p3']
        self.d3 = hparams['d3']
        self.use_cyclic = hparams.get('use_cyclic', True)
        self.cyclic_shift = CyclicShiftOperator()

        self.X = X  # Original input tensors
        self.Y = Y  # Labels
        self.Z = Z  # Optional covariates
        self.n = len(X)

        # Generate X2 as a cyclically shifted version of X if use_cyclic is True
        if self.use_cyclic:
            self.X2 = np.array([self.cyclic_shift.cs(x, forward=True) for x in self.X])
        else:
            self.X2 = self.X  # No shift if use_cyclic=False

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        X1 = self.X[idx]
        X2 = self.X2[idx]
        Y = self.Y[idx]
        if self.Z is not None:
            Z_sample = self.Z[idx]
            return X1, X2, Y, Z_sample
        return X1, X2, Y        