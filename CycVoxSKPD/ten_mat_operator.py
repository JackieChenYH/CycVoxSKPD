import numpy as np

class Ten2MatOperator:
    def __init__(self, p1, d1, p2, d2, p3=1, d3=1):
        self.p1 = p1
        self.d1 = d1
        self.p2 = p2
        self.d2 = d2
        self.p3 = p3
        self.d3 = d3

    def forward(self, C):
        if C.ndim == 2:  # Matrix case
            m, n = C.shape
            assert m == self.p1 * self.d1 and n == self.p2 * self.d2, f"Matrix dimension wrong !!!!!Expected m: {self.p1 * self.d1}, n: {self.p2 * self.d2}; Got m: {m}, n: {n}"
            
            RC = []
            for i in range(self.p1):
                for j in range(self.p2):
                    Cij = C[self.d1 * i:self.d1 * (i + 1), self.d2 * j:self.d2 * (j + 1)]
                    RC.append(Cij.flatten())
            return np.stack(RC, axis=1).T

        elif C.ndim == 3:  # Tensor case
            m, n, d = C.shape
            assert m == self.p1 * self.d1 and n == self.p2 * self.d2 and d == self.p3 * self.d3, f"Tensor dimension wrong !!!! Expected m: {self.p1 * self.d1}, n: {self.p2 * self.d2}, d: {self.p3 * self.d3}; Got m: {m}, n: {n}, d: {d}"
            
            RC = []
            for i in range(self.p1):
                for j in range(self.p2):
                    for k in range(self.p3):
                        Cij = C[self.d1 * i:self.d1 * (i + 1), self.d2 * j:self.d2 * (j + 1), self.d3 * k:self.d3 * (k + 1)]
                        RC.append(Cij.flatten())
            return np.stack(RC, axis=1).T
        else:
            raise ValueError("Input must be a 2D or 3D array.")

class Mat2TenOperator:
    def __init__(self, p1, d1, p2, d2, p3=1, d3=1):
        self.p1 = p1
        self.d1 = d1
        self.p2 = p2
        self.d2 = d2
        self.p3 = p3
        self.d3 = d3

    def forward(self, RC):
        if self.p3 == 1 and self.d3 == 1: # Matrix case
            p1p2, d1d2 = RC.shape
            C = np.zeros((self.p1 * self.d1, self.p2 * self.d2))
            for i in range(p1p2):
                Block = RC[i, :].reshape(self.d1, self.d2)
                ith = i // self.p2
                jth = i % self.p2
                C[self.d1 * ith:self.d1 * (ith + 1), self.d2 * jth:self.d2 * (jth + 1)] = Block
            return C
        else:                               # Tensor case
            p1p2p3, d1d2d3 = RC.shape
            C = np.zeros((self.p1 * self.d1, self.p2 * self.d2, self.p3 * self.d3))
            p2p3 = self.p2 * self.p3
            for i in range(p1p2p3):
                Block = RC[i, :].reshape(self.d1, self.d2, self.d3)
                ith = i // p2p3
                reminder = i % p2p3
                jth = reminder // self.p3
                kth = reminder % self.p3
                C[self.d1 * ith:self.d1 * (ith + 1), self.d2 * jth:self.d2 * (jth + 1), self.d3 * kth:self.d3 * (kth + 1)] = Block
            return C


class TranslationShiftOperator:
    def __init__(self, shift_amount=(1, 1, 1)):
        self.shift_amount = shift_amount  # Tuple (s1, s2, s3) for shifts in each dimension

    def cs(self, tensor, forward=True):
        """Interface compatible with CyclicShiftOperator's cs method."""
        if forward:
            return self.forward(tensor)
        else:
            return self.backward(tensor)

    def forward(self, tensor):
        """Shift tensor forward by shift_amount, padding with zeros."""
        if tensor.ndim == 2:
            return self._shift_2d(tensor, self.shift_amount[:2])
        elif tensor.ndim == 3:
            return self._shift_3d(tensor, self.shift_amount)
        else:
            raise ValueError("Input must be a 2D matrix or 3D tensor.")

    def backward(self, tensor):
        """Shift tensor backward by the inverse of shift_amount, padding with zeros."""
        if tensor.ndim == 2:
            return self._shift_2d(tensor, (-self.shift_amount[0], -self.shift_amount[1]))
        elif tensor.ndim == 3:
            return self._shift_3d(tensor, (-self.shift_amount[0], -self.shift_amount[1], -self.shift_amount[2]))
        else:
            raise ValueError("Input must be a 2D matrix or 3D tensor.")

    def _shift_2d(self, tensor, shift):
        s1, s2 = shift
        p1, p2 = tensor.shape
        shifted = np.zeros_like(tensor)
        if s1 >= 0 and s2 >= 0:
            shifted[s1:, s2:] = tensor[:-s1, :-s2] if s1 < p1 and s2 < p2 else shifted[s1:, s2:]
        elif s1 < 0 and s2 < 0:
            shifted[:s1, :s2] = tensor[-s1:, -s2:]
        return shifted

    def _shift_3d(self, tensor, shift):
        s1, s2, s3 = shift
        p1, p2, p3 = tensor.shape
        shifted = np.zeros_like(tensor)
        if s1 >= 0 and s2 >= 0 and s3 >= 0:
            shifted[s1:, s2:, s3:] = tensor[:-s1, :-s2, :-s3] if s1 < p1 and s2 < p2 and s3 < p3 else shifted[s1:, s2:, s3:]
        elif s1 < 0 and s2 < 0 and s3 < 0:
            shifted[:s1, :s2, :s3] = tensor[-s1:, -s2:, -s3:]
        return shifted

class CyclicShiftOperator:
    """
    Class to perform cyclic shifts on 2D matrices and 3D tensors along each axis.

    Methods:
    --------
    cs(tensor, forward=True):
        Performs a cyclic shift on the input tensor or matrix. If `forward` is True, it
        shifts elements forward; otherwise, it shifts them backward.
    """

    def __init__(self):
        """Initializes the CyclicShiftOperator class."""
        pass

    def cs(self, tensor, forward=True, original_dims=None, scaling_factors=None, custom_shift_amounts=None):
        if tensor.ndim == 2:
            p1, p2 = tensor.shape
            if original_dims and scaling_factors:
                orig_p1, orig_p2 = original_dims
                d1, d2 = scaling_factors
                mid_p1 = orig_p1 // 2 * d1
                mid_p2 = orig_p2 // 2 * d2
            else:
                mid_p1 = p1 // 2
                mid_p2 = p2 // 2
            # print(f"2D Shift amounts: mid_p1={mid_p1}, mid_p2={mid_p2}")

            idx1 = list(range(p1))
            idx2 = list(range(p2))

            if forward:
                idx1 = idx1[-mid_p1:] + idx1[:-mid_p1]
                idx2 = idx2[-mid_p2:] + idx2[:-mid_p2]
            else:
                idx1 = idx1[mid_p1:] + idx1[:mid_p1]
                idx2 = idx2[mid_p2:] + idx2[:mid_p2]

            shifted_tensor = tensor[idx1, :]
            shifted_tensor = shifted_tensor[:, idx2]

        elif tensor.ndim == 3:
            p1, p2, p3 = tensor.shape
            if original_dims and scaling_factors:
                orig_p1, orig_p2, orig_p3 = original_dims
                d1, d2, d3 = scaling_factors
                mid_p1 = orig_p1 // 2 * d1
                mid_p2 = orig_p2 // 2 * d2
                mid_p3 = orig_p3 // 2 * d3
            else:
                mid_p1 = p1 // 2
                mid_p2 = p2 // 2
                mid_p3 = p3 // 2
            # print(f"3D Shift amounts: mid_p1={mid_p1}, mid_p2={mid_p2}, mid_p3={mid_p3}")

            idx1 = list(range(p1))
            idx2 = list(range(p2))
            idx3 = list(range(p3))

            if forward:
                idx1 = idx1[-mid_p1:] + idx1[:-mid_p1]
                idx2 = idx2[-mid_p2:] + idx2[:-mid_p2]
                idx3 = idx3[-mid_p3:] + idx3[:-mid_p3]
            else:
                idx1 = idx1[mid_p1:] + idx1[:mid_p1]
                idx2 = idx2[mid_p2:] + idx2[:mid_p2]
                idx3 = idx3[mid_p3:] + idx3[:mid_p3]

            shifted_tensor = tensor[idx1, :, :]
            shifted_tensor = shifted_tensor[:, idx2, :]
            shifted_tensor = shifted_tensor[:, :, idx3]

        else:
            raise ValueError("Input must be a 2D matrix or a 3D tensor.")

        return shifted_tensor