import numpy as np
from BPNN_Decoder.utils import load_code


def encode_and_transmission(G_matrix, SNR, batch_size, noise_io, rng=0):
    """
    Generate some random bits, encode it to valid codewords and simulate transmission
    """
    K, N = np.shape(G_matrix)
    if rng == 0:
        x_bits = np.random.randint(0, 2, size=(batch_size, K))
    else:
        x_bits = rng.randint(0, 2, size=(batch_size, K))

    # Coding
    u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # G_matrix

    s_mod = u_coded_bits * (-2) + 1  # BPSK modulation
    ch_noise_normalize = noise_io.generate_noise(batch_size)  # plus the noise
    ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
    ch_noise = ch_noise_normalize * ch_noise_sigma

    y_receive = s_mod + ch_noise
    LLRs = y_receive * 2.0 / (ch_noise_sigma * ch_noise_sigma)

    return x_bits, u_coded_bits, s_mod, ch_noise, y_receive, LLRs


class Code:
    """
    Base Code class which simply handles that we need to have a K,
    which is 0 by default, trimming the first K bits as the output of
    the code
    """

    K: int = 0

    def dec_src_bits(self, output):
        return output[:, 0 : self.K]


class LDPC(Code):
    def __init__(self, N, K, file_G, file_H):
        self.N = N
        self.K = K
        self.G_matrix, self.H_matrix = self.init_LDPC_G_H(file_G, file_H)

    def init_LDPC_G_H(self, file_G, file_H):
        G_matrix_row_col = np.loadtxt(file_G, dtype=np.int32)
        H_matrix_row_col = np.loadtxt(file_H, dtype=np.int32)
        G_matrix = np.zeros([self.K, self.N], dtype=np.int32)
        H_matrix = np.zeros([self.N - self.K, self.N], dtype=np.int32)
        G_matrix[G_matrix_row_col[:, 0], G_matrix_row_col[:, 1]] = 1
        H_matrix[H_matrix_row_col[:, 0], H_matrix_row_col[:, 1]] = 1
        return G_matrix, H_matrix


class BCH(Code):
    def __init__(self, H_filename, G_filename):
        self.code = load_code(H_filename, G_filename)
        self.N = self.code.N
        self.K = self.code.K
        self.G_matrix = self.code.G
        self.H_matrix = self.code.H
