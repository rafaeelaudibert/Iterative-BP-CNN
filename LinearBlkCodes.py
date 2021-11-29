import numpy as np
import NeuralBPNN.utils as NeuralBPNNUtils


def encode_and_transmission(G_matrix, SNR, batch_size, noise_io, rng=0):
    """
    Generate some random bits, encode it to valid codewords and simulate transmission
    """
    K, N = np.shape(G_matrix)
    if rng == 0:
        x_bits = np.random.randint(0, 2, size=(batch_size, K))
    else:
        x_bits = rng.randint(0, 2, size=(batch_size, K))

    u_coded_bits = np.mod(np.matmul(x_bits, G_matrix), 2)  # Coding
    s_mod = u_coded_bits * (-2) + 1  # BPSK modulation
    ch_noise_normalize = noise_io.generate_noise(batch_size)  # plus the noise

    ch_noise_sigma = np.sqrt(1 / np.power(10, SNR / 10.0) / 2.0)
    ch_noise = ch_noise_normalize * ch_noise_sigma

    y_receive = s_mod + ch_noise
    LLR = y_receive * 2.0 / (ch_noise_sigma * ch_noise_sigma)

    return x_bits, u_coded_bits, s_mod, ch_noise, y_receive, LLR


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
        self.load_code(H_filename, G_filename)

    def load_code(self, H_filename, G_filename):
        # parity-check matrix; Tanner graph parameters
        with open(H_filename) as f:
            # get n and m (n-k) from first line
            n, m = [int(s) for s in f.readline().split(" ")]
            k = n - m

            var_degrees = np.zeros(n).astype(np.int)  # degree of each variable node
            chk_degrees = np.zeros(m).astype(np.int)  # degree of each check node

            # initialize H
            H = np.zeros([m, n]).astype(np.int)
            max_var_degree, max_chk_degree = [int(s) for s in f.readline().split(" ")]
            f.readline()  # ignore two lines
            f.readline()

            # create H, sparse version of H, and edge index matrices
            # (edge index matrices used to calculate source and destination nodes during belief propagation)
            var_edges = [[] for _ in range(0, n)]
            for i in range(0, n):
                row_string = f.readline().split(" ")
                var_edges[i] = [(int(s) - 1) for s in row_string[:-1]]
                var_degrees[i] = len(var_edges[i])
                H[var_edges[i], i] = 1

            chk_edges = [[] for _ in range(0, m)]
            for i in range(0, m):
                row_string = f.readline().split(" ")
                chk_edges[i] = [(int(s) - 1) for s in row_string[:-1]]
                chk_degrees[i] = len(chk_edges[i])

            d = [[] for _ in range(0, n)]
            edge = 0
            for i in range(0, n):
                for j in range(0, var_degrees[i]):
                    d[i].append(edge)
                    edge += 1

            u = [[] for _ in range(0, m)]
            edge = 0
            for i in range(0, m):
                for j in range(0, chk_degrees[i]):
                    v = chk_edges[i][j]
                    for e in range(0, var_degrees[v]):
                        if i == var_edges[v][e]:
                            u[i].append(d[v][e])

            num_edges = H.sum()

        if G_filename == "":
            G = []
        else:
            if "BCH" or "LDPC" in H_filename:  # dear God please fix this
                G = np.loadtxt(G_filename).astype(np.int)
                G = G.transpose()
            else:
                P = np.loadtxt(G_filename, skiprows=2)
                G = np.vstack([P.transpose(), np.eye(k)]).astype(np.int)

        self.H, self.H_matrix = (H, H)
        self.G, self.G_matrix = (G, G)
        self.var_degrees = var_degrees
        self.chk_degrees = chk_degrees
        self.num_edges = num_edges
        self.u = u
        self.d = d
        self.n = n
        self.N, self.n = (n, n)
        self.m = m
        self.K, self.k = (k, k)
        self.K = k
