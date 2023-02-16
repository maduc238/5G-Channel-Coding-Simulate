import tensorflow as tf
import numpy as np
import scipy as sp
from tensorflow.keras.layers import Layer
from importlib_resources import files, as_file
from . import codes 
import numbers

class LDPC5GEncoder(Layer):
    def __init__(self, k, n, num_bits_per_symbol=None, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert isinstance(k, numbers.Number), "k phải là một số."
        assert isinstance(n, numbers.Number), "n phải là một số."
        k = int(k)
        n = int(n)
        self._dtype = dtype

        if k>8448:
            raise ValueError("k quá lớn.")
        if k<12:
            raise ValueError("k quá nhỏ.")

        if n>(316*384):
            raise ValueError("n quá lớn.")
        if n<0:
            raise ValueError("n là số âm.")

        self._k = k
        self._n = n
        self._coderate = k / n
        self._check_input = True

        if self._coderate>(11/12):
            raise ValueError(f"Không hỗ trợ coderate (r>11/12); n={n}, k={k}.")
        if self._coderate<(1/5):
            raise ValueError("Không hỗ trợ coderate (r<1/5).")

        # khởi tạo basegraph
        self._bg = self._sel_basegraph(self._k, self._coderate)
        # kích thước lifting, index, K=(22)*Z cho bg1 và K=(10)Z cho bg2
        self._z, self._i_ls, self._k_b = self._sel_lifting(self._k, self._bg)
        self._bm = self._load_basegraph(self._i_ls, self._bg)

        # tổng số bit của từ mã
        self._n_ldpc = self._bm.shape[1] * self._z
        self._k_ldpc = self._k_b * self._z

        pcm = self._lift_basegraph(self._bm, self._z)

        pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = self._gen_submat(self._bm, self._k_b, self._z, self._bg)

        self._pcm = pcm

        self._pcm_a_ind = self._mat_to_ind(pcm_a)
        self._pcm_b_inv_ind = self._mat_to_ind(pcm_b_inv)
        self._pcm_c1_ind = self._mat_to_ind(pcm_c1)
        self._pcm_c2_ind = self._mat_to_ind(pcm_c2)

        self._num_bits_per_symbol = num_bits_per_symbol
        if num_bits_per_symbol is not None:
            self._out_int, self._out_int_inv  = self._generate_out_int(self._n,
                                                    self._num_bits_per_symbol)

    @property
    def k(self):
        """Số lượng của input information bits."""
        return self._k

    @property
    def n(self):
        "Số lượng của output codeword bits."
        return self._n

    @property
    def coderate(self):
        """Coderate của LDPC code sau rate-matching."""
        return self._coderate

    @property
    def k_ldpc(self):
        """Số lượng của LDPC information bits sau rate-matching."""
        return self._k_ldpc

    @property
    def n_ldpc(self):
        """Số lượng của LDPC codeword bits trước rate-matching."""
        return self._n_ldpc

    @property
    def pcm(self):
        """Parity-check matrix từ code parameters."""
        return self._pcm

    @property
    def z(self):
        """Lifting factor của basegraph."""
        return self._z

    @property
    def num_bits_per_symbol(self):
        """Thứ tự modulation cho rate-matching."""
        return self._num_bits_per_symbol

    @property
    def out_int(self):
        """Output interleaver sequence trong 5.4.2.2."""
        return self._out_int

    @property
    def out_int_inv(self):
        """Inverse output interleaver sequence trong 5.4.2.2."""
        return self._out_int_inv

    def _generate_out_int(self, n, num_bits_per_symbol):
        assert(n%1==0), "n phải là int."
        assert(num_bits_per_symbol%1==0), "num_bits_per_symbol phải là int."
        n = int(n)
        assert(n>0), "n phải là a positive integer."
        assert(num_bits_per_symbol>0), \
                    "num_bits_per_symbol phải là a positive integer."
        num_bits_per_symbol = int(num_bits_per_symbol)

        assert(n%num_bits_per_symbol==0),\
            "n phải là a multiple of num_bits_per_symbol."

        perm_seq = np.zeros(n, dtype=int)
        for j in range(int(n/num_bits_per_symbol)):
            for i in range(num_bits_per_symbol):
                perm_seq[i + j*num_bits_per_symbol] \
                    = int(i * int(n/num_bits_per_symbol) + j)

        perm_seq_inv = np.argsort(perm_seq)
        return perm_seq, perm_seq_inv

    def _sel_basegraph(self, k, r):
        """Chọn basegraph dựa theo [3GPPTS38212_LDPC]_."""
        if k <= 292:
            bg = "bg2"
        elif k <= 3824 and r <= 0.67:
            bg = "bg2"
        elif r <= 0.25:
            bg = "bg2"
        else:
            bg = "bg1"
        return bg

    def _load_basegraph(self, i_ls, bg):
        """Hàm hỗ trợ để load BG từ file csv."""
        # khởi tạo các ma trận toàn -1
        if bg=="bg1":
            bm = np.zeros([46, 68]) - 1
        elif bg=="bg2":
            bm = np.zeros([42, 52]) - 1
        else:
            raise ValueError("Basegraph này không hỗ trợ.")

        source = files(codes).joinpath(f"5G_{bg}.csv")
        with as_file(source) as codes.csv:
            bg_csv = np.genfromtxt(codes.csv, delimiter=";")

        r_ind = 0
        for r in np.arange(2, bg_csv.shape[0]):
            if not np.isnan(bg_csv[r, 0]):
                r_ind = int(bg_csv[r, 0])
            c_ind = int(bg_csv[r, 1])
            value = bg_csv[r, i_ls + 2]
            bm[r_ind, c_ind] = value

        return bm

    def _lift_basegraph(self, bm, z):
        num_nonzero = np.sum(bm>=0)
        r_idx = np.zeros(z*num_nonzero)
        c_idx = np.zeros(z*num_nonzero)
        data = np.ones(z*num_nonzero)
        im = np.arange(z)
        idx = 0
        for r in range(bm.shape[0]):
            for c in range(bm.shape[1]):
                if bm[r,c]==-1: # -1 is used as all-zero matrix placeholder
                    pass #do nothing (sparse)
                else:
                    # roll matrix by bm[r,c]
                    c_roll = np.mod(im+bm[r,c], z)
                    # append rolled identity matrix to pcm
                    r_idx[idx*z:(idx+1)*z] = r*z + im
                    c_idx[idx*z:(idx+1)*z] = c*z + c_roll
                    idx += 1

        pcm = sp.sparse.csr_matrix((data,(r_idx, c_idx)), shape=(z*bm.shape[0], z*bm.shape[1]))
        return pcm

    def _sel_lifting(self, k, bg):
        # kích thước của LDPC lifting như bảng 2.1
        s_val = [[2, 4, 8, 16, 32, 64, 128, 256],
                [3, 6, 12, 24, 48, 96, 192, 384],
                [5, 10, 20, 40, 80, 160, 320],
                [7, 14, 28, 56, 112, 224],
                [9, 18, 36, 72, 144, 288],
                [11, 22, 44, 88, 176, 352],
                [13, 26, 52, 104, 208],
                [15, 30, 60, 120, 240]]

        if bg == "bg1":
            k_b = 22
        else:
            if k > 640:
                k_b = 10
            elif k > 560:
                k_b = 9
            elif k > 192:
                k_b = 8
            else:
                k_b = 6

        # tìm min của Z
        min_val = 100000
        z = 0
        i_ls = 0
        i = -1
        for s in s_val:
            i += 1
            for s1 in s:
                x = k_b *s1     # nhân của từng phần tử trong lifting với k_b
                if  x >= k:     # đảm bảo lớn hơn chiều dài tin
                    if x < min_val:
                        min_val = x
                        z = s1
                        i_ls = i

        # đặt K=22*Z cho bg1 và K=10Z cho bg2
        if bg == "bg1":
            k_b = 22
        else:
            k_b = 10

        return z, i_ls, k_b

    def _gen_submat(self, bm, k_b, z, bg):
        g = 4
        mb = bm.shape[0]

        bm_a = bm[0:g, 0:k_b]
        bm_b = bm[0:g, k_b:(k_b+g)]
        bm_c1 = bm[g:mb, 0:k_b]
        bm_c2 = bm[g:mb, k_b:(k_b+g)]

        hm_a = self._lift_basegraph(bm_a, z)
        hm_c1 = self._lift_basegraph(bm_c1, z)
        hm_c2 = self._lift_basegraph(bm_c2, z)

        hm_b_inv = self._find_hm_b_inv(bm_b, z, bg)

        return hm_a, hm_b_inv, hm_c1, hm_c2

    def _find_hm_b_inv(self, bm_b, z, bg):
        pm_a= int(bm_b[0,0])
        if bg=="bg1":
            pm_b_inv = int(-bm_b[1, 0])
        else:
            pm_b_inv = int(-bm_b[2, 0])

        hm_b_inv = np.zeros([4*z, 4*z])
        im = np.eye(z)
        am = np.roll(im, pm_a, axis=1)
        b_inv = np.roll(im, pm_b_inv, axis=1)
        ab_inv = np.matmul(am, b_inv)

        # row 0
        hm_b_inv[0:z, 0:z] = b_inv
        hm_b_inv[0:z, z:2*z] = b_inv
        hm_b_inv[0:z, 2*z:3*z] = b_inv
        hm_b_inv[0:z, 3*z:4*z] = b_inv

        # row 1
        hm_b_inv[z:2*z, 0:z] = im + ab_inv
        hm_b_inv[z:2*z, z:2*z] = ab_inv
        hm_b_inv[z:2*z, 2*z:3*z] = ab_inv
        hm_b_inv[z:2*z, 3*z:4*z] = ab_inv

        # row 2
        if bg=="bg1":
            hm_b_inv[2*z:3*z, 0:z] = ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = im + ab_inv
        else:
            hm_b_inv[2*z:3*z, 0:z] = im + ab_inv
            hm_b_inv[2*z:3*z, z:2*z] = im + ab_inv
            hm_b_inv[2*z:3*z, 2*z:3*z] = ab_inv
            hm_b_inv[2*z:3*z, 3*z:4*z] = ab_inv

        # row 3
        hm_b_inv[3*z:4*z, 0:z] = ab_inv
        hm_b_inv[3*z:4*z, z:2*z] = ab_inv
        hm_b_inv[3*z:4*z, 2*z:3*z] = ab_inv
        hm_b_inv[3*z:4*z, 3*z:4*z] = im + ab_inv

        return sp.sparse.csr_matrix(hm_b_inv)

    def _mat_to_ind(self, mat):
        m = mat.shape[0]
        n = mat.shape[1]

        c_idx, r_idx, _ = sp.sparse.find(mat.transpose())
        n_max = np.max(mat.getnnz(axis=1))
        gat_idx = np.zeros([m, n_max]) + n

        r_val = -1
        c_val = 0
        for idx in range(len(c_idx)):
            if r_idx[idx] != r_val:
                r_val = r_idx[idx]
                c_val = 0
            gat_idx[r_val, c_val] = c_idx[idx]
            c_val += 1

        gat_idx = tf.cast(tf.constant(gat_idx), tf.int32)
        return gat_idx

    def _matmul_gather(self, mat, vec):
        bs = tf.shape(vec)[0]
        vec = tf.concat([vec, tf.zeros([bs, 1], dtype=self.dtype)], 1)
        retval = tf.gather(vec, mat, batch_dims=0, axis=1)
        retval = tf.reduce_sum(retval, axis=-1)
        return retval

    def _encode_fast(self, s):
        """Main encoding function based on gathering function."""
        p_a = self._matmul_gather(self._pcm_a_ind, s)
        p_a = self._matmul_gather(self._pcm_b_inv_ind, p_a)
        p_b_1 = self._matmul_gather(self._pcm_c1_ind, s)
        p_b_2 = self._matmul_gather(self._pcm_c2_ind, p_a)
        p_b = p_b_1 + p_b_2
        c = tf.concat([s, p_a, p_b], 1)
        c_uint8 = tf.cast(c, tf.uint8)
        c_bin = tf.bitwise.bitwise_and(c_uint8, tf.constant(1, tf.uint8))
        c = tf.cast(c_bin, self.dtype)
        c = tf.expand_dims(c, axis=-1)
        return c

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """"Build layer."""
        assert (input_shape[-1]==self._k), "Dữ liệu phải có chiều dài là k."
        assert (len(input_shape)>=2), "Rank của input phải lớn hơn 2."

    def call(self, inputs):
        tf.debugging.assert_type(inputs, self.dtype, "Sai dạng input.")

        # Biến đổi input thành dạng [...,k]
        input_shape = inputs.get_shape().as_list()
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(inputs, new_shape)

        if self._check_input:
            tf.debugging.assert_equal(
                tf.reduce_min(
                    tf.cast(
                        tf.logical_or(
                            tf.equal(u, tf.constant(0, self.dtype)),
                            tf.equal(u, tf.constant(1, self.dtype)),
                            ), self.dtype)),
                tf.constant(1, self.dtype), "Input phải ở dạng binary.")
            self._check_input = False

        batch_size = tf.shape(u)[0]

        u_fill = tf.concat([u, tf.zeros([batch_size, self._k_ldpc-self._k], self.dtype)], 1)
        c = self._encode_fast(u_fill)
        c = tf.reshape(c, [batch_size, self._n_ldpc])
        c_no_filler1 = tf.slice(c, [0, 0], [batch_size, self._k])
        c_no_filler2 = tf.slice(c, [0, self._k_ldpc], [batch_size, self._n_ldpc-self._k_ldpc])
        c_no_filler = tf.concat([c_no_filler1, c_no_filler2], 1)
        c_short = tf.slice(c_no_filler, [0, 2*self._z], [batch_size, self.n])
        if self._num_bits_per_symbol is not None:
            c_short = tf.gather(c_short, self._out_int, axis=-1)
        output_shape = input_shape[0:-1] + [self.n]
        output_shape[0] = -1
        c_reshaped = tf.reshape(c_short, output_shape)

        return tf.cast(c_reshaped, self._dtype)