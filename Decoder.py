import tensorflow as tf
import numpy as np
import scipy as sp
from tensorflow.keras.layers import Layer
from sionna.fec.utils import llr2mi
from . import codes 
import matplotlib.pyplot as plt

class LDPCBPDecoder(Layer):
    def __init__(self,
                 pcm,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=output_dtype, **kwargs)

        assert isinstance(trainable, bool), 'trainable phải là bool.'
        assert isinstance(hard_out, bool), 'hard_out phải là bool.'
        assert isinstance(track_exit, bool), 'track_exit phải là bool.'
        assert isinstance(cn_type, str) , 'cn_type phải là str.'
        assert isinstance(num_iter, int), 'num_iter phải là int.'
        assert num_iter>=0, 'num_iter phải dương.'
        assert isinstance(stateful, bool), 'stateful phải là bool.'
        assert isinstance(output_dtype, tf.DType), 'output_dtype phải là tf.Dtype.'

        if isinstance(pcm, np.ndarray):
            assert np.array_equal(pcm, pcm.astype(bool)), 'PC matrix phải là binary.'
        elif isinstance(pcm, sp.sparse.csr_matrix):
            assert np.array_equal(pcm.data, pcm.data.astype(bool)), 'PC matrix phải là binary.'
        elif isinstance(pcm, sp.sparse.csc_matrix):
            assert np.array_equal(pcm.data, pcm.data.astype(bool)), 'PC matrix phải là binary.'
        else:
            raise TypeError("Không hỗ trợ dtype của pcm.")

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype phải là {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder sử dụng tf.float32 cho internal calculations.')

        # init decoder parameters
        self._pcm = pcm
        self._trainable = trainable
        self._cn_type = cn_type
        self._hard_out = hard_out
        self._track_exit = track_exit
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._stateful = stateful
        self._output_dtype = output_dtype

        self._atanh_clip_value = 1 - 1e-7
        self._llr_max = tf.constant(20., tf.float32)

        # init code parameters
        self._num_cns = pcm.shape[0]
        self._num_vns = pcm.shape[1]

        if isinstance(pcm, np.ndarray):
            pcm = sp.sparse.csr_matrix(pcm)
        self._cn_con, self._vn_con, _ = sp.sparse.find(pcm)

        # parity-check matrix
        self._num_edges = len(self._vn_con)

        self._ind_cn = np.argsort(self._cn_con)

        self._ind_cn_inv = np.argsort(self._ind_cn)

        self._vn_row_splits = self._gen_node_mask_row(self._vn_con)
        self._cn_row_splits = self._gen_node_mask_row(self._cn_con[self._ind_cn])
        if self._cn_type=='boxplus':
            self._cn_update = self._cn_update_tanh
        elif self._cn_type=='boxplus-phi':
            self._cn_update = self._cn_update_phi
        elif self._cn_type=='minsum':
            self._cn_update = self._cn_update_minsum
        else:
            raise ValueError('Unknown node type.')

        self._has_weights = False
        if self._trainable:
            self._has_weights = True
            self._edge_weights = tf.Variable(tf.ones(self._num_edges),trainable=self._trainable,dtype=tf.float32)
        self._ie_c = 0
        self._ie_v = 0

    @property
    def pcm(self):
        """Parity-check matrix của LDPC code."""
        return self._pcm

    @property
    def num_cns(self):
        """Số lượng của check nodes."""
        return self._num_cns

    @property
    def num_vns(self):
        """Số lượng của variable nodes."""
        return self._num_vns

    @property
    def num_edges(self):
        """Số lượng của edges in decoding graph."""
        return self._num_edges

    @property
    def has_weights(self):
        """Chỉ định nếu decoder has trainable weights."""
        return self._has_weights

    @property
    def edge_weights(self):
        """Trainable weights của BP decoder."""
        if not self._has_weights:
            return []
        else:
            return self._edge_weights

    @property
    def output_dtype(self):
        """Output dtype của decoder."""
        return self._output_dtype

    @property
    def ie_c(self):
        return self._ie_c

    @property
    def ie_v(self):
        return self._ie_v

    @property
    def num_iter(self):
        "Số lượng của decoding iterations."
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter):
        "Số lượng của decoding iterations."
        assert isinstance(num_iter, int), 'num_iter phải là int.'
        assert num_iter>=0, 'num_iter cannot be negative.'
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

    @property
    def llr_max(self):
        """Giá trị LLR lớn nhất được dùng cho internal calculations và rate-matching."""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value):
        assert value>=0, 'llr_max cannot be negative.'
        self._llr_max = tf.cast(value, dtype=tf.float32)

    def show_weights(self, size=7):
        if self._has_weights:
            weights = self._edge_weights.numpy()
            plt.figure(figsize=(size,size))
            plt.hist(weights, density=True, bins=20, align='mid')
            plt.xlabel('weight value')
            plt.ylabel('density')
            plt.grid(True, which='both', axis='both')
            plt.title('Weight Distribution')
        else:
            print("Không có weights.")

    def _gen_node_mask(self, con):
        ind = np.argsort(con)
        con = con[ind]

        node_mask = []

        cur_node = 0
        cur_mask = []
        for i in range(self._num_edges):
            if con[i] == cur_node:
                cur_mask.append(ind[i])
            else:
                node_mask.append(cur_mask)
                cur_mask = [ind[i]]
                cur_node += 1
        node_mask.append(cur_mask)
        return node_mask

    def _gen_node_mask_row(self, con):
        node_mask = [0]
        cur_node = 0
        for i in range(self._num_edges):
            if con[i] != cur_node:
                node_mask.append(i)
                cur_node += 1
        node_mask.append(self._num_edges) # last element
        return node_mask

    def _vn_update(self, msg, llr_ch):
        x = tf.reduce_sum(msg, axis=1)
        x = tf.add(x, llr_ch)
        x = tf.ragged.map_flat_values(lambda x, y, row_ind : x + tf.gather(y, row_ind),-1.*msg,x,msg.value_rowids())
        return x

    def _extrinsic_min(self, msg):
        num_val = tf.shape(msg)[0]
        msg = tf.transpose(msg, (1,0))
        msg = tf.expand_dims(msg, axis=1)
        id_mat = tf.eye(num_val)
        msg = (tf.tile(msg, (1, num_val, 1)) + 1000. * id_mat)
        msg = tf.math.reduce_min(msg, axis=2)
        msg = tf.transpose(msg, (1,0))
        return msg

    def _where_ragged(self, msg):
        return tf.where(tf.equal(msg, 0), tf.ones_like(msg) * 1e-12, msg)

    def _where_ragged_inv(self, msg):
        msg_mod =  tf.where(tf.less(tf.abs(msg), 1e-7),tf.zeros_like(msg),msg)
        return msg_mod

    def _cn_update_tanh(self, msg):
        msg = msg / 2
        msg = tf.ragged.map_flat_values(tf.tanh, msg)
        msg = tf.ragged.map_flat_values(self._where_ragged, msg)
        msg_prod = tf.reduce_prod(msg, axis=1)
        msg = tf.ragged.map_flat_values(lambda x, y, row_ind : x * tf.gather(y, row_ind),msg**-1,msg_prod,msg.value_rowids())
        msg = tf.ragged.map_flat_values(self._where_ragged_inv, msg)
        msg = tf.clip_by_value(msg,clip_value_min=-self._atanh_clip_value,clip_value_max=self._atanh_clip_value)
        msg = 2 * tf.ragged.map_flat_values(tf.atanh, msg)
        return msg

    def _phi(self, x):
        x = tf.clip_by_value(x, clip_value_min=8.5e-8, clip_value_max=16.635532)
        return tf.math.log(tf.math.exp(x)+1) - tf.math.log(tf.math.exp(x)-1)

    def _cn_update_phi(self, msg):
        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),tf.ones_like(sign_val),sign_val)
        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_val = tf.ragged.map_flat_values(lambda x, y, row_ind : x * tf.gather(y, row_ind),sign_val,sign_node,sign_val.value_rowids())
        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign
        msg = tf.ragged.map_flat_values(self._phi, msg)
        msg_sum = tf.reduce_sum(msg, axis=1)
        msg = tf.ragged.map_flat_values(lambda x, y, row_ind : x + tf.gather(y, row_ind),-1.*msg,msg_sum,msg.value_rowids())
        msg = self._stop_ragged_gradient(sign_val) * tf.ragged.map_flat_values(self._phi, msg)
        return msg

    def _stop_ragged_gradient(self, rt):
        return rt.with_flat_values(tf.stop_gradient(rt.flat_values))

    def _sign_val_minsum(self, msg):
        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),tf.ones_like(sign_val),sign_val)
        return sign_val

    def _cn_update_minsum_mapfn(self, msg):
        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)
        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_val = self._stop_ragged_gradient(sign_val) * tf.expand_dims(sign_node, axis=1)
        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign
        msg_e = tf.map_fn(self._extrinsic_min, msg, infer_shape=False)
        msg_fv = msg_e.flat_values
        msg_fv = tf.ensure_shape(msg_fv, msg.flat_values.shape)
        msg_e = msg.with_flat_values(msg_fv)
        msg = sign_val * msg_e
        return msg

    def _cn_update_minsum(self, msg):
        LARGE_VAL = 10000.
        msg = tf.clip_by_value(msg,clip_value_min=-self._llr_max,clip_value_max=self._llr_max)
        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)
        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_val = tf.ragged.map_flat_values(lambda x, y, row_ind: tf.multiply(x, tf.gather(y, row_ind)),self._stop_ragged_gradient(sign_val),sign_node,sign_val.value_rowids())
        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign
        min_val = tf.reduce_min(msg, axis=1, keepdims=True)
        msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind: x- tf.gather(y, row_ind),msg,tf.squeeze(min_val, axis=1),msg.value_rowids())
        msg = tf.ragged.map_flat_values(lambda x: tf.where(tf.equal(x, 0), LARGE_VAL, x),msg_min1)
        min_val2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val
        node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*LARGE_VAL-1.)
        double_min = 0.5*(1-tf.sign(node_sum))
        min_val_e = (1-double_min) * min_val + (double_min) * min_val2
        msg_e = tf.where(msg==LARGE_VAL, min_val_e, min_val)
        msg_e = tf.ragged.map_flat_values(lambda x: tf.ensure_shape(x, msg.flat_values.shape),msg_e)

        msg = tf.ragged.map_flat_values(tf.multiply,sign_val,msg_e)

        return msg

    def _mult_weights(self, x):
        x = tf.transpose(x, (1, 0))
        x = tf.math.multiply(x, self._edge_weights)
        x = tf.transpose(x, (1, 0))
        return x

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        # Raise AssertionError if shape of x is invalid
        if self._stateful:
            assert(len(input_shape)==2), \
                "For stateful decoding, a tuple of two inputs is expected."
            input_shape = input_shape[0]

        assert (input_shape[-1]==self._num_vns), 'Last dimension phải dài n.'
        assert (len(input_shape)>=2), 'Input phải có rank ít nhất là 2.'

    def call(self, inputs):
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')
        # internal calculations still in tf.float32
        llr_ch = tf.cast(llr_ch, tf.float32)
        # clip llrs for numerical stability
        llr_ch = tf.clip_by_value(llr_ch,clip_value_min=-self._llr_max,clip_value_max=self._llr_max)
        # last dim phải dài n
        tf.debugging.assert_equal(tf.shape(llr_ch)[-1],self._num_vns,'Last dimension phải là dài n.')
        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        self._cn_mask_tf = tf.ragged.constant(self._gen_node_mask(self._cn_con),row_splits_dtype=tf.int32)
        llr_ch = tf.transpose(llr_ch_reshaped, (1,0))
        llr_ch = -1. * llr_ch
        if not self._stateful or msg_vn is None:
            msg_shape = tf.stack([tf.constant(self._num_edges),tf.shape(llr_ch)[1]],axis=0)
            msg_vn = tf.zeros(msg_shape, dtype=tf.float32)
        else:
            msg_vn = msg_vn.flat_values

        if self._track_exit:
            self._ie_c = tf.zeros(self._num_iter+1)
            self._ie_v = tf.zeros(self._num_iter+1)

        def dec_iter(llr_ch, msg_vn, it):
            it += 1
            msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))
            msg_vn = self._vn_update(msg_vn, llr_ch)
            if self._track_exit:
                mi = llr2mi(-1. * msg_vn.flat_values)
                self._ie_v = tf.tensor_scatter_nd_add(self._ie_v,tf.reshape(it, (1, 1)),tf.reshape(mi, (1)))

            if self._has_weights:
                msg_vn = tf.ragged.map_flat_values(self._mult_weights,msg_vn)
            msg_cn = tf.gather(msg_vn.flat_values, self._cn_mask_tf, axis=None)
            msg_cn = self._cn_update(msg_cn)
            if self._track_exit:
                mi = llr2mi(-1.*msg_cn.flat_values)
                self._ie_c = tf.tensor_scatter_nd_add(self._ie_c,tf.reshape(it, (1, 1)),tf.reshape(mi, (1)))

            msg_vn = tf.gather(msg_cn.flat_values, self._ind_cn_inv, axis=None)
            return llr_ch, msg_vn, it

        def dec_stop(llr_ch, msg_vn, it):
            return tf.less(it, self._num_iter)

        it = tf.constant(0)
        _, msg_vn, _ = tf.while_loop(dec_stop,dec_iter,(llr_ch, msg_vn, it),parallel_iterations=1,maximum_iterations=self._num_iter)
        msg_vn = tf.RaggedTensor.from_row_splits(values=msg_vn,row_splits=tf.constant(self._vn_row_splits, tf.int32))
        x_hat = tf.add(llr_ch, tf.reduce_sum(msg_vn, axis=1))
        x_hat = tf.transpose(x_hat, (1,0))
        x_hat = -1. * x_hat

        if self._hard_out:
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        output_shape = llr_ch_shape
        output_shape[0] = -1
        x_reshaped = tf.reshape(x_hat, output_shape)
        x_out = tf.cast(x_reshaped, self._output_dtype)

        if not self._stateful:
            return x_out
        else:
            return x_out, msg_vn

class LDPC5GDecoder(LDPCBPDecoder):
    def __init__(self,
                 encoder,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 return_infobits=True,
                 prune_pcm=True,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 **kwargs):

        self._encoder = encoder
        pcm = encoder.pcm

        assert isinstance(return_infobits, bool), 'return_info phải là bool.'
        self._return_infobits = return_infobits

        assert isinstance(output_dtype, tf.DType), 'output_dtype phải là tf.DType.'
        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError('output_dtype phải là {tf.float16, tf.float32, tf.float64}.')
        self._output_dtype = output_dtype

        assert isinstance(stateful, bool), 'stateful phải là bool.'
        self._stateful = stateful

        assert isinstance(prune_pcm, bool), 'prune_pcm phải là bool.'
        
        self._prune_pcm = prune_pcm
        if prune_pcm:
            dv = np.sum(pcm, axis=0)
            last_pos = encoder._n_ldpc
            for idx in range(encoder._n_ldpc-1, 0, -1):
                if dv[0, idx]==1:
                    last_pos = idx
                else:
                    break
            k_filler = self.encoder.k_ldpc - self.encoder.k
            nb_punc_bits = ((self.encoder.n_ldpc - k_filler) - self.encoder.n - 2*self.encoder.z)
            self._n_pruned = np.max((last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = encoder._n_ldpc - self._n_pruned
            pcm = pcm[:-self._nb_pruned_nodes, :-self._nb_pruned_nodes]
            assert(self._nb_pruned_nodes>=0), "Internal error: số nodes phải là dương."
        else:
            self._nb_pruned_nodes = 0
            # no pruning; same length as before
            self._n_pruned = encoder._n_ldpc

        super().__init__(pcm,
                         trainable,
                         cn_type,
                         hard_out,
                         track_exit,
                         num_iter=num_iter,
                         stateful=stateful,
                         output_dtype=output_dtype,
                         **kwargs)

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery."""
        return self._encoder

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build model."""
        if self._stateful:
            input_shape = input_shape[0]

        # check input dimensions for consistency
        assert (input_shape[-1]==self.encoder.n), 'Last dimension phải có chiều dài n.'
        assert (len(input_shape)>=2), 'Inputs phải có rank ít nhất là 2.'

        self._old_shape_5g = input_shape

    def call(self, inputs):
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Input dtype không hợp lệ.')
        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, llr_ch_shape[-1]]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,self._encoder.out_int_inv,axis=-1)
        llr_5g = tf.concat([tf.zeros([batch_size, 2*self.encoder.z], self._output_dtype),llr_ch_reshaped],1)
        k_filler = self.encoder.k_ldpc - self.encoder.k # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler) - self.encoder.n - 2*self.encoder.z)
        llr_5g = tf.concat([llr_5g,tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],self._output_dtype)],1)
        x1 = tf.slice(llr_5g, [0,0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,[0, self.encoder.k],[batch_size, nb_par_bits])
        z = -tf.cast(self._llr_max, self._output_dtype) * tf.ones([batch_size, k_filler], self._output_dtype)

        llr_5g = tf.concat([x1, z, x2], 1)

        if not self._stateful:
            x_hat = super().call(llr_5g)
        else:
            x_hat,msg_vn = super().call([llr_5g, msg_vn])

        if self._return_infobits:
            u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)
            u_out = tf.cast(u_reshaped, self._output_dtype)

            if not self._stateful:
                return u_out
            else:
                return u_out, msg_vn

        else:
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])
            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])
            x_no_filler2 = tf.slice(x,[0, self.encoder.k_ldpc],[batch_size,self._n_pruned-self.encoder.k_ldpc])
            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)
            x_short = tf.slice(x_no_filler,[0, 2*self.encoder.z],[batch_size, self.encoder.n])
            # rate-matching output
            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            # Reshape x_short
            llr_ch_shape[0] = -1
            x_short= tf.reshape(x_short, llr_ch_shape)
            x_out = tf.cast(x_short, self._output_dtype)
            if not self._stateful:
                return x_out
            else:
                return x_out, msg_vn
