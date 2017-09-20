import dynet as dy
import math
import numpy as np


class AffineTransform(object):
    def __init__(self, input_size, output_size, model):
        self.input_size = input_size
        self.output_size = output_size
        self.p_W = model.add_parameters(dim=(output_size, input_size))
        self.p_b = model.add_parameters(dim=output_size, init=dy.ConstInitializer(0.0))

    def transduce(self, input):
        W = dy.parameter(self.p_W)
        b = dy.parameter(self.p_b)
        out = dy.affine_transform([b, W, input])
        return out

    def __repr__(self):
        return "AffineTransform ({} --> {})".format(self.input_size, self.output_size)


class LayerNorm(object):
    def __init__(self, d_hid, model):
        self.p_g = model.add_parameters(dim=d_hid, init=dy.ConstInitializer(1.0))
        self.p_b = model.add_parameters(dim=d_hid, init=dy.ConstInitializer(0.0))

    def transduce(self, input):
        g = dy.parameter(self.p_g)
        b = dy.parameter(self.p_b)
        return dy.layer_norm(input, g, b)

    def __repr__(self):
        return "LayerNorm module"


class PositionwiseFeedForward(object):
    """ A two-layer Feed-Forward-Network."""

    def __init__(self, size, hidden_size, model):
        """
        Args:
            size(int): the size of input for the first-layer of the FFN.
            hidden_size(int): the hidden layer size of the second-layer
                              of the FNN.
            droput(float): dropout probability(0-1.0).
        """
        self.w_1 = AffineTransform(size, hidden_size, model)
        self.w_2 = AffineTransform(hidden_size, size, model)
        self.layer_norm = LayerNorm(size, model)

    def transduce(self, x, p):
        residual = x
        output = dy.dropout(self.w_2.transduce(dy.rectify(self.w_1.transduce(x))), p)
        return self.layer_norm.transduce(output + residual)


class MultiHeadedAttention(object):
    def __init__(self, head_count, model_dim, model):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        # Linear Projection of keys
        self.linear_keys = AffineTransform(model_dim, head_count * self.dim_per_head, model)

        # Linear Projection of values
        self.linear_values = AffineTransform(model_dim, head_count * self.dim_per_head, model)

        # Linear Projection of query
        self.linear_query = AffineTransform(model_dim, head_count * self.dim_per_head, model)

        # Layer Norm Module
        self.layer_norm = LayerNorm(model_dim, model)

    def transduce(self, input, key, value, query, mask=None, p=0.1):
        sent_expr_list = list(input)
        # residual = dy.concatenate_to_batch(list(query))
        residual = query

        # Finding the number words in a sentence
        per_sent_words = len(sent_expr_list)
        batch_size = sent_expr_list[0].dim()[1]
        total_words = per_sent_words * batch_size

        def shape_projection(x):
            # CHECKS
            # dim_ = x.dim()
            # d, w = dim_[0][0], dim_[1]
            # assert (d == self.model_dim)
            # assert (total_words == w)
            # CHECK

            return dy.reshape(x, (per_sent_words, self.dim_per_head), batch_size=batch_size * self.head_count)

        # Concatenate all the words together for doing vectorized affine transform
        # key_up = shape_projection(self.linear_keys.transduce(dy.concatenate_to_batch(list(key))))
        key_up = shape_projection(self.linear_keys.transduce(key))
        value_up = shape_projection(self.linear_keys.transduce(value))
        query_up = shape_projection(self.linear_keys.transduce(query))

        scaled = query_up * dy.transpose(key_up)
        scaled = scaled / math.sqrt(self.dim_per_head)

        # Apply Mask here


        # Computing Softmax here. Doing double transpose here, as softmax in dynet is applied to each column
        # May be Optimized ? // Dynet Tricks ??
        attn = dy.transpose(dy.softmax(dy.transpose(scaled)))

        # Applying dropout to attention
        drop_attn = dy.dropout(attn, p)

        # Computing weighted attention score
        attn_prod = drop_attn * value_up

        # Reshaping the attn_prod to input query dimensions
        temp = dy.reshape(attn_prod, (per_sent_words, self.dim_per_head * self.head_count),
                          batch_size=batch_size)
        temp = dy.transpose(temp)
        out = dy.reshape(temp, (self.model_dim,), batch_size=total_words)

        # Adding dropout and layer normalization
        res = dy.dropout(out, p) + residual
        ret = self.layer_norm.transduce(res)

        return ret

    def __repr__(self):
        return "MultiHeadedAttention from `Attention is all you need` paper"


class TransformerEncoderLayer(object):
    def __init__(self, size, rnn_size, model, head_count=8, hidden_size=2048):
        self.self_attn = MultiHeadedAttention(head_count, size, model)
        self.feed_forward = PositionwiseFeedForward(size, hidden_size, model)

    def set_dropout(self, dropout):
        self.dropout = dropout

    def transduce(self, input):
        seq_len = len(input)
        model_dim = input[0].dim()[0][0]
        MBS = input[0].dim()[1]
        input_ = dy.concatenate_cols(list(input))
        input_ = dy.reshape(input_, (model_dim,), batch_size=MBS * seq_len)

        mid = self.self_attn.transduce(input, input_, input_, input_, mask=input.mask, p=self.dropout)
        out = self.feed_forward.transduce(mid, p=self.dropout)

        # Check for Nan
        assert (np.isnan(out.npvalue()).any() == False)

        out_list = []
        for i in range(len(input)):
            indexes = map(lambda x: x+i, range(0, seq_len * MBS, seq_len))
            out_list.append(dy.pick_batch_elems(out, indexes))

        return out_list
