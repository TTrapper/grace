import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import numpy as np
import plotly.express as px

import argparse
from random import shuffle
import random
import time

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask=None):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

class GaussianNoise(tf.keras.layers.Layer):

    def __init__(self, proportion):
        super(GaussianNoise, self).__init__()
        self.proportion = proportion

    def call(self, inputs, training):
        if not training:
            return inputs
        stddev = self.proportion * tf.math.reduce_std(inputs)
        noise = tf.random.normal(tf.shape(inputs), stddev=stddev, dtype=inputs.dtype)
        return inputs + noise

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

class Model(tf.keras.Model):

    def __init__(self, d_model, seqlen, enc_depth, dropout):
        super().__init__()
        self.step = 0
        self.num_heads = 8
        self.seqlen = seqlen
        self.d_model = d_model
        self.maskid = 0

        self.generator = self.make_backbone(d_model, dropout)
        self.denoiser = self.generator

        self.cosine_loss = tf.keras.losses.CosineSimilarity(
                reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
        )
        self.xe_loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
        )

        self.sig_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE
        )
        self.bi_xe_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def make_backbone(self, d_model, dropout):
        nheads = self.num_heads

        condition_inputs = tf.keras.Input(shape=(None, 256), name='condition_inputs') # one-hot bytes
        inputs0 = tf.keras.Input(shape=(None, 256), name='inputs0') # one-hot bytes
        inputs1 = tf.keras.Input(shape=(None, 256), name='inputs1') # one-hot bytes
        inputs = tf.concat([inputs0, inputs1], axis=-1)

        condlen = tf.shape(condition_inputs)[1]
        seqlen = tf.shape(inputs0)[1]
        batchsize = tf.shape(inputs0)[0]

        maxlen = 512
        condition_pos = tf.reverse(tf.range(condlen), axis=[0])
        condition_pos = tf.one_hot(condition_pos, depth=maxlen)[tf.newaxis, :, :]
        condition_pos = tf.tile(condition_pos, [batchsize, 1, 1])
        condition = tf.concat([condition_inputs, condition_pos], axis=-1)
        condition = tf.keras.layers.Dense(d_model, tf.nn.relu)(condition)

        latent_q = tf.one_hot(tf.range(seqlen + 1), depth=maxlen)[tf.newaxis, :, :]
        latent_q = tf.tile(latent_q, [batchsize, 1, 1])
        latent_inputs = tf.concat([latent_q[:, :1, :], inputs], axis=1) # concat class token
        latent_q = tf.concat([latent_inputs, latent_q], axis=-1)
        latent_q = tf.keras.layers.Dense(d_model, tf.nn.relu)(latent_q)

        outs = []
        attention_weights = []
        q = latent_q
        kv = condition
        nlayers = 32
        for i in range(nlayers):
            attn_out, attn_weights = MultiHeadAttention(d_model, nheads)(kv, kv, q)
            attention_weights.append(attn_weights)
            attn_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_out + q)
            attn_out = tf.keras.layers.Dropout(rate=dropout)(attn_out)

            c = tf.keras.layers.Dense(2 * d_model, tf.nn.relu)(attn_out)
            c = tf.keras.layers.Dropout(rate=dropout)(c)
            c = tf.keras.layers.Dense(d_model, None)(c)
            q = tf.keras.layers.LayerNormalization(epsilon=1e-6)(c + attn_out)

            kv = q # self attention
            if i % 8 == 7:
                kv = tf.keras.layers.LayerNormalization(epsilon=1e-6)(condition)
            outs.append(q)

        # Do attention over all layer outputs
        x = tf.stack(outs, axis=2)
        x = tf.reshape(x, [batchsize * (1+seqlen), nlayers, d_model])
        x, x_out_attn = MultiHeadAttention(d_model, nheads)(x, x, x[:, -1:, :]) # Attn over layers
        x = tf.reshape(x, [batchsize, 1 + seqlen, d_model])
        # Split class token
        cls = x[:, 0, :]
        x = x[:, 1:, :]

        self.class_layers = [tf.keras.layers.Dense(d_model, tf.nn.relu), tf.keras.layers.Dense(1)] # to comapre cls outputs from 2 different runs

        self.descriminator_layer = tf.keras.layers.Dense(1)
        descriminator_logits = self.descriminator_layer(x)

        denoise_logits = tf.keras.layers.Dense(256)(x)
        gen_logits = tf.keras.layers.Dense(256)(x)
        generator = tf.keras.Model(
                inputs=(condition_inputs, inputs0,  inputs1),
                outputs=(
                    cls,
                    x,
                    denoise_logits,
                    gen_logits,
                    descriminator_logits,
                    attention_weights
                )
        )

        return generator


    def correct(self, x, y):
        correct = tf.argmax(x, axis=-1) == tf.argmax(y, axis=-1)
        return tf.cast(correct, x.dtype)

    def confidently_correct(self, x, y, min_confidence=0.99):
        correct = tf.argmax(x, axis=-1) == tf.argmax(y, axis=-1)
        confident = tf.reduce_max(x, axis=-1) > min_confidence
        return tf.cast(correct & confident, x.dtype)

    def sample_logits(self, logits, temperature):
        batchsize = tf.shape(logits)[0]
        seqlen = tf.shape(logits)[1]
        sampled = tf.random.categorical(tf.reshape(logits/temperature, [-1, 256]), 1)
        sampled = tf.reshape(sampled, [batchsize, seqlen])
        return tf.one_hot(sampled, depth=256)

    def forward_pass(self, inputs, training):
        maskid_hot = tf.one_hot(self.maskid, depth=256)[tf.newaxis, tf.newaxis, :]
        padid_hot = tf.one_hot(self.maskid+1, depth=256)[tf.newaxis, tf.newaxis, :]

        gen_masked = inputs[0]
        gen_masked_hot = tf.one_hot(gen_masked, depth=256)
        condition = tf.one_hot(inputs[1], depth=256)
        clean = inputs[2]
        hot_clean = tf.one_hot(clean, depth=256)

        batchsize = tf.shape(gen_masked)[0]
        seqlen = tf.shape(gen_masked)[1]


        # Run generator

        noise = tf.random.normal([batchsize, seqlen, 256], 0, 0.1)
        gen_cls, _, _, gen_logits, gen_is_real_logits, attn_weights = self.generator((
                                                    condition,
                                                    noise+gen_masked_hot,
                                                    noise+gen_masked_hot))
        gen_prob = tf.nn.softmax(gen_logits)
        gen_hot = tf.one_hot(tf.argmax(gen_logits, axis=-1), depth=256)
        gen_hot = tf.stop_gradient(gen_hot)
        gen_is_real = tf.stop_gradient(tf.nn.sigmoid(gen_is_real_logits))


       ######
       # Mask real+gen inputs according to descrimnator scores
#        median = tf.sort(gen_is_real, axis=1)[:, seqlen//2, tf.newaxis]
#        x = tf.where(gen_is_real > median, gen_hot, hot_clean)
#        y = tf.where(gen_is_real < median, gen_hot, hot_clean)
#        coin = tf.random.uniform([batchsize, 1, 1], 0, 1)
#        x = tf.where(coin < 0.5, x, y)
        sample = tf.random.uniform([batchsize, seqlen, 1], 0, 1)
        x = tf.where(sample < 0.5, gen_hot, hot_clean)

        # Run the denoiser to learn a language model
        _, denoise_x, denoise_logits, _, denoise_is_real_logits, d_attn_weights = self.denoiser(
            (condition, x, x)
        )
        denoise_prob = tf.nn.softmax(denoise_logits)
        denoise_is_real_prob = tf.nn.sigmoid(denoise_is_real_logits)
        denoise_is_fake_prob = (1 - denoise_is_real_prob)


        #######
        # Pass generated outputs through the model to produce training targets
        regen_in = tf.stop_gradient(gen_prob + gen_is_real*denoise_prob) 
        regen_cls, regen_x, regen_logits, _, regen_is_real, _ = self.denoiser(
            (condition, regen_in, regen_in),
            training=False
        )
        regen_is_real = tf.stop_gradient(tf.nn.sigmoid(regen_is_real))
        regen_prob = tf.nn.softmax(regen_logits)
        regen_is_fake_prob = (1 - regen_is_real)


        ################
        # Rerun denoiser
        in0 = denoise_prob
        _, redenoise_x, redenoise_logits, _, redenoise_is_real_logits, d_attn_weights = self.denoiser(
            (condition, in0, in0)
        )
        redenoise_hot = tf.one_hot(tf.argmax(redenoise_logits, axis=-1), depth=256)
        redenoise_is_real_prob = tf.nn.sigmoid(redenoise_is_real_logits)
        redenoise_is_fake_prob = (1 - redenoise_is_real_prob)

        # Losses
        no_train_denoise_is_fake_prob = tf.stop_gradient(denoise_is_fake_prob)
        no_train_redenoise_is_fake_prob = tf.stop_gradient(redenoise_is_fake_prob)
        """
        kernel = self.descriminator_layer.kernel.value()
        bias = self.descriminator_layer.bias.value()
        logits = tf.matmul(a=denoise_x, b=kernel)
        logits = tf.nn.bias_add(logits, bias)
        no_train_denoise_is_fake_prob = (1 - tf.nn.sigmoid(logits))

        logits = tf.matmul(a=redenoise_x, b=kernel)
        logits = tf.nn.bias_add(logits, bias)
        no_train_redenoise_is_fake_prob = (1 - tf.nn.sigmoid(logits))
        """
        gen_loss = self.xe_loss(tf.stop_gradient(regen_prob), gen_logits)
        copy_loss = self.xe_loss(gen_masked_hot, gen_logits, tf.where(gen_masked==0, 0, 1))
        denoise_sample_weights = tf.nn.sigmoid((no_train_denoise_is_fake_prob - 0.5)/0.05)
        denoise_loss = self.xe_loss(hot_clean, denoise_logits,
                denoise_sample_weights)
        redenoise_sample_weights = tf.nn.sigmoid((no_train_redenoise_is_fake_prob - 0.5)/0.05)
        redenoise_loss = self.xe_loss(hot_clean, redenoise_logits,
                redenoise_sample_weights)

        # Descriminator losses
        denoise_is_real_targets = self.correct(denoise_logits, hot_clean)
        redenoise_is_real_targets = self.correct(redenoise_logits, hot_clean)
#        gen_is_real_targets = self.correct(denoise_prob, gen_hot)
        descrim_loss = self.sig_loss(denoise_is_real_targets, denoise_is_real_logits)*(1/2)
        descrim_loss += self.sig_loss(redenoise_is_real_targets, redenoise_is_real_logits)*(1/2)
#        descrim_loss += self.sig_loss(gen_is_real_targets, gen_is_real_logits)*(1/3)
        
        gen_cls = tf.reshape(gen_cls, [batchsize//2, 2, self.d_model])
        true_continuation = tf.concat([gen_cls[:, 0], gen_cls[:, 1]], axis=-1)
        false_continuation = tf.concat([gen_cls[:, 1], gen_cls[:, 0]], axis=-1)
        gen_is_continued = tf.concat([true_continuation, false_continuation], axis=0)
        for layer in self.class_layers:
            gen_is_continued = layer(gen_is_continued)
        gen_is_continued_tgt = tf.concat([tf.ones((batchsize//2)), tf.zeros(batchsize//2)], axis=0)
        gen_cls_loss = self.sig_loss(gen_is_continued_tgt, gen_is_continued)

        """
        diversity_loss = 0
        clean_sum = tf.reduce_sum(hot_clean, axis=1)
        gen_sum = tf.reduce_sum(gen_prob, axis=1)
        regen_sum = tf.reduce_sum(regen_prob, axis=1)
        for shift in range(1, 2):
            clean_similarity = self.cosine_loss(clean_sum, tf.roll(clean_sum, shift=shift, axis=0))
            clean_similarity = 1 - ((1 + clean_similarity) / 2)

            loss = -self.cosine_loss(regen_sum, tf.roll(regen_sum, shift=shift, axis=0))
            diversity_loss += tf.nn.relu((loss - clean_similarity))
        """

        diversity_loss = 0
        for shift in range(1, 2):
            targets = tf.stop_gradient(tf.roll(gen_prob, shift=shift, axis=0))
            sample_weights = tf.stop_gradient(1 - gen_is_real)
            diversity_loss -= self.cosine_loss(targets, gen_prob, sample_weights)
            shift = tf.random.uniform((), 1, 3, tf.int32)
#            targets = tf.stop_gradient(tf.roll(gen_prob, shift=1, axis=1))
#            diversity_loss -= self.cosine_loss(targets, gen_prob, sample_weights)
#        realism_loss = self.cosine_loss(tf.stop_gradient(clean_sum), gen_sum)

        """
        gen_sum = tf.reduce_sum(gen_prob, axis=1)
        condition_sum = tf.reduce_sum(condition, axis=1)
        match = tf.concat([gen_sum, condition_sum], axis=-1)
        nomatch = tf.concat([gen_sum, tf.roll(condition_sum, shift=1, axis=0)], axis=-1)
        for layer in self.class_layers:
            match = layer(match)
        for layer in self.class_layers:
            nomatch = layer(nomatch)
        is_match_tgt = tf.concat([tf.ones_like(match), tf.zeros_like(nomatch)], axis=0)
        is_match = tf.concat([match, nomatch], axis=0)
        is_match_loss = self.sig_loss(is_match_tgt, is_match)
        """
        return ((gen_loss+copy_loss,
                 denoise_loss,
                 redenoise_loss,
                 descrim_loss,
                 gen_cls_loss,
                 1*diversity_loss,
                ),
                {'denoiser_in':(x, x),
                 'gen_logits': gen_logits,
                 'generated':gen_prob,
                 'regen_in':regen_in,
                 'regen_prob':regen_prob,
                 'denoise_prob':denoise_prob,
                 'gen_is_real':gen_is_real,
                 'regen_is_real':regen_is_real,
                 'denoise_is_real':denoise_is_real_prob,
                 'attn_weights': attn_weights,
                 'd_attn_weights': d_attn_weights
                 })

    def call(self, inputs, training):
        return self.forward_pass(inputs, training)[0]

def p_to_str(logits_or_probs):
    return to_str(tf.argmax(logits_or_probs, axis=-1))

def to_str(array):
    lines = []
    array = tf.where(array == 0, ord('_'), array)
    array = tf.where(array == 1, ord('+'), array)
    for line in array.numpy().astype(np.uint8):
        lines.append(bytes(line).decode('utf-8', 'replace'))
    return lines

def to_int(span, padlen):
    span = [int(c) for c in span]
    if padlen > 0:
        span = padlen * [0] + span[padlen:-padlen] + padlen * [0]
    return span

def corrupt_data(span, corrupt_rate):
    corrupt = np.random.randint(0, 256, len(span))
    corrupt = np.zeros_like(span) # FIXME zero out, should be a specal MASK token
    corrupt_mask = np.random.binomial(1, corrupt_rate, len(span))
    return np.where(corrupt_mask, corrupt, span), corrupt_mask

def scale(x):
    x -= tf.reduce_min(x, axis=-1, keepdims=True)
    x /= tf.reduce_sum(x, axis=-1, keepdims=True)
    return x


def make_dataset(model, fpath, batchsize, seqlen, condlen, gramlen, shuffle, training):
    with open(fpath, 'rb') as f:
        text = f.read()
    num_bytes = len(text)
    num_examples = num_bytes // seqlen
    print(f'{num_examples // batchsize} batches')
    right_masklen = 16
    not_masklen = seqlen - right_masklen

    text = np.frombuffer(text, dtype=np.uint8)

    def gen():
        start = 0
        n = 0
        mask = np.array(4*[0, 0, 0, 0, 0, 0, 0, 0, 1])[:seqlen]
        for idx in range(num_examples):
            softmax_temp = max(1, 100 - (model.step / 1000))
            if idx % 2 == 0:
                start = random.randint(condlen, num_bytes - (1 + seqlen))
            else:
                start += condlen + random.randint(0, 16)

                if start < condlen:
                    start = condlen
                if start >= (num_bytes - seqlen):
                    start = num_bytes - seqlen
            condition = text[start - condlen: start]
            center = text[start : start + seqlen]
#            print(start)
#            print(idx%2, to_str([condition])[0], '|', to_str([center])[0])
#            print('\n\n===============\n=============\n\n')

            alt_start = start#random.randint(condlen, num_bytes - (1 + seqlen))
            alt_center = text[alt_start : alt_start + seqlen]
            gen_masked = alt_center*mask

#            mask_rate = 0.25
#            sample = np.random.uniform(size=(condlen))
#            condition = np.where(sample < mask_rate, model.maskid, condition)

            yield ((gen_masked, condition, center),
                    (center, 1,1,1,1,1))
    dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature = (
                (
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(condlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                ),
                (
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    tf.TensorSpec(shape=(), dtype=tf.int32),


                ),

            )

    )
    return dataset.batch(batchsize)

def plot(name, p, is_real=None):
    nticks = 256
    ticktext = [chr(i) for i in range(256)]
    out_str = p_to_str(p)[0]
    if is_real is not None:
        nticks += 1
        p = np.concatenate([is_real, p], axis=-1)
        ticktext = ['is_real'] + ticktext
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(nticks)),
        ticktext = ticktext
    )
    yaxis = dict(
        tickmode = 'array',
        tickvals = list(range(len(out_str))),
        ticktext = list(out_str)
    )
    fig = px.imshow(p[0], aspect='auto')
    fig.update_layout(xaxis=xaxis, yaxis=yaxis)
    fig.write_html(f'{name}.html')

def attention_plot(name, attn_weights, condition_strings):
    batch_element = 0
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(len(condition_strings[0]))),
        ticktext = list(condition_strings[batch_element])
    )
    condition_indices = set([0, 8, 16, 24, 32]) # FIXME should get these from the model
    for idx, aw in enumerate(attn_weights):
        aw = aw[batch_element]
        nhead, nquery, nkey = aw.shape
        aw = np.reshape(aw, [nhead*nquery, nkey])
        fig = px.imshow(aw, aspect='auto')
        if idx in condition_indices:
            fig.update_layout(xaxis=xaxis)
        fig.write_html(f'{name}_{idx}.html')

def char_predict(model, dataset, num):
    inputs, targets = next(iter(dataset)) # NOTE this always grabs the 1st batch because its a new iterator each time
    cnd_str = to_str(inputs[1])
    target_str = to_str(targets[0])
    out = model.forward_pass(inputs, training=False)

    generated = out[1]['generated']
    denoise_prob = out[1]['denoise_prob']
    denoise_is_real = out[1]['denoise_is_real']
    gen_logits = out[1]['gen_logits']
    gen_is_real = out[1]['gen_is_real']
    regen_is_real = out[1]['regen_is_real']
    gen_is_real_idx = np.argsort(np.mean(gen_is_real[:, :, 0], axis=1))
    regen_prob = out[1]['regen_prob']
    regen_in = out[1]['regen_in']
    denoiser_x, denoiser_y = out[1]['denoiser_in']
    attn_weights = out[1]['attn_weights']
    # Plots
    attention_plot('x_attn', out[1]['attn_weights'], cnd_str)
    attention_plot('x_d_attn', out[1]['d_attn_weights'], cnd_str)
    plot('x_gen_prob_best', generated[0:1], gen_is_real[0:1])
    plot('x_regen_prob_best', regen_prob[0:1], regen_is_real[0:1])
    plot('x_real_prob_best', denoise_prob[0:1], denoise_is_real[0:1])
    # Print outs
    masked_str = to_str(inputs[0])
    regen_str = p_to_str(regen_prob)
    regen_in_str = p_to_str(regen_in)
    generated = p_to_str(generated)
    denoiser_x = p_to_str(denoiser_x)
    denoise_str = p_to_str(denoise_prob)
    print('--\n')
    batchsize = len(gen_is_real)
    indexes = [gen_is_real_idx[0],gen_is_real_idx[batchsize//2], gen_is_real_idx[1 + (batchsize//2)],  gen_is_real_idx[-1]]
    for i in indexes:
        print(cnd_str[i] + '|' + regen_str[i])
        print('-----------------------------')
        print(cnd_str[i] + '|' + generated[i])
        print('-----------------------------')
        print(cnd_str[i] + '|' + regen_in_str[i])
        print('-----------------------------')
        print(cnd_str[i] + '|' + target_str[i])
        print('-----------------------------')
        print(cnd_str[i] + '|' + denoise_str[i])
        print('-----------------------------')
        print(cnd_str[i] + '|' + denoiser_x[i])
        print('\n========================================\n')

class NormLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.maximum(0.0, tf.norm(y_pred, axis=-1) - 75)

class PassthroughLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    d_model = 256
    enc_depth = 8
    dropout = 0.01
    batchsize = 120
    condlen = 64
    seqlen = 32
    gramlen = 5

    model = Model(d_model, seqlen, enc_depth, dropout)

    dataset = make_dataset(model, './train.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=True)
    valid_dataset = make_dataset(model, './valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=False)

    inputs, targets = next(iter(dataset))
    condition_strings = to_str(inputs[1])
    input_strings = to_str(inputs[0])
    target_strings = to_str(targets[0])
    for idx in range(8):
        print(condition_strings[idx] + '|' + input_strings[idx])
        print(condition_strings[idx] + '|' + target_strings[idx])
        print('\n========================================\n')
    print('-----------------------')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    xe_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    norm_loss = NormLoss()
    through_loss = PassthroughLoss()

    loss = through_loss
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1,1,1,1,1,1,1,1], metrics=[])


    logits = model(inputs) # Running the model builds all the layers
    print(model.summary())
    if args.restore:
        model.load_weights('./saves/checkpoint')

    char_predict(model, dataset, 16)

    if args.train:
        def save(epoch, logs):
            model.save_weights('./saves/checkpoint', overwrite=True, save_format=None, options=None)
            return
        save_callback = tf.keras.callbacks.LambdaCallback( on_epoch_end=save)

        def run_char_predict(batch, logs):
            if batch % 500 == 0:
                char_predict(model, valid_dataset, 4)
            if batch % 500 == 499:
                model.save_weights('./saves/checkpoint', overwrite=True, save_format=None, options=None)
        char_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=run_char_predict)

        def step(batch, logs):
            model.step += 1
        step_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=step)

        callbacks = [save_callback, char_callback, step_callback]
        model.fit(dataset, epochs=4, validation_data=valid_dataset, callbacks=callbacks)

    if args.eval:
        model.evaluate(valid_dataset)
