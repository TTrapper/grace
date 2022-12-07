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

        self.generator, self.descriminator = self.make_backbone(d_model, dropout)
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

        self.sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

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

        latent_q = tf.one_hot(tf.range(seqlen), depth=maxlen)[tf.newaxis, :, :]
        latent_q = tf.tile(latent_q, [batchsize, 1, 1])
        latent_q = tf.concat([inputs, latent_q], axis=-1)
        latent_q = tf.keras.layers.Dense(d_model, tf.nn.relu)(latent_q)

        outs = []
        attention_weights = []
        q = latent_q
        kv = condition
        for i in range(16):
            attn_out, attn_weights = MultiHeadAttention(d_model, nheads)(kv, kv, q)
            attention_weights.append(attn_weights)
            attn_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_out + q)
            attn_out = tf.keras.layers.Dropout(rate=dropout)(attn_out)

            c = tf.keras.layers.Dense(2 * d_model, tf.nn.relu)(attn_out)
            c = tf.reshape(c, [batchsize, tf.shape(c)[1] // 4, 4 * 2 * d_model])
            c = tf.keras.layers.Dense(4 * 2 * d_model, tf.nn.relu)(c)
            c = tf.reshape(c, [batchsize, seqlen, 2 * d_model])

            c = tf.keras.layers.Dropout(rate=dropout)(c)
            c = tf.keras.layers.Dense(d_model, None)(c)
            layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            q = layernorm(c + attn_out)
            kv = q # self attention
            if i % 8 == 0:
                outs.append(q)
            if i % 8 == 7:
                kv = layernorm(condition) # FIXME use a new layernorm
                outs[-1] += q
#                q += latent_q

        descriminator_out = tf.keras.layers.Dense(1)(tf.reduce_sum(tf.stack(outs), axis=0))
        x = tf.concat(outs, axis=-1)
        denoise_logits = tf.keras.layers.Dense(256)(x)
        fake_logits = tf.keras.layers.Dense(256)(x)
        gen_logits = tf.keras.layers.Dense(256)(x)
        generator = tf.keras.Model(
                inputs=(condition_inputs, inputs0,  inputs1),
                outputs=(
                    denoise_logits,
                    fake_logits,
                    gen_logits,
                    descriminator_out,
                    attention_weights
                )
        )

        descriminator = tf.keras.Model(
                inputs=(condition_inputs, inputs0,  inputs1),
                outputs=descriminator_out
        )
        return generator, descriminator


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
        _, _, gen_logits, gen_is_real, attn_weights = self.generator((
                                                    condition,
                                                    gen_masked_hot,
                                                    gen_masked_hot))
        gen_prob = tf.nn.softmax(gen_logits)
        gen_hot = tf.one_hot(tf.argmax(gen_logits, axis=-1), depth=256)
        gen_hot = tf.stop_gradient(gen_hot)
        gen_is_real = tf.stop_gradient(tf.nn.sigmoid(gen_is_real))

        # Pass generated outputs through the model to produce training targets
        in0 = tf.stop_gradient(gen_prob)
#        percentile = tf.sort(gen_is_real, axis=1)[:, seqlen//2, tf.newaxis]
#        in0 = tf.where(gen_is_real > percentile, in0, gen_masked_hot)
        regen_logits, _, _, regen_is_real, _ = self.denoiser((condition, in0, in0), training=False)
        regen_is_real = tf.stop_gradient(tf.nn.sigmoid(regen_is_real))
        regen_prob = tf.nn.softmax(regen_logits)

        # Mask real+gen inputs according to descrimnator scores
        sample = tf.random.uniform([batchsize, seqlen, 1], 0, 1)
        x = tf.where(0.5 < sample, gen_hot, hot_clean)

        # Run the denoiser to learn a language model
        denoise_logits, _, _, denoise_is_real_logits, d_attn_weights = self.denoiser(
            (condition, x, x)
        )
        denoise_prob = tf.nn.softmax(denoise_logits)
        denoise_is_real_prob = tf.stop_gradient(tf.nn.sigmoid(denoise_is_real_logits))
        denoise_is_fake_prob = (1 - denoise_is_real_prob)

        # Rerun denoiser
        in0 = denoise_prob
        redenoise_logits, _, _, redenoise_is_real_logits, d_attn_weights = self.denoiser(
            (condition, in0, in0)
        )
        redenoise_hot = tf.one_hot(tf.argmax(redenoise_logits, axis=-1), depth=256)
        redenoise_is_real_prob = tf.stop_gradient(tf.nn.sigmoid(redenoise_is_real_logits))
        redenoise_is_fake_prob = (1 - redenoise_is_real_prob)

        # Losses
        gen_loss = self.xe_loss(tf.stop_gradient(regen_prob), gen_logits)
        copy_loss = self.xe_loss(gen_masked_hot, gen_logits, tf.where(gen_masked == self.maskid, 0, 1))
        regen_loss = self.xe_loss(redenoise_hot, regen_logits, self.correct(regen_logits, redenoise_hot))
        denoise_sample_weights = denoise_is_fake_prob#tf.nn.sigmoid((denoise_is_fake_prob - 0.5)/0.05)
        denoise_loss = self.xe_loss(hot_clean, denoise_logits, denoise_sample_weights)
        redenoise_sample_weights = redenoise_is_fake_prob#tf.nn.sigmoid((redenoise_is_fake_prob - 0.5)/0.05)
        redenoise_loss = self.xe_loss(hot_clean, redenoise_logits, redenoise_sample_weights)
        # Descriminator losses
        denoise_is_real_targets = self.correct(denoise_logits, hot_clean)
        redenoise_is_real_targets = self.correct(redenoise_logits, hot_clean)
        descrim_loss = self.sig_loss(denoise_is_real_targets, denoise_is_real_logits)
        descrim_loss += self.sig_loss(redenoise_is_real_targets, redenoise_is_real_logits)
        # Diversity loss
        gen_sum = tf.reduce_sum(gen_prob, axis=1)
        clean_sum = tf.reduce_sum(hot_clean, axis=1)
        clean_similarity = self.cosine_loss(clean_sum, tf.roll(clean_sum, shift=1, axis=0))
        clean_similarity = (1 + clean_similarity) / 2
        diversity_loss = -self.cosine_loss(gen_sum, tf.roll(gen_sum, shift=1, axis=0))
        diversity_loss = tf.nn.relu(diversity_loss - (1 - clean_similarity))
        return ((denoise_loss,
                 redenoise_loss,
                 gen_loss + copy_loss,
                 descrim_loss,
                 100*diversity_loss
                ),
                {'denoiser_in':(x, x),
                 'gen_logits': gen_logits,
                 'generated':gen_prob,
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
        for idx in range(num_examples):
            softmax_temp = max(1, 100 - (model.step / 1000))

            start = random.randint(condlen, num_bytes - (1 + seqlen))
            condition = text[start - condlen: start]
            center = text[start : start + seqlen]

            alt_start = random.randint(condlen, num_bytes - (1 + seqlen))
            alt_center = text[alt_start : alt_start + seqlen]

            mask_rate = 1.0
            sample = np.random.uniform(size=(seqlen))
            gen_masked = np.where(sample < mask_rate, model.maskid, alt_center)
            yield ((gen_masked, condition, center),
                    (center, center, center, center, 1))
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
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
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
    condition_indices = set([0, 8]) # FIXME should get these from the model
    for idx, aw in enumerate(attn_weights):
        aw = aw[batch_element]
        aw = np.reshape(aw, [model.num_heads*model.seqlen, -1])
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
    denoiser_x, denoiser_y = out[1]['denoiser_in']
    attn_weights = out[1]['attn_weights']
    attention_plot('x_attn', out[1]['attn_weights'], cnd_str)
    attention_plot('x_d_attn', out[1]['d_attn_weights'], cnd_str)
    plot('x_gen_prob_best', generated[0:1], gen_is_real[0:1])
    plot('x_regen_prob_best', regen_prob[0:1], regen_is_real[0:1])
    plot('x_real_prob_best', denoise_prob[0:1], denoise_is_real[0:1])

    masked_str = to_str(inputs[0])
    regen_str = p_to_str(regen_prob)
    generated = p_to_str(generated)
    denoiser_x = p_to_str(denoiser_x)
    denoise_str = p_to_str(denoise_prob)
    print('--\n')
    indexes = [gen_is_real_idx[0],gen_is_real_idx[127], gen_is_real_idx[128],  gen_is_real_idx[-1]]
    for i in indexes:
        print(cnd_str[i] + '|' + regen_str[i])
        print('--')
        print(cnd_str[i] + '|' + generated[i])
        print('--')
        print(cnd_str[i] + '|' + masked_str[i])
        print('--')
        print(cnd_str[i] + '|' + target_str[i])
        print('--')
        print(cnd_str[i] + '|' + denoise_str[i])
        print('--')
        print(cnd_str[i] + '|' + denoiser_x[i])
        print('\n------\n')

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
    batchsize = 256
    condlen = 64
    seqlen = 32
    gramlen = 5

    model = Model(d_model, seqlen, enc_depth, dropout)

    dataset = make_dataset(model, './guten-wot-train.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=True)
    valid_dataset = make_dataset(model, './wot_valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=False, training=False)
    inference_dataset = make_dataset(model, './wot_valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=False)

    inputs, targets = next(iter(dataset))
    condition_strings = to_str(inputs[1])
    input_strings = to_str(inputs[0])
    target_strings = to_str(targets[2])
    for idx in range(8):
        print(condition_strings[idx] + '|' + input_strings[idx])
        print(condition_strings[idx] + '|' + target_strings[idx])
        print('--')
    print('-----------------------')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    xe_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    norm_loss = NormLoss()
    through_loss = PassthroughLoss()

    loss = through_loss
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1,1,1,1,1], metrics=[])


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
                char_predict(model, inference_dataset, 4)
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
