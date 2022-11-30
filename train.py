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

    def __init__(self, d_model, seqlen, enc_depth, dropout, gen_noise):
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
            q = tf.keras.layers.LayerNormalization(epsilon=1e-6)(c + attn_out)
            kv = q
            if i % 8 == 7:
                kv = condition
                outs.append(q)
    
        descriminator_out = tf.keras.layers.Dense(1)(tf.reduce_sum(tf.stack(outs), axis=0))
        x = tf.concat(outs, axis=-1)
        real_logits = tf.keras.layers.Dense(256)(x)
        fake_logits = tf.keras.layers.Dense(256)(x)
        gen_logits = tf.keras.layers.Dense(256)(x)
        generator = tf.keras.Model(
                inputs=(condition_inputs, inputs0,  inputs1),
                outputs=(
                    real_logits,
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
        alt_clean = tf.one_hot(inputs[1], depth=256)

        alt_denoise_masked = inputs[2]
        alt_denoise_masked_hot = tf.one_hot(alt_denoise_masked, depth=256)
        condition = tf.one_hot(inputs[3], depth=256)
        clean = inputs[4]
        hot_clean = tf.one_hot(clean, depth=256)

        batchsize = tf.shape(gen_masked)[0]
        seqlen = tf.shape(gen_masked)[1]

        # Run generator
        _, _, gen_logits, _, attn_weights = self.generator((
                                                    condition,
                                                    gen_masked_hot,
                                                    gen_masked_hot))
        gen_prob = tf.nn.softmax(gen_logits)
        gen_hot = tf.one_hot(tf.argmax(gen_logits, axis=-1), depth=256)
        gen_hot = tf.stop_gradient(gen_hot)

        # Denoise generated outputs to produce training targets
        in0 = tf.where(gen_masked_hot == maskid_hot, tf.stop_gradient(gen_prob), gen_masked_hot)
        regen_logits, _, _, gen_is_real, _ = self.denoiser((condition, in0, in0), training=False)
        gen_is_real = tf.stop_gradient(tf.nn.sigmoid(gen_is_real))  
        regen_prob = tf.nn.softmax(regen_logits)
        regen_hot = tf.one_hot(tf.argmax(regen_logits, axis=-1), depth=256)

        # Mask real/gen inputs according to descrimnator scores
        sample = tf.random.uniform([batchsize, seqlen, 1], 0, 1)
        median = tf.sort(gen_is_real, axis=1)[:, seqlen//2]
#        print('\n:MEDIAN',median)#[:, :, tf.newaxis])
        x = tf.where(gen_is_real > median[:, :, tf.newaxis], gen_hot, hot_clean)
        
        # Run the denoiser to learn a language model
        real_logits, fake_logits, _, is_real_logits, d_attn_weights = self.denoiser((condition, x, x))
        real_prob = tf.nn.softmax(real_logits)
        real_hot = tf.one_hot(tf.argmax(real_logits, axis=-1), depth=256)
        re_real_logits, _, _, is_real_logits, d_attn_weights = self.denoiser((condition, real_prob, real_prob))

        # Run the descriminator on mixed inputs to learn to distinguish between real/fake
        is_real_input_ids = tf.cast(tf.argmax(x, axis=-1), clean.dtype)
        is_real_targets = tf.where(is_real_input_ids == clean, 1, 0)
        is_real_prob = tf.stop_gradient(tf.nn.sigmoid(is_real_logits))

        # Losses
        gen_loss = self.xe_loss(regen_hot, gen_logits)
        real_loss = self.xe_loss(hot_clean, real_logits)
        fake_loss = self.xe_loss(hot_clean, re_real_logits)
        descrim_loss = self.sig_loss(is_real_targets, is_real_logits)

        """
        copy_loss = 1*self.xe_loss(hot_clean, gen_logits, tf.where(gen_masked == self.maskid, 0, 1))
        clean_sum = tf.reduce_sum(hot_clean, axis=1)
        gen_sum = tf.reduce_sum(gen_prob, axis=1)
        regen_sum = tf.reduce_sum(regen_prob, axis=1)
        real_sum = tf.reduce_sum(real_prob, axis=1)
        realism_loss = 10*self.cosine_loss(clean_sum, gen_sum)
        realism_loss += 1*self.cosine_loss(clean_sum, regen_sum)
        realism_loss += 1*self.cosine_loss(clean_sum, real_sum)
        clean_sim = (1 + self.cosine_loss(hot_clean, tf.roll(hot_clean, shift=1, axis=0))) / 2
        diversity_loss = -1*self.cosine_loss(gen_prob, tf.roll(gen_prob, shift=1, axis=0))
        diversity_loss = 1*tf.nn.relu(diversity_loss - (1 - clean_sim))
        """
        return ((real_loss,
                 fake_loss,
                 gen_loss,
                 descrim_loss
                ),
                {'denoiser_in':(x, x),
                 'gen_logits': gen_logits,
                 'generated':gen_prob,
                 'regen_hot':regen_hot,
                 'regen_prob':regen_prob,
                 'real_prob':real_prob,
                 'gen_hot':gen_hot,
                 'gen_is_real':gen_is_real,
                 'attn_weights': attn_weights,
                 'd_attn_weights': d_attn_weights
                 })

    def call(self, inputs, training):
        return self.forward_pass(inputs, training)[0]


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


# TODO can this be vectorized/sped up?
def local_shuffle(span, pvals):
    # Normalize pvals: np.multinomial internally casts to float64, which can cause sum to exceed 1
    pvals = np.array(pvals).astype(float)
    pvals /= pvals.sum()
    # Sample swap indices
    swapidx = np.argmax(np.random.multinomial(1, pvals, (len(span))), axis=1)
    swapped = len(span)*[False]
    span = list(span)
    i = 0
    while i < len(span):
        r = swapidx[i]
        if (i + r) >= len(span) or swapped[i]:
            i += 1
        else:
            swapped[i] = True
            swapped[i + r] = True
            swap = span[i + r]
            span[i + r] = span[i]
            span[i] = swap
            i += 1
    return span


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
            alt_start = random.randint(condlen, num_bytes - (1 + seqlen))
            condition = text[start - condlen: start]
            condition_tgt = text[1 + (start - condlen): 1 + start]
            center = text[start : start + seqlen]
            alt_center = text[alt_start: alt_start + seqlen]

            mask_rate = 0.5
            sample = np.random.uniform(size=(seqlen))
            pvals = scale([0.5*(1 - 0.5)**x for x in range(seqlen)]) # scaled geometric distribution
            denoise_masked = np.where(sample < mask_rate, model.maskid, center)

            sample = np.random.uniform(size=(seqlen))
            alt_denoise_masked = np.where(sample < mask_rate, model.maskid, alt_center)

            mask_rate = 0.9
            gen_masked = np.where(sample < mask_rate, model.maskid, center)


            yield ((gen_masked, alt_center, alt_denoise_masked, condition, center),
                    (center, center, center, center))
    dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature = (
                (
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(condlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                ),
                (
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                ),

            )

    )
    return dataset.batch(batchsize)



def plots(gen_prob, regen_prob, real_prob, idx, name):
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(256)),
        ticktext = [chr(i) for i in range(256)]
    )
    fig = px.imshow(tf.nn.softmax(real_prob[idx], axis=-1), aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_real_prob_{name}.html')

    fig = px.imshow(gen_prob[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_gen_prob_{name}.html')

    fig = px.imshow(regen_prob[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_regen_prob_{name}.html')

def char_predict(model, dataset, num):
    inputs, targets = next(iter(dataset)) # NOTE this always grabs the 1st batch because its a new iterator each time
    cnd_str = to_str(inputs[3])
    target_strings = to_str(targets[0])
    out = model.forward_pass(inputs, training=False)

    rng = np.random.default_rng()
    zero_ins = [i[:1] for i in inputs]
    zero_condition = zero_ins[3]
    zero_ins[0] = tf.zeros_like(zero_ins[0])
    full_gen_out = model.forward_pass(zero_ins , training=False)
    full_gen_score = tf.nn.sigmoid(full_gen_out[0][-3]).numpy()
    full_gen_out = full_gen_out[1]['generated']

    generated = out[1]['generated']
    gen_hot = out[1]['gen_hot']
    real_prob = out[1]['real_prob']
    gen_logits = out[1]['gen_logits']
    gen_is_real = out[1]['gen_is_real']
    print('\ngen_is_real: ', np.mean(np.mean(gen_is_real, axis=1)))
    regen_prob = out[1]['regen_prob']
    denoiser_x, denoiser_y = out[1]['denoiser_in']
    attn_weights = out[1]['attn_weights']
    for idx, aw in enumerate(attn_weights):
        aw = aw[0] # 1st batch element
        aw = np.reshape(aw, [model.num_heads*model.seqlen, -1])
        fig = px.imshow(aw, aspect='auto')
        fig.write_html(f'x_attn_{idx}.html')
    attn_weights = out[1]['d_attn_weights']
    for idx, aw in enumerate(attn_weights):
        aw = aw[0] # 1st batch element
        aw = np.reshape(aw, [model.num_heads*model.seqlen, -1])
        fig = px.imshow(aw, aspect='auto')
        fig.write_html(f'x_d_attn_{idx}.html')
    
    gen_with_score = np.concatenate([generated, gen_is_real], axis=-1)
    plots(gen_with_score, regen_prob, real_prob, 0, 'worst')
    plots(gen_with_score, regen_prob, real_prob, 1, 'best')

    full_gen_out = tf.argmax(full_gen_out, axis=-1)
    generated = tf.argmax(generated, axis=-1)
    regen_hot = tf.argmax(regen_prob, axis=-1)
    real_hot = tf.argmax(real_prob, axis=-1)
    gen_hot = tf.argmax(gen_hot, axis=-1)
    denoiser_x = tf.argmax(denoiser_x, axis=-1)
    denoiser_y = tf.argmax(denoiser_y, axis=-1)


    zero_ins = to_str(zero_ins[0])
    zero_condition = to_str(zero_condition)
    full_gen_out = to_str(full_gen_out)
    generated = to_str(generated)
    regen_hot = to_str(regen_hot)
    masked_str = to_str(inputs[0])
    masked_denoiser_str = to_str(inputs[1])
    masked_denoiser_alt_str = to_str(inputs[2])
    real_str = to_str(real_hot)
    gen_hot_str = to_str(gen_hot)

    denoiser_x = to_str(denoiser_x)
    denoiser_y = to_str(denoiser_y)

    print('--\n')
    """
    print(zero_condition[0] + '|' + full_gen_out[0])
    print('--')
    print(zero_condition[0] + '|' + zero_ins[0])
    print('\n------\n')
    """
    for i in range(num):
        print(cnd_str[i] + '|' + regen_hot[i])
        print('--')
        print(cnd_str[i] + '|' + generated[i])
        print('--')
        print(cnd_str[i] + '|' + masked_str[i])
        print('--')
        print(cnd_str[i] + '|' + target_strings[i])
        print('--')
        print(cnd_str[i] + '|' + real_str[i])
#        print('--')
#        print(cnd_str[i] + '|' + denoiser_y[i])
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
    gen_noise = 0.0
    gen_noise_decay = 0.9


    model = Model(d_model, seqlen, enc_depth, dropout, gen_noise)

    dataset = make_dataset(model, './guten-wot-train.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=True)
    valid_dataset = make_dataset(model, './wot_valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=False, training=False)
    inference_dataset = make_dataset(model, './wot_valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=False)

    inputs, targets = next(iter(dataset))
    condition_strings = to_str(inputs[3])
    input_strings = to_str(inputs[0])
    denoise_strings = to_str(inputs[1])
    alt_denoise_strings = to_str(inputs[2])
    target_strings = to_str(targets[0])
    for idx in range(8):
        print(condition_strings[idx] + '|' + input_strings[idx])
        print(condition_strings[idx] + '|' + denoise_strings[idx])
        print(condition_strings[idx] + '|' + alt_denoise_strings[idx])
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
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1,1,1,1], metrics=[])


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
