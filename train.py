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
#  scaled_attention_logits = matmul_qk / 0.25


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

  def call(self, v, k, q, mask):
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
        self.char_out_embed_size = 64

        backbone, inputs = self.make_backbone(d_model, dropout)
        self.generator = self.make_generator(backbone, inputs)
        self.descriminator = self.make_descriminator(backbone, inputs)
        untrainable_backbone, untrainable_inputs = self.make_backbone(d_model, dropout)
        self.untrainable_descriminator = self.make_descriminator(untrainable_backbone, untrainable_inputs)
        self.untrainable_descriminator.trainable = False

        self.maxvar = tf.math.reduce_variance(tf.one_hot(0, depth=256))
        self.last_predict = None
        self.last_gen_loss = 0.69
        self.cur_gen_loss = 0.69
        self.use_gen_ratio = 1/4
        self.noise = gen_noise
        self.xe_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.cosine_loss = tf.keras.losses.CosineSimilarity(reduction=tf.keras.losses.Reduction.NONE)
        self.sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def make_backbone(self, d_model, dropout):
        nheads = self.num_heads

        condition_inputs = tf.keras.Input(shape=(None, 256)) # one-hot bytes
        inputs = tf.keras.Input(shape=(None, 256)) # one-hot bytes
        mask_ahead = tf.keras.Input(shape=()) # flag 1 if should use look ahead mask else 0
        condlen = tf.shape(condition_inputs)[1]
        seqlen = tf.shape(inputs)[1]
        pos = tf.one_hot(tf.range(condlen + seqlen), depth=128)[tf.newaxis, :, :]
        pos = tf.tile(pos, [tf.shape(inputs)[0], 1, 1])
        x = tf.concat([condition_inputs, inputs], axis=1)
        x = tf.concat([x, pos], axis=-1)
        x = tf.keras.layers.Dense(d_model, tf.nn.relu)(x)
        outs = []
        attention_weights = []
        q = x[:, condlen:]
        condition = x[:, :condlen]


        look_ahead_mask = 0 * mask_ahead * create_look_ahead_mask(seqlen)
        condition_not_masked = tf.zeros((seqlen, condlen))
        for i in range(8):
            mask = tf.concat([condition_not_masked, look_ahead_mask], axis=1) if i % 8 == 0 else look_ahead_mask
            mask = mask[tf.newaxis, tf.newaxis, :, :]
            attn_out, attn_weights = MultiHeadAttention(d_model, nheads)(x,x,q, mask)
            attention_weights.append(attn_weights)
            attn_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attn_out + q)
            attn_out = tf.keras.layers.Dropout(rate=dropout)(attn_out)

            c = tf.keras.layers.Dense(2 * d_model, tf.nn.relu)(attn_out)
            c = tf.reshape(c, [tf.shape(inputs)[0], tf.shape(c)[1] // 4, 4 * 2 * d_model])
            c = tf.keras.layers.Dense(4 * 2 * d_model, tf.nn.relu)(c)
            c = tf.reshape(c, [tf.shape(inputs)[0], seqlen, 2 * d_model])

            c = tf.keras.layers.Dropout(rate=dropout)(c)
            c = tf.keras.layers.Dense(d_model, None)(c)
            q = tf.keras.layers.LayerNormalization(epsilon=1e-6)(c + attn_out)
            x = q
            if i % 8 == 7:
                x = tf.concat([condition, q], axis=1)
                outs.append(q)
        x = tf.concat(outs, axis=-1)
        return tf.keras.Model(inputs=(condition_inputs, inputs, mask_ahead), outputs=(x, attention_weights)), (condition_inputs, inputs, mask_ahead)

    def make_generator(self, backbone, inputs):
        gen_logits = tf.keras.layers.Dense(256)(backbone.outputs[0])
        return tf.keras.Model(inputs=inputs, outputs=(gen_logits, (backbone.outputs[1:])))

    def make_descriminator(self, backbone, inputs):
        logits = tf.keras.layers.Dense(1)(backbone.outputs[0])
        lm_logits = tf.keras.layers.Dense(256)(backbone.outputs[0])
        return tf.keras.Model(inputs=inputs, outputs=(logits, lm_logits, (backbone.outputs[1:]) ))

    def confidently_correct(self, x, y, min_confidence=0.99):
        correct = tf.argmax(x, axis=-1) == tf.argmax(y, axis=-1)
        confident = tf.reduce_max(x, axis=-1) > min_confidence
        return tf.cast(correct & confident, x.dtype)

    def forward_pass(self, inputs, training):
        not_masked = tf.where(inputs[0] == 0, 0.0, 1.0)
        masked_inputs = tf.one_hot(inputs[0], depth=256)
        condition_inputs = tf.one_hot(inputs[1], depth=256)
        hot_clean = inputs[2]
        real_corrupt_mask = inputs[3]

        batchsize = tf.shape(masked_inputs)[0]
        seqlen = tf.shape(masked_inputs)[1]

        gen_logits, attn_weights = self.generator((condition_inputs, masked_inputs, 0))
        generated = tf.nn.softmax(gen_logits / 1)
        real = hot_clean
        fake = tf.nn.softmax(gen_logits / 1)

        label_smoothing = 0.5#16/256
        step = 0.1
        t = tf.ones([batchsize, seqlen, 1], dtype=tf.float32)
        prev_direction = None
        for _ in range(100):
            fake = tf.nn.softmax(gen_logits / t)
            maxval = tf.reduce_max(fake, axis=-1, keepdims=True)

            direction = tf.where(maxval > label_smoothing, 1, -1)
            if prev_direction is not None:
                overshoot = tf.where(direction != prev_direction, 1, 0)
                step = tf.where(overshoot == 1, step / 2, step)
            prev_direction = direction

            t = tf.where(maxval > label_smoothing, t + step, t - step)
            t = tf.where(t <= 0.0, 1e-16, t)

        predicted = tf.argmax(generated, axis=-1)
        hot_predicted = tf.one_hot(predicted, depth=256)
        max_value = tf.reduce_max(fake, axis=-1, keepdims=True)
        hot_clean = tf.cast(hot_clean, fake.dtype)
        value_at_correct_place = tf.reduce_sum(hot_clean * fake, axis=-1, keepdims=True)
        real = tf.where(hot_clean == 1, max_value, fake)
        real = tf.where(hot_predicted == 1, value_at_correct_place, real)


        # TODO move this to dataset pipeline and pass in a 1/0 mask
        corrupt_rate = tf.random.uniform((batchsize, 1, 1), minval=0.0, maxval=0.5)
        corrupt_mask = tf.random.uniform(tf.shape(inputs[0][:, :, tf.newaxis]), maxval=1)
        corrupt_mask = tf.where(corrupt_mask < corrupt_rate, 1.0, 0.0)
        real = tf.where(corrupt_mask == 1.0, fake, real)
        real = tf.stop_gradient(real)


#        self.untrainable_descriminator.set_weights(self.descriminator.get_weights())
        is_real_fake, lm_logits_fake, d_attn_weights = self.untrainable_descriminator((condition_inputs, fake, 1))
        is_real_false = is_real_fake
        is_real_fake_corrupt = is_real_fake # FIXME
        is_real_true, lm_logits, _ = self.descriminator((condition_inputs, tf.stop_gradient(real), 1))


        is_real_fake_loss = self.sig_loss(tf.ones_like(corrupt_mask), is_real_fake)
        correct_mask = self.confidently_correct(real, generated, min_confidence=0.0)
        correct_mask = tf.where((correct_mask[:, :, tf.newaxis] == 1) | (corrupt_mask == 1), 1, 0) # could just use correct directly?
        is_real_fake_corrupt_loss = self.sig_loss(correct_mask, is_real_fake_corrupt)
        correct_mask = self.confidently_correct(real, hot_clean, min_confidence=0.0)
        is_real_true_loss = self.sig_loss(correct_mask, is_real_true)

        is_fake = 1 - tf.nn.sigmoid(is_real_fake)[:, :, 0]
        is_real = 1 - is_fake

        lm_fake_loss = self.xe_loss(hot_clean, lm_logits_fake)
        correct_mask = self.confidently_correct(generated, hot_clean, min_confidence=0.0)
        lm_fake_loss *= tf.where(correct_mask == 1, 1.0, -1.0)

        lm_fake_loss = self.xe_loss(tf.one_hot(tf.argmax(lm_logits_fake, axis=-1), depth=256), gen_logits, (1 - not_masked))
        gen_loss = self.xe_loss(hot_clean, gen_logits, not_masked)
        norm_loss = tf.nn.relu((tf.norm(gen_logits, axis=-1) - 20))
        return ((
                 lm_logits,
                 lm_fake_loss,
                 is_real_fake_corrupt_loss,
                 is_real_fake_loss,
                 is_real_true_loss,
                 gen_loss,
                 norm_loss),
                {'logits':gen_logits,
                 'gen_logits': gen_logits,
                 'generated':generated,
                 'real':real,
                 'fake':fake,
                 'noised':tf.nn.softmax(lm_logits_fake),
                 'is_real_fake': is_real_fake,
                 'is_real_false': is_real_false,
                 'is_real_true': is_real_true,
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
            condition_tgt = text[1 + (start - condlen): 1 + start]
            center = text[start : start + seqlen]

            masked_in = center
            masked_in, corrupt_mask = corrupt_data(masked_in, np.random.uniform(0.5, 1))

            is_real_fake  = seqlen * [1]
            is_real_false = seqlen * [0]
            is_real_true  = seqlen * [1]

            hot_clean = tf.one_hot(center, depth=256)
            yield ((masked_in, condition, hot_clean, corrupt_mask),
                    (center, center, center, is_real_fake, is_real_true, center, center))
    dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature = (
                (
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(condlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen, 256), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.float32),
                ),
                (
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                    tf.TensorSpec(shape=(seqlen), dtype=tf.int32),
                ),

            )

    )
    return dataset.batch(batchsize)



def plots(generated, fake, real, logits, gen_logits, noised, idx, name):
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(256)),
        ticktext = [chr(i) for i in range(256)]
    )
    fig = px.imshow(tf.nn.softmax(logits[idx], axis=-1), aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_lm_{name}.html')

    fig = px.imshow(gen_logits[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_gen_log_{name}.html')

    fig = px.imshow(generated[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_gen_{name}.html')

    fig = px.imshow(real[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'y_{name}.html')

    fig = px.imshow(fake[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_fake_{name}.html')

    fig = px.imshow(noised[idx], aspect='auto')
    fig.update_layout(xaxis=xaxis)
    fig.write_html(f'x_noised_{name}.html')

def char_predict(model, dataset, num):
    inputs, targets = next(iter(dataset)) # NOTE this always grabs the 1st batch because its a new iterator each time
    cnd_str = to_str(inputs[1])
    target_strings = to_str(targets[0])
    is_real_fake = targets[-3]
    is_real_false = targets[-2]
    is_real_true = targets[-1]
    out = model.forward_pass(inputs, training=False)

    zero_ins = [i[:1] for i in inputs]
    zero_ins[0] = tf.zeros_like(zero_ins[0])
    full_gen_out = model.forward_pass(zero_ins , training=False)
    full_gen_score = tf.nn.sigmoid(full_gen_out[0][-3]).numpy()
    full_gen_out = full_gen_out[1]['generated']

    fake_decision = tf.nn.sigmoid(out[1]['is_real_fake'][:, :, 0]).numpy()
    false_decision = tf.nn.sigmoid(out[1]['is_real_false'][:, :, 0]).numpy()
    true_decision = tf.nn.sigmoid(out[1]['is_real_true'][:, :, 0]).numpy()
    right_masklen = 32 # FIXME should come from the data
    sort_idx = np.argsort(np.sum(fake_decision[:, -right_masklen:] , axis=1))
    median_idx = sort_idx[len(sort_idx) // 2]

    generated = out[1]['generated']
    fake = out[1]['fake']
    real = out[1]['real']
    logits = out[1]['logits']
    gen_logits = out[1]['gen_logits']
    noised = out[1]['noised']
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

    plots(generated, fake, real, logits, gen_logits, noised, sort_idx[0], 'worst')
    plots(generated, fake, real, logits, gen_logits, noised, sort_idx[-1], 'best')


    print('--\n')
    print(to_str(tf.argmax(full_gen_out, axis=-1))[0])
    print()
    predicted = to_str(tf.argmax(logits, axis=-1))

    generated = tf.argmax(generated, axis=-1)
    logits = tf.argmax(logits, axis=-1)
    real = tf.argmax(real, axis=-1)
    fake = tf.argmax(fake, axis=-1)


    generated = to_str(generated)
    lm_str = to_str(logits)
    masked_str = to_str(inputs[0])
    real_str = to_str(real)
    fake_str = to_str(fake)

    middle = len(sort_idx) // 2
#    for i in np.concatenate([sort_idx[:1], sort_idx[middle-num:middle+num], sort_idx[-1:]], axis=0):
    for i in range(num+2):
        print((np.mean(fake_decision[i][-right_masklen:])))
        print(cnd_str[i] + '|' + generated[i])
        print('--')
        print(cnd_str[i] + '|' + target_strings[i])
        print('--')
        print(cnd_str[i] + '|' + masked_str[i])
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
    batchsize = 2048
    condlen = 16
    seqlen = 8
    gramlen = 5
    gen_noise = 0.0
    gen_noise_decay = 0.9


    model = Model(d_model, seqlen, enc_depth, dropout, gen_noise)

    dataset = make_dataset(model, './guten-wot-train.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=True)
    valid_dataset = make_dataset(model, './wot_valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=False, training=False)
    inference_dataset = make_dataset(model, './wot_valid.txt', batchsize, seqlen, condlen, gramlen, shuffle=True, training=False)

    inputs, targets = next(iter(dataset))
    condition_strings = to_str(inputs[1])
    input_strings = to_str(inputs[0])
    target_strings = to_str(targets[0])
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

    loss = (cat_loss, through_loss , through_loss, through_loss, through_loss, through_loss, through_loss)
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1,1, 0, 0,0,1,1], metrics=[])


    logits = model(inputs) # Running the model builds all the layers
    print(model.summary())
    if args.restore:
        dweights = {}
        for w in model.weights:
            if 'descrim' in w.name:
                dweights[w.name] = w.read_value()
        model.load_weights('./unregressive')
        model.untrainable_descriminator.set_weights(model.descriminator.get_weights())

    char_predict(model, dataset, 8)

    if args.train:
        def save(epoch, logs):
            model.save_weights('./unregressive', overwrite=True, save_format=None, options=None)
            return
        save_callback = tf.keras.callbacks.LambdaCallback( on_epoch_end=save)

        def copy_weights(batch, logs):
            model.untrainable_descriminator.set_weights(model.descriminator.get_weights())
            return

        copy_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=copy_weights)
        def run_char_predict(batch, logs):
            if batch % 500 == 0:
                char_predict(model, inference_dataset, 2)
            if batch % 500 == 499:
                model.save_weights('./unregressive.batch', overwrite=True, save_format=None, options=None)

        char_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=run_char_predict)

        def step(batch, logs):
            model.step += 1
        step_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=step)

        callbacks = [save_callback, copy_callback, char_callback, step_callback]
        model.fit(dataset, epochs=4, validation_data=valid_dataset, callbacks=callbacks)

    if args.eval:
        model.evaluate(valid_dataset)

    """
    with tf.GradientTape() as tape:

        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        logits = model(inputs, training=True)  # Logits for this minibatch

        # Compute the loss value for this minibatch.
        loss_value = loss(targets, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    for g,v in zip(grads, model.trainable_weights):
        print(v.name)
        print(g)
        print('-----')
    exit()
    """


