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
            padlen = tf.math.floormod(4 - seqlen, 4)
#           pad = tf.zeros((tf.shape(c)[0], padlen, tf.shape(c)[2],), dtype=c.dtype)
#            c = tf.concat([c, pad], axis=1)
            c = tf.reshape(c, [tf.shape(inputs)[0], tf.shape(c)[1] // 4, 4 * 2 * d_model])
#            c = (1-mask_ahead) * tf.keras.layers.Dense(4 * 2 * d_model, tf.nn.relu)(c) + mask_ahead * c # FIXME inefficient: don't compute when not needed
            c = tf.keras.layers.Dense(4 * 2 * d_model, tf.nn.relu)(c)
            c = tf.reshape(c, [tf.shape(inputs)[0], (seqlen + padlen), 2 * d_model])
#            c = c[:, :-padlen]
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

        x = tf.keras.layers.Dense(32)(backbone.outputs[1])
        x = tf.keras.layers.Reshape((32, 1))(x)
        descriminator = tf.keras.Model(inputs=inputs, outputs=(x, (backbone.outputs[1:])))
        return descriminator

    def scale(self, x):
        x -= tf.reduce_min(x, axis=-1, keepdims=True)
        x /= tf.reduce_sum(x, axis=-1, keepdims=True)
        return x

    def add_noise(self, x, std):
        return x + tf.random.normal(tf.shape(x), stddev=std, dtype=x.dtype)
    
    def make_noise(self, x, std):
        return tf.random.normal(tf.shape(x), stddev=std, dtype=tf.float32)

    def confidently_correct(self, x, y, min_confidence=0.99):
        correct = tf.argmax(x, axis=-1) == tf.argmax(y, axis=-1)
        confident = tf.reduce_max(x, axis=-1) > min_confidence
        return tf.cast(correct & confident, x.dtype)


    def hide_probabilities(self, x):
        maxval = tf.reduce_max(x, axis=-1, keepdims=True)
        mask = tf.where(x == maxval, 1.0, 0.0)
        x *= (mask / x)
        return x
    
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
#        choice = tf.cast(tf.random.uniform(shape=(batchsize, 1, 1), minval=0, maxval=2, dtype=tf.int32), corrupt_rate.dtype)
#        corrupt_rate = choice * corrupt_rate + (1 - choice) * (1 - corrupt_rate)

        corrupt_mask = tf.random.uniform(tf.shape(inputs[0][:, :, tf.newaxis]), maxval=1)
        corrupt_mask = tf.where(corrupt_mask < corrupt_rate, 1.0, 0.0)
#        corrupt_mask = tf.where(inputs[0][:, :, tf.newaxis] == 0, 1.0, 0.0) # FIXME DONT DO this overriding with inputs mask because descrim will not see many real inputs (inputs mask is above 50%)
        real = tf.where(corrupt_mask == 1.0, fake, real)

        real = tf.stop_gradient(real)
        corrupt_fake = tf.where(corrupt_mask == 1.0, fake, real) # NOTE same as real, but where gradients can flow through fake portion
 

#        self.untrainable_descriminator.set_weights(self.descriminator.get_weights())
        is_real_fake, lm_logits_fake, d_attn_weights = self.untrainable_descriminator((condition_inputs, fake, 1))
        is_real_false = is_real_fake
#        is_real_fake_corrupt, lm_logits_fake, _ = self.untrainable_descriminator((condition_inputs, corrupt_fake, 1))
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

#        same = tf.where(predicted == tf.roll(predicted, shift=1, axis=0), 1.0, 0.0)
#        diff_loss = self.xe_loss(hot_clean, gen_logits) * same
#        diff_loss += -self.xe_loss(tf.roll(hot_predicted, axis=0, shift=1), gen_logits) * (1 - correct_mask) * same
#        diff_loss *= is_fake

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


    def Oldforward_pass(self, inputs, training):
        masked_inputs = tf.one_hot(inputs[0], depth=256)
        condition_inputs = tf.one_hot(inputs[1], depth=256)
        hot_clean = inputs[2]
        real_corrupt_mask = inputs[3]

        batchsize = tf.shape(masked_inputs)[0]
        seqlen = tf.shape(masked_inputs)[1]

        noise = 1 * self.make_noise(masked_inputs, 0.01)

        gen_logits, attn_weights = self.generator((condition_inputs, 1 * noise + masked_inputs, 0))
        generated = tf.nn.softmax(gen_logits / 1)
        fake = gen_logits
        real = hot_clean



        """
        is_real_fake = tf.ones((batchsize, seqlen, 1))
        is_real_false = tf.zeros((batchsize,seqlen, 1))
        is_real_true = tf.ones((batchsize, seqlen, 1))
        return ((
                 gen_logits,
                 is_real_fake,
                 is_real_false,
                 is_real_true),
                {'logits':lm_logits,
                 'gen_logits': gen_logits,
                 'generated':generated,
                 'real':real,
                 'fake':fake,
                 'is_real_fake': is_real_fake,
                 'is_real_false': is_real_false,
                 'is_real_true': is_real_fake,
                 'attn_weights': attn_weights
                 })
        """


        fake = tf.nn.softmax(gen_logits / 1)
        real = tf.nn.softmax(real / 1)

        
        
        
        
        label_smoothing = 2/256#0.50
        step = 0.1
        t = tf.ones([batchsize, seqlen, 1], dtype=tf.float32)
        prev_direction = None
        for _ in range(1000):
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




#        fake = self.scale(gen_logits)
        corrupt_mask = tf.random.uniform(tf.shape(inputs[0][:, :, tf.newaxis]), maxval=1)

#        real = tf.where(hot_clean ==1, 0.125, (1 - 0.125)/255)
#        print(tf.reduce_sum(real, axis=-1))
#        real = self.scale(noise + hot_clean)

#        label_smoothing = tf.reduce_max(fake, axis=-1, keepdims=True)
#        real = tf.where(hot_clean == 1, label_smoothing, (1 - label_smoothing) / 255) 
        uncorrupted_real_noised = real
        # TODO move this to dataset pipeline and pass in a 1/0 mask
        corrupt_rate = tf.random.uniform((batchsize, 1, 1), minval=0.5, maxval=1.0)
        choice = tf.cast(tf.random.uniform(shape=(batchsize, 1, 1), minval=0, maxval=2, dtype=tf.int32), corrupt_rate.dtype)
        corrupt_rate = choice * corrupt_rate + (1 - choice) * (1 - corrupt_rate)

        corrupt_mask = tf.where(corrupt_mask < corrupt_rate, 1.0, 0.0)
        real = tf.where(corrupt_mask == 1.0, fake, real)
#        real = tf.where(real_corrupt_mask[:,:,tf.newaxis] == 0, real, fake)
        real = tf.stop_gradient(real)
        corrupt_fake = tf.where(corrupt_mask == 1.0, fake, real) # NOTE same as real, but where gradients can flow through fake portion

#        fake = tf.where(corrupt_mask == 1.0, fake, tf.stop_gradient(real))


        """
        is_real_fake = self.untrainable_descriminator((condition_inputs, fake))
        is_real_false = self.descriminator((condition_inputs, tf.stop_gradient(fake)))
        is_real_true = self.descriminator((condition_inputs, tf.stop_gradient(real)))

        gen_correct_mask = self.confidently_correct(generated, hot_clean, min_confidence=0.0)
        is_real_false_loss = self.sig_loss(gen_correct_mask, is_real_false)
        """
        """
        cnd_window = 4 
        window = 16
#        fake_condition_inputs = [condition_inputs[:, -cnd_window:]]
#        fake_condition_inputs = fake_condition_inputs + [fake[:, i:i+cnd_window] for i in range(0, 32, cnd_window)][:-1]
#        fake_condition_inputs = tf.concat(fake_condition_inputs, axis=0)
        fake_windowed = tf.concat([fake[:, i:i+window] for i in range(0, 32, window)], axis=0)
        fake_condition_inputs = tf.zeros_like(fake_windowed[:, :0, :])

#        real_condition_inputs = [condition_inputs[:, -cnd_window:]]
#        real_condition_inputs = real_condition_inputs + [real[:, i:i+cnd_window] for i in range(0, 32, cnd_window)][:-1]
#        real_condition_inputs = tf.concat(real_condition_inputs, axis=0)
        real_windowed = tf.concat([real[:, i:i+window] for i in range(0, 32, window)], axis=0)
        real_condition_inputs = tf.zeros_like(real_windowed[:, :0, :])
        """

 #       fake = tf.concat([fake[:, :1, :], tf.zeros_like(fake[:, 1:, :])], axis=1)
 #       real = tf.concat([real[:, :1, :], tf.zeros_like(real[:, 1:, :])], axis=1)
#        generated = tf.concat([generated[:, :1, :], tf.zeros_like(generated[:, 1:, :])], axis=1)
        
        fake_windowed = fake # FIXME hide probabilities
        real_windowed = real
        fake_condition_inputs = condition_inputs
        real_condition_inputs = condition_inputs

        is_real_fake_windowed, _, d_attn_weights = self.untrainable_descriminator((fake_condition_inputs, fake_windowed, 1))
        is_real_fake_corrupt, lm_logits_fake, _ = self.untrainable_descriminator((fake_condition_inputs, corrupt_fake, 1))
        is_real_false_windowed = is_real_fake_windowed # FIXME
        is_real_true_windowed, lm_logits, _ = self.descriminator((real_condition_inputs, tf.stop_gradient(real_windowed), 1))
        
        is_real_fake_windowed = tf.reshape(is_real_fake_windowed, [batchsize, seqlen, 1])
        is_real_false_windowed = tf.reshape(is_real_false_windowed, [batchsize, seqlen, 1])
        is_real_true_windowed = tf.reshape(is_real_true_windowed, [batchsize, seqlen, 1])

        # FIXME overriding 
        is_real_fake = is_real_fake_windowed
        is_real_false = is_real_false_windowed
        is_real_true = is_real_true_windowed

#        real_correct_mask = self.confidently_correct(real, hot_clean, min_confidence=0.0)
        is_real_fake_loss = self.sig_loss(tf.ones_like(corrupt_mask), is_real_fake)
        is_real_fake_corrupt_loss = self.sig_loss(corrupt_mask, is_real_fake_corrupt)
        is_real_true_loss = self.sig_loss(1 - corrupt_mask, is_real_true)



#        gen_correct_mask = self.confidently_correct(generated, hot_clean, min_confidence=0.0)
#        is_real_false_loss = self.sig_loss(gen_correct_mask, is_real_false)
#        is_real_false_loss = self.sig_loss(tf.zeros_like(is_real_false[:, :, 0]), is_real_false)

        is_fake = 1 - tf.nn.sigmoid(is_real_fake)[:, :, 0]
        is_real = 1 - is_fake
        shifts = list(range(1, 9))
#        cos = [-self.cosine_loss(fake, tf.roll(fake, shift=shift, axis=0)) for shift in shifts]
        cos = [self.weighted_cos(gen_logits, is_fake, shift) for shift in shifts]
        cos = sum(cos) / len(cos)
        """
        fake_noised = tf.where(tf.one_hot(tf.argmax(fake, axis=-1), depth=256) == 1, label_smoothing, (1 - label_smoothing) / 255) 
        logs_noised = tf.where(tf.one_hot(tf.argmax(lm_logits, axis=-1), depth=256) == 1, label_smoothing, (1 - label_smoothing) / 255)

        ave_noised = (fake_noised + logs_noised) / 2
        flat = (1 / 256) * tf.ones_like(fake)
        #lm_fake_loss =  1e4*self.mse(tf.stop_gradient(ave_noised), fake)
        lm_fake_loss =  self.xe_loss(tf.stop_gradient(tf.nn.softmax(lm_logits)), gen_logits)


        lm_fake_loss *= tf.nn.relu(is_fake - 0.5)#NOTE gradient now relates the 2 losses, meaning the model "knows" it can reduce one by improving the other
        print(lm_fake_loss)
        """
        lm_fake_loss = self.xe_loss(tf.stop_gradient(hot_predicted), lm_logits_fake)

#        is_real_fake_loss *= tf.where(lm_fake_loss > 0.1, 0.0, 1.0) # gradients not related
#        is_real_true_loss *= tf.where(lm_fake_loss > 0.1, 0.0, 1.0)
        return ((
                 lm_logits,
                 lm_fake_loss,
                 is_real_fake_corrupt_loss,
                 is_real_fake_loss,
                 is_real_true_loss),
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

    def same_predict(self, x):
        predict = tf.argmax(x, axis=-1)
        return tf.where(predict == tf.roll(predict, shift=1, axis=0), 1.0, 0.0)

    def weighted_cos(self, x, is_fake, shift):
        y = tf.roll(x, shift=shift, axis=0)        
        xidx = tf.argmax(x, axis=-1)
        yidx = tf.argmax(y, axis=-1)
        mask = 2 * tf.nn.relu(is_fake - 0.5) * tf.where(yidx == xidx, 1.0, 0.0)

        random_targets = tf.one_hot(tf.random.uniform(tf.shape(xidx), minval=0, maxval=256, dtype=tf.int32), depth=256)
#        return self.xe_loss(random_targets, x) * mask
        return 1*self.cosine_loss(tf.stop_gradient(y), x) * mask

    def call(self, inputs, training):
        return self.forward_pass(inputs, training)[0]


def to_str(array):
    lines = []
    array = tf.where(array == 0, ord('_'), array)
    array = tf.where(array == 1, ord('+'), array)
    for line in array.numpy().astype(np.uint8):
        lines.append(bytes(line).decode('utf-8', 'replace'))
    return lines

def get_span(text, start, seqlen):
    return text[start : start + seqlen]

def to_int(span, padlen):
    span = [int(c) for c in span]
    if padlen > 0:
        span = padlen * [0] + span[padlen:-padlen] + padlen * [0]
    return span

def corrupt_data(span, corrupt_rate):
    corrupt = np.random.randint(0, 256, len(span))
    corrupt = np.zeros_like(span) # FIXME zero out
    corrupt_mask = np.random.binomial(1, corrupt_rate, len(span))
    return np.where(corrupt_mask, corrupt, span), corrupt_mask

def span_mask(span, masklen):
    assert masklen <= len(span)
#    unmasklen = (len(span) - masklen) // 2
#    span = [span[:unmasklen], masklen * [0], span[-(len(span) - masklen - unmasklen):]]
    span = [span[:-masklen], masklen * [0]]
    return np.concatenate(span, axis=0)

def word_mask(span):
    # FIXME this is obviously bad and slow
    maskfirst = np.random.binomial(1, 0.5)
    split_indices = np.where((span == 32) | (span == 10))[0]
    span = np.split(span, split_indices)
    span = [w if i % 2 == maskfirst else np.zeros_like(w) for i,w in enumerate(span)]
    return np.concatenate(span)

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

    text += b''.join((seqlen - len(text) % seqlen)  * [b'\x01']) # NOTE mask is 0, pad is 1.
                                                                 # Model fills in masks, so we don't
                                                                 # want it to see masks as just a
                                                                 # copy through.
                                                                 # TODO use special tokens outside the byte range
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
#            masked_in = word_mask(center)
            masked_in, corrupt_mask = corrupt_data(masked_in, np.random.uniform(0.5, 1))
#            masked_in = condition[-seqlen:] # FIXME
#            masked_in = span_mask(masked_in, 3)

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

#    decisions = [tf.nn.sigmoid(y).numpy() for y in out[1][-3:]]
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

    """
    embeds = model.embeddings(tf.range(32))
    fig = px.imshow(embeds, aspect='auto')
    fig.write_html(f'x_embeds.html')

    for i, attn_block in enumerate(out[1]['attn']):
        fig = px.imshow(attn_block[0, :, 0, :], aspect='auto', zmin=0, zmax=1) # NOTE there are multiple heads but only one query
        fig.write_html(f'x_attn_{i}.html')
    example_x = out[1]['example_x']
    fig = px.imshow(example_x, aspect='auto')
    fig.write_html(f'x_example.html')
    """
    plots(generated, fake, real, logits, gen_logits, noised, sort_idx[0], 'worst')
    plots(generated, fake, real, logits, gen_logits, noised, sort_idx[-1], 'best')


    print('--\n')
#    print(full_gen_score[0, :, 0])
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
#        print(cnd_str[i] + '|' + lm_str[i])
#        print('--')
        print(cnd_str[i] + '|' + generated[i])
        print('--')
        print(cnd_str[i] + '|' + target_strings[i])
        print('--')
        print(cnd_str[i] + '|' + masked_str[i])
#        print(fake_decision[i])
#        print(false_decision[i])
#        print(fake_decision[i] - false_decision[i])
#        print(is_real_fake[i].numpy())
#        print(is_real_false[i].numpy())
#        print('--')
#        print(masked_str[i])
        print('\n------\n')

def unscramble(seq, indices):
    result = []
    indices = np.where(indices >= seq.shape[1], 0, indices)
    for row, idx in zip(np.array(seq), indices):
        result.append(row[np.argsort(idx)])
    return np.array(result)

def embed_data(model, dataset):
    embeds = []
    text = []
    for inputs, targets in iter(dataset):
        out = model.forward_pass(inputs, training=False)
        for x in out[1]['embeds']:
            embeds.append(x)
#            print(np.std(x))
        for x in inputs[1]:
            text.extend(to_str(x))
    embeds = np.concatenate(embeds, axis=0)
    assert embeds.shape[0] == len(text)

    reducer = umap.UMAP()
    embeds = reducer.fit_transform(embeds)
    fig = px.scatter(x=embeds[:, 0], y=embeds[:, 1], hover_name=text)
    fig.show()

class UnorderedLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()
        self.bxe = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        super(UnorderedLoss, self).__init__()

    def call(self, y_true, y_pred):
        batchsize = tf.shape(y_true)[0]
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        on_value = 0.9
        y_true = tf.one_hot(y_true, depth=256, on_value=on_value, off_value=(1 - on_value) / 255)




        """
        predicted = tf.argmax(y_pred, axis=-1)
        clean_inputs = y_true
        max_value = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        value_at_correct_place = tf.reduce_sum(clean_inputs * y_pred, axis=-1, keepdims=True)
        real = tf.where(clean_inputs == on_value, max_value, y_pred)
        real = tf.where(tf.one_hot(predicted, depth=256) == 1, value_at_correct_place, real)
        real = tf.stop_gradient(real)
        y_true = real
        """

        y_pred = tf.reshape(y_pred, [-1, 4, 256]) # FIXME assumes seqlen is divisible by 8
        y_true = tf.reshape(y_true, [-1, 4, 256])

        y_pred = tf.reduce_sum(y_pred, axis=1, keepdims=True)
        y_true = tf.reduce_sum(y_true, axis=1, keepdims=True)

        y_pred = tf.reshape(y_pred, [batchsize, -1, 256])
        y_true = tf.reshape(y_true, [batchsize, -1, 256])

        """
        fig = px.imshow(y_pred[0], aspect='auto')
        fig.show()
        fig = px.imshow(y_true[0], aspect='auto')
        fig.show()
        """

        return self.bxe(y_true, y_pred)

class NormLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.maximum(0.0, tf.norm(y_pred, axis=-1) - 75)

class PassthroughLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return y_pred

class NoLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return 0.0

class VarLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return -tf.math.reduce_variance(y_pred, axis=-1)

class ControlledLoss(tf.keras.losses.Loss):

    def __init__(self):
        self.sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        super(ControlledLoss, self).__init__()

    def call(self, y_true, y_pred):
        loss = self.sig_loss(y_true, y_pred)
        loss = (loss - 0.68) ** 2
        return loss

class WeightedLoss(tf.keras.losses.Loss):

    def __init__(self):
        self.xe_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        super(WeightedLoss, self).__init__()

    def call(self, y_true, y_pred):
        weights = y_pred[:, :, -1:]
        y_pred = y_pred[:, :, :-1]

        return self.xe_loss(y_true, y_pred, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--load_d_weights', action='store_true')
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

#    dataset = make_dataset(model, './wot_train.txt.dedupe', batchsize, seqlen, padlen, gramlen, shuffle=True, training=True)
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


#    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    cat_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    xe_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    smooth_sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.95)
    sig_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    unordered_loss = UnorderedLoss()
    controlled_loss = ControlledLoss()
    weighted_loss = WeightedLoss()
    norm_loss = NormLoss()
    no_loss = NoLoss()
    through_loss = PassthroughLoss()

    loss = (cat_loss, through_loss , through_loss, through_loss, through_loss, through_loss, through_loss)
#    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1,1, 0, 1,1,1,1], metrics=[])
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1,1, 0, 0,0,1,1], metrics=[])


    logits = model(inputs) # Running the model builds all the layers
    print(model.summary())
    if args.restore:
        dweights = {}
        for w in model.weights:
            if 'descrim' in w.name:
                dweights[w.name] = w.read_value()
        model.load_weights('./unregressive')
        if args.load_d_weights:
            model.descriminator_backbone.set_weights(model.generator_backbone.get_weights())
        model.untrainable_descriminator.set_weights(model.descriminator.get_weights())

#        for w in model.weights:
#            if 'descrim' in w.name:
#                w.assign(dweights[w.name])


#    print(unordered_loss(targets[0], model(inputs)[0]))
    char_predict(model, dataset, 8)

    if args.visualize:
        char_predict(model, dataset, 5)
        embed_data(model, inference_dataset)

    if args.train:
        def save(epoch, logs):
#            model.save('./unregressive/', overwrite=True, include_optimizer=False, save_format='tf')
            model.save_weights('./unregressive', overwrite=True, save_format=None, options=None)
            return
        save_callback = tf.keras.callbacks.LambdaCallback( on_epoch_end=save)

        def copy_weights(batch, logs):
            model.untrainable_descriminator.set_weights(model.descriminator.get_weights())
            return
        copy_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=copy_weights)
        """
        def run_char_predict(epoch, logs):
            char_predict(model, inference_dataset, 2)
            print(f'Global step:{model.step}\tg_ratio: {model.use_gen_ratio}\tnoise:{model.noise}')
        char_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=run_char_predict)
        """
        def run_char_predict(batch, logs):
            if batch % 500 == 0:
                char_predict(model, inference_dataset, 2)
            if batch % 500 == 499:
                model.save_weights('./unregressive.batch', overwrite=True, save_format=None, options=None)

        char_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=run_char_predict)

        def step(batch, logs):
#            gen_loss = logs['output_3_loss'] # FIXME generator loss should have a namev
#            print('\n', logs['output_7_loss'], logs['output_8_loss'])
#            model.use_gen_ratio = min(1, max(0, 4 - 5 * gen_loss))
#            model.use_gen_ratio = min(1, max(0, -(1/3.30685) * gen_loss + 1.2))
#            if batch % 100 == 0:
#                char_predict(model, inference_dataset, 1)

#            if gen_loss > 0.69315 and gen_loss > model.last_gen_loss:
#                model.noise = min(.9, model.noise + 0.05)
#            elif gen_loss < 0.69315 and gen_loss < model.last_gen_loss:
#                model.noise = max(0, model.noise - 0.0001)
#            model.last_gen_loss = gen_loss

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


