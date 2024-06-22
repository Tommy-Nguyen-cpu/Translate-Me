import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super().__init__()

        self.d_model = tf.cast(d_model, dtype = tf.float32) # Cast to float.

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype = tf.float32) # Cast to float
        arg1 = tf.math.rsqrt(step) # Calculates reciprocal square root of step ( step ^ (-.5)).
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) # Increases for the first "warmup_steps" and decreases afterwards.
