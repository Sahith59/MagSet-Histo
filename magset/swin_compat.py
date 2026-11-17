"""Exact integer window counting for the pinned TensorFlow Swin implementation."""


def install_integer_window_reverse():
    import tensorflow as tf
    from transformers.models.swin import modeling_tf_swin as module

    original = module.window_reverse

    def integer_window_reverse(windows, window_size, height, width):
        # Padded spatial dimensions are exact multiples of the window size.
        # Avoid float division followed by truncation in graph optimization.
        size = tf.cast(window_size, tf.int32)
        height, width = tf.cast(height, tf.int32), tf.cast(width, tf.int32)
        windows_per_image = (height // size) * (width // size)
        count = tf.shape(windows)[0]
        checks = [tf.debugging.assert_equal(height % size, 0),
                  tf.debugging.assert_equal(width % size, 0),
                  tf.debugging.assert_equal(count % windows_per_image, 0)]
        with tf.control_dependencies(checks):
            batch = count // windows_per_image
            values = tf.reshape(windows, (batch, height // size, width // size, size, size, -1))
            values = tf.transpose(values, (0, 1, 3, 2, 4, 5))
            return tf.reshape(values, (batch, height, width, -1))

    module.window_reverse = integer_window_reverse
    return original, integer_window_reverse


def check_window_reverse():
    """Technical output/gradient tests only; no data, model or fitted parameters."""
    import numpy as np
    import tensorflow as tf
    from transformers.models.swin import modeling_tf_swin as module

    original, repaired = install_integer_window_reverse()
    rows=[]; original_failed=False
    for height in (7, 14, 28, 56):
        for batch in (1, 2, 8, 32):
            x = tf.constant(np.arange(batch*height*height*3).reshape(batch,height,height,3),tf.float32)
            window=tf.constant(7,tf.int32)
            def roundtrip(value):
                shape=tf.shape(value)
                return repaired(module.window_partition(value,window),window,shape[1],shape[2])
            graph=tf.function(roundtrip,input_signature=[tf.TensorSpec([None,None,None,3],tf.float32)])
            eager=roundtrip(x); traced=graph(x)
            raw=original(module.window_partition(x,window),window,tf.shape(x)[1],tf.shape(x)[2])
            assert np.array_equal(raw.numpy(),x.numpy()),'Original eager inverse differs'
            with tf.GradientTape() as tape:
                tape.watch(x);out=graph(x);loss=tf.reduce_sum(out*out)
            gradient=tape.gradient(loss,x)
            windows=module.window_partition(x,window)
            upstream=tf.constant(((np.arange(batch*height*height*3)%127)-63).reshape(batch,height,height,3),tf.float32)
            with tf.GradientTape()as tape:
                tape.watch(windows)
                loss=tf.reduce_sum(repaired(windows,window,height,height)*upstream)
            window_gradient=tape.gradient(loss,windows)
            errors=[float(tf.reduce_max(tf.abs(eager-x))),float(tf.reduce_max(tf.abs(eager-traced))),
                    max(float(tf.reduce_max(tf.abs(gradient-2*x))),
                        float(tf.reduce_max(tf.abs(window_gradient-module.window_partition(upstream,window)))))]
            assert errors==[0.,0.,0.],errors
            def old_roundtrip(value):
                shape=tf.shape(value)
                return original(module.window_partition(value,window),window,shape[1],shape[2])
            try:
                old_graph=tf.function(old_roundtrip,input_signature=[tf.TensorSpec([None,None,None,3],tf.float32)])
                if not np.array_equal(old_graph(x).numpy(),x.numpy()):original_failed=True
            except (tf.errors.InvalidArgumentError,ValueError):
                original_failed=True
            rows.append({'height':height,'width':height,'batch':batch,'channels':3})
    return {'status':'accepted exact integer window reversal','cases':rows,
            'maximum_reconstruction_abs':0.0,'maximum_eager_graph_abs':0.0,'maximum_gradient_abs':0.0,
            'original_graph_reproduced_failure':bool(original_failed),'training_math_change':False}
