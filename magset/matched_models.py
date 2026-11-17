"""Three-crop CNN/Swin readers for the separately registered matched experiment.

Importing this module needs no TensorFlow. Models accept raw RGB pixels shaped
``(batch, 3, 224, 224, 3)`` and return binary-two and malignant-four probabilities.
Loss masking, splitting, augmentation, optimization and case pooling belong to
the experiment driver. Only weight-only H5 checkpoint replay is required here.
"""
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re


CROPS = 3
IMAGE_SIZE = 224
CNN_IMAGENET_SHA256 = 'fd2529efca107725477325b0a3de53521c37873ba2910a2600a6b26209d95285'
SWIN_REVISION = 'd00d478bfaf1f417d34a1186673ad4b4be264c51'
SWIN_FILES = {
    'config.json': 'b99b2d1d80b05ff592979047b4a34fb459dd53b82c8036bae1c2a9d1ea2c5ed3',
    'preprocessor_config.json': 'cf47f0a7c8ff723066a62ca5cf536ba0f530ed1e797a0eab270f1304e2f880db',
    'tf_model.h5': '613ffde402c38d5626003db04e3578c64a32ddee8c1272333524315faaa4c2c7',
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _checked_file(path, expected_sha256):
    if path is None:
        raise ValueError('A local pretrained weight path is required')
    path = Path(path)
    if not path.is_file():
        raise ValueError('Required local pretrained file is missing')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Local file differs from pinned original pretrained weights')
    return str(path.resolve())


@lru_cache(maxsize=1)
def _runtime():
    if os.environ.setdefault('TF_USE_LEGACY_KERAS', '1') != '1':
        raise RuntimeError('Start a fresh legacy-Keras process before building models')
    import tensorflow as tf
    import tf_keras as keras
    if tf.__version__ != '2.15.1' or keras.__version__ != '2.15.1':
        raise RuntimeError('Matched models require TensorFlow 2.15.1 and tf_keras 2.15.1')
    if not tf.keras.__name__.startswith('tf_keras'):
        raise RuntimeError('Mixed Keras namespaces are not permitted')
    return tf, keras


def normalize_cnn_inputs(pixels):
    """Keep raw 0..255 pixels for EfficientNetV2's included preprocessing."""
    tf, _ = _runtime()
    return tf.cast(pixels, tf.float32)


def normalize_swin_inputs(pixels):
    """Normalize raw NHWC RGB values; conversion to NCHW occurs in the reader."""
    tf, _ = _runtime()
    values = tf.cast(pixels, tf.float32) / 255.0
    mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    return (values - mean) / std


def average_crop_descriptors(descriptors, batch_size, descriptor_width):
    """Keep each field's three reader descriptors together when averaging."""
    tf, _ = _runtime()
    fields = tf.reshape(descriptors, (batch_size, CROPS, descriptor_width))
    return tf.reduce_mean(fields, axis=1)


def _verify_swin_loading(backbone, info, tf):
    import numpy as np
    allowed = {'swin/encoder/layers.%d/blocks.%d/attention/self/relative_position_index:0' % (stage, block)
               for stage, depth in enumerate((2, 2, 6, 2)) for block in range(depth)}
    raw_missing=info.get('missing_keys')
    if not isinstance(raw_missing,list)or len(raw_missing)!=len(set(raw_missing))or not isinstance(info.get('mismatched_keys'),list):
        raise ValueError('Swin pretrained diagnostics incomplete or duplicated')
    missing = set(raw_missing)
    if missing not in (set(), allowed) or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Swin pretrained loading has unresolved missing/mismatched weights')
    unexpected = info.get('unexpected_keys')
    if not isinstance(unexpected,list)or len(unexpected)!=len(set(unexpected))or any(re.search(r'(^|/)classifier(?:/|$)',name)is None for name in unexpected):
        raise ValueError('Swin pretrained loading has unexpected non-classifier weights')
    coords = np.stack(np.meshgrid(np.arange(7), np.arange(7), indexing='ij')).reshape(2, -1)
    delta = coords[:, :, None] - coords[:, None, :]
    expected = ((delta[0] + 6) * 13 + delta[1] + 6).astype(np.int32)
    found = set()
    for weight in backbone.weights:
        match = re.search(r'(swin/encoder/layers\.\d+/blocks\.\d+/attention/self/relative_position_index:0)$', weight.name)
        if match:
            name = match.group(1)
            if name in found or weight.trainable or weight.dtype != tf.int32 or not np.array_equal(weight.numpy(), expected):
                raise ValueError('Swin deterministic position-index buffer differs')
            found.add(name)
    if found != allowed:
        raise ValueError('Swin deterministic position-index coverage differs')
    return {'missing_deterministic_buffers': len(missing), 'verified_position_buffers': len(found),
            'unresolved_missing_weights': 0, 'mismatched_weights': 0,
            'unused_classifier_weights': len(unexpected)}


@lru_cache(maxsize=1)
def _model_classes():
    tf, keras = _runtime()
    layers = keras.layers

    class TransformerEncoderBlock(layers.Layer):
        def __init__(self, d_model=512, num_heads=8, dff=1024, dropout_rate=0.2, **kwargs):
            super().__init__(**kwargs)
            self.d_model, self.num_heads = int(d_model), int(num_heads)
            self.dff, self.dropout_rate = int(dff), float(dropout_rate)
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads,
                                                       dropout=dropout_rate)
            self.ffn = keras.Sequential([layers.Dense(dff, activation='gelu'),
                                         layers.Dropout(dropout_rate), layers.Dense(d_model)])
            self.norm1 = layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(dropout_rate)
            self.dropout2 = layers.Dropout(dropout_rate)

        def call(self, inputs, training=False):
            attended = self.attention(inputs, inputs, training=training)
            values = self.norm1(inputs + self.dropout1(attended, training=training))
            feed_forward = self.ffn(values, training=training)
            return self.norm2(values + self.dropout2(feed_forward, training=training))

        def get_config(self):
            return dict(super().get_config(), d_model=self.d_model, num_heads=self.num_heads,
                        dff=self.dff, dropout_rate=self.dropout_rate)

    class CLSTokenAndPositionalEncoding(layers.Layer):
        def __init__(self, num_patches=49, d_model=512, **kwargs):
            super().__init__(**kwargs)
            self.num_patches, self.d_model = int(num_patches), int(d_model)
            self.cls_token = self.add_weight(name='cls_token', shape=(1, 1, d_model),
                                             initializer='zeros', trainable=True)
            self.pos_emb = self.add_weight(name='positional_embedding', shape=(1, num_patches + 1, d_model),
                                           initializer='random_normal', trainable=True)

        def call(self, inputs):
            cls = tf.broadcast_to(self.cls_token, (tf.shape(inputs)[0], 1, self.d_model))
            return tf.concat((cls, inputs), axis=1) + self.pos_emb

        def get_config(self):
            return dict(super().get_config(), num_patches=self.num_patches, d_model=self.d_model)

    class DualHeads(layers.Layer):
        def __init__(self, stop_binary_gradient=False, l2_reg=1e-4, **kwargs):
            super().__init__(**kwargs)
            self.stop_binary_gradient, self.l2_reg = bool(stop_binary_gradient), float(l2_reg)
            l2 = keras.regularizers.l2(l2_reg)
            self.s1_dense = layers.Dense(128, activation='gelu', kernel_regularizer=l2, name='s1_dense')
            self.s1_drop = layers.Dropout(0.4, name='s1_drop')
            self.binary = layers.Dense(2, activation='softmax', name='binary_output')
            self.s2_dense1 = layers.Dense(256, activation='gelu', kernel_regularizer=l2, name='s2_dense1')
            self.s2_drop1 = layers.Dropout(0.4, name='s2_drop1')
            self.s2_dense2 = layers.Dense(128, activation='gelu', kernel_regularizer=l2, name='s2_dense2')
            self.s2_drop2 = layers.Dropout(0.3, name='s2_drop2')
            self.subtype = layers.Dense(4, activation='softmax', name='subtype_output')

        def call(self, descriptor, training=False):
            binary = self.binary(self.s1_drop(self.s1_dense(descriptor), training=training))
            condition = tf.stop_gradient(binary) if self.stop_binary_gradient else binary
            values = tf.concat((descriptor, condition), axis=-1)
            values = self.s2_drop1(self.s2_dense1(values), training=training)
            values = self.s2_drop2(self.s2_dense2(values), training=training)
            return {'binary_output': binary, 'subtype_output': self.subtype(values)}

        def get_config(self):
            return dict(super().get_config(), stop_binary_gradient=self.stop_binary_gradient, l2_reg=self.l2_reg)

    def crop_batch(inputs):
        inputs = tf.ensure_shape(inputs, (None, CROPS, IMAGE_SIZE, IMAGE_SIZE, 3))
        return tf.reshape(inputs, (-1, IMAGE_SIZE, IMAGE_SIZE, 3))

    class MatchedCNN(keras.Model):
        def __init__(self, cnn_weights, **kwargs):
            kwargs.setdefault('name', 'matched_cnn')
            super().__init__(**kwargs)
            self.kind, self.cnn_weights = 'cnn', str(cnn_weights)
            self.backbone = keras.applications.EfficientNetV2B0(
                input_shape=(224, 224, 3), include_top=False, weights=self.cnn_weights,
                include_preprocessing=True)
            for index, layer in enumerate(self.backbone.layers):
                layer.trainable = index >= len(self.backbone.layers) - 60
            patches = layers.Reshape((49, 1280))(self.backbone.output)
            patches = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-4), name='patch_projection')(patches)
            patches = layers.LayerNormalization(epsilon=1e-6, name='patch_norm')(patches)
            patches = layers.Dropout(0.2, name='patch_dropout')(patches)
            patches = CLSTokenAndPositionalEncoding(name='cls_pos_encoding')(patches)
            for index in range(4):
                patches = TransformerEncoderBlock(name='transformer_block_%d' % index)(patches)
            descriptor = layers.LayerNormalization(epsilon=1e-6, name='cls_norm')(patches[:, 0, :])
            self.crop_reader = keras.Model(self.backbone.inputs, descriptor, name='cnn_crop_reader')
            self.heads = DualHeads(stop_binary_gradient=False, name='dual_heads')

        def call(self, inputs, training=False):
            pixels = normalize_cnn_inputs(crop_batch(inputs))
            # Propagate training: trainable final-backbone BN follows its original semantics.
            descriptor = self.crop_reader(pixels, training=training)
            descriptor = average_crop_descriptors(descriptor, tf.shape(inputs)[0], 512)
            return self.heads(descriptor, training=training)

        def get_config(self):
            return dict(super().get_config(), cnn_weights=self.cnn_weights)

    class MatchedSwin(keras.Model):
        def __init__(self, swin_snapshot, **kwargs):
            kwargs.setdefault('name', 'matched_swin')
            super().__init__(**kwargs)
            import transformers
            from transformers import TFSwinModel
            from magset.swin_compat import install_integer_window_reverse
            if transformers.__version__ != '4.46.3' or not issubclass(TFSwinModel, keras.Model):
                raise RuntimeError('Swin requires Transformers 4.46.3 with tf_keras models')
            install_integer_window_reverse()  # Install before from_pretrained creates/traces the backbone.
            self.kind, self.swin_snapshot = 'swin', str(swin_snapshot)
            self.backbone, loading = TFSwinModel.from_pretrained(
                self.swin_snapshot, local_files_only=True, use_safetensors=False, output_loading_info=True)
            self.pretrained_loading = _verify_swin_loading(self.backbone, loading, tf)
            self.backbone.swin.embeddings.trainable = False
            for index, stage in enumerate(self.backbone.swin.encoder.layers):
                stage.trainable = index >= 2
            self.norm = layers.LayerNormalization(epsilon=1e-6, name='field_norm')
            self.heads = DualHeads(stop_binary_gradient=True, name='dual_heads')

        def call(self, inputs, training=False):
            pixels = normalize_swin_inputs(crop_batch(inputs))
            pixels = tf.transpose(pixels, (0, 3, 1, 2))
            descriptor = self.backbone(pixel_values=pixels, training=training).pooler_output
            descriptor = average_crop_descriptors(self.norm(descriptor), tf.shape(inputs)[0], 768)
            return self.heads(descriptor, training=training)

        def get_config(self):
            return dict(super().get_config(), swin_snapshot=self.swin_snapshot)

    return MatchedCNN, MatchedSwin


def build_model(kind, cnn_weights=None, swin_snapshot=None):
    """Build one model from pinned original pretraining, never a fitted checkpoint.

    The initial inference call creates all head weights for immediate H5 loading
    and complete parameter receipts. Training and RNG seeding are caller-owned.
    """
    if kind not in ('cnn', 'swin'):
        raise ValueError('Model kind must be cnn or swin')
    if kind == 'cnn':
        source = _checked_file(cnn_weights, CNN_IMAGENET_SHA256)
    else:
        if swin_snapshot is None:
            raise ValueError('A local pinned Swin snapshot directory is required')
        source = str(Path(swin_snapshot).resolve())
        for name, digest in SWIN_FILES.items():
            _checked_file(Path(source) / name, digest)
    cnn_class, swin_class = _model_classes()
    model = cnn_class(source) if kind == 'cnn' else swin_class(source)
    tf, _ = _runtime()
    model(tf.zeros((1, CROPS, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32), training=False)
    return model


def architecture_receipt(model):
    """Return actual trainability and parameter counts without private file paths."""
    if getattr(model, 'kind', None) not in ('cnn', 'swin') or not model.built:
        raise ValueError('An initialized matched model is required')
    count = lambda weights: sum(math.prod(int(size) for size in weight.shape) for weight in weights)
    receipt = {'kind': model.kind, 'input_shape': [None, 3, 224, 224, 3],
               'input_scale': 'raw RGB 0..255', 'crop_pool': 'mean per-crop normalized descriptors before heads',
               'outputs': {'binary_output': 2, 'subtype_output': 4},
               'subtype_binary_stop_gradient': model.heads.stop_binary_gradient,
               'parameters': {'all': count(model.weights), 'trainable': count(model.trainable_weights),
                              'nontrainable': count(model.non_trainable_weights)}}
    if model.kind == 'cnn':
        flags = [{'name': layer.name, 'trainable': bool(layer.trainable)} for layer in model.backbone.layers]
        receipt.update(backbone='EfficientNetV2B0', pretrained_sha256=CNN_IMAGENET_SHA256,
                       backbone_layer_flags=flags, trainable_last_backbone_layers=sum(row['trainable'] for row in flags),
                       transformer_blocks=4, descriptor_dimensions=512, descriptor_point='per-crop cls_norm',
                       preprocessing='built-in EfficientNetV2 preprocessing; no external division',
                       backbone_training='propagated; frozen layers retain Keras trainability semantics')
    else:
        receipt.update(backbone='Swin-Tiny', pretrained_revision=SWIN_REVISION,
                       pretrained_files=dict(SWIN_FILES), descriptor_dimensions=768,
                       descriptor_point='per-crop added LayerNorm of pooler_output',
                       embeddings_trainable=bool(model.backbone.swin.embeddings.trainable),
                       stage_trainable=[bool(stage.trainable) for stage in model.backbone.swin.encoder.layers],
                       preprocessing='divide255 then ImageNet mean/std, NHWC to NCHW',
                       pretrained_loading=dict(model.pretrained_loading), integer_window_reverse=True)
    return receipt
