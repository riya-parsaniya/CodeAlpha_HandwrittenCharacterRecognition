"""
models.py
---------
CNN model architectures for Handwritten Character Recognition:
  1. Simple CNN    — lightweight baseline
  2. Deep CNN      — deeper with more filters
  3. ResNet-style  — skip connections (best accuracy)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Dense, Dropout, BatchNormalization,
    Flatten, Activation, Add, Input,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE


# ── 1. Simple CNN (Baseline) ──────────────────────────────────────────────────

def build_simple_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Lightweight 3-block CNN — fast to train, good baseline.
    Expected accuracy: ~88–91%
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ], name='Simple_CNN')

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── 2. Deep CNN ───────────────────────────────────────────────────────────────

def build_deep_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Deeper CNN with double convolution blocks (VGG-style).
    Expected accuracy: ~91–94%
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape,
               kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.30),

        # Classifier
        Flatten(),
        Dense(512, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(256),
        Activation('relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax'),
    ], name='Deep_CNN')

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── 3. ResNet-style CNN (Best) ────────────────────────────────────────────────

def residual_block(x, filters, stride=1, downsample=False):
    """Single residual block with optional downsampling shortcut."""
    shortcut = x

    # Main path
    x = Conv2D(filters, (3, 3), strides=stride, padding='same',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), padding='same',
               kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    # Shortcut path (match dimensions if needed)
    if downsample:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same',
                          kernel_regularizer=l2(1e-4))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def build_resnet_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Custom ResNet-style CNN with skip connections.
    Expected accuracy: ~93–96%
    """
    inputs = Input(shape=input_shape)

    # Stem
    x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2, downsample=True)
    x = Dropout(0.2)(x)

    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2, downsample=True)
    x = Dropout(0.3)(x)

    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2, downsample=True)
    x = Dropout(0.3)(x)

    # Global average pooling instead of Flatten (fewer params, better generalisation)
    x = GlobalAveragePooling2D()(x)

    # Classifier
    x = Dense(256, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='ResNet_CNN')
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Model summary helper ──────────────────────────────────────────────────────

def get_all_models():
    """Return dict of all three model builders."""
    return {
        'Simple_CNN': build_simple_cnn,
        'Deep_CNN'  : build_deep_cnn,
        'ResNet_CNN': build_resnet_cnn,
    }
