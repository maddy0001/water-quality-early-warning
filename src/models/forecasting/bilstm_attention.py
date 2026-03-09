"""
============================================================================
Bidirectional LSTM with Attention Mechanism
Spatio-Temporal Water Quality pH Forecasting
============================================================================
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os

# GPU/CPU auto-detection
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info logs

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import FORECASTING_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()

# Import TF globally to be used by the custom layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

@keras.utils.register_keras_serializable()
class SumOverTime(layers.Layer):
    """
    Custom layer to replace Lambda and avoid module pickling errors.
    Sums the attended context vectors across the sequence length (axis 1).
    """
    def __init__(self, **kwargs):
        super(SumOverTime, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        return super(SumOverTime, self).get_config()


def build_bilstm_attention(input_shape: Tuple,
                           lstm_units: List[int] = None,
                           dropout: float = 0.3,
                           attention_units: int = 64,
                           dense_units: List[int] = None,
                           output_size: int = 1) -> 'Model':
    """
    Build Bidirectional LSTM with Attention mechanism
    
    Architecture:
      Input → BiLSTM(128) → Dropout → BiLSTM(64) → Attention → Dense(32) → Dense(1)
    """
    tf.random.set_seed(RANDOM_SEED)

    lstm_units = lstm_units or [128, 64]
    dense_units = dense_units or [32, 16]

    inputs = keras.Input(shape=input_shape, name='input')

    # First Bidirectional LSTM (return sequences for attention)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[0], return_sequences=True,
                    dropout=dropout, recurrent_dropout=dropout * 0.5),
        name='bilstm_1'
    )(inputs)

    # Second Bidirectional LSTM (return sequences for attention)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units[1], return_sequences=True,
                    dropout=dropout, recurrent_dropout=dropout * 0.5),
        name='bilstm_2'
    )(x)

    # Attention Mechanism
    # Query = last hidden state, Key/Value = all hidden states
    attention_scores = layers.Dense(1, activation='tanh', name='attention_scores')(x)
    attention_weights = layers.Softmax(axis=1, name='attention_weights')(attention_scores)
    context = layers.Multiply(name='context')([x, attention_weights])
    
    # REPLACED: Lambda layer removed, using custom subclassed layer instead
    attended = SumOverTime(name='attended_output')(context)

    # Dense layers
    x = layers.Dense(dense_units[0], activation='relu', name='dense_1')(attended)
    x = layers.Dropout(dropout * 0.5, name='dropout_dense')(x)
    x = layers.Dense(dense_units[1], activation='relu', name='dense_2')(x)
    outputs = layers.Dense(output_size, activation='linear', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='BiLSTM_Attention')

    return model


class BiLSTMAttentionModel:
    """
    Bidirectional LSTM with Attention for water quality forecasting
    """

    def __init__(self, config: Dict = None):
        self.config = config or FORECASTING_CONFIG
        self.model = None
        self.history = None
        self.metrics = {}
        self.input_shape = None

    def build(self, input_shape: Tuple, verbose: bool = True):
        """Build and compile the model"""
        self.input_shape = input_shape

        self.model = build_bilstm_attention(
            input_shape=input_shape,
            lstm_units=self.config.get('lstm_units', [128, 64]),
            dropout=self.config.get('lstm_dropout', 0.3),
            attention_units=self.config.get('attention_units', 64),
            dense_units=self.config.get('dense_units', [32, 16]),
            output_size=1
        )

        lr = self.config.get('learning_rate', 0.001)

        try:
            from tensorflow.keras.optimizers import Adam
            self.model.compile(
                optimizer=Adam(learning_rate=lr),
                loss='mse',
                metrics=['mae']
            )
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            raise

        if verbose:
            logger.info("  BiLSTM + Attention model built:")
            logger.info(f"    Input shape: {input_shape}")
            logger.info(f"    LSTM units: {self.config.get('lstm_units', [128, 64])}")
            logger.info(f"    Total parameters: {self.model.count_params():,}")
            
            # Log GPU status
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    logger.info(f"    Device: GPU ({gpus[0].name})")
                else:
                    logger.info(f"    Device: CPU")
            except:
                pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              verbose: bool = True) -> Dict:
        """Train the BiLSTM model"""
        if self.model is None:
            self.build(input_shape=X_train.shape[1:], verbose=verbose)

        if verbose:
            logger.subsection("Training BiLSTM + Attention")
            logger.info(f"  Train samples: {len(X_train)}")
            logger.info(f"  Val samples: {len(X_val)}")
            logger.info(f"  Epochs: {self.config.get('epochs', 100)}")
            logger.info(f"  Batch size: {self.config.get('batch_size', 32)}")

        # Callbacks
        callbacks = self._get_callbacks(verbose)

        start = time.time()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )

        train_time = time.time() - start
        epochs_run = len(self.history.history['loss'])

        self.metrics['train_time'] = train_time
        self.metrics['epochs_run'] = epochs_run
        self.metrics['final_train_loss'] = self.history.history['loss'][-1]
        self.metrics['final_val_loss'] = self.history.history['val_loss'][-1]

        if verbose:
            logger.success(f"  Training complete!")
            logger.info(f"  Epochs run: {epochs_run} / {self.config.get('epochs', 100)}")
            logger.info(f"  Time: {train_time:.1f}s")
            logger.info(f"  Final train loss: {self.metrics['final_train_loss']:.6f}")
            logger.info(f"  Final val loss:   {self.metrics['final_val_loss']:.6f}")

        return self.metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> Dict:
        """Evaluate forecasting performance"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test, verbose=0).flatten()
        y_true = y_test.flatten()

        metrics = self._compute_metrics(y_true, y_pred)
        self.metrics.update(metrics)
        self.metrics['y_pred'] = y_pred
        self.metrics['y_true'] = y_true

        if verbose:
            logger.subsection("BiLSTM + Attention Evaluation")
            logger.info(f"  RMSE:  {metrics['rmse']:.6f}")
            logger.info(f"  MAE:   {metrics['mae']:.6f}")
            logger.info(f"  MAPE:  {metrics['mape']:.2f}%")
            logger.info(f"  R²:    {metrics['r2']:.4f}")

        return metrics

    def _compute_metrics(self, y_true: np.ndarray,
                        y_pred: np.ndarray) -> Dict:
        """Compute forecasting metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        return {'rmse': float(rmse), 'mae': float(mae),
                'mape': float(mape), 'r2': float(r2)}

    def _get_callbacks(self, verbose: bool) -> list:
        """Build training callbacks"""
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 15),
                restore_best_weights=True,
                verbose=1 if verbose else 0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.get('reduce_lr_factor', 0.5),
                patience=self.config.get('reduce_lr_patience', 7),
                min_lr=self.config.get('min_lr', 1e-7),
                verbose=1 if verbose else 0
            )
        ]

        return callbacks

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        return self.model.predict(X, verbose=0).flatten()

    def save(self, filepath: Path = None):
        """Save model"""
        filepath = filepath or MODEL_PATHS['bilstm_attention']
        filepath = str(filepath).replace('.h5', '.keras') # UPGRADE FORMAT
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.success(f"BiLSTM model saved: {Path(filepath).name}")

    def load(self, filepath: Path = None):
        """Load model"""
        filepath = filepath or MODEL_PATHS['bilstm_attention']
        filepath = str(filepath).replace('.h5', '.keras') # UPGRADE FORMAT
        
        self.model = keras.models.load_model(
            filepath, 
            custom_objects={'SumOverTime': SumOverTime} # REGISTER CUSTOM LAYER
        )
        logger.success(f"BiLSTM model loaded: {Path(filepath).name}")
        return self

    def get_training_history(self) -> Dict:
        """Return training history"""
        if self.history is None:
            return {}
        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }


if __name__ == "__main__":
    # Quick sanity test
    X_dummy = np.random.rand(100, 30, 11).astype(np.float32)
    y_dummy = np.random.rand(100, 1).astype(np.float32)

    model = BiLSTMAttentionModel()
    model.build(input_shape=(30, 11), verbose=True)
    print(f"✓ BiLSTM model built. Parameters: {model.model.count_params():,}")