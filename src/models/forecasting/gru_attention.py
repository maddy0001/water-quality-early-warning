"""
============================================================================
GRU with Attention Mechanism
Gated Recurrent Unit for Water Quality pH Forecasting
Lighter and often faster than LSTM with comparable accuracy
============================================================================
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple, List
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import FORECASTING_CONFIG, MODEL_PATHS, RANDOM_SEED
from utils.logger import get_logger
from utils.file_handler import FileHandler

logger = get_logger()


def build_gru_attention(input_shape: Tuple,
                        gru_units: List[int] = None,
                        dropout: float = 0.3,
                        dense_units: List[int] = None,
                        output_size: int = 1) -> 'Model':
    """
    Build GRU with Attention mechanism
    
    Architecture:
      Input → GRU(128, return_seq) → GRU(64, return_seq)
            → Attention → Dense(32) → Dense(1)
    
    GRU vs LSTM:
    - GRU has fewer parameters → faster training
    - Similar accuracy on many time-series tasks
    - Good for comparison in paper
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        tf.random.set_seed(RANDOM_SEED)
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    gru_units = gru_units or [128, 64]
    dense_units = dense_units or [32, 16]

    inputs = keras.Input(shape=input_shape, name='input')

    # GRU layers (return sequences for attention)
    x = layers.GRU(gru_units[0], return_sequences=True,
                   dropout=dropout, recurrent_dropout=dropout * 0.5,
                   name='gru_1')(inputs)

    x = layers.GRU(gru_units[1], return_sequences=True,
                   dropout=dropout, recurrent_dropout=dropout * 0.5,
                   name='gru_2')(x)

    # Attention mechanism (same as BiLSTM)
    attention_scores = layers.Dense(1, activation='tanh', name='attention_scores')(x)
    attention_weights = layers.Softmax(axis=1, name='attention_weights')(attention_scores)
    context = layers.Multiply(name='context')([x, attention_weights])
    attended = layers.Lambda(
        lambda t: __import__('tensorflow').reduce_sum(t, axis=1),
        name='attended_output'
    )(context)

    # Dense layers
    x = layers.Dense(dense_units[0], activation='relu', name='dense_1')(attended)
    x = layers.Dropout(dropout * 0.5, name='dropout_dense')(x)
    x = layers.Dense(dense_units[1], activation='relu', name='dense_2')(x)
    outputs = layers.Dense(output_size, activation='linear', name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='GRU_Attention')
    return model


class GRUAttentionModel:
    """
    GRU with Attention for water quality forecasting
    Designed as a comparison model against BiLSTM
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

        self.model = build_gru_attention(
            input_shape=input_shape,
            gru_units=self.config.get('gru_units', [128, 64]),
            dropout=self.config.get('gru_dropout', 0.3),
            dense_units=self.config.get('dense_units', [32, 16]),
            output_size=1
        )

        from tensorflow.keras.optimizers import Adam
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.get('learning_rate', 0.001)),
            loss='mse',
            metrics=['mae']
        )

        if verbose:
            logger.info("  GRU + Attention model built:")
            logger.info(f"    Input shape: {input_shape}")
            logger.info(f"    GRU units: {self.config.get('gru_units', [128, 64])}")
            logger.info(f"    Total parameters: {self.model.count_params():,}")

            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                device = f"GPU ({gpus[0].name})" if gpus else "CPU"
                logger.info(f"    Device: {device}")
            except:
                pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              verbose: bool = True) -> Dict:
        """Train the GRU model"""
        if self.model is None:
            self.build(input_shape=X_train.shape[1:], verbose=verbose)

        if verbose:
            logger.subsection("Training GRU + Attention")
            logger.info(f"  Train samples: {len(X_train)}")
            logger.info(f"  Val samples:   {len(X_val)}")

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

        self.metrics.update({
            'train_time': train_time,
            'epochs_run': epochs_run,
            'final_train_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1]
        })

        if verbose:
            logger.success(f"  Training complete in {train_time:.1f}s")
            logger.info(f"  Epochs: {epochs_run} | Val loss: {self.metrics['final_val_loss']:.6f}")

        return self.metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 verbose: bool = True) -> Dict:
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained.")

        y_pred = self.model.predict(X_test, verbose=0).flatten()
        y_true = y_test.flatten()

        metrics = self._compute_metrics(y_true, y_pred)
        self.metrics.update(metrics)
        self.metrics['y_pred'] = y_pred
        self.metrics['y_true'] = y_true

        if verbose:
            logger.subsection("GRU + Attention Evaluation")
            logger.info(f"  RMSE:  {metrics['rmse']:.6f}")
            logger.info(f"  MAE:   {metrics['mae']:.6f}")
            logger.info(f"  MAPE:  {metrics['mape']:.2f}%")
            logger.info(f"  R²:    {metrics['r2']:.4f}")

        return metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute regression metrics"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        return {'rmse': float(rmse), 'mae': float(mae),
                'mape': float(mape), 'r2': float(r2)}

    def _get_callbacks(self, verbose: bool) -> list:
        """Build training callbacks"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        return [
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def save(self, filepath: Path = None):
        filepath = filepath or MODEL_PATHS['gru_attention']
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(filepath))
        logger.success(f"GRU model saved: {Path(filepath).name}")

    def load(self, filepath: Path = None):
        from tensorflow import keras
        filepath = filepath or MODEL_PATHS['gru_attention']
        self.model = keras.models.load_model(str(filepath))
        logger.success(f"GRU model loaded: {Path(filepath).name}")
        return self

    def get_training_history(self) -> Dict:
        if self.history is None:
            return {}
        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'mae': self.history.history.get('mae', []),
            'val_mae': self.history.history.get('val_mae', [])
        }


if __name__ == "__main__":
    import numpy as np
    X_dummy = np.random.rand(100, 30, 11).astype(np.float32)
    y_dummy = np.random.rand(100, 1).astype(np.float32)

    model = GRUAttentionModel()
    model.build(input_shape=(30, 11), verbose=True)
    print(f"✓ GRU model built. Parameters: {model.model.count_params():,}")
