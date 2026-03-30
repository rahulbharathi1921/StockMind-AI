import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os


class MultiModelPredictionEngine:
    """Advanced multi-model ensemble for stock prediction"""

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize models
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()

        # Model weights for ensemble
        self.model_weights = {
            'rf': 0.5,
            'xgb': 0.5
        }

        # Model performance tracking
        self.model_performance = {}

    def train_models(self, X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2, retrain: bool = False):
        """
        Train all models

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion for test set
            retrain: Force retrain even if models exist
        """
        if X.empty or len(X) < 100:
            print("Insufficient data for training")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self._train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test, retrain)

        # Train XGBoost
        self._train_xgboost(X_train_scaled, y_train, X_test_scaled, y_test, retrain)

        # Update model weights based on performance
        self._update_model_weights()

    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray, retrain: bool):
        """Train Random Forest model"""
        rf_path = os.path.join(self.model_dir, 'rf_model.pkl')

        if not retrain and os.path.exists(rf_path):
            try:
                self.rf_model = joblib.load(rf_path)
                print("Loaded existing Random Forest model")
            except:
                retrain = True

        if retrain or self.rf_model is None:
            print("Training Random Forest...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.rf_model.fit(X_train, y_train)
            joblib.dump(self.rf_model, rf_path)
            print("Random Forest trained and saved")

        # Evaluate
        if len(X_test) > 0:
            y_pred = self.rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            self.model_performance['rf'] = {
                'accuracy': accuracy,
                'precision': precision
            }
            print(f"Random Forest - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")

    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray, retrain: bool):
        """Train XGBoost model"""
        xgb_path = os.path.join(self.model_dir, 'xgb_model.pkl')

        if not retrain and os.path.exists(xgb_path):
            try:
                self.xgb_model = joblib.load(xgb_path)
                print("Loaded existing XGBoost model")
            except:
                retrain = True

        if retrain or self.xgb_model is None:
            print("Training XGBoost...")
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            self.xgb_model.fit(X_train, y_train)
            joblib.dump(self.xgb_model, xgb_path)
            print("XGBoost trained and saved")

        # Evaluate
        if len(X_test) > 0:
            y_pred = self.xgb_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            self.model_performance['xgb'] = {
                'accuracy': accuracy,
                'precision': precision
            }
            print(f"XGBoost - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}")

    def _update_model_weights(self):
        """Update model weights based on recent performance"""
        total_perf = 0
        weights = {}

        for model_name, perf in self.model_performance.items():
            # Use accuracy as performance metric
            model_perf = perf.get('accuracy', 0.5)
            weights[model_name] = model_perf
            total_perf += model_perf

        if total_perf > 0:
            # Normalize weights
            for model_name in weights:
                self.model_weights[model_name] = weights[model_name] / total_perf

        print(f"Model weights updated: {self.model_weights}")

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Make ensemble prediction

        Args:
            X: Feature matrix for prediction

        Returns:
            Dictionary with predictions and confidence
        """
        if X.empty or len(X) == 0:
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'bullish_prob': 0.5,
                'model_predictions': {},
                'model_probabilities': {},
                'model_weights': self.model_weights.copy()
            }

        try:
            # Scale features
            X_scaled = self.scaler.transform(X)

            predictions = {}
            probabilities = {}

            # Random Forest prediction
            if self.rf_model is not None:
                try:
                    rf_pred = self.rf_model.predict(X_scaled[-1:])[0]
                    rf_prob = self.rf_model.predict_proba(X_scaled[-1:])[0]
                    predictions['rf'] = rf_pred
                    probabilities['rf'] = rf_prob[1] if len(rf_prob) > 1 else 0.5
                except Exception as e:
                    print(f"RF prediction error: {e}")
                    predictions['rf'] = 1
                    probabilities['rf'] = 0.5

            # XGBoost prediction
            if self.xgb_model is not None:
                try:
                    xgb_pred = self.xgb_model.predict(X_scaled[-1:])[0]
                    xgb_prob = self.xgb_model.predict_proba(X_scaled[-1:])[0]
                    predictions['xgb'] = xgb_pred
                    probabilities['xgb'] = xgb_prob[1] if len(xgb_prob) > 1 else 0.5
                except Exception as e:
                    print(f"XGB prediction error: {e}")
                    predictions['xgb'] = 1
                    probabilities['xgb'] = 0.5

            # Ensemble prediction
            ensemble_prob = 0
            total_weight = 0

            for model_name, prob in probabilities.items():
                weight = self.model_weights.get(model_name, 0.5)
                ensemble_prob += prob * weight
                total_weight += weight

            if total_weight > 0:
                ensemble_prob /= total_weight
            else:
                ensemble_prob = 0.5

            # Determine direction and confidence
            direction = 'bullish' if ensemble_prob > 0.55 else 'bearish' if ensemble_prob < 0.45 else 'neutral'
            confidence = abs(ensemble_prob - 0.5) * 2  # Convert to 0-1 range

            return {
                'direction': direction,
                'confidence': min(confidence, 0.99),  # Cap at 0.99
                'bullish_prob': ensemble_prob,
                'model_predictions': predictions,
                'model_probabilities': probabilities,
                'model_weights': self.model_weights.copy()
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'bullish_prob': 0.5,
                'model_predictions': {},
                'model_probabilities': {},
                'model_weights': self.model_weights.copy()
            }

    def get_model_insights(self) -> Dict:
        """Get insights about model performance and feature importance"""
        insights = {
            'model_performance': self.model_performance,
            'model_weights': self.model_weights,
            'feature_importance': {}
        }

        # Get feature importance from Random Forest
        if self.rf_model is not None:
            try:
                importances = self.rf_model.feature_importances_
                insights['feature_importance']['rf'] = importances.tolist()
            except Exception as e:
                print(f"Error getting RF feature importance: {e}")

        # Get feature importance from XGBoost
        if self.xgb_model is not None:
            try:
                importances = self.xgb_model.feature_importances_
                insights['feature_importance']['xgb'] = importances.tolist()
            except Exception as e:
                print(f"Error getting XGB feature importance: {e}")

        return insights
