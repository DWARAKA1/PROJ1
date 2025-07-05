import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.exceptions import NotFittedError
import os

warnings.filterwarnings('ignore')

class SalesPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = ['quantity', 'unit_price']
        
    def preprocess_data(self, df):
        try:
            df = df.copy()
            # Validate required columns
            if not all(col in df.columns for col in self.feature_names):
                raise ValueError(f"Missing required columns: {self.feature_names}")
            
            # Remove invalid data
            df = df[(df['quantity'] > 0) & (df['unit_price'] > 0)]
            if len(df) == 0:
                raise ValueError("No valid data after filtering")
            
            # Feature engineering
            df['total_sales'] = df['quantity'] * df['unit_price']
            
            X = df[self.feature_names]
            y = df['total_sales']
            
            return X, y
        except Exception as e:
            raise ValueError(f"Data preprocessing failed: {str(e)}")
    
    def train(self, df):
        try:
            if df is None or len(df) < 3:
                raise ValueError("Insufficient data for training (minimum 3 samples required)")
            
            X, y = self.preprocess_data(df)
            
            # Use full dataset if too small, otherwise split
            if len(df) < 10:
                X_train, X_test = X, X
                y_train, y_test = y, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.lr_model.fit(X_train_scaled, y_train)
            self.rf_model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Evaluate
            lr_pred = self.lr_model.predict(X_test_scaled)
            rf_pred = self.rf_model.predict(X_test_scaled)
            
            # Handle edge cases for metrics
            lr_r2 = r2_score(y_test, lr_pred) if len(y_test) > 1 else 0.0
            rf_r2 = r2_score(y_test, rf_pred) if len(y_test) > 1 else 0.0
            
            results = {
                'linear_r2': lr_r2,
                'linear_rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
                'rf_r2': rf_r2,
                'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
            }
            
            # Feature importance analysis
            lr_coef = abs(self.lr_model.coef_)
            rf_importance = self.rf_model.feature_importances_
            
            results['feature_importance'] = {
                'linear_coef': dict(zip(self.feature_names, lr_coef)),
                'rf_importance': dict(zip(self.feature_names, rf_importance)),
                'most_influential': self.feature_names[np.argmax(lr_coef)]
            }
            
            return results
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def predict(self, quantity, unit_price, model_type='rf'):
        try:
            if not self.is_trained:
                raise NotFittedError("Model must be trained before making predictions")
            
            # Input validation
            if quantity <= 0 or unit_price <= 0:
                raise ValueError("Quantity and unit price must be positive")
            
            X = pd.DataFrame([[quantity, unit_price]], columns=self.feature_names)
            X_scaled = self.scaler.transform(X)
            
            if model_type == 'lr':
                prediction = self.lr_model.predict(X_scaled)[0]
            else:
                prediction = self.rf_model.predict(X_scaled)[0]
            
            # Ensure non-negative prediction
            return max(0, prediction)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def export_model(self, filepath='sales_model.pkl'):
        try:
            if not self.is_trained:
                raise NotFittedError("Model must be trained before export")
            
            model_data = {
                'scaler': self.scaler,
                'lr_model': self.lr_model,
                'rf_model': self.rf_model,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
            return filepath
        except Exception as e:
            raise RuntimeError(f"Model export failed: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath='sales_model.pkl'):
        try:
            model_data = joblib.load(filepath)
            predictor = cls()
            predictor.scaler = model_data['scaler']
            predictor.lr_model = model_data['lr_model']
            predictor.rf_model = model_data['rf_model']
            predictor.feature_names = model_data.get('feature_names', ['quantity', 'unit_price'])
            predictor.is_trained = True
            return predictor
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

# Usage example
if __name__ == "__main__":
    try:
        # Sample data structure
        sample_data = pd.DataFrame({
            'quantity': [10, 25, 5, 30, 15, 40, 8, 22, 35, 12, 18, 45, 7, 28, 33],
            'unit_price': [100, 50, 200, 75, 120, 60, 180, 90, 45, 150, 110, 55, 170, 85, 65]
        })
        
        predictor = SalesPredictor()
        results = predictor.train(sample_data)
        print("Model Performance:", results)
        print(f"Most influential feature: {results['feature_importance']['most_influential']}")
        
        # Export model
        model_path = predictor.export_model()
        print(f"Model exported to: {model_path}")
        
        # Make prediction
        prediction = predictor.predict(20, 80)
        print(f"Predicted sales: ${prediction:.2f}")
    except Exception as e:
        print(f"Error: {e}")