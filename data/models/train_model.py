#!/usr/bin/env python3
"""
Train ML model for malware detection
Uses features extracted from compiler IR
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json
import sys

class MalwareClassifier:
    def __init__(self, features_path="data/features_ml.csv", model_dir="data/models"):
        self.features_path = features_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_features(self):
        """Load feature CSV"""
        try:
            df = pd.read_csv(self.features_path)
            print(f"[✓] Loaded features: {len(df)} samples, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"[✗] Error loading features: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare features and labels for training"""
        # Remove non-feature columns
        exclude_cols = ['sha256', 'label', 'label_binary']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['label_binary'].values
        
        # Handle any NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.feature_names = feature_cols
        
        print(f"[✓] Prepared {len(feature_cols)} features")
        print(f"    Class distribution: Benign={np.sum(y==0)}, Malicious={np.sum(y==1)}")
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("\n[*] Training Random Forest classifier...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        print(f"[✓] Model trained with {self.model.n_estimators} trees")
        
        return X_train_scaled
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n[*] Evaluating model...")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0
        
        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║                   Model Performance                      ║")
        print(f"╠══════════════════════════════════════════════════════════╣")
        print(f"║ Accuracy:  {accuracy*100:6.2f}%                                        ║")
        print(f"║ Precision: {precision*100:6.2f}%                                        ║")
        print(f"║ Recall:    {recall*100:6.2f}%                                        ║")
        print(f"║ F1-Score:  {f1*100:6.2f}%                                        ║")
        print(f"║ AUC-ROC:   {auc*100:6.2f}%                                        ║")
        print(f"╚══════════════════════════════════════════════════════════╝")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Benign  Malicious")
        print(f"Actual Benign    {cm[0][0]:4d}     {cm[0][1]:4d}")
        print(f"     Malicious   {cm[1][0]:4d}     {cm[1][1]:4d}")
        
        # Classification Report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Benign', 'Malicious'],
                                   zero_division=0))
        
        # Feature Importance
        self.show_feature_importance()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def show_feature_importance(self, top_n=15):
        """Show most important features"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n📊 Top {top_n} Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12}")
        print("=" * 55)
        
        for i in range(min(top_n, len(indices))):
            idx = indices[i]
            print(f"{i+1:<6} {self.feature_names[idx]:<35} {importances[idx]:.4f}")
    
    def save_model(self, metrics):
        """Save trained model and metadata"""
        try:
            # Save model
            model_path = self.model_dir / "malware_classifier.pkl"
            joblib.dump(self.model, model_path)
            print(f"\n[✓] Model saved to: {model_path}")
            
            # Save scaler
            scaler_path = self.model_dir / "feature_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            print(f"[✓] Scaler saved to: {scaler_path}")
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'num_features': len(self.feature_names),
                'metrics': metrics,
                'model_type': 'RandomForestClassifier',
                'n_estimators': self.model.n_estimators
            }
            
            metadata_path = self.model_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"[✓] Metadata saved to: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"[✗] Error saving model: {e}")
            return False
    
    def run(self):
        """Run complete training pipeline"""
        print("╔══════════════════════════════════════════════════════════╗")
        print("║              MEEF ML Model Training                      ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()
        
        # Load features
        df = self.load_features()
        if df is None or len(df) == 0:
            print("[✗] No features to train on")
            return False
        
        # Check class balance
        if len(df['label'].unique()) < 2:
            print("[✗] Need both benign and malicious samples for training")
            return False
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split data
        print(f"\n[*] Splitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"[✓] Train set: {len(X_train)} samples")
        print(f"[✓] Test set:  {len(X_test)} samples")
        
        # Train model
        X_train_scaled = self.train_model(X_train, y_train)
        
        # Cross-validation
        print(f"\n[*] Running 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"[✓] CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save model
        success = self.save_model(metrics)
        
        if success:
            print(f"\n[✓] Training complete!")
            print(f"    Model ready for predictions")
            print(f"    Next step: python3 predict.py <sample.asm>")
        
        return success


if __name__ == "__main__":
    classifier = MalwareClassifier()
    success = classifier.run()
    sys.exit(0 if success else 1)
