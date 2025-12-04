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
    def __init__(self, features_path=None, model_dir=None):
        # Default to repository-relative paths based on this file's location
        repo_data_dir = Path(__file__).parent.parent  # data/
        if features_path is None:
            features_path = repo_data_dir / 'features_ml.csv'
        if model_dir is None:
            model_dir = Path(__file__).parent  # data/models/

        self.features_path = Path(features_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.feature_names = None
        
    def load_features(self):
        """Load feature CSV"""
        try:
            if not self.features_path.exists():
                raise FileNotFoundError(f"Features file not found: {self.features_path}")

            df = pd.read_csv(str(self.features_path))
            print(f"[✓] Loaded features: {len(df)} samples, {len(df.columns)} columns")
            print(f"    (features path: {self.features_path})")
            return df
        except Exception as e:
            print(f"[✗] Error loading features: {e}")
            print(f"    Expected features CSV at: {self.features_path}")
            return None
    
    def prepare_data(self, df):

        import json
        import joblib
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline

        class MalwareClassifier:
            def __init__(self, features_path=None, model_dir=None):
                if model_dir is None:
                    model_dir = Path(__file__).parent
                self.features_path = Path(features_path) if features_path else Path(__file__).parents[2] / "data" / "features" / "features.csv"
                self.model_dir = Path(model_dir)
                self.model_dir.mkdir(parents=True, exist_ok=True)
        
                self.model = None
                self.scaler = None
                self.feature_names = None
        
            def load_features(self):
                """Load feature CSV"""
                try:
                    if not self.features_path.exists():
                        print(f"[✗] Features file not found: {self.features_path}")
                        return None
                    df = pd.read_csv(self.features_path)
                    print(f"[✓] Loaded features: {self.features_path} ({len(df)} rows)")
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
                """Train Pipeline: StandardScaler -> LDA -> RandomForest"""
                print("\n[*] Training Pipeline: StandardScaler -> LDA -> RandomForest...")
        
                # Use LDA only when there are at least 2 classes (we assume binary)
                lda = LinearDiscriminantAnalysis(n_components=1)
                scaler = StandardScaler()
                rf = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )
                pipeline = Pipeline([
                    ('scaler', scaler),
                    ('lda', lda),
                    ('rf', rf)
                ])
        
                pipeline.fit(X_train, y_train)
                self.model = pipeline
                # keep scaler externally if you like, for debug — but pipeline has it
                self.scaler = None
        
                print(f"[✓] Pipeline trained: RF trees={pipeline.named_steps['rf'].n_estimators}")
                return pipeline
    
            def evaluate_model(self, X_test, y_test):
                """Evaluate model performance"""
                print("\n[*] Evaluating model...")
        
                # Predictions
                y_pred = self.model.predict(X_test)
                if hasattr(self.model, "predict_proba"):
                    y_pred_proba = self.model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = np.zeros_like(y_pred, dtype=float)
        
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
        
                # Feature Importance (RF steps after LDA will have transformed features; still show RF importances)
                try:
                    rp = self.model.named_steps['rf']
                    importances = rp.feature_importances_
                    # LDA reduces features, but importances are on transformed space; show if present.
                    print("\nTop RF importances (on LDA feature dims):")
                    for idx, val in enumerate(np.sort(importances)[::-1][:10]):
                        print(f"    dim_{idx}: {val:.4f}")
                except Exception:
                    pass
        
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                }
    
            def show_feature_importance(self, top_n=15):
                """If RF expects raw features we can show importances; if LDA transforms then importance is less meaningful."""
                try:
                    rf = self.model.named_steps['rf']
                    importances = rf.feature_importances_
                    indices = np.argsort(importances)[::-1]
            
                    print(f"\n📊 Top {top_n} Most Important Features (post-LDA dims):")
                    print(f"{'Rank':<6} {'Feature':<35} {'Importance':<12}")
                    print("=" * 55)
                    for i in range(min(top_n, len(indices))):
                        idx = indices[i]
                        print(f"{i+1:<6} dim_{idx:<35} {importances[idx]:.4f}")
                except Exception:
                    pass
    
            def save_model(self, metrics):
                """Save trained model and metadata"""
                try:
                    # Save model (pipeline)
                    model_path = self.model_dir / "malware_classifier.pkl"
                    joblib.dump(self.model, model_path)
                    print(f"\n[✓] Model saved to: {model_path}")
            
                    # Save metadata
                    metadata = {
                        'feature_names': self.feature_names,
                        'num_features': len(self.feature_names) if self.feature_names else 0,
                        'metrics': metrics,
                        'model_type': 'Pipeline(StandardScaler->LDA->RandomForest)',
                        'n_estimators': self.model.named_steps['rf'].n_estimators if hasattr(self.model.named_steps['rf'], 'n_estimators') else None
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
                    print("[✗] Not enough classes to train on")
                    return False
        
                # Prepare data
                X, y = self.prepare_data(df)
        
                # Split data (stratified to keep class ratios)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
                # Train
                pipeline = self.train_model(X_train, y_train)
        
                # Evaluate
                metrics = self.evaluate_model(X_test, y_test)
        
                # Save
                saved_ok = self.save_model(metrics)
        
                return saved_ok

        if __name__ == "__main__":
            import sys
            c = MalwareClassifier()
            ok = c.run()
            sys.exit(0 if ok else 1)
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
