"""
Automated Loan Approval Prediction Analysis
============================================
This script automates the entire machine learning pipeline for loan approval prediction.

Usage:
    python run_analysis.py

This script will:
1. Load and explore the dataset
2. Preprocess the data
3. Train multiple ML models
4. Evaluate and compare models
5. Save the best model
6. Generate visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

class LoanApprovalAnalysis:
    """Complete pipeline for loan approval prediction."""

    def __init__(self, data_path='loan_approval_dataset.csv'):
        """Initialize the analysis pipeline."""
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.scaler = None
        self.label_encoders = {}
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.target_col = None

        print("="*70)
        print("LOAN APPROVAL PREDICTION ANALYSIS")
        print("="*70 + "\n")

    def load_data(self):
        """Load the dataset."""
        print("Step 1: Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")

        # Display basic info
        print("Dataset Preview:")
        print(self.df.head())
        print(f"\nDataset Info:")
        print(self.df.info())
        print()

    def identify_target(self):
        """Identify the target column."""
        if 'loan_status' in self.df.columns:
            self.target_col = 'loan_status'
        elif 'Loan_Status' in self.df.columns:
            self.target_col = 'Loan_Status'
        else:
            # Find likely target column
            target_candidates = [col for col in self.df.columns
                               if 'status' in col.lower() or 'approved' in col.lower()]
            self.target_col = target_candidates[0] if target_candidates else self.df.columns[-1]

        print(f"Target column identified: {self.target_col}")
        print(f"Target distribution:\n{self.df[self.target_col].value_counts()}\n")

    def preprocess_data(self):
        """Preprocess the data."""
        print("Step 2: Preprocessing data...")
        self.df_processed = self.df.copy()

        # Identify column types
        numerical_cols = self.df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.df_processed.select_dtypes(include=['object']).columns.tolist()

        numerical_features = [col for col in numerical_cols if col != self.target_col]
        categorical_features = [col for col in categorical_cols if col != self.target_col]

        # Handle missing values
        for col in numerical_features:
            if self.df_processed[col].isnull().sum() > 0:
                median_value = self.df_processed[col].median()
                self.df_processed[col].fillna(median_value, inplace=True)

        for col in categorical_features:
            if self.df_processed[col].isnull().sum() > 0:
                mode_value = self.df_processed[col].mode()[0]
                self.df_processed[col].fillna(mode_value, inplace=True)

        print(f"✓ Missing values handled")

        # Remove duplicates
        before = len(self.df_processed)
        self.df_processed.drop_duplicates(inplace=True)
        after = len(self.df_processed)
        print(f"✓ Removed {before - after} duplicate rows")

        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            self.df_processed[col] = le.fit_transform(self.df_processed[col])
            self.label_encoders[col] = le

        # Encode target if categorical
        if self.df_processed[self.target_col].dtype == 'object':
            le_target = LabelEncoder()
            self.df_processed[self.target_col] = le_target.fit_transform(self.df_processed[self.target_col])
            self.label_encoders[self.target_col] = le_target

        print(f"✓ Categorical variables encoded")
        print(f"✓ Final dataset shape: {self.df_processed.shape}\n")

    def prepare_data(self):
        """Prepare data for modeling."""
        print("Step 3: Preparing data for modeling...")

        # Separate features and target
        X = self.df_processed.drop(columns=[self.target_col])
        y = self.df_processed[self.target_col]

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✓ Data split: {len(self.X_train)} training, {len(self.X_test)} testing")

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        print(f"✓ Features scaled")

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(X_train_scaled, self.y_train)

        print(f"✓ Class imbalance handled with SMOTE")
        print(f"  Before: {dict(pd.Series(self.y_train).value_counts())}")
        print(f"  After: {dict(pd.Series(self.y_train_balanced).value_counts())}\n")

        # Store scaled test data
        self.X_test_scaled = X_test_scaled

    def train_models(self):
        """Train multiple ML models."""
        print("Step 4: Training models...")
        print("This may take a few minutes...\n")

        # Initialize models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
        }

        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"Training {name}...")

            # Train
            model.fit(self.X_train_balanced, self.y_train_balanced)

            # Predict
            y_pred = model.predict(self.X_test_scaled)

            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train_balanced, self.y_train_balanced,
                                       cv=5, scoring='accuracy')

            self.results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std()
            })

            print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_scores.mean():.4f}\n")

        print("✓ All models trained\n")

    def evaluate_models(self):
        """Evaluate and compare models."""
        print("Step 5: Evaluating models...")

        # Create results dataframe
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('Accuracy', ascending=False)

        print("\n" + "="*70)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*70)
        print(results_df.to_string(index=False))
        print("="*70 + "\n")

        # Select best model
        self.best_model_name = results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]

        print(f"Best Model: {self.best_model_name}")
        print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}\n")

        # Classification report
        y_pred_best = self.best_model.predict(self.X_test_scaled)
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred_best))

        return results_df

    def save_models(self):
        """Save the best model and artifacts."""
        print("Step 6: Saving model and artifacts...")

        # Create models directory
        os.makedirs('models', exist_ok=True)

        # Save model
        with open('models/best_loan_model.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)

        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save encoders
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)

        # Save feature names
        feature_names = self.X_train.columns.tolist()
        with open('models/feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)

        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': accuracy_score(self.y_test, self.best_model.predict(self.X_test_scaled)),
            'f1_score': f1_score(self.y_test, self.best_model.predict(self.X_test_scaled), average='weighted'),
            'features': feature_names,
            'target_column': self.target_col,
            'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open('models/model_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print("✓ Model saved to 'models/best_loan_model.pkl'")
        print("✓ All artifacts saved\n")

    def generate_visualizations(self):
        """Generate key visualizations."""
        print("Step 7: Generating visualizations...")

        # Confusion matrix
        y_pred_best = self.best_model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred_best)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Confusion matrix saved")

        # Model comparison
        results_df = pd.DataFrame(self.results).sort_values('Accuracy', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(results_df))
        width = 0.2

        ax.bar(x - 1.5*width, results_df['Accuracy'], width, label='Accuracy')
        ax.bar(x - 0.5*width, results_df['Precision'], width, label='Precision')
        ax.bar(x + 0.5*width, results_df['Recall'], width, label='Recall')
        ax.bar(x + 1.5*width, results_df['F1-Score'], width, label='F1-Score')

        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Model comparison chart saved")
        print()

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        try:
            self.load_data()
            self.identify_target()
            self.preprocess_data()
            self.prepare_data()
            self.train_models()
            results_df = self.evaluate_models()
            self.save_models()
            self.generate_visualizations()

            print("="*70)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nBest Model: {self.best_model_name}")
            print(f"Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
            print(f"F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
            print(f"\nModel saved in 'models/' directory")
            print(f"Visualizations saved in current directory")
            print(f"\nTo make predictions, run: python predict.py")
            print("="*70 + "\n")

            return True

        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point."""
    # Check if dataset exists
    if not os.path.exists('loan_approval_dataset.csv'):
        print("Error: loan_approval_dataset.csv not found!")
        print("Please ensure the dataset is in the current directory.")
        return

    # Run analysis
    analysis = LoanApprovalAnalysis()
    success = analysis.run_complete_analysis()

    if success:
        print("\nNext steps:")
        print("1. Review the generated visualizations")
        print("2. Open the Jupyter notebook for detailed analysis")
        print("3. Use predict.py to make predictions on new data")


if __name__ == "__main__":
    main()
