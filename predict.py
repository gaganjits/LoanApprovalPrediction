"""
Loan Approval Prediction Script
================================
Standalone script for making loan approval predictions using the trained model.

Usage:
    python predict.py

    Or import in your code:
    from predict import LoanApprovalPredictor
    predictor = LoanApprovalPredictor()
    result = predictor.predict(applicant_data)
"""

import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class LoanApprovalPredictor:
    """
    A class for making loan approval predictions using a trained machine learning model.
    """

    def __init__(self, model_dir='models'):
        """
        Initialize the predictor by loading saved model artifacts.

        Args:
            model_dir (str): Directory containing saved model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        self.metadata = None
        self.target_column = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load all necessary model artifacts."""
        try:
            # Load model
            with open(f'{self.model_dir}/best_loan_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded successfully")

            # Load scaler
            with open(f'{self.model_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded successfully")

            # Load label encoders
            with open(f'{self.model_dir}/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            print(f"✓ Label encoders loaded successfully")

            # Load feature names
            with open(f'{self.model_dir}/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✓ Feature names loaded successfully")

            # Load metadata
            with open(f'{self.model_dir}/model_metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
                self.target_column = self.metadata.get('target_column', 'loan_status')
            print(f"✓ Metadata loaded successfully")

            print(f"\n{'='*60}")
            print(f"Model Information:")
            print(f"  Algorithm: {self.metadata.get('model_name', 'Unknown')}")
            print(f"  Accuracy: {self.metadata.get('accuracy', 0):.4f}")
            print(f"  F1-Score: {self.metadata.get('f1_score', 0):.4f}")
            print(f"  Training Date: {self.metadata.get('training_date', 'Unknown')}")
            print(f"  Features: {len(self.feature_names)}")
            print(f"{'='*60}\n")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.model_dir}' directory. "
                f"Please ensure you have trained the model first by running the Jupyter notebook."
            )
        except Exception as e:
            raise Exception(f"Error loading model artifacts: {str(e)}")

    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction.

        Args:
            input_data (dict or pd.DataFrame): Applicant data

        Returns:
            np.ndarray: Preprocessed and scaled features
        """
        # Convert to DataFrame if dictionary
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df.columns and col != self.target_column:
                try:
                    # Handle unseen categories
                    df[col] = df[col].map(
                        lambda x: x if x in encoder.classes_ else encoder.classes_[0]
                    )
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    print(f"Warning: Could not encode column '{col}': {str(e)}")

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
                print(f"Warning: Missing feature '{feature}', using default value 0")

        # Select and order features
        df = df[self.feature_names]

        # Scale features
        scaled_data = self.scaler.transform(df)

        return scaled_data

    def predict(self, applicant_data, return_details=True):
        """
        Make a loan approval prediction for an applicant.

        Args:
            applicant_data (dict or pd.DataFrame): Applicant information
            return_details (bool): Whether to return detailed prediction information

        Returns:
            dict: Prediction results including decision and probability
        """
        # Preprocess input
        processed_data = self.preprocess_input(applicant_data)

        # Make prediction
        prediction = self.model.predict(processed_data)[0]

        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(processed_data)[0]
            probability_approved = probabilities[1]
            probability_rejected = probabilities[0]
        else:
            probability_approved = None
            probability_rejected = None

        # Prepare result
        result = {
            'prediction': int(prediction),
            'decision': 'Approved' if prediction == 1 else 'Rejected',
            'probability_approved': float(probability_approved) if probability_approved is not None else None,
            'probability_rejected': float(probability_rejected) if probability_rejected is not None else None,
            'confidence': float(max(probabilities)) if probability_approved is not None else None
        }

        if return_details:
            result['model_name'] = self.metadata.get('model_name', 'Unknown')
            result['model_accuracy'] = self.metadata.get('accuracy', 0)

        return result

    def predict_batch(self, applicants_df):
        """
        Make predictions for multiple applicants.

        Args:
            applicants_df (pd.DataFrame): DataFrame with multiple applicant records

        Returns:
            pd.DataFrame: Original data with predictions and probabilities
        """
        results_df = applicants_df.copy()

        predictions = []
        probabilities = []

        for idx, row in applicants_df.iterrows():
            result = self.predict(row.to_dict(), return_details=False)
            predictions.append(result['decision'])
            probabilities.append(result['probability_approved'])

        results_df['Prediction'] = predictions
        results_df['Approval_Probability'] = probabilities

        return results_df

    def get_feature_importance(self, top_n=10):
        """
        Get feature importance from the model if available.

        Args:
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)

            return importance_df.head(top_n)

        elif hasattr(self.model, 'coef_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_[0]
            }).sort_values('Coefficient', ascending=False, key=abs)

            return importance_df.head(top_n)

        else:
            return None


def interactive_prediction():
    """
    Interactive command-line interface for making predictions.
    """
    print("\n" + "="*60)
    print("LOAN APPROVAL PREDICTION SYSTEM")
    print("="*60 + "\n")

    try:
        # Initialize predictor
        predictor = LoanApprovalPredictor()

        print("\nEnter applicant information (press Enter to skip optional fields):\n")

        # Collect input from user
        applicant_data = {}

        # Example fields - adjust based on your actual dataset
        print("Note: Fields depend on your dataset. Here's an example:")
        print("\nRequired fields (examples):")

        try:
            # You'll need to adjust these fields based on your actual dataset
            age = input("Age: ")
            if age:
                applicant_data['age'] = int(age)

            income = input("Annual Income: ")
            if income:
                applicant_data['income'] = float(income)

            loan_amount = input("Loan Amount: ")
            if loan_amount:
                applicant_data['loan_amount'] = float(loan_amount)

            credit_score = input("Credit Score: ")
            if credit_score:
                applicant_data['credit_score'] = int(credit_score)

            # Add more fields as needed

        except ValueError as e:
            print(f"\nError: Invalid input format. Please enter numeric values where required.")
            return

        if not applicant_data:
            print("\nNo data provided. Using sample data for demonstration...")
            # Use sample data
            applicant_data = {
                'age': 35,
                'income': 50000,
                'loan_amount': 200000,
                'credit_score': 720
            }

        print(f"\n{'='*60}")
        print("Making prediction...")
        print(f"{'='*60}\n")

        # Make prediction
        result = predictor.predict(applicant_data)

        # Display results
        print(f"{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"\nDecision: {result['decision']}")

        if result['probability_approved'] is not None:
            print(f"Approval Probability: {result['probability_approved']:.2%}")
            print(f"Rejection Probability: {result['probability_rejected']:.2%}")
            print(f"Confidence: {result['confidence']:.2%}")

        print(f"\nModel Used: {result['model_name']}")
        print(f"Model Accuracy: {result['model_accuracy']:.4f}")
        print(f"\n{'='*60}\n")

        # Show feature importance
        print("\nTop Important Features:")
        print("="*60)
        importance = predictor.get_feature_importance(top_n=5)
        if importance is not None:
            print(importance.to_string(index=False))
        else:
            print("Feature importance not available for this model.")
        print("="*60)

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("1. You have trained the model by running the Jupyter notebook")
        print("2. The 'models' directory exists with all required files")
        print("3. All dependencies are installed")


def example_usage():
    """
    Example usage of the LoanApprovalPredictor class.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Using LoanApprovalPredictor Programmatically")
    print("="*60 + "\n")

    try:
        # Initialize predictor
        predictor = LoanApprovalPredictor()

        # Example applicant data
        applicant_data = {
            'age': 35,
            'income': 50000,
            'loan_amount': 200000,
            'credit_score': 720,
            # Add other features as needed
        }

        print("Sample Applicant Data:")
        for key, value in applicant_data.items():
            print(f"  {key}: {value}")

        # Make prediction
        result = predictor.predict(applicant_data)

        print(f"\n{'='*60}")
        print("Prediction Result:")
        print(f"{'='*60}")
        print(f"Decision: {result['decision']}")
        if result['probability_approved']:
            print(f"Approval Probability: {result['probability_approved']:.2%}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    """
    Main entry point for the script.
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--example':
        example_usage()
    else:
        # Run interactive prediction
        try:
            interactive_prediction()
        except KeyboardInterrupt:
            print("\n\nPrediction cancelled by user.")
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("\nFor example usage, run: python predict.py --example")
