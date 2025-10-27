"""
Test Prediction Script
======================
Demonstrates the loan approval prediction system with real examples.
"""

from predict import LoanApprovalPredictor
import pandas as pd

def test_predictions():
    """Test the prediction system with sample applicants."""

    print("\n" + "="*70)
    print("LOAN APPROVAL PREDICTION - TEST SCENARIOS")
    print("="*70 + "\n")

    # Initialize predictor
    predictor = LoanApprovalPredictor()

    # Test cases with different profiles
    test_cases = [
        {
            "name": "High-Quality Applicant",
            "data": {
                'loan_id': 9001,
                ' no_of_dependents': 1,
                ' education': ' Graduate',
                ' self_employed': ' No',
                ' income_annum': 8000000,  # 8 million annual income
                ' loan_amount': 15000000,  # 15 million loan
                ' loan_term': 10,
                ' cibil_score': 800,  # Excellent credit
                ' residential_assets_value': 6000000,
                ' commercial_assets_value': 8000000,
                ' luxury_assets_value': 10000000,
                ' bank_asset_value': 5000000
            }
        },
        {
            "name": "Average Applicant",
            "data": {
                'loan_id': 9002,
                ' no_of_dependents': 2,
                ' education': ' Graduate',
                ' self_employed': ' No',
                ' income_annum': 5000000,  # 5 million annual income
                ' loan_amount': 12000000,  # 12 million loan
                ' loan_term': 12,
                ' cibil_score': 720,  # Good credit
                ' residential_assets_value': 4000000,
                ' commercial_assets_value': 3000000,
                ' luxury_assets_value': 5000000,
                ' bank_asset_value': 3000000
            }
        },
        {
            "name": "High-Risk Applicant",
            "data": {
                'loan_id': 9003,
                ' no_of_dependents': 4,
                ' education': ' Not Graduate',
                ' self_employed': ' Yes',
                ' income_annum': 2000000,  # 2 million annual income
                ' loan_amount': 15000000,  # 15 million loan (high ratio)
                ' loan_term': 20,
                ' cibil_score': 600,  # Poor credit
                ' residential_assets_value': 1000000,
                ' commercial_assets_value': 500000,
                ' luxury_assets_value': 1000000,
                ' bank_asset_value': 800000
            }
        },
        {
            "name": "Self-Employed Professional",
            "data": {
                'loan_id': 9004,
                ' no_of_dependents': 0,
                ' education': ' Graduate',
                ' self_employed': ' Yes',
                ' income_annum': 7000000,  # 7 million annual income
                ' loan_amount': 10000000,  # 10 million loan
                ' loan_term': 8,
                ' cibil_score': 780,  # Very good credit
                ' residential_assets_value': 8000000,
                ' commercial_assets_value': 12000000,
                ' luxury_assets_value': 6000000,
                ' bank_asset_value': 4000000
            }
        },
        {
            "name": "Young Professional",
            "data": {
                'loan_id': 9005,
                ' no_of_dependents': 0,
                ' education': ' Graduate',
                ' self_employed': ' No',
                ' income_annum': 4000000,  # 4 million annual income
                ' loan_amount': 8000000,  # 8 million loan
                ' loan_term': 15,
                ' cibil_score': 740,  # Good credit
                ' residential_assets_value': 2000000,
                ' commercial_assets_value': 1000000,
                ' luxury_assets_value': 3000000,
                ' bank_asset_value': 2000000
            }
        }
    ]

    # Test each case
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*70}")

        # Display applicant profile
        data = test_case['data']
        print("\nApplicant Profile:")
        print(f"  Education: {data[' education'].strip()}")
        print(f"  Employment: {'Self-Employed' if data[' self_employed'].strip() == 'Yes' else 'Salaried'}")
        print(f"  Dependents: {data[' no_of_dependents']}")
        print(f"  Annual Income: ₹{data[' income_annum']:,}")
        print(f"  Loan Amount: ₹{data[' loan_amount']:,}")
        print(f"  Loan Term: {data[' loan_term']} months")
        print(f"  CIBIL Score: {data[' cibil_score']}")
        print(f"  Total Assets: ₹{data[' residential_assets_value'] + data[' commercial_assets_value'] + data[' luxury_assets_value'] + data[' bank_asset_value']:,}")

        # Make prediction
        result = predictor.predict(data, return_details=True)

        # Display results
        print(f"\n{'─'*70}")
        print("PREDICTION RESULT:")
        print(f"{'─'*70}")
        print(f"  Decision: {result['decision']} {'✅' if result['decision'] == 'Approved' else '❌'}")
        if result['probability_approved'] is not None:
            print(f"  Approval Probability: {result['probability_approved']:.2%}")
            print(f"  Rejection Probability: {result['probability_rejected']:.2%}")
            print(f"  Confidence: {result['confidence']:.2%}")

        # Risk assessment
        if result['probability_approved'] is not None:
            prob = result['probability_approved']
            if prob >= 0.8:
                risk = "LOW RISK"
                risk_emoji = "🟢"
            elif prob >= 0.6:
                risk = "MEDIUM RISK"
                risk_emoji = "🟡"
            else:
                risk = "HIGH RISK"
                risk_emoji = "🔴"
            print(f"  Risk Assessment: {risk} {risk_emoji}")

        print(f"{'='*70}\n")

        # Store results
        results.append({
            'Applicant': test_case['name'],
            'Decision': result['decision'],
            'Probability': f"{result['probability_approved']:.2%}" if result['probability_approved'] else "N/A",
            'Confidence': f"{result['confidence']:.2%}" if result['confidence'] else "N/A"
        })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL PREDICTIONS")
    print(f"{'='*70}\n")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print(f"\n{'='*70}")

    # Statistics
    approved = sum(1 for r in results if r['Decision'] == 'Approved')
    rejected = sum(1 for r in results if r['Decision'] == 'Rejected')

    print(f"\nStatistics:")
    print(f"  Total Applications Tested: {len(results)}")
    print(f"  Approved: {approved} ({approved/len(results)*100:.1f}%)")
    print(f"  Rejected: {rejected} ({rejected/len(results)*100:.1f}%)")
    print(f"\n{'='*70}\n")


def load_and_predict_from_csv():
    """Load test data from CSV and make predictions."""

    print("\n" + "="*70)
    print("BATCH PREDICTION FROM DATASET")
    print("="*70 + "\n")

    try:
        # Load dataset
        df = pd.read_csv('loan_approval_dataset.csv')

        # Take a sample of 5 records
        sample_df = df.drop(columns=[' loan_status']).sample(n=5, random_state=42)

        print("Making predictions for 5 random applicants from the dataset...\n")

        # Initialize predictor
        predictor = LoanApprovalPredictor()

        # Make predictions
        results_df = predictor.predict_batch(sample_df)

        # Display results
        print("Results:")
        print("="*70)
        display_cols = ['loan_id', ' cibil_score', ' income_annum', ' loan_amount',
                       'Prediction', 'Approval_Probability']
        print(results_df[display_cols].to_string(index=False))
        print("="*70 + "\n")

    except FileNotFoundError:
        print("Dataset file not found. Skipping batch prediction test.\n")
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}\n")


if __name__ == "__main__":
    """Run all tests."""

    try:
        # Test individual predictions
        test_predictions()

        # Test batch predictions
        load_and_predict_from_csv()

        print("\nAll tests completed successfully! ✅")
        print("\nTo use this system in your application:")
        print("  1. Import: from predict import LoanApprovalPredictor")
        print("  2. Initialize: predictor = LoanApprovalPredictor()")
        print("  3. Predict: result = predictor.predict(applicant_data)")
        print("\nSee predict.py for more details.\n")

    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
