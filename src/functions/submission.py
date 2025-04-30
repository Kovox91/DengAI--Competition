import pandas as pd

def create_submission(predictions: pd.DataFrame, validation_data: pd.DataFrame, submission_file_path: str):
    """
    Creates a submission file for the competition.
    Args:
        predictions (pd.DataFrame): The predictions made by the model.
        validation_data (pd.DataFrame): The validation data used to make the predictions.
        submission_file_path (str): The path to save the submission file.
    """
    # Create a submission DataFrame
    submission_df = pd.DataFrame({
        'city': validation_data['city'],
        'year': validation_data['year'],
        'weekofyear': validation_data['weekofyear'],
        'total_cases': predictions
    })

    # Save the submission DataFrame to a CSV file
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file created at {submission_file_path}")