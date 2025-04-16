import boto3
import os
import json

# Initialize clients (ensure region is set to ca-central-1)
s3 = boto3.client('s3', region_name="ca-central-1")
runtime = boto3.client('sagemaker-runtime', region_name="ca-central-1")

# The SageMaker endpoint name should be set as an environment variable in the Lambda function
ENDPOINT_NAME = os.environ.get("SM_ENDPOINT_NAME")

def lambda_handler(event, context):
    # Log the incoming event for debugging
    print("Received event:", json.dumps(event))
    
    # Extract bucket and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Only process files in the "incoming/" folder
    if not key.startswith("incoming/"):
        print("File not in incoming folder; skipping:", key)
        return
    
    try:
        # Download the CSV file content from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        print("Downloaded file from S3:", key)
        
        # Invoke the SageMaker endpoint using sagemaker-runtime.
        # Note: Explicitly set Accept to a supported value (here "text/csv")
        endpoint_response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Accept='text/csv',   # Ensure the output function accepts this type
            Body=csv_content
        )
        
        result = endpoint_response['Body'].read().decode('utf-8')
        print("Received prediction result:", result)
        
        # Define the destination key for predictions by replacing "incoming/" with "predictions/"
        prediction_key = key.replace("incoming/", "predictions/")
        
        # Upload the prediction result to S3
        s3.put_object(Bucket=bucket, Key=prediction_key, Body=result)
        print(f"Uploaded predictions to s3://{bucket}/{prediction_key}")
        
        return {
            'statusCode': 200,
            'body': f"Processed {key} and saved predictions to {prediction_key}"
        }
    
    except Exception as e:
        print("Error processing file {}: {}".format(key, str(e)))
        raise
