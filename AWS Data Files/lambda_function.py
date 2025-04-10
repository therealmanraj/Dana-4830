import boto3
import pandas as pd
import numpy as np
import io
import json

# Initialize the S3 client
s3 = boto3.client('s3')
bucket_name = 'dana-minicapstone'
data_key = 'data/hvac_model_zones.csv'

# Define the HVAC columns exactly as in your CSV header (without "Date")
hvac_columns = [
    "Environment:Site Outdoor Air Relative Humidity[%]",
    "Environment:Site Outdoor Air Dewpoint Temperature[C]",
    "Electricity:HVAC[J]",
    "Environment:Site Outdoor Air Wetbulb Temperature[C]",
    "CLASSROOM_BOT ZN:Zone People Occupant Count[]",
    "CONFROOM_BOT_1 ZN:Zone People Occupant Count[]",
    "CONFROOM_BOT_2 ZN:Zone People Occupant Count[]",
    "CONFROOM_MID_1 ZN:Zone People Occupant Count[]",
    "CONFROOM_MID_2 ZN:Zone People Occupant Count[]",
    "CONFROOM_TOP_1 ZN:Zone People Occupant Count[]",
    "CONFROOM_TOP_2 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_BOT_1 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_BOT_2 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_BOT_3 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_BOT_4 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_MID_1 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_MID_2 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_MID_3 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_TOP_1 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_TOP_2 ZN:Zone People Occupant Count[]",
    "ENCLOSEDOFFICE_TOP_3 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_BOT_1 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_BOT_2 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_BOT_3 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_MID_1 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_MID_2 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_MID_3 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_MID_4 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_TOP_1 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_TOP_2 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_TOP_3 ZN:Zone People Occupant Count[]",
    "OPENOFFICE_TOP_4 ZN:Zone People Occupant Count[]",
    "Environment:Site Day Type Index[]",
    "Environment:Site Outdoor Air Drybulb Temperature[C]"
]

def generate_random_data(num_rows=100):
    data = {}
    # For each HVAC column, generate random data based on assumptions
    for col in hvac_columns:
        if "Relative Humidity" in col:
            # Random percentage between 0 and 100
            data[col] = np.random.uniform(0, 100, num_rows)
        elif "Dewpoint" in col:
            # Temperature between -10 and 30 Celsius
            data[col] = np.random.uniform(-10, 30, num_rows)
        elif "Electricity:HVAC" in col:
            # Energy values between 1e6 and 2e6 Joules
            data[col] = np.random.uniform(1e6, 2e6, num_rows)
        elif "Wetbulb" in col:
            # Temperature range for wetbulb: 0 to 40 Celsius
            data[col] = np.random.uniform(0, 40, num_rows)
        elif "Occupant Count" in col:
            # Random integer occupant counts between 0 and 50
            data[col] = np.random.randint(0, 51, num_rows)
        elif "Day Type Index" in col:
            # Binary day type index (0 or 1)
            data[col] = np.random.choice([0, 1], num_rows)
        elif "Drybulb" in col:
            # Random drybulb temperature between -10 and 40 Celsius
            data[col] = np.random.uniform(-10, 40, num_rows)
        else:
            # Default random value if needed
            data[col] = np.random.uniform(0, 100, num_rows)
    
    return pd.DataFrame(data)

def lambda_handler(event, context):
    # Generate 100 new random rows of HVAC data (without Date column)
    new_data_df = generate_random_data(100)
    
    # Retrieve the existing dataset from S3
    try:
        response = s3.get_object(Bucket=bucket_name, Key=data_key)
        existing_df = pd.read_csv(io.BytesIO(response['Body'].read()))
    except s3.exceptions.NoSuchKey:
        # If file doesn't exist, create a new DataFrame with the expected columns
        existing_df = pd.DataFrame(columns=hvac_columns)
    
    # Append the new random data to the existing data
    updated_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    
    # Convert the updated DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    updated_df.to_csv(csv_buffer, index=False)
    
    # Upload the updated CSV back to S3
    s3.put_object(Bucket=bucket_name, Key=data_key, Body=csv_buffer.getvalue())
    
    return {
        'statusCode': 200,
        'body': json.dumps('100 new rows added to hvac_model_zones.csv')
    }
