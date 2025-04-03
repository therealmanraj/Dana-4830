# DANA 4830

Team:

- Alex
- Andres
- Isa
- Mani

The data is limited because github does not allow more data to be hosted. To access the whole data put the year in the folder and it would extract all the data.

Files:

Extraction of data from csv for models.ipynb extracts the csv from the folders.

HVAC_Model.ipynb is for the HVAC prediction.

Occupancy_Model.ipynb is for the occupancy prediction.

---

S3 Bucket Setup and Data Upload:

- We created an S3 bucket named dana-minicapstone.

- Uploaded initial datasets (e.g., hvac_model_zones.csv, occupancy_model_zones.csv) to the bucket using Python (boto3), AWS CLI, or the AWS Console.

IAM Configuration:

- Created an IAM user (e.g., dana-4830) for programmatic access.

- Ensured that the IAM user/role has the necessary S3 permissions (using a custom policy or managed policies).

Lambda Function Creation:

- Developed a Python script that simulates daily data ingestion by generating 100 new random rows of HVAC data.

- The Lambda function retrieves the existing CSV from the S3 bucket, appends new random data, and re-uploads the file.

- Tested the function locally via the Lambda console and scheduled it (if needed) with EventBridge.

Dependency Management:

- Encountered issues with missing modules (like pandas and numpy), timeout and resources.

- Resolved these by creating a Lambda layer (or packaging dependencies) so that the function can import these libraries.

- Timeout and resources were increased to fix the issue.
