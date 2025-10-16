# AWS Deployment Files

This directory contains all the files needed to deploy the Marketing Tool to AWS using CloudFormation.

## Files

- **`cloudformation-template.yaml`** - Main CloudFormation template defining the AWS infrastructure
- **`cloudformation-parameters.json`** - Template parameters file (update with your values)
- **`deploy.sh`** - Automated deployment script
- **`AWS_DEPLOYMENT.md`** - Comprehensive deployment guide

## Quick Start

1. Set your environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export API_KEY="your_32_character_minimum_api_key_here"
   export DATABASE_PASSWORD="your_database_password_here"
   export MONGODB_PASSWORD="your_mongodb_password_here"
   ```

2. Run the deployment script:
   ```bash
   cd deploy
   ./deploy.sh -e production -r us-east-1
   ```

3. Build and push your Docker image (see AWS_DEPLOYMENT.md for details)

## Manual Deployment

If you prefer to deploy manually:

```bash
cd deploy
aws cloudformation create-stack \
  --stack-name marketing-tool-production \
  --template-body file://cloudformation-template.yaml \
  --parameters file://cloudformation-parameters.json \
  --capabilities CAPABILITY_IAM
```

## Documentation

See `AWS_DEPLOYMENT.md` for detailed deployment instructions, troubleshooting, and configuration options.
