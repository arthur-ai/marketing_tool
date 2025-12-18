# AWS Deployment Guide

This guide explains how to deploy the Marketing Tool to AWS using CloudFormation.

## Overview

The CloudFormation template creates a complete AWS infrastructure including:

- **VPC** with public and private subnets across multiple AZs
- **ECS Fargate** cluster for running the application
- **Application Load Balancer** for traffic distribution
- **RDS PostgreSQL** database for structured data
- **ElastiCache Redis** for caching and rate limiting
- **DocumentDB** (MongoDB-compatible) for content storage
- **CloudWatch** for logging and monitoring
- **ECR** repository for Docker images
- **Secrets Manager** for secure credential storage

## Prerequisites

1. **AWS CLI** installed and configured
2. **Docker** installed (for building images)
3. **Required credentials**:
   - OpenAI API key
   - API authentication key (32+ characters)
   - Database password (8+ characters)
   - MongoDB password (8+ characters)

## Quick Start

### 1. Set Environment Variables

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
export API_KEY="your_32_character_minimum_api_key_here"
export DATABASE_PASSWORD="your_database_password_here"
export MONGODB_PASSWORD="your_mongodb_password_here"
```

### 2. Deploy Using the Script

```bash
# Deploy to production
./deploy.sh -e production -r us-east-2

# Deploy with custom domain
./deploy.sh -e production -d api.mycompany.com -c arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012

# Dry run to see what would be deployed
./deploy.sh --dry-run -e staging
```

### 3. Build and Push Docker Image

After deployment, build and push your Docker image. **Note**: The Dockerfile automatically downloads the UDPipe English model required for SEO keywords engine operations.

```bash
# Get the ECR repository URI from the stack outputs
ECR_URI=$(aws cloudformation describe-stacks --stack-name marketing-tool-production --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' --output text)

# Build the image (includes UDPipe model download)
docker build -t $ECR_URI:latest -f deploy/docker/Dockerfile .

# Login to ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $ECR_URI

# Push the image
docker push $ECR_URI:latest

# Update the ECS service
aws ecs update-service --cluster marketing-tool-production-cluster --service marketing-tool-production-service --force-new-deployment
```

**Important**: If you're rebuilding an existing image, make sure to use the updated Dockerfile that includes the UDPipe model download step. The model is automatically downloaded during the Docker build process.

## Manual Deployment

If you prefer to deploy manually:

### 1. Create Stack

```bash
aws cloudformation create-stack \
  --stack-name marketing-tool-production \
  --template-body file://cloudformation-template.yaml \
  --parameters file://cloudformation-parameters.json \
  --capabilities CAPABILITY_IAM
```

### 2. Update Parameters

Edit `cloudformation-parameters.json` and replace the placeholder values:

- `YOUR_OPENAI_API_KEY_HERE` → Your actual OpenAI API key
- `YOUR_API_KEY_HERE_32_CHARS_MINIMUM` → Your API key (32+ chars)
- `YOUR_DATABASE_PASSWORD_HERE` → Your database password (8+ chars)
- `YOUR_MONGODB_PASSWORD_HERE` → Your MongoDB password (8+ chars)

### 3. Monitor Deployment

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name marketing-tool-production

# Watch stack events
aws cloudformation describe-stack-events --stack-name marketing-tool-production
```

## Configuration

### Environment-Specific Settings

The template automatically adjusts settings based on the environment:

| Setting | Development | Staging | Production |
|---------|-------------|---------|------------|
| Instance Type | t3.medium | t3.medium | t3.medium |
| Desired Count | 1 | 2 | 2+ |
| Database Multi-AZ | No | No | Yes |
| Backup Retention | 1 day | 1 day | 7 days |
| Deletion Protection | No | No | Yes |
| Debug Mode | Yes | No | No |

### Custom Domain Setup

To use a custom domain:

1. **Create ACM Certificate**:
   ```bash
   aws acm request-certificate \
     --domain-name api.mycompany.com \
     --validation-method DNS
   ```

2. **Validate Certificate**:
   - Add the DNS validation record to your domain
   - Wait for validation to complete

3. **Deploy with Domain**:
   ```bash
   ./deploy.sh -e production -d api.mycompany.com -c arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012
   ```

## Monitoring and Logs

### CloudWatch Dashboard

A CloudWatch dashboard is automatically created with:
- ECS service metrics (CPU, Memory)
- Load balancer metrics (Request count, Response time)

### Application Logs

Logs are available in CloudWatch Logs:
- Log Group: `/ecs/marketing-tool-{environment}`
- Log Stream: `ecs/marketing-tool-api/{task-id}`

### Health Checks

The application includes health check endpoints:
- **Health**: `GET /api/v1/health`
- **Readiness**: `GET /api/v1/ready`

## Scaling

### Horizontal Scaling

Update the desired count:
```bash
aws ecs update-service \
  --cluster marketing-tool-production-cluster \
  --service marketing-tool-production-service \
  --desired-count 5
```

### Vertical Scaling

Update the CloudFormation stack with new instance types:
```bash
aws cloudformation update-stack \
  --stack-name marketing-tool-production \
  --use-previous-template \
  --parameters ParameterKey=InstanceType,ParameterValue=t3.large
```

## Security

### Network Security

- Application runs in private subnets
- Database and cache in isolated subnets
- Security groups restrict access between components
- All data encrypted in transit and at rest

### Credential Management

- API keys stored in AWS Secrets Manager
- Database passwords encrypted
- IAM roles follow least privilege principle

### SSL/TLS

- HTTPS enabled when certificate provided
- HTTP to HTTPS redirect configured
- Modern TLS versions only

## Troubleshooting

### Common Issues

1. **Stack Creation Fails**:
   - Check IAM permissions
   - Verify parameter values
   - Check for resource limits

2. **ECS Tasks Not Starting**:
   - Check CloudWatch logs
   - Verify Docker image exists in ECR
   - Check security group rules

3. **Database Connection Issues**:
   - Verify security group allows port 5432
   - Check database endpoint
   - Verify credentials

### Useful Commands

```bash
# Check ECS service status
aws ecs describe-services --cluster marketing-tool-production-cluster --services marketing-tool-production-service

# View recent logs
aws logs tail /ecs/marketing-tool-production --follow

# Check load balancer health
aws elbv2 describe-target-health --target-group-arn $(aws elbv2 describe-target-groups --names marketing-tool-production-tg --query 'TargetGroups[0].TargetGroupArn' --output text)

# Get stack outputs
aws cloudformation describe-stacks --stack-name marketing-tool-production --query 'Stacks[0].Outputs'
```

## Cost Optimization

### Development Environment

- Use `t3.small` instances
- Single AZ deployment
- Minimal backup retention
- Spot instances for non-critical workloads

### Production Environment

- Multi-AZ for high availability
- Reserved instances for predictable workloads
- CloudWatch log retention policies
- Auto Scaling based on metrics

## Cleanup

To remove all resources:

```bash
# Delete the CloudFormation stack
aws cloudformation delete-stack --stack-name marketing-tool-production

# Wait for deletion to complete
aws cloudformation wait stack-delete-complete --stack-name marketing-tool-production
```

**Note**: This will delete all data in the databases. Make sure to backup any important data before cleanup.

## Support

For issues or questions:
1. Check the CloudFormation console for stack events
2. Review CloudWatch logs for application errors
3. Check the troubleshooting section above
4. Review AWS documentation for specific services
