# AWS Deployment via GitHub Actions

This document describes how to deploy the Marketing Tool to AWS using GitHub Actions.

## Overview

The AWS deployment workflow automates the entire deployment process:
- Builds Docker image
- Pushes to Amazon ECR
- Deploys infrastructure using CloudFormation
- Updates ECS service with new image
- Performs health checks

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** configured (for initial setup)
3. **GitHub Secrets** configured (see below)

## Required GitHub Secrets

Navigate to **Settings** → **Secrets and variables** → **Actions** and add:

### AWS Credentials
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key

### Application Secrets
- `OPENAI_API_KEY`: OpenAI API key for AI features
- `API_KEY`: API authentication key (minimum 32 characters)
- `DATABASE_PASSWORD`: RDS database password (minimum 8 characters)
- `MONGODB_PASSWORD`: MongoDB password (minimum 8 characters)

### Optional Secrets
- `DOMAIN_NAME`: Custom domain for the application (e.g., `api.mycompany.com`)
- `CERTIFICATE_ARN`: ACM certificate ARN for HTTPS (required if using custom domain)

## How to Deploy

### Method 1: Manual Deployment (Recommended for Admin)

1. Go to **Actions** → **Deploy to AWS**
2. Click **"Run workflow"**
3. Select:
   - **Environment**: `development`, `staging`, or `production`
   - **Region**: AWS region (default: `us-east-1`)
4. Click **"Run workflow"**

### Method 2: Tag-based Deployment

Push a version tag to trigger automatic deployment to production:

```bash
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## Deployment Process

The workflow performs the following steps:

1. **Build Docker Image**
   - Builds the application using the `Dockerfile`
   - Tags with version, branch, and SHA
   - Caches layers for faster builds

2. **Push to ECR**
   - Authenticates with Amazon ECR
   - Pushes tagged images

3. **Deploy Infrastructure**
   - Creates/updates CloudFormation stack
   - Provisions:
     - VPC with public/private subnets
     - Application Load Balancer
     - ECS Fargate cluster
     - RDS PostgreSQL database
     - ElastiCache Redis
     - CloudWatch logging
     - Security groups and IAM roles

4. **Update ECS Service**
   - Forces new deployment with latest image
   - Waits for service to stabilize
   - Performs rolling update

5. **Health Check**
   - Verifies application is responding
   - Retries up to 10 times
   - Fails deployment if unhealthy

## Monitoring Deployment

### View Progress

1. **GitHub Actions**:
   - Go to **Actions** tab
   - Click on the running workflow
   - View real-time logs

2. **AWS Console**:
   - [CloudFormation Console](https://console.aws.amazon.com/cloudformation/)
   - [ECS Console](https://console.aws.amazon.com/ecs/)
   - [ECR Console](https://console.aws.amazon.com/ecr/)

### Deployment Summary

After deployment, check the workflow summary for:
- Application URL
- ECR repository URI
- CloudFormation stack status
- Quick links to AWS consoles

## Environment-Specific Deployments

### Development
```bash
# Manual trigger only
# Select "development" environment in workflow UI
```

### Staging
```bash
# Manual trigger only
# Select "staging" environment in workflow UI
```

### Production
```bash
# Via tag
git tag v1.0.0
git push origin v1.0.0

# Or manual trigger
# Select "production" environment in workflow UI
```

## Stack Outputs

After deployment, the CloudFormation stack provides:

- **ALBURL**: Application Load Balancer URL
- **ECRRepositoryURI**: Docker image repository
- **ClusterName**: ECS cluster name
- **ServiceName**: ECS service name
- **VPCId**: VPC identifier
- **DatabaseEndpoint**: RDS endpoint

## Troubleshooting

### Deployment Fails

1. **Check CloudFormation Events**:
   ```bash
   aws cloudformation describe-stack-events \
     --stack-name marketing-tool-production \
     --region us-east-1
   ```

2. **Check ECS Service**:
   ```bash
   aws ecs describe-services \
     --cluster marketing-tool-production-cluster \
     --services marketing-tool-production-service \
     --region us-east-1
   ```

3. **View Application Logs**:
   - Go to CloudWatch Logs
   - Log group: `/ecs/marketing-tool-production`

### Health Check Fails

```bash
# Check service status
aws ecs describe-services \
  --cluster marketing-tool-production-cluster \
  --services marketing-tool-production-service

# Check ALB target health
aws elbv2 describe-target-health \
  --target-group-arn <target-group-arn>
```

### Rollback

If deployment fails, CloudFormation will automatically rollback. To manually rollback:

```bash
# Rollback to previous version
aws cloudformation cancel-update-stack \
  --stack-name marketing-tool-production \
  --region us-east-1
```

## Cost Optimization

### Development/Staging
- Use smaller instance types (t3.small)
- Enable scheduled scaling
- Stop non-production environments after hours

### Production
- Use Auto Scaling
- Enable RDS automated backups
- Monitor CloudWatch metrics
- Set up billing alerts

## Security Best Practices

1. **Rotate Secrets Regularly**
   - Update GitHub secrets
   - Rotate AWS access keys
   - Change database passwords

2. **Enable MFA**
   - For AWS root account
   - For IAM users with admin access

3. **Use IAM Roles**
   - Consider using OIDC federation instead of long-lived keys
   - Grant least privilege access

4. **Monitor Access**
   - Enable CloudTrail
   - Set up AWS Config rules
   - Review security groups regularly

## CI/CD Pipeline Integration

The deployment workflow integrates with your existing CI:

```
┌─────────────┐
│  Push Code  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Run CI    │  ← Full_Test, Formatting, Security
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Manual     │
│  Approval   │  ← Admin triggers deployment
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Deploy    │  ← Build, Push, Deploy, Health Check
│   to AWS    │
└─────────────┘
```

## Additional Resources

- [AWS CloudFormation Template](../deploy/aws/cloudformation-template.yaml)
- [Deployment Script](../deploy/aws/deploy.sh)
- [AWS Deployment Guide](../deploy/AWS_DEPLOYMENT.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review CloudFormation events
3. Check application logs in CloudWatch
4. Open an issue in the repository
