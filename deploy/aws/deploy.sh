#!/bin/bash

# Marketing Tool AWS Deployment Script
# This script deploys the marketing tool to AWS using CloudFormation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="prod"
PROJECT_NAME="marketing-tool"
REGION="us-east-2"
STACK_NAME=""
DOMAIN_NAME=""
CERTIFICATE_ARN=""
EXISTING_VPC_ID=""
EXISTING_PUBLIC_SUBNET_1_ID=""
EXISTING_PUBLIC_SUBNET_2_ID=""
EXISTING_DB_SUBNET_1_ID=""
EXISTING_DB_SUBNET_2_ID=""
EXISTING_DATABASE_ENDPOINT=""
EXISTING_DATABASE_NAME=""
EXISTING_REDIS_ENDPOINT=""
EXISTING_S3_BUCKET_NAME=""
FRONTEND_CONTAINER_IMAGE=""
FRONTEND_DEPLOY_VERSION="latest"
FRONTEND_DOCKER_REGISTRY_USERNAME=""
FRONTEND_DOCKER_REGISTRY_PASSWORD=""
DRY_RUN=false
FORCE_UPDATE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Marketing Tool to AWS using CloudFormation

OPTIONS:
    -e, --environment ENV     Environment (dev|stag|prod) [default: prod]
    -n, --project-name NAME   Project name [default: marketing-tool]
    -r, --region REGION       AWS region [default: us-east-2]
    -s, --stack-name NAME     CloudFormation stack name [default: marketing-tool-ENV]
    -v, --deploy-version VER  Deploy version to force new resources [default: v1]
    -d, --domain DOMAIN       Domain name for the application (required). Frontend: DomainName, API: DomainName/api/*
    -c, --certificate ARN     ACM certificate ARN for HTTPS (optional)
    --existing-vpc-id ID           Existing VPC ID (required)
    --existing-public-subnet-1 ID   Existing public subnet 1 ID (required)
    --existing-public-subnet-2 ID   Existing public subnet 2 ID (required)
    --existing-public-subnet-3 ID   Existing public subnet 3 ID (required)
    --existing-db-subnet-1 ID       Existing database subnet 1 ID (required)
    --existing-db-subnet-2 ID       Existing database subnet 2 ID (required)
    --existing-db-subnet-3 ID       Existing database subnet 3 ID (required - use Arthur Engine's private subnet 3)
    --existing-database-endpoint ENDPOINT  Existing RDS database endpoint (optional - your RDS endpoint with marketing_tool_main database)
    --existing-database-name NAME          Database name (optional - default: marketing_tool_main)
    --existing-redis-endpoint ENDPOINT     Existing Redis endpoint (optional)
    --existing-s3-bucket-name NAME         Existing S3 bucket name (optional)
    --frontend-image URI                   Frontend Docker image URI from ECR (optional - e.g., 123456789.dkr.ecr.us-east-1.amazonaws.com/marketing-frontend:latest)
    --frontend-version VERSION            Frontend Docker image version/tag (optional - default: latest)
    --frontend-backend-api-url URL        Backend API URL for frontend (optional - auto-inferred as https://DomainName/api if not provided)
    --frontend-docker-registry-username USERNAME Docker registry username (required when deploying frontend)
    --frontend-docker-registry-password PASSWORD Docker registry password (required when deploying frontend)
    -f, --force               Force update even if no changes detected
    --dry-run                 Show what would be deployed without actually deploying
    -h, --help                Show this help message

REQUIRED ENVIRONMENT VARIABLES:
    OPENAI_API_KEY           OpenAI API key
    API_KEY                  API authentication key (32+ characters)
    DATABASE_PASSWORD        Database master password (stored in AWS Secrets Manager)
    REDIS_PASSWORD           Redis authentication token (16-128 characters)

OPTIONAL ENVIRONMENT VARIABLES:
    EXISTING_MONGODB_ENDPOINT  Existing MongoDB/DocumentDB endpoint (optional - provide to use existing MongoDB)
    ARTHUR_BASE_URL            Base URL for Arthur API (optional - default: http://localhost:3030)
    ARTHUR_API_KEY             API key for Arthur authentication (optional - leave empty to disable telemetry)
    ARTHUR_TASK_ID             Task ID for Arthur (optional - leave empty to disable telemetry, must have is_agentic=True if provided)
    OTEL_SERVICE_NAME          OpenTelemetry service name (optional - default: marketing-tool)
    OTEL_DEPLOYMENT_ENVIRONMENT OpenTelemetry deployment environment (optional - default: production)
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT Capture message content in telemetry (optional - default: true)

EXAMPLES:
    # Deploy to production
    $0 -e prod -r us-west-2

    # Deploy with custom domain
    $0 -e prod -d api.mycompany.com -c arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012

    # Dry run to see what would be deployed
    $0 --dry-run -e stag

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -n|--project-name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -s|--stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        -d|--domain)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        -c|--certificate)
            CERTIFICATE_ARN="$2"
            shift 2
            ;;
        -v|--deploy-version)
            DEPLOY_VERSION="$2"
            shift 2
            ;;
        --existing-vpc-id)
            EXISTING_VPC_ID="$2"
            shift 2
            ;;
        --existing-public-subnet-1)
            EXISTING_PUBLIC_SUBNET_1_ID="$2"
            shift 2
            ;;
        --existing-public-subnet-2)
            EXISTING_PUBLIC_SUBNET_2_ID="$2"
            shift 2
            ;;
        --existing-public-subnet-3)
            EXISTING_PUBLIC_SUBNET_3_ID="$2"
            shift 2
            ;;
        --existing-db-subnet-1)
            EXISTING_DB_SUBNET_1_ID="$2"
            shift 2
            ;;
        --existing-db-subnet-2)
            EXISTING_DB_SUBNET_2_ID="$2"
            shift 2
            ;;
        --existing-db-subnet-3)
            EXISTING_DB_SUBNET_3_ID="$2"
            shift 2
            ;;
        --existing-database-endpoint)
            EXISTING_DATABASE_ENDPOINT="$2"
            shift 2
            ;;
        --existing-database-name)
            EXISTING_DATABASE_NAME="$2"
            shift 2
            ;;
        --existing-redis-endpoint)
            EXISTING_REDIS_ENDPOINT="$2"
            shift 2
            ;;
        --existing-s3-bucket-name)
            EXISTING_S3_BUCKET_NAME="$2"
            shift 2
            ;;
        --frontend-image)
            FRONTEND_CONTAINER_IMAGE="$2"
            shift 2
            ;;
        --frontend-version)
            FRONTEND_DEPLOY_VERSION="$2"
            shift 2
            ;;
        --frontend-backend-api-url)
            FRONTEND_BACKEND_API_URL="$2"
            shift 2
            ;;
        --frontend-docker-registry-username)
            FRONTEND_DOCKER_REGISTRY_USERNAME="$2"
            shift 2
            ;;
        --frontend-docker-registry-password)
            FRONTEND_DOCKER_REGISTRY_PASSWORD="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_UPDATE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default stack name if not provided
if [[ -z "$STACK_NAME" ]]; then
    STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
fi

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|stag|prod)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be one of: dev, stag, prod"
    exit 1
fi

# Check required environment variables
print_info "Checking required environment variables..."

required_vars=("OPENAI_API_KEY" "API_KEY" "DATABASE_PASSWORD" "REDIS_PASSWORD")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    print_error "Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please set these variables and try again."
    exit 1
fi

# Validate API key length
if [[ ${#API_KEY} -lt 32 ]]; then
    print_error "API_KEY must be at least 32 characters long"
    exit 1
fi

# Validate Redis password length
if [[ ${#REDIS_PASSWORD} -lt 16 ]] || [[ ${#REDIS_PASSWORD} -gt 128 ]]; then
    print_error "REDIS_PASSWORD must be between 16 and 128 characters long"
    exit 1
fi

# Check required infrastructure parameters
print_info "Checking required infrastructure parameters..."
required_infra=("EXISTING_VPC_ID" "EXISTING_PUBLIC_SUBNET_1_ID" "EXISTING_PUBLIC_SUBNET_2_ID" "EXISTING_PUBLIC_SUBNET_3_ID" "EXISTING_DB_SUBNET_1_ID" "EXISTING_DB_SUBNET_2_ID" "EXISTING_DB_SUBNET_3_ID")
missing_infra=()

for var in "${required_infra[@]}"; do
    eval "value=\$$var"
    if [[ -z "$value" ]]; then
        missing_infra+=("$var")
    fi
done

if [[ ${#missing_infra[@]} -gt 0 ]]; then
    print_error "Missing required infrastructure parameters:"
    for var in "${missing_infra[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please provide these parameters using --existing-* flags and try again."
    exit 1
fi

# Check required domain parameters
print_info "Checking required domain parameters..."
if [[ -z "$DOMAIN_NAME" ]]; then
    print_error "DOMAIN_NAME environment variable is required"
    exit 1
fi

if [[ -z "$HOSTED_ZONE_ID" ]]; then
    print_error "HOSTED_ZONE_ID environment variable is required"
    exit 1
fi

print_success "All required environment variables and infrastructure parameters are set"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured. Please run 'aws configure' first."
    exit 1
fi

# Check if Docker is available (for building and pushing images)
if ! command -v docker &> /dev/null; then
    print_warning "Docker is not installed. You'll need to build and push the Docker image manually."
fi

# Set AWS region
export AWS_DEFAULT_REGION="$REGION"

print_info "Deployment Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  Project Name: $PROJECT_NAME"
echo "  Stack Name: $STACK_NAME"
echo "  Region: $REGION"
echo "  VPC ID: $EXISTING_VPC_ID"
echo "  Domain: ${DOMAIN_NAME:-'Not specified'}"
echo "  Frontend: ${DOMAIN_NAME:-'N/A'}"
echo "  API: ${DOMAIN_NAME:-'N/A'}/api/*"
echo "  Certificate: ${CERTIFICATE_ARN:-'Not specified'}"
echo "  Existing Database: ${EXISTING_DATABASE_ENDPOINT:-'Will create new'}"
if [[ -n "$EXISTING_DATABASE_ENDPOINT" ]]; then
    echo "  Database Name: ${EXISTING_DATABASE_NAME:-marketing_tool_main}"
fi
echo "  Existing Redis: ${EXISTING_REDIS_ENDPOINT:-'Will create new'}"
echo "  Existing S3 Bucket: ${EXISTING_S3_BUCKET_NAME:-'Will create new'}"
echo "  Frontend Image: ${FRONTEND_CONTAINER_IMAGE:-'Not deploying frontend'}"
if [[ -n "$FRONTEND_CONTAINER_IMAGE" ]]; then
    echo "  Frontend Version: ${FRONTEND_DEPLOY_VERSION}"
    echo "  Frontend Backend API URL: Auto-inferred as https://${DOMAIN_NAME:-'N/A'}/api"
fi
echo "  Dry Run: $DRY_RUN"
echo ""

# Build CloudFormation parameters
DEPLOY_VERSION="${DEPLOY_VERSION:-v1}"
PARAMETERS="ParameterKey=DeployVersion,ParameterValue=$DEPLOY_VERSION"
PARAMETERS="$PARAMETERS ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
PARAMETERS="$PARAMETERS ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME"
PARAMETERS="$PARAMETERS ParameterKey=OpenAIApiKey,ParameterValue=$OPENAI_API_KEY"
PARAMETERS="$PARAMETERS ParameterKey=ApiKey,ParameterValue=$API_KEY"
PARAMETERS="$PARAMETERS ParameterKey=DatabasePassword,ParameterValue=$DATABASE_PASSWORD"
PARAMETERS="$PARAMETERS ParameterKey=RedisPassword,ParameterValue=$REDIS_PASSWORD"
if [[ -n "$EXISTING_MONGODB_ENDPOINT" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=ExistingMongoDBEndpoint,ParameterValue=$EXISTING_MONGODB_ENDPOINT"
fi

# Required domain parameters
PARAMETERS="$PARAMETERS ParameterKey=DomainName,ParameterValue=$DOMAIN_NAME"
PARAMETERS="$PARAMETERS ParameterKey=HostedZoneId,ParameterValue=$HOSTED_ZONE_ID"

if [[ -n "$CERTIFICATE_ARN" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=CertificateArn,ParameterValue=$CERTIFICATE_ARN"
fi

# Required infrastructure parameters
PARAMETERS="$PARAMETERS ParameterKey=ExistingVpcId,ParameterValue=$EXISTING_VPC_ID"
PARAMETERS="$PARAMETERS ParameterKey=ExistingPublicSubnet1Id,ParameterValue=$EXISTING_PUBLIC_SUBNET_1_ID"
PARAMETERS="$PARAMETERS ParameterKey=ExistingPublicSubnet2Id,ParameterValue=$EXISTING_PUBLIC_SUBNET_2_ID"
PARAMETERS="$PARAMETERS ParameterKey=ExistingPublicSubnet3Id,ParameterValue=$EXISTING_PUBLIC_SUBNET_3_ID"
PARAMETERS="$PARAMETERS ParameterKey=ExistingDatabaseSubnet1Id,ParameterValue=$EXISTING_DB_SUBNET_1_ID"
PARAMETERS="$PARAMETERS ParameterKey=ExistingDatabaseSubnet2Id,ParameterValue=$EXISTING_DB_SUBNET_2_ID"
PARAMETERS="$PARAMETERS ParameterKey=ExistingDatabaseSubnet3Id,ParameterValue=$EXISTING_DB_SUBNET_3_ID"

if [[ -n "$EXISTING_DATABASE_ENDPOINT" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=ExistingDatabaseEndpoint,ParameterValue=$EXISTING_DATABASE_ENDPOINT"
    # Use provided database name or default
    DB_NAME="${EXISTING_DATABASE_NAME:-marketing_tool_main}"
    PARAMETERS="$PARAMETERS ParameterKey=ExistingDatabaseName,ParameterValue=$DB_NAME"
fi

if [[ -n "$EXISTING_REDIS_ENDPOINT" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=ExistingRedisEndpoint,ParameterValue=$EXISTING_REDIS_ENDPOINT"
fi

if [[ -n "$EXISTING_S3_BUCKET_NAME" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=ExistingS3BucketName,ParameterValue=$EXISTING_S3_BUCKET_NAME"
fi

# Frontend parameters
if [[ -n "$FRONTEND_CONTAINER_IMAGE" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=FrontendContainerImage,ParameterValue=$FRONTEND_CONTAINER_IMAGE"
    # Use backend version if frontend version not specified
    if [[ -z "$FRONTEND_DEPLOY_VERSION" ]] || [[ "$FRONTEND_DEPLOY_VERSION" = "latest" ]]; then
        FRONTEND_DEPLOY_VERSION="$DEPLOY_VERSION"
        print_info "Using backend version ($DEPLOY_VERSION) for frontend"
    fi
    PARAMETERS="$PARAMETERS ParameterKey=FrontendDeployVersion,ParameterValue=$FRONTEND_DEPLOY_VERSION"
    # API URL is optional - will be auto-inferred from DomainName if not provided
    if [[ -n "$FRONTEND_BACKEND_API_URL" ]]; then
        PARAMETERS="$PARAMETERS ParameterKey=FrontendBackendApiUrl,ParameterValue=$FRONTEND_BACKEND_API_URL"
    fi
    # Docker registry credentials - always required for docker.arthur.ai
    if [[ -z "$FRONTEND_DOCKER_REGISTRY_USERNAME" ]] || [[ -z "$FRONTEND_DOCKER_REGISTRY_PASSWORD" ]]; then
        print_error "Frontend Docker registry username and password are required when deploying frontend"
        exit 1
    fi
    PARAMETERS="$PARAMETERS ParameterKey=FrontendDockerRegistryUsername,ParameterValue=$FRONTEND_DOCKER_REGISTRY_USERNAME"
    PARAMETERS="$PARAMETERS ParameterKey=FrontendDockerRegistryPassword,ParameterValue=$FRONTEND_DOCKER_REGISTRY_PASSWORD"
fi

# Telemetry parameters (optional)
ARTHUR_BASE_URL="${ARTHUR_BASE_URL:-http://localhost:3030}"
ARTHUR_API_KEY="${ARTHUR_API_KEY:-}"
ARTHUR_TASK_ID="${ARTHUR_TASK_ID:-}"
OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-marketing-tool}"
OTEL_DEPLOYMENT_ENVIRONMENT="${OTEL_DEPLOYMENT_ENVIRONMENT:-production}"
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="${OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:-true}"

PARAMETERS="$PARAMETERS ParameterKey=ArthurBaseUrl,ParameterValue=$ARTHUR_BASE_URL"
if [[ -n "$ARTHUR_API_KEY" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=ArthurApiKey,ParameterValue=$ARTHUR_API_KEY"
else
    PARAMETERS="$PARAMETERS ParameterKey=ArthurApiKey,ParameterValue="
fi
if [[ -n "$ARTHUR_TASK_ID" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=ArthurTaskId,ParameterValue=$ARTHUR_TASK_ID"
else
    PARAMETERS="$PARAMETERS ParameterKey=ArthurTaskId,ParameterValue="
fi
PARAMETERS="$PARAMETERS ParameterKey=OtelServiceName,ParameterValue=$OTEL_SERVICE_NAME"
PARAMETERS="$PARAMETERS ParameterKey=OtelDeploymentEnvironment,ParameterValue=$OTEL_DEPLOYMENT_ENVIRONMENT"
PARAMETERS="$PARAMETERS ParameterKey=OtelInstrumentationGenaiCaptureMessageContent,ParameterValue=$OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"

# Validate CloudFormation template
print_info "Validating CloudFormation template..."
if [ -f "validate-template.sh" ]; then
    if ! ./validate-template.sh 2>&1 | grep -q "ERROR"; then
        print_success "Template validation passed"
    else
        print_warning "Template validation found issues, but continuing..."
    fi
else
    print_warning "Validation script not found, skipping template validation"
fi

# Validate template with AWS CLI
print_info "Validating template with AWS CloudFormation..."
if aws cloudformation validate-template --template-body file://cloudformation-template.yaml --region "$REGION" &> /tmp/cf-validate.log; then
    print_success "AWS CloudFormation template validation passed"
else
    print_error "AWS CloudFormation template validation failed:"
    cat /tmp/cf-validate.log
    print_error "Please fix the template errors before deploying"
    exit 1
fi

# Check if stack exists
STACK_EXISTS=false
if aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" &> /dev/null; then
    STACK_EXISTS=true
    print_info "Stack $STACK_NAME already exists"
else
    print_info "Stack $STACK_NAME does not exist, will create new stack"
fi

# Prepare CloudFormation command
if [[ "$STACK_EXISTS" == true ]]; then
    COMMAND="update-stack"
    ACTION="update"
else
    COMMAND="create-stack"
    ACTION="create"
fi

CF_COMMAND="aws cloudformation $COMMAND --stack-name $STACK_NAME --template-body file://cloudformation-template.yaml --parameters $PARAMETERS --capabilities CAPABILITY_IAM --region $REGION"

if [[ "$DRY_RUN" == true ]]; then
    print_info "DRY RUN: Would execute the following command:"
    echo "$CF_COMMAND"
    echo ""
    print_info "This would $ACTION the CloudFormation stack with the above parameters."
    exit 0
fi

# Deploy the stack
print_info "Deploying CloudFormation stack..."

if eval "$CF_COMMAND"; then
    print_success "CloudFormation stack $ACTION initiated successfully"

    print_info "Waiting for stack $ACTION to complete..."
    if aws cloudformation wait "stack-${ACTION}-complete" --stack-name "$STACK_NAME" --region "$REGION"; then
        print_success "Stack $ACTION completed successfully!"

        # Get stack outputs
        print_info "Retrieving stack outputs..."
        aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query 'Stacks[0].Outputs' --output table

        # Get the ALB URL
        ALB_URL=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query 'Stacks[0].Outputs[?OutputKey==`ALBURL`].OutputValue' --output text)
        if [[ -n "$ALB_URL" ]]; then
            print_success "Application is available at: $ALB_URL"
        fi

        # Get ECR repository URI
        ECR_URI=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' --output text)
        if [[ -n "$ECR_URI" ]]; then
            print_info "ECR Repository URI: $ECR_URI"
            echo ""
            print_info "To build and push your Docker image:"
            echo "  # Build the image (includes spaCy model download)"
            echo "  # Note: Run this from the project root directory"
            echo "  docker build -t $ECR_URI:latest -f deploy/docker/Dockerfile ."
            echo ""
            echo "  # Login to ECR"
            echo "  aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI"
            echo ""
            echo "  # Push the image"
            echo "  docker push $ECR_URI:latest"
            echo ""
            echo "  # Update the ECS service to use the new image"
            echo "  aws ecs update-service --cluster $PROJECT_NAME-$ENVIRONMENT-cluster --service $PROJECT_NAME-$ENVIRONMENT-service --force-new-deployment"
            echo ""
            print_info "Note: The Dockerfile automatically downloads the spaCy language model (en_core_web_sm) required for SEO keywords engine operations."
        fi

    else
        print_error "Stack $ACTION failed or timed out"
        print_info "Check the CloudFormation console for details: https://console.aws.amazon.com/cloudformation/home?region=$REGION#/stacks"
        exit 1
    fi
else
    print_error "Failed to $ACTION CloudFormation stack"
    exit 1
fi

print_success "Deployment completed successfully!"
