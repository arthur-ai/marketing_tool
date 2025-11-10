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
ENVIRONMENT="production"
PROJECT_NAME="marketing-tool"
REGION="us-east-1"
STACK_NAME=""
DOMAIN_NAME=""
CERTIFICATE_ARN=""
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
    -e, --environment ENV     Environment (development|staging|production) [default: production]
    -n, --project-name NAME   Project name [default: marketing-tool]
    -r, --region REGION       AWS region [default: us-east-1]
    -s, --stack-name NAME     CloudFormation stack name [default: marketing-tool-ENV]
    -d, --domain DOMAIN       Domain name for the application (optional)
    -c, --certificate ARN     ACM certificate ARN for HTTPS (optional)
    -f, --force               Force update even if no changes detected
    --dry-run                 Show what would be deployed without actually deploying
    -h, --help                Show this help message

REQUIRED ENVIRONMENT VARIABLES:
    OPENAI_API_KEY           OpenAI API key
    API_KEY                  API authentication key (32+ characters)
    DATABASE_PASSWORD        Database master password (8+ characters)

OPTIONAL ENVIRONMENT VARIABLES:
    ENABLE_MONGODB           Enable MongoDB/DocumentDB (true/false, default: false)
    MONGODB_PASSWORD         MongoDB password (8+ characters, required only if ENABLE_MONGODB=true)

EXAMPLES:
    # Deploy to production
    $0 -e production -r us-west-2

    # Deploy with custom domain
    $0 -e production -d api.mycompany.com -c arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012

    # Dry run to see what would be deployed
    $0 --dry-run -e staging

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
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be one of: development, staging, production"
    exit 1
fi

# Check required environment variables
print_info "Checking required environment variables..."

required_vars=("OPENAI_API_KEY" "API_KEY" "DATABASE_PASSWORD")
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

# Validate password length
if [[ ${#DATABASE_PASSWORD} -lt 8 ]]; then
    print_error "DATABASE_PASSWORD must be at least 8 characters long"
    exit 1
fi

# Check MongoDB configuration
ENABLE_MONGODB="${ENABLE_MONGODB:-false}"
if [[ "$ENABLE_MONGODB" == "true" ]]; then
    if [[ -z "$MONGODB_PASSWORD" ]]; then
        print_error "MONGODB_PASSWORD is required when ENABLE_MONGODB=true"
        exit 1
    fi
    if [[ ${#MONGODB_PASSWORD} -lt 8 ]]; then
        print_error "MONGODB_PASSWORD must be at least 8 characters long"
        exit 1
    fi
    print_info "MongoDB will be enabled"
else
    print_info "MongoDB will be disabled (set ENABLE_MONGODB=true to enable)"
fi

print_success "All required environment variables are set"

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
echo "  Domain: ${DOMAIN_NAME:-'Not specified'}"
echo "  Certificate: ${CERTIFICATE_ARN:-'Not specified'}"
echo "  Dry Run: $DRY_RUN"
echo ""

# Build CloudFormation parameters
PARAMETERS="ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
PARAMETERS="$PARAMETERS ParameterKey=ProjectName,ParameterValue=$PROJECT_NAME"
PARAMETERS="$PARAMETERS ParameterKey=OpenAIApiKey,ParameterValue=$OPENAI_API_KEY"
PARAMETERS="$PARAMETERS ParameterKey=ApiKey,ParameterValue=$API_KEY"
PARAMETERS="$PARAMETERS ParameterKey=DatabasePassword,ParameterValue=$DATABASE_PASSWORD"
PARAMETERS="$PARAMETERS ParameterKey=EnableMongoDB,ParameterValue=$ENABLE_MONGODB"

# Add MongoDB password only if MongoDB is enabled
# Note: We don't pass the parameter at all when disabled, so CloudFormation uses the default empty string
if [[ "$ENABLE_MONGODB" == "true" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=MongoDBPassword,ParameterValue=$MONGODB_PASSWORD"
fi

if [[ -n "$DOMAIN_NAME" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=DomainName,ParameterValue=$DOMAIN_NAME"
fi

if [[ -n "$CERTIFICATE_ARN" ]]; then
    PARAMETERS="$PARAMETERS ParameterKey=CertificateArn,ParameterValue=$CERTIFICATE_ARN"
fi

# Check if stack exists
STACK_EXISTS=false
if aws cloudformation describe-stacks --stack-name "$STACK_NAME" &> /dev/null; then
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

CF_COMMAND="aws cloudformation $COMMAND --stack-name $STACK_NAME --template-body file://cloudformation-template.yaml --parameters $PARAMETERS --capabilities CAPABILITY_IAM"

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
    if aws cloudformation wait "stack-${ACTION}-complete" --stack-name "$STACK_NAME"; then
        print_success "Stack $ACTION completed successfully!"

        # Get stack outputs
        print_info "Retrieving stack outputs..."
        aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query 'Stacks[0].Outputs' --output table

        # Get the ALB URL
        ALB_URL=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query 'Stacks[0].Outputs[?OutputKey==`ALBURL`].OutputValue' --output text)
        if [[ -n "$ALB_URL" ]]; then
            print_success "Application is available at: $ALB_URL"
        fi

        # Get ECR repository URI
        ECR_URI=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' --output text)
        if [[ -n "$ECR_URI" ]]; then
            print_info "ECR Repository URI: $ECR_URI"
            echo ""
            print_info "To build and push your Docker image:"
            echo "  # Build the image (using Uvicorn)"
            echo "  docker build -t $ECR_URI:latest ."
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
            print_info "Note: This deployment uses Uvicorn as the ASGI server, optimized for FastAPI applications."
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
