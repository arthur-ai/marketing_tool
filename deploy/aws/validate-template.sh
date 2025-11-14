#!/bin/bash

# CloudFormation Template Validation Script
# Provides multiple validation methods for the CloudFormation template

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TEMPLATE_FILE="${1:-cloudformation-template.yaml}"
REGION="${2:-us-east-2}"
AWS_PROFILE="${3:-}"

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

# Show usage if help requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [TEMPLATE_FILE] [REGION] [AWS_PROFILE]"
    echo ""
    echo "Arguments:"
    echo "  TEMPLATE_FILE    CloudFormation template file (default: cloudformation-template.yaml)"
    echo "  REGION          AWS region (default: us-east-2)"
    echo "  AWS_PROFILE     AWS profile to use (optional, e.g., sandbox)"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 cloudformation-template.yaml us-east-2"
    echo "  $0 cloudformation-template.yaml us-east-2 sandbox"
    exit 0
fi

# Check if template file exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    print_error "Template file not found: $TEMPLATE_FILE"
    exit 1
fi

print_info "Validating CloudFormation template: $TEMPLATE_FILE"
echo ""

# Method 1: AWS CLI validate-template (most reliable)
print_info "Method 1: AWS CloudFormation validate-template"
AWS_CMD="aws"
if [ -n "$AWS_PROFILE" ]; then
    AWS_CMD="aws --profile $AWS_PROFILE"
    print_info "Using AWS profile: $AWS_PROFILE"
fi

if $AWS_CMD cloudformation validate-template \
    --template-body "file://$TEMPLATE_FILE" \
    --region "$REGION" &> /tmp/cf-validate.log; then
    print_success "AWS CloudFormation validation passed"
else
    print_error "AWS CloudFormation validation failed:"
    cat /tmp/cf-validate.log
    echo ""
fi

# Method 2: cfn-lint (if installed)
print_info "Method 2: cfn-lint (CloudFormation Linter)"
if command -v cfn-lint &> /dev/null; then
    if cfn-lint "$TEMPLATE_FILE" 2>&1 | tee /tmp/cfn-lint.log; then
        print_success "cfn-lint validation passed"
    else
        print_warning "cfn-lint found issues (see output above)"
    fi
    echo ""
else
    print_warning "cfn-lint not installed. Install with: pip install cfn-lint"
    echo ""
fi

# Method 3: Check for common YAML syntax issues
print_info "Method 3: YAML syntax check"
if command -v yamllint &> /dev/null; then
    if yamllint "$TEMPLATE_FILE" &> /tmp/yamllint.log; then
        print_success "YAML syntax validation passed"
    else
        print_warning "YAML syntax issues found:"
        cat /tmp/yamllint.log
    fi
    echo ""
elif command -v python3 &> /dev/null; then
    # Basic YAML parsing check with Python
    if python3 -c "import yaml; yaml.safe_load(open('$TEMPLATE_FILE'))" 2>&1; then
        print_success "YAML syntax validation passed (basic check)"
    else
        print_error "YAML syntax validation failed"
    fi
    echo ""
fi

# Method 4: Check for required sections
print_info "Method 4: Template structure validation"
MISSING_SECTIONS=()

if ! grep -q "^AWSTemplateFormatVersion:" "$TEMPLATE_FILE"; then
    MISSING_SECTIONS+=("AWSTemplateFormatVersion")
fi

if ! grep -q "^Parameters:" "$TEMPLATE_FILE"; then
    MISSING_SECTIONS+=("Parameters")
fi

if ! grep -q "^Resources:" "$TEMPLATE_FILE"; then
    MISSING_SECTIONS+=("Resources")
fi

if [ ${#MISSING_SECTIONS[@]} -eq 0 ]; then
    print_success "Required template sections found"
else
    print_error "Missing required sections: ${MISSING_SECTIONS[*]}"
fi
echo ""

# Method 5: Check for common CloudFormation issues
print_info "Method 5: Common CloudFormation pattern checks"

ISSUES_FOUND=0

# Check for Fn::If syntax issues
if grep -q "Fn::If:" "$TEMPLATE_FILE"; then
    print_warning "Found 'Fn::If:' syntax - ensure it's used correctly (consider using !If shorthand)"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

# Check for undefined conditions in !If statements
# Extract conditions from Conditions section
CONDITIONS=$(awk '/^Conditions:/{flag=1; next} /^[A-Z]/{flag=0} flag && /^  [A-Za-z]+:/{print $1}' "$TEMPLATE_FILE" | sed 's/://g')
USED_CONDITIONS=$(grep -oE "!If \[([A-Za-z]+)" "$TEMPLATE_FILE" | sed 's/!If \[//' | sort -u)

for cond in $USED_CONDITIONS; do
    if ! echo "$CONDITIONS" | grep -q "^$cond$"; then
        print_warning "Condition '$cond' used in !If but not defined in Conditions section"
        ISSUES_FOUND=$((ISSUES_FOUND + 1))
    fi
done

if [ $ISSUES_FOUND -eq 0 ]; then
    print_success "Common pattern checks passed"
fi
echo ""

# Summary
print_info "Validation complete!"
print_info "For best results, ensure AWS CLI is configured and cfn-lint is installed:"
print_info "  pip install cfn-lint"
print_info "  yamllint (optional): pip install yamllint"
