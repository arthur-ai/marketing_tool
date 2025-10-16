# Security Guide

This document provides comprehensive information about the security features and best practices for the Marketing Project API.

## 🔒 Security Overview

The Marketing Project API implements enterprise-grade security features to protect against common web vulnerabilities and ensure secure operation in production environments.

## 🛡️ Security Features

### **1. Authentication & Authorization**

#### API Key Authentication
- **Format**: 32+ character alphanumeric keys
- **Storage**: Environment variables or Kubernetes secrets
- **Validation**: Real-time format and strength validation
- **Roles**: Admin, User, Viewer with different permission levels

```bash
# Example API key configuration
API_KEY=your-secure-api-key-32-chars-minimum
API_KEY_1=admin-key-with-admin-role
API_KEY_1_ROLE=admin
API_KEY_2=user-key-with-user-role
API_KEY_2_ROLE=user
```

#### Role-Based Access Control (RBAC)
- **Admin**: Full access to all endpoints and operations
- **User**: Access to content analysis and pipeline operations
- **Viewer**: Read-only access to content sources and status

### **2. Rate Limiting & Attack Prevention**

#### Advanced Rate Limiting
- **Per-IP Limiting**: Configurable requests per minute per IP
- **Per-User Limiting**: User-based rate limiting
- **Burst Protection**: Configurable burst limits
- **IP Whitelisting**: Trusted IP bypass for rate limits

```bash
# Rate limiting configuration
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
RATE_LIMIT_BURST_LIMIT=100
SECURITY_IP_WHITELIST=127.0.0.1,::1,192.168.1.0/24
```

#### Attack Detection
- **Rapid-Fire Detection**: Identifies rapid consecutive requests
- **Suspicious Pattern Detection**: Detects unusual access patterns
- **Automatic Blocking**: Temporary IP blocking for suspicious activity
- **Risk Scoring**: Dynamic risk assessment for requests

### **3. Input Validation & Sanitization**

#### SQL Injection Prevention
- **Pattern Detection**: Identifies SQL injection attempts
- **Query Sanitization**: Automatic query parameter sanitization
- **Parameterized Queries**: Enforced use of parameterized queries
- **Database-Specific Validation**: Different validation for SQL, MongoDB, Redis

#### XSS (Cross-Site Scripting) Protection
- **Script Detection**: Identifies and blocks script injection
- **Content Sanitization**: Automatic HTML content sanitization
- **Output Encoding**: Proper output encoding for all responses
- **CSP Headers**: Content Security Policy headers

#### Command Injection Prevention
- **Shell Command Detection**: Identifies shell command injection
- **Command Sanitization**: Sanitizes command parameters
- **Restricted Execution**: Limits command execution capabilities
- **Audit Logging**: Logs all command execution attempts

### **4. Security Monitoring & Auditing**

#### Comprehensive Audit Logging
- **Authentication Events**: Login attempts, API key usage
- **Authorization Events**: Permission checks, access denials
- **Input Validation Events**: Validation failures, attack attempts
- **Rate Limiting Events**: Rate limit violations, IP blocks
- **System Events**: Configuration changes, security updates

#### Security Event Types
```python
class SecurityEventType(Enum):
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_SUCCESS = "authz_success"
    AUTHORIZATION_FAILURE = "authz_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    ATTACK_DETECTED = "attack_detected"
    IP_BLOCKED = "ip_blocked"
    SECURITY_ALERT = "security_alert"
```

#### Risk Scoring
- **Dynamic Assessment**: Real-time risk calculation
- **Multi-Factor Analysis**: Considers multiple security factors
- **Threshold-Based Actions**: Automatic responses to high-risk events
- **Historical Analysis**: Learns from past security events

## 🔧 Security Configuration

### **Environment Variables**

#### Core Security Settings
```bash
# Security audit
SECURITY_AUDIT_ENABLED=true
SECURITY_AUDIT_LOG_FILE=logs/security_audit.log

# Rate limiting
SECURITY_RATE_LIMIT_ENABLED=true
SECURITY_ATTACK_DETECTION_ENABLED=true
SECURITY_IP_WHITELIST=127.0.0.1,::1
SECURITY_BLOCK_DURATION=300
SECURITY_SUSPICIOUS_THRESHOLD=5

# API authentication
API_KEY=your-secure-api-key
API_KEY_1=admin-key
API_KEY_1_ROLE=admin
```

#### Database Security
```bash
# Database connections with security
POSTGRES_URL=postgresql://user:password@localhost:5432/db
MONGODB_URL=mongodb://user:password@localhost:27017/db
REDIS_URL=redis://:password@localhost:6379/0
```

### **Kubernetes Security**

#### Secret Management
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: marketing-project-secrets
type: Opaque
data:
  API_KEY: <base64-encoded-api-key>
  OPENAI_API_KEY: <base64-encoded-openai-key>
  POSTGRES_PASSWORD: <base64-encoded-password>
```

#### Security Context
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
      - ALL
```

## 🚨 Security Best Practices

### **1. API Key Management**
- Use strong, randomly generated API keys (32+ characters)
- Rotate API keys regularly
- Store keys securely (environment variables, secrets management)
- Monitor API key usage and revoke compromised keys

### **2. Network Security**
- Use HTTPS in production
- Implement proper CORS policies
- Use trusted host middleware
- Consider VPN or private networks for sensitive deployments

### **3. Database Security**
- Use encrypted connections (SSL/TLS)
- Implement proper access controls
- Regular security updates
- Monitor database access logs

### **4. Monitoring & Alerting**
- Set up security event monitoring
- Configure alerts for suspicious activity
- Regular security log review
- Implement incident response procedures

### **5. Input Validation**
- Validate all input data
- Use parameterized queries
- Implement proper error handling
- Sanitize user-generated content

## 🔍 Security Testing

### **Automated Security Tests**
```bash
# Run security tests
python test_security_and_database.py

# Run with specific security focus
pytest tests/ -k security -v

# Run security linting
bandit -r src/ -f json -o security-report.json
```

### **Manual Security Testing**
```bash
# Test API authentication
curl -H "X-API-Key: invalid-key" http://localhost:8000/api/v1/analyze

# Test rate limiting
for i in {1..200}; do curl http://localhost:8000/api/v1/health; done

# Test input validation
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"content": {"id": "test", "title": "<script>alert(1)</script>", "content": "test", "type": "blog_post"}}'
```

## 📊 Security Monitoring

### **Key Metrics to Monitor**
- Authentication success/failure rates
- Rate limit violations
- Input validation failures
- Attack detection events
- IP blocking events
- Security risk scores

### **Alert Thresholds**
- High authentication failure rate (>10% in 5 minutes)
- Multiple rate limit violations from same IP
- Detection of attack patterns
- High security risk scores
- Unusual access patterns

## 🛠️ Incident Response

### **Security Incident Response Plan**
1. **Detection**: Monitor security events and alerts
2. **Assessment**: Evaluate the severity and impact
3. **Containment**: Isolate affected systems
4. **Investigation**: Analyze logs and determine root cause
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Update security measures

### **Emergency Contacts**
- Security Team: security@company.com
- DevOps Team: devops@company.com
- Management: management@company.com

## 📚 Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

## 🔄 Security Updates

This security guide is regularly updated to reflect new threats and best practices. Please check for updates regularly and ensure your deployment follows the latest recommendations.

For security-related questions or to report vulnerabilities, please contact the security team at security@company.com.
