# Marketing Project Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Marketing Project API to a Kubernetes cluster.

## Files Overview

- `namespace.yaml` - Creates the marketing-project namespace
- `configmap.yaml` - Configuration values for the application
- `secret.yaml` - Production secrets (base64 encoded)
- `secret-template.yaml` - Template for secrets (copy and fill with actual values)
- `deployment.yaml` - Main application deployment
- `service.yaml` - Service to expose the application
- `ingress.yaml` - Ingress for external access with TLS
- `hpa.yaml` - Horizontal Pod Autoscaler for automatic scaling
- `cronjob.yaml` - CronJob to trigger the marketing pipeline
- `kustomization.yaml` - Kustomize configuration for environment management

## Quick Start

### Option 1: Using Kustomize (Recommended)

```bash
# Deploy everything at once
kubectl apply -k k8s/

# Or deploy to a specific environment
kubectl apply -k k8s/ --namespace=marketing-project-staging
```

### Option 2: Manual Deployment

1. **Create the namespace:**
   ```bash
   kubectl apply -f k8s/namespace.yaml
   ```

2. **Create secrets:**
   ```bash
   # Copy the template and fill with actual values
   cp k8s/secret-template.yaml k8s/secret-custom.yaml
   # Edit secret-custom.yaml with your actual base64-encoded secrets
   kubectl apply -f k8s/secret-custom.yaml
   ```

3. **Deploy the application:**
   ```bash
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   kubectl apply -f k8s/hpa.yaml
   kubectl apply -f k8s/cronjob.yaml
   ```

## Configuration

### Environment Variables

The application uses the following environment variables:

- `API_KEY` - Main API key for authentication
- `API_KEY_1` - Admin API key
- `API_KEY_2` - User API key
- `OPENAI_API_KEY` - OpenAI API key for AI processing
- `TEMPLATE_VERSION` - Version of prompts to use (default: v1)
- `LOG_LEVEL` - Logging level (default: INFO)
- `DEBUG` - Debug mode (default: false)

### Resource Requirements

- **CPU:** 250m request, 500m limit
- **Memory:** 256Mi request, 512Mi limit

### Scaling

The HPA is configured to:
- Scale based on CPU (70% utilization) and memory (80% utilization)
- Scale from 3 to 10 replicas
- Use conservative scaling policies

## API Endpoints

The deployment exposes the following endpoints:

- `GET /api/v1/health` - Health check endpoint
- `GET /api/v1/ready` - Readiness check endpoint
- `POST /api/v1/analyze` - Content analysis endpoint
- `POST /api/v1/pipeline` - Marketing pipeline endpoint
- `GET /api/v1/content-sources` - Content sources management
- `GET /docs` - API documentation (Swagger UI)

## Monitoring

The deployment includes:
- Liveness and readiness probes
- Resource limits and requests
- Health check endpoints
- Performance monitoring middleware
- Security audit logging

## Security

- API key authentication
- Rate limiting and attack detection
- Input validation and sanitization
- Secrets stored in Kubernetes secrets
- TLS termination at the ingress level
- Resource limits prevent resource exhaustion
- Read-only secret mounts

## Troubleshooting

1. **Check pod status:**
   ```bash
   kubectl get pods -n marketing-project
   ```

2. **View logs:**
   ```bash
   kubectl logs -f deployment/marketing-project-api -n marketing-project
   ```

3. **Check events:**
   ```bash
   kubectl get events -n marketing-project
   ```

4. **Test the service:**
   ```bash
   kubectl port-forward service/marketing-project-service 8000:80 -n marketing-project
   curl http://localhost:8000/api/v1/health
   ```

5. **Check HPA status:**
   ```bash
   kubectl get hpa -n marketing-project
   ```

6. **Check CronJob status:**
   ```bash
   kubectl get cronjobs -n marketing-project
   kubectl get jobs -n marketing-project
   ```

## Environment-Specific Deployments

### Development
```bash
kubectl apply -k k8s/ --namespace=marketing-project-dev
```

### Staging
```bash
kubectl apply -k k8s/ --namespace=marketing-project-staging
```

### Production
```bash
kubectl apply -k k8s/ --namespace=marketing-project
```

## Updating the Deployment

1. **Update image:**
   ```bash
   kubectl set image deployment/marketing-project-api api=marketing-project-api:v2.0.0 -n marketing-project
   ```

2. **Rolling update:**
   ```bash
   kubectl rollout restart deployment/marketing-project-api -n marketing-project
   ```

3. **Check rollout status:**
   ```bash
   kubectl rollout status deployment/marketing-project-api -n marketing-project
   ```

## Cleanup

To remove the deployment:

```bash
kubectl delete -k k8s/
# or
kubectl delete namespace marketing-project
```
