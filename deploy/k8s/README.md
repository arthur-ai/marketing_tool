# Marketing Project Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Marketing Project API to a Kubernetes cluster.

## Files Overview

- `namespace.yaml` - Creates the marketing-project namespace
- `configmap.yaml` - Configuration values for the application (includes Redis settings)
- `secret.yaml` - Production secrets (base64 encoded)
- `secret-template.yaml` - Template for secrets (copy and fill with actual values)
- `redis-deployment.yaml` - **[Phase 2 Required]** Redis deployment with persistence
- `deployment.yaml` - Main application deployment
- `worker-deployment.yaml` - **[Phase 2 Required]** ARQ worker deployment for background jobs
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

3. **Create ConfigMap (includes Redis configuration):**
   ```bash
   kubectl apply -f k8s/configmap.yaml
   ```

4. **Deploy Redis (REQUIRED - Phase 2):**
   ```bash
   kubectl apply -f k8s/redis-deployment.yaml

   # Wait for Redis to be ready
   kubectl wait --for=condition=ready pod -l app=redis -n marketing-project --timeout=120s
   ```

5. **Deploy the application:**
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   kubectl apply -f k8s/hpa.yaml
   ```

6. **Deploy ARQ Workers (REQUIRED - Phase 2):**
   ```bash
   kubectl apply -f k8s/worker-deployment.yaml
   ```

7. **Optional: Deploy CronJob:**
   ```bash
   kubectl apply -f k8s/cronjob.yaml
   ```

## Configuration

### Environment Variables

The application uses the following environment variables:

**Required (Phase 2+):**
- `REDIS_HOST` - Redis hostname (set to "redis-service" in ConfigMap)
- `REDIS_PORT` - Redis port (default: 6379)
- `REDIS_DATABASE` - Redis database number (default: 0)

**Optional:**
- `OPENAI_API_KEY` - OpenAI API key for AI processing
- `CONTENT_API_KEY` - External content API key
- `TEMPLATE_VERSION` - Version of prompts to use (default: v1)
- `LOG_LEVEL` - Logging level (default: INFO)
- `DEBUG` - Debug mode (default: false)
- `ARQ_MAX_JOBS` - Max concurrent jobs per worker (default: 10)
- `ARQ_JOB_TIMEOUT` - Job timeout in seconds (default: 600)
- `ARQ_WORKER_COUNT` - Number of worker replicas (default: 2)

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
   # API logs
   kubectl logs -f deployment/marketing-project-api -n marketing-project

   # Worker logs
   kubectl logs -f deployment/marketing-project-worker -n marketing-project

   # Redis logs
   kubectl logs -f deployment/redis -n marketing-project
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
