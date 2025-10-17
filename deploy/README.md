# Deployment Guide

This directory contains all deployment configurations and scripts for the Marketing Tool.

## Directory Structure

```
deploy/
├── README.md                    # This file
├── docker/                      # Docker Compose configurations
│   ├── README.md               # Docker setup guide
│   ├── docker-compose.yml      # Main compose file (API only)
│   ├── docker-compose.postgres.yml
│   ├── docker-compose.mongodb.yml
│   ├── docker-compose.redis.yml
│   ├── docker-compose.full.yml
│   └── nginx/                  # Nginx configuration
├── aws/                        # AWS CloudFormation deployment
│   ├── cloudformation-template.yaml
│   ├── cloudformation-parameters.json
│   ├── deploy.sh
│   └── deploy-aws.sh
└── k8s/                        # Kubernetes configurations
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    └── ...
```

## Quick Start

### Development (Docker)
```bash
cd deploy/docker
docker-compose up
```

### Production (AWS)
```bash
cd deploy/aws
./deploy.sh -e production
```

### Kubernetes
```bash
kubectl apply -f deploy/k8s/
```

## Deployment Options

1. **Docker Compose** - For development and local testing
2. **AWS CloudFormation** - For production deployment on AWS
3. **Kubernetes** - For container orchestration

See individual README files in each subdirectory for detailed instructions.
