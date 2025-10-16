# Performance Guide

This document provides comprehensive information about the performance features, monitoring, and optimization strategies for the Marketing Project API.

## 🚀 Performance Overview

The Marketing Project API is designed for high performance with built-in monitoring, caching, and optimization features to ensure optimal performance in production environments.

## 📊 Performance Features

### **1. Real-time Performance Monitoring**

#### Metrics Collection
- **Request/Response Times**: Detailed timing for all API requests
- **Memory Usage**: Real-time memory consumption tracking
- **CPU Usage**: CPU utilization monitoring
- **Request Size**: Request and response size tracking
- **Error Rates**: Success and failure rate monitoring

#### Performance Metrics
```python
@dataclass
class PerformanceMetrics:
    timestamp: datetime
    endpoint: str
    method: str
    response_time: float
    status_code: int
    memory_usage: float
    cpu_usage: float
    request_size: int
    response_size: int
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
```

#### Monitoring Endpoints
- **Performance Summary**: `/api/v1/performance/summary`
- **Endpoint Statistics**: `/api/v1/performance/endpoints`
- **Slow Requests**: `/api/v1/performance/slow-requests`
- **Error Requests**: `/api/v1/performance/error-requests`

### **2. Intelligent Caching System**

#### Cache Manager
- **LRU Eviction**: Least Recently Used eviction policy
- **TTL Support**: Time-to-live for cache entries
- **Size Limits**: Configurable cache size limits
- **Hit Rate Monitoring**: Cache performance metrics

```python
# Cache configuration
PERFORMANCE_CACHE_ENABLED=true
PERFORMANCE_CACHE_TTL=300
PERFORMANCE_CACHE_MAX_SIZE=1000
```

#### Cache Statistics
- **Hit Rate**: Percentage of cache hits
- **Miss Rate**: Percentage of cache misses
- **Size**: Current cache size
- **Evictions**: Number of evicted entries

### **3. Database Connection Pooling**

#### Connection Pool Features
- **Pool Management**: Automatic connection pool management
- **Health Checks**: Regular connection health monitoring
- **Timeout Handling**: Configurable connection timeouts
- **Retry Logic**: Automatic retry for failed connections

```python
# Connection pool configuration
DATABASE_POOL_ENABLED=true
DATABASE_POOL_MAX_CONNECTIONS=10
DATABASE_POOL_MIN_CONNECTIONS=2
DATABASE_POOL_TIMEOUT=300
DATABASE_POOL_RETRY_ATTEMPTS=3
```

#### Supported Databases
- **PostgreSQL**: Full connection pooling support
- **MongoDB**: Connection pool with health monitoring
- **Redis**: Connection pool with failover
- **SQLite**: File-based connection management

### **4. Query Optimization**

#### SQL Query Optimization
- **Query Analysis**: Automatic query complexity analysis
- **Index Suggestions**: Recommended database indexes
- **Query Caching**: Automatic query result caching
- **Parameter Optimization**: Query parameter optimization

#### MongoDB Query Optimization
- **Query Planning**: MongoDB query execution planning
- **Index Usage**: Index utilization analysis
- **Aggregation Optimization**: Pipeline optimization
- **Projection Optimization**: Field selection optimization

### **5. Load Testing Framework**

#### Test Types
- **Basic Load Test**: Standard load testing
- **Stress Test**: High-load stress testing
- **Spike Test**: Sudden load spike testing
- **Endurance Test**: Long-duration testing
- **Comprehensive Test**: Full feature testing

#### Load Test Configuration
```python
@dataclass
class LoadTestConfig:
    base_url: str
    endpoints: List[Dict[str, Any]]
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    think_time_seconds: float = 1.0
    timeout_seconds: int = 30
```

## 🔧 Performance Configuration

### **Environment Variables**

#### Core Performance Settings
```bash
# Performance monitoring
PERFORMANCE_MONITORING_ENABLED=true
PERFORMANCE_METRICS_ENABLED=true
PERFORMANCE_CACHE_ENABLED=true
PERFORMANCE_CACHE_TTL=300
PERFORMANCE_CACHE_MAX_SIZE=1000

# Connection pooling
PERFORMANCE_CONNECTION_POOL_MAX_SIZE=10
PERFORMANCE_CONNECTION_POOL_MIN_SIZE=2
PERFORMANCE_QUERY_OPTIMIZATION_ENABLED=true

# Load testing
LOAD_TEST_ENABLED=false
LOAD_TEST_CONCURRENT_USERS=10
LOAD_TEST_DURATION_SECONDS=60
LOAD_TEST_RAMP_UP_SECONDS=10
```

#### Database Performance
```bash
# PostgreSQL
POSTGRES_URL=postgresql://user:password@localhost:5432/db
POSTGRES_POOL_SIZE=10

# MongoDB
MONGODB_URL=mongodb://user:password@localhost:27017/db
MONGODB_MAX_POOL_SIZE=10

# Redis
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=10
```

### **Kubernetes Performance**

#### Resource Limits
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

#### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: marketing-project-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: marketing-project-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📈 Performance Monitoring

### **Key Performance Indicators (KPIs)**

#### Response Time Metrics
- **Average Response Time**: Mean response time across all requests
- **95th Percentile**: 95% of requests complete within this time
- **99th Percentile**: 99% of requests complete within this time
- **Maximum Response Time**: Longest request processing time

#### Throughput Metrics
- **Requests Per Second**: API request throughput
- **Concurrent Users**: Number of simultaneous users
- **Peak Load**: Maximum load handled
- **Sustained Load**: Long-term load capacity

#### Resource Utilization
- **CPU Usage**: CPU utilization percentage
- **Memory Usage**: Memory consumption in MB
- **Database Connections**: Active database connections
- **Cache Hit Rate**: Cache effectiveness percentage

### **Performance Dashboards**

#### Real-time Monitoring
- **Live Metrics**: Real-time performance data
- **Request Tracking**: Individual request monitoring
- **Error Tracking**: Error rate and type monitoring
- **Resource Usage**: Live resource utilization

#### Historical Analysis
- **Trend Analysis**: Performance trends over time
- **Peak Analysis**: Peak load identification
- **Anomaly Detection**: Unusual performance patterns
- **Capacity Planning**: Future capacity requirements

## 🧪 Load Testing

### **Running Load Tests**

#### Basic Load Test
```bash
python run_load_test.py --url http://localhost:8000 --test basic
```

#### Stress Test
```bash
python run_load_test.py --url http://localhost:8000 --test stress
```

#### Comprehensive Test
```bash
python run_load_test.py --url http://localhost:8000 --test all --output results.json
```

#### Custom Load Test
```python
from marketing_project.performance.load_testing import LoadTestConfig, LoadTester

config = LoadTestConfig(
    base_url="http://localhost:8000",
    endpoints=[
        {
            "path": "/api/v1/analyze",
            "method": "POST",
            "headers": {"X-API-Key": "your-key"},
            "data": {"content": {"id": "test", "title": "Test", "content": "Test content", "type": "blog_post"}}
        }
    ],
    concurrent_users=50,
    duration_seconds=300
)

async with LoadTester(config) as tester:
    result = await tester.run_test()
    tester.print_summary()
```

### **Load Test Results**

#### Performance Metrics
- **Total Requests**: Number of requests processed
- **Successful Requests**: Number of successful requests
- **Failed Requests**: Number of failed requests
- **Error Rate**: Percentage of failed requests
- **Average Response Time**: Mean response time
- **95th Percentile**: 95% response time
- **Requests Per Second**: Throughput rate

#### Analysis
- **Performance Bottlenecks**: Identify slow components
- **Capacity Limits**: Determine maximum capacity
- **Error Patterns**: Analyze failure patterns
- **Optimization Opportunities**: Identify improvement areas

## 🔧 Performance Optimization

### **Database Optimization**

#### Query Optimization
- **Index Usage**: Ensure proper index utilization
- **Query Analysis**: Analyze slow queries
- **Connection Pooling**: Optimize connection management
- **Caching**: Implement query result caching

#### Database Configuration
```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

### **Application Optimization**

#### Code Optimization
- **Async/Await**: Use asynchronous programming
- **Connection Reuse**: Reuse database connections
- **Caching**: Implement intelligent caching
- **Memory Management**: Optimize memory usage

#### Configuration Optimization
```python
# FastAPI optimization
app = FastAPI(
    title="Marketing Project API",
    description="High-performance marketing API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)
```

### **Infrastructure Optimization**

#### Kubernetes Optimization
- **Resource Limits**: Set appropriate resource limits
- **Horizontal Scaling**: Configure HPA properly
- **Node Affinity**: Optimize pod placement
- **Network Policies**: Optimize network traffic

#### Docker Optimization
- **Multi-stage Builds**: Optimize image size
- **Layer Caching**: Optimize build times
- **Health Checks**: Implement proper health checks
- **Resource Limits**: Set container limits

## 📊 Performance Monitoring Tools

### **Built-in Monitoring**
- **Performance Dashboard**: Real-time performance metrics
- **API Metrics**: Endpoint-specific performance data
- **System Metrics**: Server resource utilization
- **Custom Metrics**: Application-specific metrics

### **External Monitoring**
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **ELK Stack**: Log analysis and monitoring

### **Monitoring Configuration**
```yaml
# Prometheus configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'marketing-project'
      static_configs:
      - targets: ['marketing-project-service:80']
```

## 🚨 Performance Alerts

### **Alert Thresholds**
- **High Response Time**: > 2 seconds average
- **High Error Rate**: > 5% error rate
- **High CPU Usage**: > 80% CPU utilization
- **High Memory Usage**: > 90% memory usage
- **Low Cache Hit Rate**: < 70% cache hit rate

### **Alert Configuration**
```yaml
# Grafana alert rules
groups:
- name: marketing-project
  rules:
  - alert: HighResponseTime
    expr: avg(response_time) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
```

## 📚 Performance Best Practices

### **1. Database Performance**
- Use appropriate indexes
- Optimize queries
- Implement connection pooling
- Monitor query performance

### **2. Caching Strategy**
- Cache frequently accessed data
- Use appropriate TTL values
- Monitor cache hit rates
- Implement cache invalidation

### **3. API Design**
- Use pagination for large datasets
- Implement proper error handling
- Use compression for large responses
- Optimize request/response formats

### **4. Infrastructure**
- Use appropriate resource limits
- Implement horizontal scaling
- Monitor resource utilization
- Plan for peak loads

### **5. Monitoring**
- Set up comprehensive monitoring
- Configure appropriate alerts
- Regular performance reviews
- Capacity planning

## 🔄 Performance Maintenance

### **Regular Tasks**
- **Performance Reviews**: Weekly performance analysis
- **Load Testing**: Monthly load testing
- **Capacity Planning**: Quarterly capacity assessment
- **Optimization**: Continuous optimization

### **Performance Checklist**
- [ ] Monitor key performance metrics
- [ ] Review and optimize slow queries
- [ ] Check cache hit rates
- [ ] Verify resource utilization
- [ ] Run load tests regularly
- [ ] Update performance baselines
- [ ] Review and update alerts

## 📞 Performance Support

For performance-related questions or issues:
- **Performance Team**: performance@company.com
- **DevOps Team**: devops@company.com
- **Documentation**: [Performance Guide](performance.md)

This performance guide is regularly updated to reflect new features and best practices. Please check for updates regularly and ensure your deployment follows the latest recommendations.
