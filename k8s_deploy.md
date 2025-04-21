# Kubernetes Deployment Guide

This guide explains how to deploy the Multi-Agent AI System on a Kubernetes cluster for large-scale operation.

## Prerequisites

- Kubernetes cluster (e.g., Google Kubernetes Engine, Amazon EKS, Azure AKS, or local with minikube/kind)
- `kubectl` CLI configured to access your cluster
- Docker installed locally
- Container registry access (e.g., Docker Hub, GCR, ECR)

## Deployment Steps

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t multi-agent-system:latest .

# Tag for your registry
docker tag multi-agent-system:latest your-registry/multi-agent-system:latest

# Push to registry
docker push your-registry/multi-agent-system:latest
```

### 2. Update Image References

Edit the following files to use your image repository:

- `kubernetes/agent-deployment.yaml`
- `kubernetes/scaling-agents.yaml`

Replace `multi-agent-system:latest` with `your-registry/multi-agent-system:latest`.

### 3. Create Persistent Volumes

```bash
# Create persistent volume claims
kubectl apply -f kubernetes/pvc.yaml
```

### 4. Deploy Redis for Job Queue

```bash
# Deploy Redis
kubectl apply -f kubernetes/redis.yaml
```

### 5. Deploy the Main Application

```bash
# Deploy main application
kubectl apply -f kubernetes/agent-deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### 6. Deploy Agent Workers for Scaling

```bash
# Deploy agent workers
kubectl apply -f kubernetes/scaling-agents.yaml
```

### 7. Check Deployment Status

```bash
# Check pods
kubectl get pods

# Check services
kubectl get services
```

### 8. Access the UI

```bash
# Forward port (if not using LoadBalancer or Ingress)
kubectl port-forward svc/multi-agent-system 8080:80
```

Then access http://localhost:8080 in your browser.

## Horizontal Scaling

The system automatically scales agent workers based on CPU utilization. By default, it scales from 1 to 50 workers, which can support many concurrent users.

To modify scaling parameters:

```bash
# Edit HPA
kubectl edit hpa agent-worker-hpa
```

## Model Management

Models are stored on persistent volumes. To add or update models:

1. Create a job to download models:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-download
spec:
  template:
    spec:
      containers:
      - name: model-downloader
        image: curlimages/curl
        command: ["/bin/sh", "-c"]
        args:
        - |
          curl -L "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" -o "/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        volumeMounts:
        - name: models-volume
          mountPath: /models
      restartPolicy: OnFailure
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
```

2. Apply this job:

```bash
kubectl apply -f model-download-job.yaml
```

## Production Considerations

For production deployments:

1. **Resource Allocation**: Adjust CPU/memory in deployment files based on model size
2. **Security**: Add appropriate authentication and TLS for API endpoints
3. **Monitoring**: Set up Prometheus and Grafana for system monitoring
4. **Backup**: Implement backup for the persistent volumes
5. **High Availability**: Deploy multiple replicas across availability zones

## Cleanup

To remove all resources:

```bash
kubectl delete -f kubernetes/scaling-agents.yaml
kubectl delete -f kubernetes/service.yaml
kubectl delete -f kubernetes/agent-deployment.yaml
kubectl delete -f kubernetes/redis.yaml
kubectl delete -f kubernetes/pvc.yaml
```

## Troubleshooting

Common issues and solutions:

### Pods Not Starting
Check for resource issues:
```bash
kubectl describe pod <pod-name>
```

### Models Not Found
Check persistent volume:
```bash
kubectl exec -it <pod-name> -- ls -la /app/models
```

### Redis Connection Issues
Verify Redis service:
```bash
kubectl get svc redis-service
kubectl exec -it <pod-name> -- ping redis-service
``` 