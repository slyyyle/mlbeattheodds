# MLB Betting Analytics - Production Deployment Guide

This guide covers deploying the MLB betting analytics platform to production environments with proper security, monitoring, and scalability considerations.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for production)
- **Storage**: 50GB+ available disk space
- **CPU**: 4+ cores recommended

### Required Services
- **Database**: DuckDB (included) or PostgreSQL for larger deployments
- **Cache**: Redis (optional but recommended)
- **Web Server**: Nginx (for reverse proxy)
- **Process Manager**: systemd or Docker

### External Dependencies
- **The Odds API**: Active subscription with sufficient quota
- **Email Service**: SMTP server for notifications (optional)
- **Monitoring**: Sentry account for error tracking (optional)

## 🚀 Deployment Options

### Option 1: Traditional Server Deployment

#### 1. Server Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and system dependencies
sudo apt install -y python3.9 python3.9-pip python3.9-venv git nginx redis-server

# Create application user
sudo useradd -m -s /bin/bash mlbbetting
sudo usermod -aG sudo mlbbetting
```

#### 2. Application Setup
```bash
# Switch to application user
sudo su - mlbbetting

# Clone repository
git clone https://github.com/your-repo/mlbetting.git
cd mlbetting

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,features,results} logs models monitoring config
```

#### 3. Configuration
```bash
# Copy environment template
cp config/env_template.txt .env

# Edit configuration
nano .env
```

**Required Environment Variables:**
```bash
# The Odds API
ODDS_API_KEY=your_actual_api_key_here
ODDS_DAYS_AHEAD=7

# Database
DATABASE_PATH=/home/mlbbetting/mlbetting/data/warehouse.duckdb

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Security
SECRET_KEY=your_very_secure_secret_key_here
ALLOWED_HOSTS=your-domain.com,localhost

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
LOG_LEVEL=INFO

# Notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=alerts@your-domain.com

# Risk Management
INITIAL_BANKROLL=10000
MAX_DAILY_RISK=0.10
MIN_EDGE_THRESHOLD=0.02
```

#### 4. Database Initialization
```bash
# Initialize database
python src/storage/init_db.py

# Run initial data ingestion
python src/ingest/ingest_games.py
python src/ingest/ingest_odds.py

# Build initial features
python src/features/build_features.py

# Train initial models
python -c "
from src.ml.models import MLBBettingModel
import pandas as pd
import numpy as np

# Load features and train model
model = MLBBettingModel(model_type='ensemble')
# Add your training code here based on available data
print('Initial model training completed')
"
```

#### 5. Systemd Service Setup
```bash
# Create systemd service file
sudo nano /etc/systemd/system/mlbbetting-api.service
```

**Service Configuration:**
```ini
[Unit]
Description=MLB Betting Analytics API
After=network.target

[Service]
Type=exec
User=mlbbetting
Group=mlbbetting
WorkingDirectory=/home/mlbbetting/mlbetting
Environment=PATH=/home/mlbbetting/mlbetting/venv/bin
ExecStart=/home/mlbbetting/mlbetting/venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Automated Pipeline Service:**
```bash
sudo nano /etc/systemd/system/mlbbetting-pipeline.service
```

```ini
[Unit]
Description=MLB Betting Analytics Automated Pipeline
After=network.target

[Service]
Type=exec
User=mlbbetting
Group=mlbbetting
WorkingDirectory=/home/mlbbetting/mlbetting
Environment=PATH=/home/mlbbetting/mlbetting/venv/bin
ExecStart=/home/mlbbetting/mlbetting/venv/bin/python src/pipeline/automated_pipeline.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

#### 6. Nginx Configuration
```bash
sudo nano /etc/nginx/sites-available/mlbbetting
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # API proxy
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files (React build)
    location / {
        root /home/mlbbetting/mlbetting/ui/build;
        try_files $uri $uri/ /index.html;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

#### 7. SSL Certificate Setup
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Enable automatic renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

#### 8. Start Services
```bash
# Enable and start services
sudo systemctl enable nginx redis-server
sudo systemctl enable mlbbetting-api mlbbetting-pipeline

sudo systemctl start nginx redis-server
sudo systemctl start mlbbetting-api mlbbetting-pipeline

# Check status
sudo systemctl status mlbbetting-api
sudo systemctl status mlbbetting-pipeline
```

### Option 2: Docker Deployment

#### 1. Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/{raw,features,results} logs models monitoring

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Docker Compose
```yaml
version: '3.8'

services:
  mlbbetting-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_PATH=/app/data/warehouse.duckdb
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - redis
    restart: unless-stopped

  mlbbetting-pipeline:
    build: .
    command: python src/pipeline/automated_pipeline.py
    environment:
      - DATABASE_PATH=/app/data/warehouse.duckdb
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ui/build:/usr/share/nginx/html
      - /etc/letsencrypt:/etc/letsencrypt
    depends_on:
      - mlbbetting-api
    restart: unless-stopped

volumes:
  redis_data:
```

#### 3. Deploy with Docker
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f mlbbetting-api

# Scale API service
docker-compose up -d --scale mlbbetting-api=3
```

## 🔒 Security Considerations

### 1. API Security
```python
# Add to src/api/app.py
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

# Protect sensitive endpoints
@app.get("/admin/metrics")
async def get_admin_metrics(token: dict = Depends(verify_token)):
    # Admin-only endpoint
    pass
```

### 2. Environment Security
```bash
# Set proper file permissions
chmod 600 .env
chmod 700 data/ logs/ models/

# Use a dedicated user
sudo useradd -r -s /bin/false mlbbetting-service

# Firewall configuration
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable
```

### 3. Database Security
```python
# Add to database configuration
import os
from cryptography.fernet import Fernet

# Encrypt sensitive data
def encrypt_sensitive_data(data: str) -> str:
    key = os.getenv('ENCRYPTION_KEY').encode()
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()
```

## 📊 Monitoring and Alerting

### 1. Application Monitoring
```python
# Add to main application
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv('SENTRY_DSN'),
    integrations=[FastApiIntegration(auto_enable=False)],
    traces_sample_rate=0.1,
)
```

### 2. System Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Set up log rotation
sudo nano /etc/logrotate.d/mlbbetting
```

```
/home/mlbbetting/mlbetting/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 mlbbetting mlbbetting
}
```

### 3. Health Checks
```python
# Add comprehensive health check
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Database check
    try:
        con = duckdb.connect(DATABASE_PATH)
        con.execute("SELECT 1").fetchone()
        con.close()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {e}"
        health_status["status"] = "unhealthy"
    
    # API check
    try:
        response = requests.get(f"{ODDS_API_URL}/sports", timeout=5)
        health_status["checks"]["odds_api"] = "healthy" if response.status_code == 200 else "degraded"
    except Exception as e:
        health_status["checks"]["odds_api"] = f"unhealthy: {e}"
    
    return health_status
```

## 🔄 Backup and Recovery

### 1. Data Backup
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/mlbbetting"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp /home/mlbbetting/mlbetting/data/warehouse.duckdb $BACKUP_DIR/warehouse_$DATE.duckdb

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /home/mlbbetting/mlbetting/models/

# Backup configuration
cp /home/mlbbetting/mlbetting/.env $BACKUP_DIR/env_$DATE.txt

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.duckdb" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

# Add to crontab: 0 2 * * * /home/mlbbetting/backup.sh
```

### 2. Disaster Recovery
```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
RESTORE_DIR="/home/mlbbetting/mlbetting"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop services
sudo systemctl stop mlbbetting-api mlbbetting-pipeline

# Restore database
cp $BACKUP_FILE $RESTORE_DIR/data/warehouse.duckdb

# Restore models if available
if [ -f "${BACKUP_FILE%.*}_models.tar.gz" ]; then
    tar -xzf "${BACKUP_FILE%.*}_models.tar.gz" -C $RESTORE_DIR/
fi

# Start services
sudo systemctl start mlbbetting-api mlbbetting-pipeline

echo "Restore completed"
```

## 📈 Performance Optimization

### 1. Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_games_date ON raw.games(date);
CREATE INDEX idx_odds_date ON raw.ml_odds(date);
CREATE INDEX idx_props_date ON raw.so_props(date);

-- Optimize DuckDB settings
PRAGMA memory_limit='4GB';
PRAGMA threads=4;
```

### 2. API Optimization
```python
# Add caching
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Cache expensive endpoints
@app.get("/odds")
@cache(expire=300)  # 5 minutes
async def get_odds():
    # Expensive operation
    pass
```

### 3. Model Optimization
```python
# Model serving optimization
import joblib
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model():
    return joblib.load("models/latest/ensemble.joblib")

# Batch predictions
def predict_batch(features_batch):
    model = load_model()
    return model.predict(features_batch)
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Monitor memory
   htop
   # Adjust model batch sizes
   # Use model quantization for deployment
   ```

2. **API Rate Limits**
   ```python
   # Implement exponential backoff
   import time
   import random
   
   def api_call_with_backoff(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return func()
           except RateLimitError:
               wait_time = (2 ** attempt) + random.uniform(0, 1)
               time.sleep(wait_time)
       raise Exception("Max retries exceeded")
   ```

3. **Database Locks**
   ```python
   # Use connection pooling
   import duckdb
   from contextlib import contextmanager
   
   @contextmanager
   def get_db_connection():
       conn = duckdb.connect(DATABASE_PATH)
       try:
           yield conn
       finally:
           conn.close()
   ```

### Log Analysis
```bash
# View application logs
tail -f /home/mlbbetting/mlbetting/logs/app.log

# Filter for errors
grep ERROR /home/mlbbetting/mlbetting/logs/app.log

# Monitor API access
tail -f /var/log/nginx/access.log | grep mlbbetting
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks
1. **Daily**: Check system health, review alerts
2. **Weekly**: Review model performance, update features
3. **Monthly**: Security updates, backup verification
4. **Quarterly**: Full system audit, performance review

### Support Contacts
- **System Admin**: admin@your-domain.com
- **Data Team**: data@your-domain.com
- **On-Call**: +1-XXX-XXX-XXXX

This deployment guide provides a comprehensive foundation for running the MLB betting analytics platform in production. Adjust configurations based on your specific requirements and infrastructure. 