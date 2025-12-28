# Render Deployment Configuration

## Environment Variables to set in Render Dashboard:

# AWS S3 Configuration (if using cloud storage)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-traffic-videos-bucket

# Python Configuration
PYTHON_VERSION=3.11.0

# Build Command
pip install -r requirements_render.txt

# Start Command  
gunicorn app_cloud:app

## Render Service Settings:
- Service Type: Web Service
- Environment: Python
- Build Command: pip install -r requirements_render.txt
- Start Command: gunicorn app_cloud:app
- Instance Type: Starter (can upgrade if needed)

## Important Notes:
1. Remove large video files from repository before deployment
2. Use environment variables for sensitive configuration
3. Enable CORS for frontend communication
4. Use smaller YOLO model (yolov8n.pt) for faster processing
5. Set up cloud storage for video files
6. Configure health check endpoint (/health)

## Deployment Checklist:
☐ Remove large video files from repo
☐ Update requirements.txt
☐ Set environment variables in Render
☐ Test API endpoints
☐ Configure CORS for Vercel domain
☐ Update frontend with correct backend URL
☐ Test video streaming functionality
☐ Monitor deployment logs