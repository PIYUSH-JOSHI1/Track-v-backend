# Render.com deployment configuration

web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 60 --max-requests 1000 --preload
