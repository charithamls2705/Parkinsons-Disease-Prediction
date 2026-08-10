# ParkinsonAI Prediction App

A Django application for Parkinson's disease prediction using a pre-trained KNN model.

## Deployment

This application is prepared for deployment on Render.

### Render setup

1. Push this repository to GitHub.
2. Create a new Render Web Service.
3. Connect Render to the GitHub repository.
4. Set the following environment variables in Render:
   - `DJANGO_SECRET_KEY` (production secret key)
   - `DJANGO_DEBUG` (set to `False`)
   - `DJANGO_ALLOWED_HOSTS` (optional; defaults to `localhost 127.0.0.1 [::1]`)
   - `DJANGO_CSRF_TRUSTED_ORIGINS` (optional; include `https://<your-render-hostname>`)
   - `DATABASE_URL` (optional; if omitted, local SQLite is used)

### Build command

```bash
bash build.sh
```

### Start command

```bash
gunicorn parkinsons_project.wsgi:application
```

### Notes

- The trained model file is preserved at `models/parkinsons_knn_pipeline.joblib`.
- The project is configured to use WhiteNoise for static file serving.
- SQLite remains available locally unless `DATABASE_URL` is supplied.
