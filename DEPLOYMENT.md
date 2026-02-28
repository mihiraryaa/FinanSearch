# FinanSearch Deployment Guide

Complete guide to deploy FinanSearch on Vercel (Frontend) + Render (Backend)

## 🚀 Quick Overview

- **Frontend**: Vercel (React + Vite)
- **Backend**: Render (FastAPI + Python)
- **Total Time**: ~15 minutes
- **No GitHub Required**: Deploy directly from your local machine

---

## 📋 Prerequisites

Before you start, make sure you have:

1. Vercel account (sign up at [vercel.com](https://vercel.com))
2. Render account (sign up at [render.com](https://render.com))
3. OpenAI API key ([get one here](https://platform.openai.com/api-keys))
4. Vercel CLI installed: `npm install -g vercel`
5. Your project files ready locally

---

## Part 1: Deploy Backend to Render

### Step 1: Prepare Backend for Deployment

First, let's ensure your backend has all necessary files.

#### 1.1 Update `requirements.txt`

Make sure your `backend/requirements.txt` includes:

```txt
langchain
langchain-openai
langchain-community
faiss-cpu
unstructured
pypdf
fastapi
uvicorn[standard]
python-multipart
rank-bm25
python-dotenv
```

#### 1.2 Create `render.yaml` (Optional but Recommended)

Create a file `backend/render.yaml`:

```yaml
services:
  - type: web
    name: finansearch-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: OPENAI_API_KEY
        sync: false
      - key: PYTHON_VERSION
        value: 3.12.0
```

### Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Web Service"

2. **Connect Your Repository**
   - Connect your GitHub account
   - Select your FinanSearch repository
   - Click "Connect"

3. **Configure the Service**
   - **Name**: `finansearch-backend` (or your choice)
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**
   - Click "Advanced" → "Add Environment Variable"
   - Add: `OPENAI_API_KEY` = `your_actual_openai_api_key`
   - Add: `PYTHON_VERSION` = `3.12.0`

5. **Choose Plan**
   - Select "Free" plan (or paid for better performance)
   - Click "Create Web Service"

6. **Wait for Deployment**
   - Render will build and deploy your backend
   - Takes ~5-10 minutes
   - Once done, you'll get a URL like: `https://finansearch-backend.onrender.com`

7. **Test Your Backend**
   - Visit: `https://your-backend-url.onrender.com/docs`
   - You should see the FastAPI documentation

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Frontend for Deployment

#### 1.1 Update API Configuration

Update `frontend/vite.config.js` to handle API proxy:

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

#### 1.2 Update Frontend API Calls

Update `frontend/src/App.jsx` to use environment variable:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || '/api'
```

### Step 2: Deploy on Vercel

1. **Go to Vercel Dashboard**
   - Visit [vercel.com/dashboard](https://vercel.com/dashboard)
   - Click "Add New..." → "Project"

2. **Import Repository**
   - Click "Import Git Repository"
   - Select your FinanSearch repository
   - Click "Import"

3. **Configure Project**
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `dist` (auto-detected)

4. **Set Environment Variables**
   - Click "Environment Variables"
   - Add: `VITE_API_URL` = `https://your-backend-url.onrender.com`
   - (Replace with your actual Render backend URL from Part 1)

5. **Deploy**
   - Click "Deploy"
   - Wait ~2-3 minutes
   - You'll get a URL like: `https://finansearch.vercel.app`

---

## Part 3: Configure CORS

### Update Backend CORS Settings

Edit `backend/main.py` to allow your Vercel domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://your-app.vercel.app",  # Add your Vercel URL
        "https://*.vercel.app"  # Allow all Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Push this change to GitHub, and Render will automatically redeploy.

---

## Part 4: Update Frontend API URL

### Option A: Direct API Calls (Recommended)

Update `frontend/src/App.jsx`:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
```

Then set `VITE_API_URL` in Vercel environment variables to your Render backend URL.

### Option B: Using Vercel Rewrites

Create `vercel.json` in the `frontend` directory:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://your-backend-url.onrender.com/:path*"
    }
  ]
}
```

---

## 🎉 You're Done!

Your app should now be live at:
- **Frontend**: `https://your-app.vercel.app`
- **Backend**: `https://your-backend.onrender.com`

---

## 📝 Post-Deployment Checklist

- [ ] Test document upload functionality
- [ ] Test chat/query functionality
- [ ] Verify OpenAI API key is working
- [ ] Check CORS is properly configured
- [ ] Test on mobile devices
- [ ] Set up custom domain (optional)

---

## 🔧 Troubleshooting

### Backend Issues

**Problem**: "OPENAI_API_KEY not found"
- **Solution**: Check environment variables in Render dashboard
- Make sure the key is set correctly without quotes

**Problem**: Backend is slow or times out
- **Solution**: Render free tier spins down after inactivity
- Consider upgrading to a paid plan for always-on service
- First request after inactivity takes ~30 seconds

**Problem**: Import errors or module not found
- **Solution**: Check `requirements.txt` has all dependencies
- Trigger a manual redeploy in Render

### Frontend Issues

**Problem**: API calls failing with CORS errors
- **Solution**: Update CORS settings in `backend/main.py`
- Add your Vercel domain to `allow_origins`

**Problem**: Environment variables not working
- **Solution**: Redeploy after adding environment variables
- Make sure variable names start with `VITE_`

**Problem**: 404 errors on page refresh
- **Solution**: Add `vercel.json` with SPA configuration:

```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

---

## 💰 Cost Breakdown

### Free Tier Limits

**Render (Free)**
- 750 hours/month
- Spins down after 15 min inactivity
- 512 MB RAM
- Shared CPU

**Vercel (Free)**
- 100 GB bandwidth/month
- Unlimited deployments
- Automatic HTTPS
- Global CDN

**OpenAI API**
- Pay per use
- GPT-4o-mini: ~$0.15 per 1M input tokens
- Embeddings: ~$0.02 per 1M tokens

### Recommended Paid Plans (Optional)

- **Render Starter**: $7/month (always-on, better performance)
- **Vercel Pro**: $20/month (more bandwidth, better analytics)

---

## 🔐 Security Best Practices

1. **Never commit `.env` files** to GitHub
2. **Use environment variables** for all secrets
3. **Rotate API keys** regularly
4. **Enable rate limiting** on backend (add middleware)
5. **Use HTTPS only** (both platforms provide this)
6. **Monitor API usage** in OpenAI dashboard

---

## 🚀 Advanced: Custom Domains

### Vercel Custom Domain

1. Go to Project Settings → Domains
2. Add your domain (e.g., `finansearch.com`)
3. Update DNS records as instructed
4. Vercel handles SSL automatically

### Render Custom Domain

1. Go to Service Settings → Custom Domain
2. Add your domain (e.g., `api.finansearch.com`)
3. Update DNS records as instructed
4. Render handles SSL automatically

---

## 📊 Monitoring

### Render Monitoring

- View logs: Dashboard → Your Service → Logs
- Check metrics: Dashboard → Your Service → Metrics
- Set up alerts for downtime

### Vercel Monitoring

- View deployments: Dashboard → Your Project → Deployments
- Check analytics: Dashboard → Your Project → Analytics
- View function logs: Dashboard → Your Project → Functions

---

## 🔄 Continuous Deployment

Both platforms support automatic deployments:

1. **Push to GitHub** → Automatic deployment
2. **Pull Request** → Preview deployment (Vercel)
3. **Merge to main** → Production deployment

---

## 📞 Support

- **Render Docs**: [render.com/docs](https://render.com/docs)
- **Vercel Docs**: [vercel.com/docs](https://vercel.com/docs)
- **OpenAI Docs**: [platform.openai.com/docs](https://platform.openai.com/docs)

---

## 🎯 Next Steps

1. Add authentication (Auth0, Clerk, etc.)
2. Implement rate limiting
3. Add document storage (S3, Cloudinary)
4. Set up monitoring (Sentry, LogRocket)
5. Add analytics (Google Analytics, Plausible)
6. Implement caching (Redis)

---

**Happy Deploying! 🚀**
