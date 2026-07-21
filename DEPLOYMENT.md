# 🚀 100% Free Cloud Deployment Guide

Follow these 3 easy steps to deploy your Self-Healing RAG Assistant to the cloud completely free of charge.

---

## 🔑 Step 1: Get Your Free Groq API Key (1 Minute)

1. Go to [console.groq.com](https://console.groq.com) and sign up for free.
2. Go to **API Keys** -> Click **Create API Key**.
3. Copy your API Key (e.g. `gsk_...`).

---

## 📦 Step 2: Deploy Backend to Hugging Face Spaces (Free Docker Hosting)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. **Space Name**: `synapse-rag-backend`
3. **SDK**: Choose **Docker** -> Select **Blank**.
4. **License**: Choose MIT (or Open Source).
5. Click **Create Space**.

### Add Secret & Upload Code:
1. Inside your new Space, go to **Settings** -> **Variables and secrets** -> Click **New Secret**:
   - **Key**: `GROQ_API_KEY`
   - **Value**: *(Paste your Groq API key)*
2. Push your repository code to Hugging Face:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/synapse-rag-backend
   git push hf main
   ```
3. Hugging Face will automatically build your Docker container. Once complete, your public backend URL will be:
   `https://YOUR_USERNAME-synapse-rag-backend.hf.space`

---

## ⚡ Step 3: Deploy Frontend to Vercel (Free React Hosting)

1. Push your code to your GitHub repository.
2. Go to [vercel.com](https://vercel.com) and log in with GitHub.
3. Click **Add New...** -> **Project**.
4. Import your `Rag-Self-Healing` repository.
5. In **Framework Preset**, select **Vite**.
6. Set **Root Directory** to `frontend`.
7. Expand **Environment Variables** and add:
   - **Name**: `VITE_API_BASE_URL`
   - **Value**: `https://YOUR_USERNAME-synapse-rag-backend.hf.space`
8. Click **Deploy**!

---

### 🎉 You're Done!
Vercel will give you a live production link (e.g. `https://synapse-ai-rag.vercel.app`) that you can share with anyone in the world!
