# LLM Federated Learning Platform Free Deployment Guide

This guide outlines three paths to deploy both the Web UI and the ML Backend **100% for free**, taking advantage of generous free-tier services.

---

## ⚡ Step 1: Deploy Frontend on Vercel (Always Free)

Vercel provides a 100% free Hobby tier for hosting static web interfaces.

### Quick Deploy:
1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Run the deployment command pointing to the static folder:
   ```bash
   vercel web/static --name llm-federated-platform-ui
   ```
3. Set the project settings to production and deploy:
   ```bash
   vercel --prod
   ```
*Your frontend is now live at `https://llm-federated-platform-ui.vercel.app`!*

---

## 🚀 Step 2: Deploy the Backend (Free Hosting Options)

Choose one of the following methods to host your backend server for free:

### Option A: Local Backend with Free Tunneling (Recommended)
Since this is an **On-device training platform**, running the backend locally gives you full access to your PC's CPU/GPU and RAM without any cloud costs. You can expose your local server to the Vercel frontend using a free tunneling service.

1. **Start the local backend**:
   ```bash
   python run_platform.py --mode web
   ```
2. **Expose port 8000 using LocalTunnel (Free & No Signup)**:
   Open a new terminal and run:
   ```bash
   npx localtunnel --port 8000
   ```
   *Alternatively, you can use `ngrok`: `ngrok http 8000`.*
3. Copy the generated public URL (e.g. `https://curly-wolves-howl.locallt.ly`).
4. Paste this URL into the **Backend Server API** setting on your Vercel frontend dashboard.

---

### Option B: Deploy to Hugging Face Spaces (Free Cloud Hosting - 16GB RAM)
Hugging Face Spaces offers a **100% free** CPU tier running Docker containers with **16 GB RAM, 2 vCPUs, and 50 GB Disk Space**, which is perfect for PyTorch ML hosting.

1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a new **Space**:
   - **Space SDK**: Select **Docker**
   - **Docker Template**: Select **Blank**
   - **Space License**: MIT
   - **Visibility**: Public (free tier)
3. Under the hood, HF Spaces requires the container to run on port **7860**. 
4. Update your local `run_platform.py` or customize the startup port to `7860` for the Web UI mode:
   - Expose port `7860` in your Dockerfile.
   - Run command: `python run_platform.py --mode web` (ensure FastAPI/uvicorn runs on port `7860`).
5. Upload your files (`Dockerfile`, `requirements.txt`, `web/`, `client/`, `utils/`, etc.) directly to the Hugging Face Space repository using Git:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```
6. Hugging Face will automatically build your Docker container and host your backend API for free at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`.

---

### Option C: Deploy to Render (Free Tier - 512MB RAM Limit)
Render offers free hosting, but with a strict **512 MB RAM limit**. 

> [!WARNING]
> Because loading PyTorch and ML models in memory exceeds 512 MB, Render will likely trigger an **Out of Memory (OOM) crash** unless you use a micro-sized model (like a 2.5M parameter model) and disable all PyTorch caching.

1. Create a free account on [Render](https://render.com/).
2. Click **New +** > **Web Service** and connect your GitHub repo.
3. Configure the settings:
   - **Runtime**: `Docker`
   - **Branch**: `main`
   - **Free Plan**: $0/month (512MB RAM, 0.5 CPU)
4. Add Environment Variable:
   - `PORT`: `10000` (Render's default port)
5. Expose port `10000` in the Web service.
