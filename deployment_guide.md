# LLM Federated Learning Platform Deployment Guide

This document details the configuration and steps required to build, dockerize, and deploy the On-device LLM Fine-tuning & Federated Learning Platform.

---

## 🚀 Deployment Overview

The platform consists of a **hybrid architecture** for cloud deployment:
1. **Frontend (Vercel)**: Hosted as a fast, static Single Page App (SPA).
2. **Backend API & Flower Server (Docker container on Render/Railway/AWS)**: Containerized with **Python 3.14.2** to run the FastAPI app and Flower Federated Server.
3. **Federated Clients**: Run locally on user machines and connect to the cloud Federated Server.

---

## ⚡ Vercel Deployment Guide (Frontend UI)

Since the frontend is fully client-side and dynamic, we deploy it statically on Vercel. 

### Option A: Deployment via Vercel CLI
1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Navigate to the project directory and run the deployment command specifying the `web/static` directory as the build target:
   ```bash
   vercel web/static --name llm-federated-platform-ui
   ```
3. Follow the CLI prompts to deploy. Run `vercel --prod` to release to production.

### Option B: Deployment via GitHub integration
1. Create a new project on the **Vercel Dashboard**.
2. Connect your Git repository.
3. In **Project Settings**, configure:
   - **Root Directory**: `web/static`
   - **Framework Preset**: `Other` (No build steps required)
4. Click **Deploy**. Vercel will host the dashboard using the root `index.html` and rewrite `/static/*` requests automatically using `vercel.json`.

---

## 🐳 Docker Deployment Guide (Backend & Flower)

The backend runs the FastAPI server (serving predictions, model state, configurations, and specs) and the Flower coordinator.

### 1. Build the Docker Image
To build the Docker container using Python **3.14.2**:
```bash
docker build -t llm-federated-platform .
```

### 2. Multi-Role Container Execution
The Docker container can be launched in different modes via CLI arguments:

- **Run Server Only** (Flower coordinator on port 8080):
  ```bash
  docker run -d -p 8080:8080 llm-federated-platform python run_platform.py --mode server
  ```
- **Run Web Backend Only** (FastAPI app on port 8000):
  ```bash
  docker run -d -p 8000:8000 -e FL_SERVER_ADDRESS="<SERVER_IP>:8080" llm-federated-platform python run_platform.py --mode web
  ```
- **Run Everything** (Server + Web Backend + Client in parallel):
  ```bash
  docker run -d -p 8000:8000 -p 8080:8080 llm-federated-platform
  ```

---

## 🔗 Connecting Vercel Frontend to Docker Backend

1. Deploy the Docker backend to **Render**, **Railway**, or a cloud virtual machine, ensuring port `8000` (FastAPI) and `8080` (Flower) are exposed.
2. Note your deployed backend API URL (e.g. `https://llm-platform-backend.onrender.com`).
3. Open your Vercel deployment URL (e.g. `https://llm-federated-platform-ui.vercel.app`).
4. On the top left of the dashboard, locate the **Backend Server API** panel.
5. Paste your backend URL into the input field.
6. The dashboard will verify the connection and show a green **"Connected to backend!"** status indicator.
7. Under the hood, **CORS Middleware** configured in FastAPI will allow the cross-origin requests from Vercel to function securely.
8. Local specs, system memory status, training status, and model comparing playground will load and function dynamically from the remote Docker host!
