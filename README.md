# SUPIR - Super-Resolution AI (RunPod Deployment + REST API)

A fully integrated, production-ready deployment of the SUPIR super-resolution AI on **RunPod**, accessible via **FastAPI REST endpoints** and optionally through a **Gradio web UI**.

---

## 📌 Overview

This guide helps you deploy SUPIR on RunPod with REST API access. You'll:

* Provision a GPU-enabled RunPod instance with persistent storage
* Set up SUPIR with environment variables and virtual environment
* Launch the API server (FastAPI)
* Authenticate using an API token
* Access endpoints and download results

---

## 🔧 Prerequisites

* RunPod account: [https://runpod.io?ref=1aka98lq](https://runpod.io?ref=1aka98lq)
* Basic Linux terminal familiarity
* GitHub access: [https://github.com/sab110/SUPIR-Installation-RESTAPI.git](https://github.com/sab110/SUPIR-Installation-RESTAPI.git)

---

## 🚀 Step-by-Step Deployment Guide

### 1. 🔐 Create RunPod Instance

1. Go to RunPod and select **Secure Cloud** → **Create Pod**
2. **Template**: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1`
3. **GPU**: Choose one of the following:

   * Recommended: `A6000 / A40` (48GB)
   * Budget: `RTX 3090 / 4090` (24GB)
4. **Volume**: Enable Persistent Volume → Set to **100–150GB**
5. **Expose HTTP Port**: Add `8000` (for FastAPI)
6. Launch the pod

---

### 2. 📁 Setup Workspace

> Your default working directory should be `/workspace`

```bash
cd /workspace
```

Clone the repository:

```bash
git clone https://github.com/sab110/SUPIR-Installation-RESTAPI.git .
```

> If the directory is not empty, upload as `.zip` and extract manually

---

### 3. 🌍 Environment Setup

Set huggingface directory and permissions:

```bash
export HF_HOME="/workspace"
chmod +x RunPod_Install_SUPIR.sh
```

Run the installer:

```bash
./RunPod_Install_SUPIR.sh
```

⚠️ If model downloads freeze, rerun the command.

---

### 4. 🔑 Configure API Token

Open `.env` or create it in root directory:

```
API_TOKEN=your-secret-token
```

> This token will be used to authenticate every REST API call.

Ensure token is correctly loaded:

```bash
source .env
```

---

### 5. ▶️ Start FastAPI Server

#### ✅ Recommended: Entrypoint Script

```bash
chmod +x entrypoint.sh
./entrypoint.sh
```

This will:

* Activate Python venv
* Download and prepare models
* Start FastAPI server at `http://0.0.0.0:8000`

Check logs for:

```
Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 Testing the REST API

### Swagger UI:

```
http://<pod-id>-8000.proxy.runpod.net/docs
```

### Health Check:

```bash
curl http://<pod-ip>:8000/health
```

---

### 🔐 Authorization Header (Required)

Every call to `/job`, `/jobs`, etc. must have:

```http
Authorization: Bearer your-secret-token
```

---

### 🚀 Upload + Create Job

**POST** `/job`

* Upload your image (e.g., PNG, JPG)
* JSON settings (optional)

```bash
curl -X POST http://<pod-ip>:8000/job \
  -H "Authorization: Bearer your-secret-token" \
  -F "image=@input.jpg" \
  -F 'settings={}'
```

Returns:

```json
{
  "job_id": "...",
  "status": "pending"
}
```

---

### 📊 Check Job Status

```bash
curl -H "Authorization: Bearer your-secret-token" \
http://<pod-ip>:8000/job/{job_id}
```

---

### 🖼️ Download Result Image (SCP)

Once job completes, result will be saved at:

```
/workspace/adjustedupscaled/<job_id>_timestamp.png
```

Download to local using SCP:

```bash
scp -P <pod_port> root@<pod_ip>:/workspace/adjustedupscaled/*.png .
```

> For example:

```bash
scp -P 22032 root@69.30.85.187:/workspace/adjustedupscaled/*.png .
```

---

## 📎 Optional: Gradio UI Mode

```bash
cd SUPIR
source ./venv/bin/activate
python gradio_demo.py --loading_half_params --use_tile_vae --share
```

Access public Gradio link provided in terminal.

---

## ✅ RunPod + SUPIR Deployment Recap

* ✅ Pod setup with persistent volume & port
* ✅ Git cloned or zipped to `/workspace`
* ✅ Models downloaded
* ✅ FastAPI launched with `entrypoint.sh`
* ✅ Token-based API used successfully
* ✅ Result images downloaded from `/workspace/adjustedupscaled`

---

## 🧠 Troubleshooting

| Issue             | Fix                                             |
| ----------------- | ----------------------------------------------- |
| Swagger shows 403 | Missing Authorization header                    |
| Model stuck at 0% | Rerun install script or increase VRAM           |
| 502 Bad Gateway   | Server may have closed or crashed – restart pod |
| Token Invalid     | Confirm `.env` is sourced and token matches     |
| Can't access 8000 | Ensure port is exposed in pod setup             |

---

## 📁 Project Structure

```
├── SUPIR/
│   ├── api_server.py
│   ├── gradio_demo.py
│   ├── requirements.txt
├── entrypoint.sh
├── RunPod_Install_SUPIR.sh
├── RunPod_Start_SUPIR_Linux.sh
├── HF_model_downloader.py
├── adjustedupscaled/ (output directory)
└── .env
```

---

## 📝 License

Please refer to `LICENSE` file for original SUPIR credits and modifications.

---

**Happy Upscaling!** ✨
