# SPNC GNN Classifier

Multimodal Vietnamese textbook classifier using GraphSAGE + Fusion MLP.
14 subject classes · 3 grade levels · Inductive GNN evaluation (Khung B).

---

## ML Artifacts (required for backend)

Model weights and graph data are stored on Google Drive (~280 MB total).
Download them once before running the backend:

```bash
pip install gdown
python download_artifacts.py
```

This will populate `backend/infrastructure/ml/artifacts/` automatically.

---

# 🖥️ Frontend

1️. Clone the Repository
```bash
git clone https://github.com/giangmninfo/2026SPNCGraph.git
cd 2026SPNCGraph
```

2. Install Dependencies
```bash
npm install
```

3. Running the Project
```bash
npm run dev
```

Open the browser at:
```bash
http://localhost:3000
```
   
Or you can find the browser link in terminal where you execute the commands.

# ⚙️ Backend
1. Create Virtual Environment
Open either in PowerShell or Git Bash
```bash
cd backend
python -m venv .venv
```

Open virtual environment
```bash
cd backend
source .venv/Scripts/Activate
```

2. Packages and library installations
Make sure there is (.venv) on terminal
```bash
pip install -r Requirements-Full.txt
```

3. Setup .env
`.env` file should be put in `backend` folder
content in `.env` file will be:
```env
DATABASE_URL=postgresql://user:password@host/db
```
🔐 Replace [Password] with your actual database password.

4. Run backend

Open Git Bash and run the backend from the project root:
```bash
source backend/.venv/Scripts/Activate
python -m backend.app
```
The backend will be available at:
```bash
http://localhost:8000
```