# 🖥️ Frontend

1️. Clone the Repository
```bash
git clone https://github.com/anhleh33/SPNC_gnnclassifier.git
cd SPNC_gnnclassifier
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