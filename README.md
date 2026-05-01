# Shaastra 2026 IOH Project

Streamlit dashboard and analysis files for the IOH attendance project.

## What organizers should download

Download or clone the complete repository. The dashboard expects these files and folders to stay together:

- `dashboard.py` - Streamlit dashboard
- `analysis.py` - analysis and model helper code
- `IOH.csv` - input dataset
- `executive_summary.txt` - summary shown in the dashboard
- `plots/` - generated dashboard plots
- `requirements.txt` - Python packages needed to run the app


## Local setup

### 1. Install Python

Install Python 3.10 or newer from https://www.python.org/downloads/.

During installation on Windows, enable **Add Python to PATH**.

Check that Python is available:

```bash
python --version
```

### 2. Cloning the project
Clone the repository:

```bash
git clone https://github.com/<your-github-username>/Shaastra_2026_IOH_Project.git
cd Shaastra_2026_IOH_Project
```

### 3. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the dashboard

```bash
streamlit run dashboard.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser to view the dashboard.

## Troubleshooting

If PowerShell blocks virtual environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

If `python` is not recognized, reinstall Python and enable **Add Python to PATH**, or try:

```bash
py --version
py -m venv venv
```

If Streamlit opens but files are missing, confirm that `IOH.csv`, `executive_summary.txt`, and the `plots/` folder are in the same folder as `dashboard.py`.

## Repository upload checklist

Use `Shaastra_2026_IOH_Project` as the GitHub repository name.

Before pushing to GitHub, include:

- `.gitignore`
- `README.md`
- `requirements.txt`
- `dashboard.py`
- `analysis.py`
- `IOH.csv`
- `executive_summary.txt`
- `plots/`

Do not include:

- `venv/`
- `__pycache__/`
- `.streamlit/secrets.toml`
