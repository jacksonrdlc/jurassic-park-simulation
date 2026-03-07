#!/bin/bash
# Creates jurassic_env with Python 3.12, installs all requirements
python3.12 -m venv jurassic_env
source jurassic_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ jurassic_env ready. Activate with: source jurassic_env/bin/activate"
