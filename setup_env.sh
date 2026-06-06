#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment setup complete. To activate the environment, run: source venv/bin/activate"