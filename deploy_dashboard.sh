#!/bin/bash
# Run this ON the dashboard EC2 instance after SSH-ing in.
set -e

echo "=== Setting up BR41N.IO Dashboard ==="

# Install Python 3.10 venv
sudo apt-get update -y
sudo apt-get install -y python3.10-venv python3-pip git

# Clone repo
cd /home/ubuntu
rm -rf BR41N-hackathon
git clone https://github.com/vladamisici/BR41N-hackathon.git
cd BR41N-hackathon

# Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install dashboard deps
pip install -r dashboard/requirements.txt

# Copy dashboard data and dataset symlink
# (dashboard_data.json needs to be generated first on g5, then scp'd here)

echo "=== Setup complete ==="
echo "To start the dashboard:"
echo "  cd /home/ubuntu/BR41N-hackathon"
echo "  source .venv/bin/activate"
echo "  streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
