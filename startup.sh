#!/bin/bash
apt-get update
apt-get install -y tesseract-ocr
export PATH="/usr/bin:$PATH"
streamlit run app.py --server.port 8000 --server.address 0.0.0.0 
