#!/bin/bash

# Install requirements if needed
# pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_jsonl_viewer.py --server.port 8501 --server.address 0.0.0.0
