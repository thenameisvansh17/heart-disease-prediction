#!/bin/bash
# ============================================================
# One-time helper to push this project to a new GitHub repo.
# Edit REPO_URL below with your own repo link, then run:
#   bash setup_github.sh
# ============================================================

set -e

REPO_URL="https://github.com/YOUR_USERNAME/heart-disease-prediction.git"

echo "Initializing git repo..."
git init
git add .
git commit -m "Initial commit: Heart Disease Prediction ML project"
git branch -M main
git remote add origin "$REPO_URL"

echo "Pushing to $REPO_URL ..."
git push -u origin main

echo ""
echo "Done! Now go to https://share.streamlit.io to deploy the live demo."
