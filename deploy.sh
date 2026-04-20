#!/bin/bash

# Render API Deployment Script for RAG Support Bot

API_KEY="${RENDER_API_KEY:-YOUR_API_KEY_HERE}"
REPO_URL="https://github.com/althu2/rag-support-bot.git"
SERVICE_NAME="rag-support-bot"

if [ "$API_KEY" = "YOUR_API_KEY_HERE" ]; then
    echo "❌ Error: RENDER_API_KEY not set"
    echo "Set your API key first:"
    echo "  export RENDER_API_KEY='your_api_key_here'"
    echo ""
    echo "Get your API key from: https://dashboard.render.com/account/api-tokens"
    exit 1
fi

echo "🚀 Deploying RAG Support Bot to Render..."
echo "Service: $SERVICE_NAME"
echo "Repository: $REPO_URL"
echo ""

# Create web service via Render API
curl -X POST https://api.render.com/v1/services \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'$SERVICE_NAME'",
    "type": "web_service",
    "ownerId": null,
    "repoUrl": "'$REPO_URL'",
    "branch": "main",
    "buildCommand": "pip install -r requirements.txt",
    "startCommand": "bash start.sh",
    "envVars": [
      {
        "key": "GEMINI_API_KEY",
        "value": "REPLACE_WITH_YOUR_KEY",
        "scope": "build_service"
      }
    ],
    "plan": "free"
  }' \
  -w "\n\nStatus: %{http_code}\n"

echo ""
echo "✅ Check your deployment at: https://dashboard.render.com/"
