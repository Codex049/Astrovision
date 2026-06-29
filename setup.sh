#!/bin/bash

echo "Setting up AstroVision..."

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo ""
echo "Done! Now install Ollama for the AI chat feature:"
echo "  brew install ollama       (Mac)"
echo "  ollama pull llama3.2"
echo ""
echo "To run AstroVision:"
echo "  1. ollama serve           (in a separate terminal)"
echo "  2. source .venv/bin/activate"
echo "  3. python server.py"
echo "  4. Open http://localhost:5001"
