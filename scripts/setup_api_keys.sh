#!/bin/bash
# Setup script for API keys
# This script helps you configure API keys for OpenAI, Google, and Anthropic

echo "======================================"
echo "API Keys Setup"
echo "======================================"
echo ""

# Function to set API key
set_api_key() {
    local provider=$1
    local var_name=$2
    local current_value=$(printenv $var_name)
    
    echo "Setting up $provider API Key..."
    if [ -n "$current_value" ]; then
        echo "✓ $var_name is already set"
    else
        echo "✗ $var_name is not set"
        echo "Please enter your $provider API key (or press Enter to skip):"
        read -s api_key
        
        if [ -n "$api_key" ]; then
            echo "export $var_name='$api_key'" >> ~/.bashrc
            export $var_name="$api_key"
            echo "✓ $var_name added to ~/.bashrc"
        else
            echo "Skipped $provider"
        fi
    fi
    echo ""
}

# Setup OpenAI
set_api_key "OpenAI" "OPENAI_API_KEY"

# Setup Google
set_api_key "Google (Gemini)" "GOOGLE_API_KEY"

# Setup Anthropic
set_api_key "Anthropic (Claude)" "ANTHROPIC_API_KEY"

echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To activate the changes, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or restart your terminal."
echo ""
echo "To get API keys:"
echo "  - OpenAI: https://platform.openai.com/api-keys"
echo "  - Google: https://aistudio.google.com/app/apikey"
echo "  - Anthropic: https://console.anthropic.com/settings/keys"
