#!/bin/bash
# ============================================================================
# AI-Based Water Contamination Early Warning System
# Environment Setup Script
# ============================================================================

echo "======================================================================="
echo "   AI-Based Water Contamination Early Warning System"
echo "   Environment Setup"
echo "======================================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if command -v python3 &> /dev/null; then
    echo "✓ Python 3 found"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (y/n): " create_venv

if [ "$create_venv" = "y" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

echo ""
echo "Installing required packages..."
echo "This may take several minutes..."
echo ""

pip install --upgrade pip

# Install packages with specific flags
pip install -r requirements.txt

echo ""
echo "======================================================================="
echo "Installation Complete!"
echo "======================================================================="
echo ""
echo "To verify installation, run:"
echo "  python3 -c 'import tensorflow, sklearn, xgboost; print(\"All imports successful!\")'"
echo ""
echo "To start the project:"
echo "  1. Copy your datasets to data/raw/"
echo "     - water_potability.csv"
echo "     - water_dataset.mat"
echo "  2. Run: python main.py"
echo ""
echo "======================================================================="
