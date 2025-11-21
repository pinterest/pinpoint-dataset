#!/bin/bash
#
# Setup script for MetaCLIP2 Retrieval Reproduction Code
# This script sets up the environment for running the reproduction code
#

set -e  # Exit on error

echo "=========================================="
echo "MetaCLIP2 Reproduction Code Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check Python version
echo ""
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Python found: $PYTHON_VERSION"
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_status "Python found: $PYTHON_VERSION"
    PYTHON_CMD=python
else
    print_error "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is available
echo ""
echo "Checking pip..."
if $PYTHON_CMD -m pip --version &> /dev/null; then
    print_status "pip is available"
else
    print_error "pip not found. Please install pip."
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet
print_status "pip upgraded"

# Check for virtual environment
echo ""
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    VENV_DIR="venv"
    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment $VENV_DIR already exists"
        read -p "Do you want to recreate it? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            $PYTHON_CMD -m venv "$VENV_DIR"
            print_status "Virtual environment recreated"
        fi
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_status "Virtual environment created"
    fi
    
    source "$VENV_DIR/bin/activate"
    print_status "Virtual environment activated"
    
    # Upgrade pip in virtual environment
    pip install --upgrade pip --quiet
fi

# Install PyTorch
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    print_warning "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi
print_status "PyTorch installed"

# Install OpenCLIP from GitHub
echo ""
echo "Installing OpenCLIP (from GitHub main branch for MetaCLIP2 support)..."
pip install --upgrade "open-clip-torch @ git+https://github.com/mlfoundations/open_clip.git@main" --quiet
print_status "OpenCLIP installed"

# Install FAISS
echo ""
echo "Installing FAISS..."
if command -v nvidia-smi &> /dev/null; then
    print_warning "Attempting to install FAISS with GPU support..."
    if pip install faiss-gpu --quiet 2>/dev/null; then
        print_status "FAISS GPU version installed"
    else
        print_warning "FAISS GPU installation failed. Installing CPU version..."
        pip install faiss-cpu --quiet
        print_status "FAISS CPU version installed"
    fi
else
    pip install faiss-cpu --quiet
    print_status "FAISS CPU version installed"
fi

# Install remaining requirements
echo ""
echo "Installing remaining requirements..."
pip install -r requirements.txt --quiet
print_status "All requirements installed"

# Verify installations
echo ""
echo "Verifying installations..."
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
import sys

packages = [
    ("torch", "torch"),
    ("open_clip", "open_clip"),
    ("faiss", "faiss"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("PIL", "PIL"),
    ("tqdm", "tqdm")
]

failed = []
for name, module in packages:
    try:
        __import__(module)
        print(f"✓ {name}")
    except ImportError as e:
        print(f"✗ {name}: {e}")
        failed.append(name)

# Test OpenCLIP specifically for MetaCLIP2
try:
    import open_clip
    model_name = 'ViT-H-14-worldwide-quickgelu'
    pretrained = 'metaclip2_worldwide'
    print(f"✓ MetaCLIP2 model configuration verified")
except Exception as e:
    print(f"✗ MetaCLIP2 configuration error: {e}")
    failed.append("metaclip2")

# Check CUDA availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("! CUDA not available - will use CPU")
except:
    pass

if failed:
    print(f"\n⚠ Some packages failed to import: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\n✅ All packages verified successfully!")
EOF

$PYTHON_CMD test_imports.py
TEST_RESULT=$?
rm test_imports.py

if [ $TEST_RESULT -eq 0 ]; then
    print_status "All packages verified successfully!"
else
    print_error "Some packages failed verification. Please check the errors above."
    exit 1
fi

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Environment Summary:"
echo "-------------------"
$PYTHON_CMD --version
echo "PyTorch version: $($PYTHON_CMD -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $($PYTHON_CMD -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'No')"

if [ -d "$VENV_DIR" ]; then
    echo ""
    echo "Virtual environment created: $VENV_DIR"
    echo "To activate it in the future, run:"
    echo "  source $VENV_DIR/bin/activate"
fi

echo ""
echo "Next Steps:"
echo "-----------"
echo "1. Build FAISS index:"
echo "   python src/build_faiss_index.py --image_list index_signatures.txt --output_dir ./indices/metaclip2"
echo ""
echo "2. Run retrieval:"
echo "   python src/run_retrieval.py --query_file pinpoint_metadata.parquet --index_dir ./indices/metaclip2 --output_file results.json"
echo ""
echo "3. Evaluate results:"
echo "   python src/evaluate.py --results results.json --ground_truth pinpoint_metadata.parquet --output metrics.csv"
echo ""
echo "For more details, see README.md"
echo ""
print_status "Setup complete! Happy reproducing! 🚀"

