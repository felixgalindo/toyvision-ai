
#!/bin/bash

echo "🧠 ToyVision-AI Setup Script"
echo "---------------------------"

UNAME=$(uname)
IS_MAC=false
IS_LINUX=false
IS_WINDOWS=false

if [[ "$UNAME" == "Darwin" ]]; then
    IS_MAC=true
elif [[ "$UNAME" == "Linux" ]]; then
    if grep -qi microsoft /proc/version 2>/dev/null; then
        IS_WINDOWS=true
    else
        IS_LINUX=true
    fi
fi

# Create Python virtual environment
echo "🛠 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

install_python_requirements() {
    echo "📦 Installing Python packages into virtual environment..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

if $IS_MAC; then
    echo "🖥 Detected macOS"
    which brew >/dev/null || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install espeak portaudio
    install_python_requirements

elif $IS_LINUX; then
    echo "🐧 Detected Linux (Raspberry Pi or Debian)"
    sudo apt update && sudo apt install -y \
        libatlas-base-dev \
        espeak \
        libespeak1 \
        portaudio19-dev \
        libffi-dev \
        libjpeg-dev \
        libopenblas-dev \
        python3-pyaudio \
        python3-pip \
        python3-opencv \
        python3-venv
    install_python_requirements
else
    echo "❌ Unsupported OS: $UNAME"
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "👉 To activate the virtual environment, run:"
echo "   source venv/bin/activate"
echo ""
echo "🚀 To launch the demo:"
echo "   python demo.py"
