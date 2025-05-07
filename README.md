# 🧠 ToyVision-AI

**Real-time Toy Recognition using Edge Impulse, Python, and a Webcam — Built to Inspire the Next Generation of Engineers**

<img src="assets/demo-screenshot.png" alt="ToyVision Demo" width="600"/>

## 📌 Overview

ToyVision-AI is a Python-based object recognition project designed for engaging, educational AI demos. Using a USB webcam and a lightweight Edge Impulse model, the app can recognize toys like Pikachu or Hello Kitty in real time — and display playful messages like:

> "That's Pikachu!"

It was originally built for an **elementary school career day** to spark curiosity in artificial intelligence and engineering — and to show kids that engineering can be fun, creative, and magical.

---

## 🎯 Goals

- ✅ Make AI approachable for kids using real-time vision
- ✅ Demonstrate Edge Impulse + Python in action
- ✅ Showcase how engineers build playful, purposeful tech
- ✅ Serve as a template for other educators, engineers, and community demos

---

## 📸 Demo Video / Screenshots

*Coming soon...*  

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/felixgalindo/toyvision-ai.git
cd toyvision-ai
```

### 2. Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Run the setup script (auto-detects macOS, Linux, or WSL)

Make the script executable if needed:
```bash
chmod +x setup.sh
```

Then run the script:
```bash
./setup.sh
```

This installs all required system packages and Python libraries for your platform.

### 4. Run the demo

Make sure your USB webcam is connected and your `.eim` model is placed in the root directory (e.g., `modelfile_mac.eim`, `modelfile_linux-armv7.eim`, etc.). Then launch the demo:

```bash
python demo.py
```

Press `q` to quit the demo.

---

## 🧠 How It Works

- Uses OpenCV to stream frames from a USB camera
- Runs an Edge Impulse `.eim` model to detect toys
- Draws bounding boxes with the object name
- Speaks “That’s a [toy name]” when a new toy appears
- Works on both macOS (Apple Silicon supported) and Raspberry Pi

---

## 🛠 Tools Used

- [Edge Impulse](https://www.edgeimpulse.com/)
- Python 3
- OpenCV
- Pyttsx3 (for voice)
- Raspberry Pi or Mac

---

## 🧸 Toys Supported

The pre-trained AI model included in this project can currently recognize:

- 🟡 **Pikachu**  
- 🕷 **Spider-Man**  
- 🎀 **Hello Kitty**

To add more toys in the future, you can collect and train your own dataset using [Edge Impulse](https://www.edgeimpulse.com/).

---


## ✨ Educational Impact

Kids learn:
- What AI is and how it works
- How engineers combine math, science, and code
- That it’s possible to build cool, intelligent tools with a tiny computer

---

## 📂 Folder Structure

```
toyvision-ai/
├── demo.py
├── requirements.txt
├── setup.sh
├── README.md
├── modelfile_mac.eim         <-- macOS version of model
├── modelfile_linux-armv7.eim <-- Raspberry Pi version of model
└── assets/
    └── demo-screenshot.png
```

---
