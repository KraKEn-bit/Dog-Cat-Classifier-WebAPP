# 🧬 Neural Vision: Dog vs Cat Classifier (Pro)

A **high-performance image classification system** upgraded to **Keras 3.0**, designed to distinguish between **canine and feline specimens** using a deep **Convolutional Neural Network (CNN)**.

This project features a **professional, laboratory-grade cyber interface**, optimized inference, and both **web and API access**, making it suitable for research demos, ML portfolios, and real-world integration.

---

## 🚀 Technical Upgrades — *Rafsan’s Edition*

### 🔁 Keras 3.0 Migration
- Fully refactored to support **TensorFlow / Keras 2.16+**
- Resolved legacy `batch_input_shape` serialization issues
- Re-linked model weights to a **native Keras 3 architecture**

### 🧠 Neural Engine Optimization
- Faster inference with optimized model loading
- Clean separation between preprocessing, inference, and UI layers

### 🧪 Cyber-Lab Interface
- Complete UI overhaul with **Glassmorphism**
- Monospace typography for a **research console aesthetic**
- Animated scanning effects and real-time prediction feedback

### 📊 Dynamic Analysis Matrix
- Real-time probability tracking
- Automated “scanning” animation during inference

---

## ✨ Features

- **Advanced UI**
  - Side-by-side input/output console
  - Subtle textured backgrounds for depth
- **Cross-Platform**
  - Optimized for **Python 3.10+**
- **Dual Access Modes**
  - 🌐 Streamlit Web Dashboard (users)
  - ⚙️ CLI / API Backend (engineers)
- **Automated Preprocessing**
  - Real-time image resizing to **128×128**
  - Normalization for CNN compatibility

---

## 🛠️ Installation

### 📌 Prerequisites
- Python **3.10+**
- Virtual Environment (**recommended**)

---

### 🔧 Setup Instructions

#### 1️⃣ Clone the Repository
```bash
git clone https://github.com/KraKEn-bit/Dog-Cat-Classifier-WebAPP.git
cd Dog-Cat-Classifier-WebAPP
```


# **Create & Activate Virtual Environment:**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```


# **Install Dependencies:**
```bash
pip install -r requirements.txt
```


## **Cyber-Lab Dashboard (Streamlit):**

Launch the high-tech web interface:
```bash
streamlit run streamlit_app.py
```


# **Project Structure:**
dog-cat-classifier/
├── streamlit_app.py      # Cyber-Lab Streamlit UI
├── webapp.py             # FastAPI Backend Implementation
├── rebuild_model.py      # Keras 3 Migration Engine
├── dog_cat_fixed.keras   # Optimized Keras 3 Model Weights
├── PAWS.webp             # UI Background Texture Asset
└── README.md             # Project Documentation
