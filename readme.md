# 🏥 MediGuide - AI-Powered Medical Prescription Analyzer

## Project Overview

**MediGuide** is an AI-powered medical prescription and test report analyzer that helps patients understand their medical documents. It combines **Computer Vision**, **OCR (Optical Character Recognition)**, **Large Language Models (LLMs)**, and **Text-to-Speech** into a seamless web application.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit (Python web framework) |
| **OCR Engine** | EasyOCR (Deep Learning-based) |
| **AI/LLM** | Google Gemini 2.0 Flash |
| **Image Processing** | PIL/Pillow, NumPy |
| **Text-to-Speech** | gTTS (Google Text-to-Speech) |
| **Security** | Streamlit Secrets Management |

---

## 🧠 Core Features & How They Work

### 1. **Image Processing Algorithms (Available in Codebase)**

Four custom image processing algorithms implemented from scratch for advanced preprocessing:

#### a) **Grayscale Conversion**
```
Y = 0.2989 × R + 0.5870 × G + 0.1140 × B
```
- Uses the **luminosity method** (ITU-R BT.601 standard)
- Weights RGB channels based on human eye sensitivity
- Green has highest weight (0.587) because human eyes are most sensitive to green

#### b) **Binary Thresholding**
- Converts grayscale to pure black & white
- Uses threshold value (default: 128)
- Pixels below threshold → Black (0), above → White (255)
- Improves OCR accuracy by increasing contrast

#### c) **Bilinear Interpolation** (Image Resizing)
- Samples **4 neighboring pixels** for each new pixel
- Uses weighted average based on distance
- Formula considers top-left, top-right, bottom-left, bottom-right pixels
- Produces smoother results than nearest-neighbor

#### d) **Bicubic Interpolation** (Advanced Resizing)
- Samples **16 neighboring pixels** (4×4 grid)
- Uses **cubic polynomial function** for weight calculation
- Cubic factor: -0.5 (Mitchell-Netravali filter)
- Produces highest quality image scaling

### 2. **OCR Text Extraction**
- Uses **EasyOCR** - a deep learning-based OCR engine
- Supports multiple languages (configured for English)
- Returns bounding boxes + extracted text
- Processes saved JPG for optimal results

### 3. **AI-Powered Medical Analysis**
Using **Google Gemini 2.0 Flash**:
- Analyzes extracted prescription/report text
- Provides medical insights in simple language
- Explains complex medical terminology
- Suggests potential symptoms based on diagnosis
- Adds appropriate medical disclaimers

### 4. **Markdown Cleaning for TTS**
Custom regex-based text sanitizer that removes:
- Bold markers (`**text**`)
- Italic markers (`*text*`)
- Headers (`# ## ###`)
- Bullet points (`- * +`)
- Numbered lists (`1. 2. 3.`)
- Markdown links `[text](url)`
- Cleans up whitespace and punctuation

### 5. **Text-to-Speech Output**
- Converts AI analysis to spoken audio
- Uses Google's TTS engine
- Outputs as WAV file for browser playback

---

## 🔐 Security Implementation

- **API keys stored in `.streamlit/secrets.toml`** (not hardcoded!)
- **`.gitignore` configured** to exclude sensitive files
- **Environment variable pattern** for secure credential management

---

## 🏗️ Architecture Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Upload    │────▶│   Convert    │────▶│   EasyOCR   │
│   Image     │     │   to RGB     │     │  Extraction │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Audio     │◀────│  Text Clean  │◀────│   Gemini    │
│   Output    │     │   + gTTS     │     │   AI/LLM    │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## 📊 Key Algorithms Explained

### **Luminosity Grayscale Formula**
The ITU-R BT.601 standard for grayscale conversion weights RGB channels based on human perception - green at 58.7%, red at 29.89%, and blue at 11.4%. This produces more natural-looking grayscale images compared to simple averaging.

### **Bicubic Interpolation**
For high-quality image scaling, bicubic interpolation samples a 4×4 pixel neighborhood and uses cubic polynomial weighting. The cubic function uses a -0.5 factor (Mitchell-Netravali) to balance sharpness and ringing artifacts.

### **OCR Pipeline**
The OCR pipeline converts uploaded images to RGB format, saves as JPG for optimal processing, and feeds to the EasyOCR deep learning model for text extraction. Advanced preprocessing algorithms (grayscale, thresholding, interpolation) are available in the codebase for enhanced accuracy when needed.

---

## 🎯 Problem Statement & Solution

### Problem
Medical prescriptions and test reports are often difficult for patients to understand due to:
- Poor handwriting
- Complex medical terminology
- Lack of context about conditions

### Solution
MediGuide provides:
- Automatic text extraction from images
- AI-powered plain-English explanations
- Audio output for accessibility
- Symptom awareness based on diagnoses

---

## 📁 Project Structure

```
mediguide python/
├── app.py                 # Main Streamlit application
├── helper.py              # Image processing & AI functions
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml       # API keys (not in version control)
├── .gitignore             # Excludes sensitive files
└── PROJECT_DOCUMENTATION.md
```

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   Create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   Open `http://localhost:8501` in your browser

---

## 📈 Potential Extensions

1. **Multi-language support** - EasyOCR supports 80+ languages
2. **Handwriting recognition** - Fine-tune model for doctors' handwriting
3. **Database integration** - Store patient history
4. **Mobile app** - React Native or Flutter frontend
5. **Local LLM** - Ollama integration for privacy
6. **PDF support** - Process multi-page documents
7. **Medicine database** - Cross-reference with drug interactions

---

## 🔧 Dependencies

```
Pillow          - Image processing
streamlit       - Web application framework
easyocr         - Optical Character Recognition
gtts            - Google Text-to-Speech
google-generativeai - Gemini AI API
numpy           - Numerical computing
```

---

## 👨‍💻 Technical Highlights

- **Custom implementation** of image processing algorithms (not just library calls)
- **Secure credential management** using Streamlit secrets
- **Error handling** for API failures and missing configurations
- **Modular code structure** separating UI from business logic
- **Regex-based text sanitization** for clean TTS output

---

## 📝 License

This project is for educational purposes as part of Semester 7 coursework.

---

*Built with ❤️ using Python, Streamlit, and Google Gemini AI*
