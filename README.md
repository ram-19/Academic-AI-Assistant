# Academic-AI-Assistant
Internship project on Academic AI Assistant by using Gemini API 


# Academic AI Assistant 🤖🎓

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dependencies](https://img.shields.io/badge/dependencies-see%20requirements.txt-brightgreen)](requirements.txt)

An intelligent academic companion powered by Google Gemini AI, offering document analysis, contextual Q&A, and exam preparation support.

![Demo Interface](https://via.placeholder.com/800x400.png?text=Academic+AI+Demo+Interface)

## Features ✨

- **Multi-Format Support**: PDF, DOCX, PPTX, XLSX, Images (PNG/JPG)
- **Smart Document Analysis**
  - Chapter-wise summaries
  - Exam-focused key points
  - Technical term explanations
- **Conversational Interface**
  - Context-aware responses
  - Chat history preservation
- **Advanced Processing**
  - OCR for scanned documents
  - Semantic search using embeddings
  - Dark/Light mode toggle

## Installation 🛠️

### Basic Setup

bash
# Clone repository
git clone https://github.com/yourusername/academic-ai-assistant.git
cd academic-ai-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR
## Windows: https://github.com/UB-Mannheim/tesseract/wiki
## Mac: brew install tesseract
## Linux: sudo apt install tesseract-ocr



## Configuration ⚙️

1. Get Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Create `.env` file:

```env
GEMINI_API_KEY=your_actual_key_here
```

## Usage 🚀

```bash
streamlit run app.py

or

python -m streamlit run app.py
```

**Example Interactions:**
```
[User] Summarize chapter 3 from uploaded PDF
[AI] Chapter 3 covers neural network architectures... (key points listed)

[User] Generate exam tips from this document
[AI] 1. Focus on backpropagation equations... (exam strategies)
```
## Directory Structure 📂

```
.
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env                # Environment template
├── chat_history.db     # Conversation database
└── documents/          # Sample test files
```


## License 📜

MIT License - See [LICENSE](LICENSE) for details

> **Note**: This project uses Google's Gemini API - Ensure compliance with [Google's AI Principles](https://ai.google/principles/)

---

**Acknowledgements** 🙏
- Google Gemini API Team
- Streamlit Community
- Tesseract OCR Maintainers

**Contact** 📧: [your.email@domain.com](mailto:ramprabhathsirimalla6@gmail.com)
