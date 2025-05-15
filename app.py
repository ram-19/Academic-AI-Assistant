import os
import time
import sqlite3
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pdfplumber
import docx
import openpyxl
from pptx import Presentation
import pytesseract
import streamlit.components.v1 as components
from typing import List, Dict, Optional
import google.generativeai as genai
from google.api_core import retry

# Configure Tesseract path (Update for your OS)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load environment variables
load_dotenv()

# Database Setup
class ChatDatabase:
    def __init__(self, db_name: str = "chat_history.db"):
        self.conn = sqlite3.connect(db_name)
        self._create_tables()

    def _create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    role TEXT CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT,
                    file_context TEXT,
                    embedding BLOB
                )
            """)

    def save_message(self, role: str, content: str, file_context: str = "", embedding: bytes = b''):
        allowed_roles = ['user', 'assistant', 'system']
        role = role if role in allowed_roles else 'system'
        
        with self.conn:
            self.conn.execute(
                """INSERT INTO conversations 
                (role, content, file_context, embedding) 
                VALUES (?, ?, ?, ?)""",
                (role, content, file_context[:10000], embedding)
            )

    def get_history(self, limit: int = 100) -> List[Dict]:
        cursor = self.conn.execute(
            "SELECT role, content, timestamp FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return [{"role": row[0], "content": row[1], "timestamp": row[2]} for row in cursor.fetchall()]

# Custom CSS
PROFESSIONAL_CSS = """
<style>
    [data-theme="dark"] {
        --primary: #818cf8;
        --secondary: #a78bfa;
        --background: #1a1b26;
        --text: #a9b1d6;
    }
    .stApp {
        background-color: var(--background) !important;
        color: var(--text) !important;
    }
    .user-message {
        background: var(--primary) !important;
        border-radius: 15px 15px 0 15px !important;
        max-width: 80%;
        float: right;
        margin: 8px 0;
        padding: 12px 18px;
    }
    .bot-message {
        background: var(--background) !important;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px 15px 15px 0 !important;
        max-width: 80%;
        float: left;
        margin: 8px 0;
        padding: 12px 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
"""

class AcademicAssistant:
    MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB
    PDF_PAGE_LIMIT = 50
    TEXT_LIMIT = 50000
    EMBEDDING_MODEL = "models/embedding-001"

    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')
        
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Missing Gemini API key in .env file")
            st.stop()
            
        self.db = ChatDatabase()
        self._init_session()
        self._init_ui()
        st._config.set_option('server.maxUploadSize', 1024)

    def _init_session(self):
        defaults = {
            'theme': 'light',
            'uploaded_files': [],
            'processing': False,
            'embeddings': {}
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _init_ui(self):
        st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)
        components.html("""
            <script>
            function toggleTheme() {
                const theme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
                document.documentElement.setAttribute('data-theme', theme);
            }
            </script>
            <button onclick="toggleTheme()" style="
                position:fixed;bottom:20px;right:20px;z-index:999;
                border-radius:50%;padding:10px;border:none;
                background:#2d3436;color:white;">
                🌙/☀️
            </button>
        """, height=0)

    @retry.Retry()
    def get_embedding(self, text: str) -> List[float]:
        """Get free embeddings using Gemini"""
        try:
            return genai.embed_content(
                model=self.EMBEDDING_MODEL,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )['embedding']
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            return []

    def _process_file(self, file) -> Optional[str]:
        try:
            if file.size > self.MAX_FILE_SIZE:
                raise ValueError(f"File size exceeds 1GB limit")

            content = []
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages[:self.PDF_PAGE_LIMIT]:
                        try:
                            text = page.extract_text()
                        except:
                            text = page.crop(page.mediabox).extract_text()
                        content.append(text)
                        if len("".join(content)) >= self.TEXT_LIMIT:
                            break

            elif file.type == "text/plain":
                content.append(file.read().decode()[:self.TEXT_LIMIT])

            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = docx.Document(file)
                content = [para.text for para in doc.paragraphs][:self.TEXT_LIMIT]

            elif file.type.startswith("image/"):
                img = Image.open(file)
                content.append(pytesseract.image_to_string(img))

            processed_text = "\n".join(content)[:self.TEXT_LIMIT]
            
            # Generate and store embedding
            embedding = self.get_embedding(processed_text)
            st.session_state.embeddings[file.name] = embedding
            
            return processed_text

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")
            return None

    def _display_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Control Panel")
            uploaded_file = st.file_uploader(
                "Upload Documents (Max 1GB)",
                type=["pdf", "txt", "docx", "xlsx", "pptx", "png", "jpg", "jpeg"],
                help="Supported formats: PDF, Text, Word, Excel, PowerPoint, Images",
                accept_multiple_files=False
            )
            
            if uploaded_file:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    if content := self._process_file(uploaded_file):
                        st.session_state.uploaded_files.append({
                            "name": uploaded_file.name,
                            "content": content
                        })
                        self.db.save_message(
                            "system", 
                            f"Uploaded {uploaded_file.name}", 
                            content,
                            embedding=str(st.session_state.embeddings[uploaded_file.name]).encode()
                        )

            st.subheader("📚 History")
            for msg in self.db.get_history(10):
                st.markdown(f"**{msg['role'].title()}** ({msg['timestamp'][:16]})")
                st.caption(msg['content'][:100] + "...")

    def _display_chat(self):
        st.markdown("<h1 style='text-align:center'>🎓 Academic AI Assistant (Gemini)</h1>", unsafe_allow_html=True)

        for msg in self.db.get_history(20):
            cls = "user-message" if msg['role'] == 'user' else "bot-message"
            st.markdown(f"<div class='{cls}'>{msg['content']}</div>", unsafe_allow_html=True)

        if prompt := st.chat_input("Ask about your research..."):
            try:
                # Retrieve relevant context using embeddings
                query_embedding = self.get_embedding(prompt)
                similarities = []
                
                for file in st.session_state.uploaded_files:
                    if file['name'] in st.session_state.embeddings:
                        emb = st.session_state.embeddings[file['name']]
                        similarity = sum(a*b for a,b in zip(query_embedding, emb))
                        similarities.append((similarity, file['content']))
                
                # Get top 3 relevant contexts
                context = "\n\n".join(
                    [content for _, content in sorted(similarities, reverse=True)[:3]]
                )

                # Generate response
                response = self.model.generate_content(
                    f"Document Context:\n{context[:30000]}\n\nQuestion: {prompt}",
                    generation_config={
                        "temperature": 0.7,
                        "max_output_tokens": 2000
                    },
                    safety_settings={
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                )

                if response.text:
                    self.db.save_message("assistant", response.text)
                    st.rerun()
                else:
                    st.error("No response generated")
                    if response.prompt_feedback:
                        st.write(response.prompt_feedback)

            except Exception as e:
                st.error(f"Gemini Error: {str(e)}")
                self.db.save_message("system", f"Error: {str(e)}")

    def run(self):
        self._display_sidebar()
        self._display_chat()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Academic AI Assistant (Gemini)",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    assistant = AcademicAssistant()
    assistant.run()