# 🧠 Synapse AI: The Smart, Self-Correcting Chatbot

Synapse AI is a highly advanced, local AI chatbot built to read your documents and answer questions about them. 

Unlike basic AI bots that guess or make things up (hallucinate) when they don't know the answer, Synapse AI acts like a smart researcher. It thinks step-by-step. If it searches your document and realizes it didn't find a good answer, it stops, fixes its search strategy, tries again, or even searches the live internet to find the truth!

---

## ✨ What Makes It Special?

- **🧠 A "Thinking" Brain (LangGraph)**: Instead of a straight line, this AI uses a flowchart. It loops through steps like `Analyze` → `Search` → `Check for Mistakes` → `Generate Answer` → `Double Check`.
- **⚡ Super Fast Typing (Streaming)**: It types out answers in real-time so you never have to wait staring at a loading screen.
- **🌐 Automatic Web Search**: If your uploaded PDF doesn't have the answer to your question, the AI realizes this and automatically searches DuckDuckGo to get live results from the internet!
- **📖 Smart Reading (Semantic Chunking)**: When you upload a document, it reads it sentence-by-sentence and groups related topics together, rather than just chopping pages in half.
- **🐘 Elephant Memory**: If you ask it a question it has already answered before, it replies instantly from memory without having to think again.
- **🕵️ Detective Logic (GraphRAG)**: It automatically connects the dots. If you ask a tricky question, it builds a mini "mind map" of the characters or topics in your document to figure out how they are connected before answering.
- **🖱️ Easy Drag-and-Drop**: You don't need to be a coder to add files. Just click the paperclip icon in the chat to upload a new PDF!
- **👁️ Watch It Think**: A cool sidebar next to the chat lights up in real-time, showing you exactly what step the AI's brain is currently working on.

---

## 🛠️ Built With

- **Backend (The Brain)**: Python, FastAPI, LangGraph
- **AI Models**: Ollama (Running locally and privately on your machine)
- **Database**: ChromaDB
- **Frontend (The Look)**: React.js (Vite)

---

## 🚀 How to Run It on Your Computer

### 1. What You Need First
Make sure you have downloaded and installed:
- Python (version 3.10 or newer)
- Node.js (version 18 or newer)
- [Ollama](https://ollama.ai/) (This runs the AI models locally)

Open your terminal and download the required AI models by typing:
```bash
ollama run llama3.2
ollama pull mxbai-embed-large
```

### 2. Start the Backend (The Server)
Open your terminal, go to the project folder, and run these commands to set up Python:

```bash
# Create a virtual environment
python -m venv venv

# Turn it on (Windows)
.\venv\Scripts\activate
# Turn it on (Mac/Linux)
source venv/bin/activate

# Install the required tools
pip install -r requirements.txt

# Start the server!
uvicorn src.app:app --reload
```

### 3. Start the Frontend (The Web Interface)
Open a **new** terminal window, go to the project folder, and run:
```bash
cd frontend
npm install
npm run dev
```

Finally, open your web browser and go to: `http://localhost:5173`

---

## 🎯 How to Use It

1. **Upload a File**: Click the paperclip icon next to the chat box and upload a PDF. Wait a moment for the AI to read and process it.
2. **Ask a Question**: Type a question about your document. Look at the right sidebar to watch the AI's brain light up as it searches and thinks!
3. **Test the Internet Fallback**: Try asking a question that is *definitely not* in your document (like "What is the weather today?"). Watch how the AI realizes it doesn't know, and automatically switches to searching the web!

---

## 📜 License
Free to use, modify, and learn from!
