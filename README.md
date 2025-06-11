# 📰 News Article Summarizer & Q&A Bot

This Streamlit web app allows users to **summarize news articles** and **ask questions** based on the article content. It leverages the power of **Google's Gemini API** and **LangChain** for language understanding and **FAISS** for efficient retrieval.

🔗 **Live App**: [Click here to try it out](https://news-article-summarizer-and-app-bot.streamlit.app/)

---

## ✨ Features

- 🔗 Input any news article URL
- 📝 Automatically extract and summarize article content
- ❓ Ask follow-up questions about the article
- ⚡ Fast and accurate responses powered by Gemini 1.5 Flash
- 🧠 Uses vector embeddings + retrieval for context-aware Q&A
- 🌐 Minimal, user-friendly interface built with Streamlit

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/gajula21/news-article-summarizer-and-qa-bot.git
cd news-article-summarizer-and-qa-bot
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and add:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Make sure you have access to [Gemini](https://ai.google.dev/) and the `gemini-1.5-flash` model.

### 3. Install Dependencies

Use `pip` to install all required packages:

```bash
pip install -r requirements.txt
```

**Note**: If you face an error regarding `faiss`, install it using:

```bash
# For CPU only
pip install faiss-cpu

# OR for GPU
pip install faiss-gpu
```

---

## 🧪 Run the App Locally

```bash
streamlit run app.py
```

---

## 📦 Tech Stack

- **Streamlit** – UI framework
- **Newspaper3k** – Article parsing
- **LangChain** – LLM orchestration
- **Google Gemini API** – LLM (gemini-1.5-flash)
- **FAISS** – Vector store for Q&A retrieval
- **Pandas, NLTK** – Data handling and NLP

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to use, modify, and share!

---

## 👤 Author

**Vivek Gajula**  
🔗 [GitHub Profile](https://github.com/gajula21)

---
