## 📄 AskMyPDF – AI-powered PDF Chatbot

> Upload any PDF and ask questions directly from its content.
> Built with **Python**, **Streamlit**, **Gemini AI**, and **Pinecone**.

---

### 🔗 Live Repo

[GitHub: axixatechnologies/AskMyPDF](https://github.com/axixatechnologies/AskMyPDF)

---

### 🚀 Features

- Upload any PDF file
- AI reads and stores the content semantically
- Ask any question from the PDF
- Answers are accurate, polite, and strictly based on the uploaded document
- Uses Pinecone for vector similarity
- Powered by Google Gemini AI (Gemini Flash)

---

### 🛠 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Google Generative AI (Gemini)](https://makersuite.google.com/)
- [Pinecone Vector DB](https://www.pinecone.io/)
- [LangChain](https://www.langchain.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

---

### 📦 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/axixatechnologies/AskMyPDF.git
cd AskMyPDF
```

2. **Create a `.env` file:**

```env
# .env or use the provided .env.example
GOOGLE_API_KEY="your gemini api key"
PINECONE_API_KEY="your pinecone api key"
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the app:**

```bash
streamlit run app.py
```

---

### 📁 Example `.env` file

```env
GOOGLE_API_KEY="your gemini api key"
PINECONE_API_KEY="your pinecone api key"
```

---

### ✅ To-Do / Future Enhancements

- OCR support for scanned PDFs
- Multiple PDF session management
- PDF preview beside chat

---

### 🤝 Contributing

Pull requests and stars are welcome!
For major changes, please open an issue first.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Developed By

Axixa Technologies – AI Training Team ([Yarendra](https://github.com/Yansu07) And [Sumit](https://github.com/))
