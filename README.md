
# 🤖 AI-Powered Resume Analyzer & Job Matchmaking System

## 📌 Overview

In today’s job market, candidates often submit generic resumes that don’t align well with specific job roles. Meanwhile, recruiters struggle with the time-consuming task of manually screening applications.

This project provides an **AI-powered Resume Analyzer** that automates resume parsing, matches resumes with job descriptions, and offers **smart suggestions** to optimize hiring for both candidates and employers.

---

## 🎯 Objectives

- Parse and analyze resumes using NLP techniques
- Match resume content against job descriptions using AI models
- Generate a job-match score based on keyword and semantic relevance
- Provide actionable suggestions for resume improvement using LLMs

---

## 🧠 AI/ML Components Used

| Component              | Technology / Model                  | Purpose                                |
|------------------------|-------------------------------------|----------------------------------------|
| Resume Parsing         | `spaCy` (en_core_web_sm)            | Extract structured data from resumes   |
| Keyword Matching       | `TF-IDF + Cosine Similarity`        | Measure keyword relevance              |
| Semantic Matching      | `Sentence-BERT (all-MiniLM-L6-v2)`  | Understand context beyond keywords     |
| Resume Suggestions     | `Mistral-7B-Instruct` (LLM)         | Generate personalized feedback         |
| Experience Extraction  | Regex (upgradeable to ML)           | Detect mentions of experience duration |
| Skill Extraction       | Rule-based (upgradeable to NER)     | Identify technical and soft skills     |
| Visualization          | Streamlit / Plotly                  | Display insights & match score         |

---

## 🧱 System Architecture

```
[ Resume Upload ] 
       ↓
[ NLP Parsing (spaCy) ]
       ↓
[ TF-IDF & BERT Matching ]
       ↓
[ LLM Feedback Generation ]
       ↓
[ Results Dashboard (Streamlit) ]
```

---

## 🚀 Tech Stack

- **Language**: Python 3.10+
- **NLP**: spaCy, Sentence-BERT (`sentence-transformers`)
- **ML**: scikit-learn (TF-IDF, cosine similarity)
- **LLM**: Mistral-7B-Instruct (via Hugging Face API)
- **Frontend**: Streamlit or Flask (for UI)
- **Deployment**: Heroku / AWS / GCP

---

## 📥 Installation

```bash
git clone https://github.com/yourusername/resume-analyzer-ai.git
cd resume-analyzer-ai
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Folder Structure

```
resume-analyzer-ai/
│
├── data/                  # Sample resumes and job descriptions
├── models/                # Model configs or embeddings
├── utils/                 # Text parsing, match scoring, etc.
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📸 Example Output

![output](./screenshots/sample-output.png)

- **Resume Parsed Output**
- **Job Match Score** (e.g., 82%)
- **AI Suggestions**:
  - Add certifications like AWS or GCP
  - Emphasize team/project experience

---

## 🔮 Future Scope

- Integrate custom NER model for dynamic skill extraction
- Add support for multilingual resumes
- Improve scoring logic with learning-based feedback loops
- Resume builder with real-time AI feedback
- Connect with job boards (e.g., LinkedIn, Indeed)

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- [spaCy](https://spacy.io)
- [Hugging Face Transformers](https://huggingface.co)
- [Sentence-BERT](https://www.sbert.net/)
- [scikit-learn](https://scikit-learn.org/)
- [Mistral AI](https://huggingface.co/mistralai)
