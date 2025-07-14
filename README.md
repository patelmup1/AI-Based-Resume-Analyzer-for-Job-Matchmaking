
# 🤖 AI-Powered Resume Analyzer & Job Matchmaking System

## 📌 Overview

In today's hiring landscape, job seekers often submit generic resumes, and recruiters spend hours filtering through applications manually. This project solves both problems with an **AI-based Resume Analyzer** that parses resumes, matches them against job descriptions, and provides **smart, LLM-generated feedback**.

---

## 🎯 Objectives

- Parse resumes to extract skills, experience, and education
- Compare resume content with job descriptions using keyword & semantic similarity
- Generate match score and missing skill gaps
- Provide AI-generated improvement suggestions using LLMs (Mistral-7B)

---

## 🧠 AI/ML Components

| Feature                 | Model / Technique                  | Purpose                                      |
|------------------------|-------------------------------------|----------------------------------------------|
| Resume Parsing         | spaCy                               | Extract structured data from resumes         |
| Keyword Matching       | TF-IDF + Cosine Similarity          | Basic similarity scoring                     |
| Semantic Matching      | Sentence-BERT (`MiniLM-L6-v2`)      | Deep contextual similarity                   |
| Resume Suggestions     | Mistral-7B-Instruct (via HF)        | Generate personalized feedback               |
| Experience Detection   | Regex + NLP                         | Identify experience gaps                     |
| Skill Extraction       | Rule-based (NER upgradeable)        | Identify technical and soft skills           |
| Visualization          | Streamlit / matplotlib / seaborn    | Display match score and analytics            |

---

## 🧱 Project Structure

```
resume-analyzer-ai/
├── notebooks/
│   └── Resume_Analyzer_v2_Enhanced_Edition.ipynb   # Google Colab-compatible notebook
├── src/
│   └── resume_analyzer_v2_enhanced_edition.py      # Production-ready Python script
├── screenshots/                                    # Output charts & results
├── data/                                           # Sample resumes and job descriptions
├── requirements.txt                                # Python dependencies
└── README.md                                       # Project documentation
```

---

## 🚀 How to Run

### 🔹 1. Clone the Repository
```bash
git clone https://github.com/yourusername/resume-analyzer-ai.git
cd resume-analyzer-ai
```

### 🔹 2. (Option A) Run Notebook in Google Colab

- Open: `notebooks/Resume_Analyzer_v2_Enhanced_Edition.ipynb`
- Upload resumes via file upload cell
- Enter job description in text block
- View parsed results, match scores, charts, and suggestions

### 🔹 3. (Option B) Run Script in Python Environment

```bash
pip install -r requirements.txt
python src/resume_analyzer_v2_enhanced_edition.py
```

> ⚠️ Ensure you add your Hugging Face token for LLM features.

---

## 📸 Sample Output

![Example](./screenshots/sample-output.png)

- Parsed resume data
- Match Score (TF-IDF + BERT)
- Missing Skills and Experience Gap
- AI Suggestions (e.g., “Highlight AWS experience”)

---

## 🔮 Future Enhancements

- Train a custom NER model for dynamic skill extraction
- Multilingual support for global applicants
- Resume editor with live AI feedback
- Integration with LinkedIn / Indeed APIs
- Add fairness & bias detection for recruiter use

---

## 📚 References

- [spaCy](https://spacy.io)
- [Hugging Face Transformers](https://huggingface.co)
- [Sentence-BERT](https://www.sbert.net/)
- [Mistral AI](https://huggingface.co/mistralai)
- [scikit-learn](https://scikit-learn.org/)

---

## 📃 License

This project is licensed under the MIT License.
