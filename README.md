# NLP News Analysis: Topic Classification & Bias Detection

**A multi-stage NLP system that classifies news articles using DistilBERT and explains detected bias using Generative AI.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brandyn-Ewanek/News-Topic-Bias-Analysis/blob/main/NLP_Analysis.ipynb)

## 📰 Project Overview
Standard text classification tells you *what* a document is about, but not *why* it matters. This project bridges that gap by combining:
1.  **State-of-the-art Classification:** Fine-tuning a **DistilBERT** transformer to categorize news into 20 topics with **85.6% accuracy**.
2.  **Bias Detection:** Using VADER sentiment analysis to identify statistically significant negativity in political vs. technical topics.
3.  **Explainable AI (XAI):** Integrating **Google Gemini 1.5** to generate human-readable explanations for *why* specific articles were flagged as biased outliers.

## 🏗️ Methodology & Architecture
The project employs a "Champion vs. Challenger" approach to model selection:

### Phase 1: Classical Baselines
* **Feature Engineering:** TF-IDF Vectorization (removing stopwords, stemming).
* **Models:** Logistic Regression, Decision Trees, Random Forest (tuned), Gradient Boosting.
* **Result:** Logistic Regression was the strongest baseline (83% Test Accuracy), proving that simpler models are still competitive for high-dimensional text data.

### Phase 2: Modern Transformers (The Winner)
* **Model:** **DistilBERT** (Fine-tuned on 20 Newsgroups).
* **Performance:** Achieved **85.6% Accuracy**, outperforming all classical models.
* **Why DistilBERT?** It provided the best balance of accuracy and inference speed, capturing semantic context that TF-IDF missed.

### Phase 3: The "GenAI" Insight Layer
* **Sentiment:** VADER (Valence Aware Dictionary) calculated compound sentiment scores.
* **Bias Detection:** Identified that political categories (`talk.politics.guns`, `talk.politics.mideast`) had statistically significant negative skew compared to technical topics.
* **Generative Explanation:** An integrated function calls **Google Gemini 1.5 Flash** to analyze "Outlier Articles" (high sentiment deviation) and generate a natural language explanation for the anomaly.

## 📊 Key Results
| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **DistilBERT (Fine-Tuned)** | **85.61%** | **0.86** |
| Logistic Regression | 83.07% | 0.83 |
| Random Forest (Tuned) | 73.15% | 0.72 |
| Zero-Shot DistilBERT | 4.00% | 0.01 |

*Insight: Zero-shot classification failed completely (4%), proving the necessity of fine-tuning for domain-specific taxonomy.*

## 🛠️ Tech Stack
* **Deep Learning:** Hugging Face Transformers, PyTorch, DistilBERT.
* **Classical ML:** Scikit-Learn, Pandas, NumPy.
* **GenAI Integration:** Google Gemini API (1.5 Flash).
* **Visualization:** Matplotlib, Seaborn (WordClouds, Confusion Matrices).

## 📂 Repository Structure
* `notebooks/`: Contains the full analysis notebook (`NLP_Analysis.ipynb`).
* `reports/`: The detailed academic project report (PDF).

## 🚀 Future Improvements
* **Deployment:** Wrap the "Explainable Bias" function into a Streamlit app.
* **Model:** Test larger architectures like RoBERTa-Large for potentially higher accuracy.

## 👤 Author
**Brandyn Ewanek**
* [LinkedIn](https://www.linkedin.com/in/brandyn-ewanek-9733873b/)
* [Portfolio](https://github.com/Brandyn-Ewanek/)
