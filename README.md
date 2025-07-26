# Fake News AI

A demo project to generate and detect fake news using AI.

## Features
- **Fake News Generator:** Fine-tuned GPT-2 model generates realistic fake news articles from a prompt.
- **Fake News Detector:** TF-IDF + Logistic Regression model classifies news as Fake or Real.
- **Streamlit Web App:** User-friendly interface for both generation and detection.

## Project Structure
```
.
├── app/
│   └── streamlit_app.py         # Streamlit web app
├── data/
│   ├── Fake.csv                 # Raw fake news data
│   ├── True.csv                 # Raw real news data
│   ├── news_dataset.csv         # (Unused/empty)
│   └── processed/
│       └── cleaned_news.csv     # Preprocessed data
├── detector/
│   └── train_detector.py        # Detector training script
├── generator/
│   ├── train_generator.py       # Generator training script
│   └── generate_fake_news.py    # Sample generation script
├── models/
│   ├── bert_detector/
│   │   ├── logreg_model.joblib  # Trained detector model
│   │   └── tfidf_vectorizer.joblib # Trained vectorizer
│   └── gpt2_model/              # Fine-tuned GPT-2 model/tokenizer (to be generated)
├── utils/
│   └── preprocessing.py         # Data cleaning utilities
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd fake-news-ai
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download NLTK stopwords:**
   The first run will auto-download, or run:
   ```python
   import nltk; nltk.download('stopwords')
   ```

## Data Preparation
- Place `Fake.csv` and `True.csv` in the `data/` directory.
- Preprocessing will run automatically when training scripts are executed.

## Training
### 1. Train the Detector
```bash
python detector/train_detector.py
```
- Trains a TF-IDF + Logistic Regression model.
- Saves model and vectorizer in `models/bert_detector/`.

### 2. Train the Generator
```bash
python generator/train_generator.py
```
- Fine-tunes GPT-2 on fake news data.
- Saves model and tokenizer in `models/gpt2_model/`.

## Running the App
```bash
streamlit run app/streamlit_app.py
```
- Open the provided local URL in your browser.
- Use the Generator and Detector tabs.

## Notes
- For GPU acceleration, ensure PyTorch is installed with CUDA support.
- The detector is currently a classical ML model. For a BERT-based detector, see TODOs below.

## TODOs
- [ ] Implement a BERT-based fake news detector
- [ ] Add model evaluation scripts/notebooks
- [ ] Improve UI/UX of the Streamlit app
- [ ] Add tests and validation
- [ ] Clean up unused files

## Credits
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)
