# Sentiment Analysis with Snorkel and Weak Supervision

This lab demonstrates building a sentiment classifier for movie reviews using Snorkel's weak supervision framework without manual labeling.

## Dataset

IMDB Movie Reviews dataset (automatically downloaded)

## Prerequisites

* Python 3.8+
* Virtual environment (recommended)

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy English model:
```bash
python -m spacy download en_core_web_sm
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open `01_sentiment_tutorial.ipynb` and run all cells sequentially

## Lab Steps

1. **Import and Load Data** - Load IMDB movie reviews dataset
2. **Explore Data** - View sample reviews and data structure
3. **Write Keyword LFs** - Create simple keyword-based labeling functions
4. **Test Initial LFs** - Apply and analyze labeling functions
5. **Template-based LFs** - Build reusable LF templates
6. **Pattern Matching LFs** - Use regex for patterns like "10/10" ratings
7. **Third-party Model LFs** - Leverage TextBlob sentiment analyzer
8. **Heuristic LFs** - Implement rule-based heuristics
9. **NLP-based LFs** - Use spaCy for adjective analysis
10. **Combine All LFs** - Aggregate all 18 labeling functions
11. **Train Label Model** - Use Snorkel to combine noisy LF outputs
12. **Compare with Baseline** - Evaluate against majority voting
13. **Filter Unlabeled Data** - Remove uncovered data points
14. **Train Final Classifier** - Train discriminative model (XGBoost/LogReg/RandomForest)
15. **Evaluate Model** - View accuracy and classification report

## Key Concepts

**Labeling Functions (LFs)**: Programmatic rules that label data (can be noisy)

**Label Model**: Combines noisy LF outputs without ground truth labels

**Discriminative Classifier**: Final model trained on weak labels

## Expected Results

- **LF Coverage**: 70-80% of training data
- **Label Model Accuracy**: 75-85%
- **Final Classifier Accuracy**: 80-90%

## Model Options

Modify Cell 37 to change classifier:
```python
# Logistic Regression (default)
from sklearn.linear_model import LogisticRegression
sklearn_model = LogisticRegression(C=1e3, solver="liblinear")

# XGBoost (best performance)
from xgboost import XGBClassifier
sklearn_model = XGBClassifier(n_estimators=100, max_depth=6)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
sklearn_model = RandomForestClassifier(n_estimators=100)
```

## Files

* `01_sentiment_tutorial.ipynb` - Main tutorial notebook
* `utils_sentiment.py` - Helper functions
* `requirements.txt` - Dependencies

## Troubleshooting

**spaCy model not found**: Run `python -m spacy download en_core_web_sm`

**Dataset download fails**: Check internet connection or manually place `IMDB_Dataset.csv` in `data/` folder

**Low accuracy**: Add more labeling functions or refine existing ones
