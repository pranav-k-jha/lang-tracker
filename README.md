# 🇫🇷 French Vocabulary Tracker & Fluency Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- [![CI/CD](https://github.com/yourHere's a professional README.md for your GitHub repository: -->

### Usage

```bash
# Launch Streamlit dashboard
streamlit run src/dashboard.py

# Run unit tests
pytest tests/
```

## 🛠️ Tech Stack

### Data Processing

![Pandas](https://img.shields.io/badge/Pandas-2C2D72?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white)

### Visualization

![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

### Machine Learning

![Scikit-learn](https://img.shields.io/badge/ScikitLearn-F7931E?logo=scikit-learn&logoColor=white)

### Infrastructure

![SQLite](https://img.shields.io/badge/SQLite-07405E?logo=sqlite&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?logo=github-actions&logoColor=white)

## 📂 Data Structure

### vocab.csv

| Column           | Description                       | Example           |
| ---------------- | --------------------------------- | ----------------- |
| date             | Learning date (YYYY-MM-DD)        | 2024-02-15        |
| words_learned    | Number of new words learned       | 25                |
| notes            | Additional learning notes         | "Irregular verbs" |
| cumulative_words | Running total of words learned    | 1250              |
| week             | ISO week number                   | 7                 |
| streak           | Current consecutive learning days | 5                 |

## 📊 Sample Dashboard Metrics

| Metric               | Description                          |
| -------------------- | ------------------------------------ |
| Daily Learning Rate  | Average words learned per day        |
| Retention Efficiency | Estimated word retention percentage  |
| Peak Learning Hours  | Time of day with most productivity   |
| Vocabulary Diversity | Unique word categories learned       |
| Fluency Confidence   | ML model prediction confidence score |

## 🤝 Contributing

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- Inspired by spaced repetition research
- Built with amazing open-source tools
- Special thanks to the French learning community

## 📬 Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pranav-k-jha/)
[![Email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:pranav.jha@mail.concordia.ca)
