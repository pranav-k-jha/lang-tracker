import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime


class FrenchVocabTracker:
    def __init__(self, data_path='data/vocab.csv'):
        # Create directory if not exists
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        # Initialize empty DataFrame structure
        self.columns = ['date', 'words_learned', 'notes']

        # Create file with headers if not exists
        if not os.path.exists(data_path):
            pd.DataFrame(columns=self.columns).to_csv(data_path, index=False)

        try:
            self.df = pd.read_csv(data_path, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            self.df = pd.DataFrame(columns=self.columns)

        self._preprocess_data()

    def _preprocess_data(self):
        if not self.df.empty:
            self.df['cumulative_words'] = self.df['words_learned'].cumsum()
            self.df['week'] = self.df['date'].dt.isocalendar().week
            self.df['streak'] = self._calculate_streaks()
        else:
            # Add empty columns if no data exists
            self.df['cumulative_words'] = pd.Series(dtype='int')
            self.df['week'] = pd.Series(dtype='int')
            self.df['streak'] = pd.Series(dtype='int')

    def _calculate_streaks(self):
        streaks = []
        current_streak = 0
        for date in self.df['date'].diff().dt.days:
            if date == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
        return streaks

    def add_entry(self, words_learned, notes=""):
        new_entry = pd.DataFrame({
            'date': [datetime.today().strftime('%Y-%m-%d')],
            'words_learned': [words_learned],
            'notes': [notes]
        })
        self.df = pd.concat([self.df, new_entry], ignore_index=True)
        self._preprocess_data()

    def plot_progress(self):
        fig = px.line(self.df, x='date', y='cumulative_words',
                      title='Cumulative Vocabulary Progress',
                      labels={'cumulative_words': 'Total Words Learned'})
        fig.add_vline(x=self.df['date'].max(), line_dash="dash")
        return fig

    def predict_fluency(self, target_words=5000):
        X = self.df.index.values.reshape(-1, 1)
        y = self.df['cumulative_words'].values
        model = LinearRegression().fit(X, y)

        current_words = self.df['cumulative_words'].iloc[-1]
        days_needed = (target_words - current_words) / model.coef_[0]
        prediction_date = self.df['date'].iloc[-1] + \
            pd.Timedelta(days=days_needed)

        return {
            'current_words': current_words,
            'daily_rate': round(model.coef_[0], 1),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'confidence': model.score(X, y)
        }
