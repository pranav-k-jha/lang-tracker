import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


class FrenchVocabTracker:
    def __init__(self, data_path='data/vocab.csv'):
        # Ensure data directory exists
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        # Define expected columns
        self.columns = ['date', 'words_learned', 'notes']

        # Create the CSV with headers if it doesn't exist
        if not os.path.exists(data_path):
            pd.DataFrame(columns=self.columns).to_csv(data_path, index=False)

        # Load data, handle empty file
        try:
            self.df = pd.read_csv(data_path, parse_dates=['date'])
        except pd.errors.EmptyDataError:
            self.df = pd.DataFrame(columns=self.columns)

        self._preprocess_data()

    def _preprocess_data(self):
        if not self.df.empty:
            # Cumulative total of words learned
            self.df['cumulative_words'] = self.df['words_learned'].cumsum()
            # ISO week number
            self.df['week'] = self.df['date'].dt.isocalendar().week
            # Consecutive-day streaks
            self.df['streak'] = self._calculate_streaks()
        else:
            # Initialize empty columns to avoid KeyErrors later
            for col in ['cumulative_words', 'week', 'streak']:
                self.df[col] = pd.Series(dtype=int)

    def _calculate_streaks(self):
        if self.df.empty:
            return []
        streaks = []
        current_streak = 0
        prev_date = None
        for dt in self.df['date']:
            if prev_date and (dt - prev_date).days == 1:
                current_streak += 1
            else:
                current_streak = 0
            streaks.append(current_streak)
            prev_date = dt
        return streaks

    def get_current_words(self):
        return int(self.df['cumulative_words'].iloc[-1]) if not self.df.empty else 0

    def get_current_streak(self):
        return int(self.df['streak'].iloc[-1]) if not self.df.empty else 0

    def add_entry(self, words_learned, notes=""):
        new_row = pd.DataFrame({
            'date': [datetime.today().strftime('%Y-%m-%d')],
            'words_learned': [words_learned],
            'notes': [notes]
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self._preprocess_data()

    def plot_progress(self):
        fig = px.line(
            self.df,
            x='date',
            y='cumulative_words',
            title='Cumulative Vocabulary Progress',
            labels={'cumulative_words': 'Total Words Learned'}
        )
        fig.add_vline(x=self.df['date'].max(), line_dash="dash")
        return fig

    def predict_fluency(self, target_words=5000):
        base = {
            'current_words': self.get_current_words(),
            'daily_rate': 0,
            'prediction_date': "Insufficient data 📉",
            'confidence': 0
        }

        # Not enough data
        if self.df.empty or len(self.df[self.df['words_learned'] > 0]) < 2:
            needed = max(0, 2 - len(self.df[self.df['words_learned'] > 0]))
            base['prediction_date'] = (
                f"Need {needed} more entr{'y' if needed==1 else 'ies'}!"
                if needed > 0 else base['prediction_date']
            )
            return base

        # Prepare regression data
        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df['cumulative_words'].values

        model = LinearRegression().fit(X, y)
        rate = max(model.coef_[0], 0.1)  # avoid negative slope
        days_needed = max((target_words - y[-1]) / rate, 0)
        pred_date = self.df['date'].iloc[-1] + pd.Timedelta(days=days_needed)
        raw_conf = model.score(X, y)
        conf = min(max(raw_conf, 0), 0.99)  # clamp between 0 and 0.99

        return {
            'current_words': int(y[-1]),
            'daily_rate': round(rate, 1),
            'prediction_date': pred_date.strftime('%Y-%m-%d'),
            'confidence': conf
        }
