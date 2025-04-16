import streamlit as st
from tracker import FrenchVocabTracker
import pandas as pd


def main():
    st.set_page_config(page_title="French Learning Dashboard", layout="wide")
    tracker = FrenchVocabTracker()

    st.title("🇫🇷 Data-Driven French Learning Journey")

    # Basic Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        current_words = tracker.get_current_words()
        st.metric("Total Words Learned", current_words)

    with col2:
        current_streak = tracker.get_current_streak()
        st.metric("Current Streak",
                  f"{current_streak} 🔥" if current_streak > 0 else "No active streak")

    with col3:
        prediction = tracker.predict_fluency()
        st.metric("Fluency Prediction",
                  prediction['prediction_date'],
                  f"Confidence: {prediction['confidence']*100:.0f}%")

    # Advanced Insights Section
    st.header("🚀 Learning Analytics")

    # Create tabs for different insights
    tab1, tab2, tab3 = st.tabs(
        ["Learning Intensity", "Vocabulary Diversity", "Learning Patterns"])

    # Advanced Insights from get_advanced_insights()
    insights = tracker.get_advanced_insights()

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Words/Day",
                      insights['learning_intensity']['avg_words_per_day'])
        with col2:
            st.metric("Most Productive Week",
                      insights['learning_intensity']['most_productive_week'] or "N/A")
        with col3:
            st.metric("Consistency Score",
                      f"{insights['learning_intensity']['consistency_score']}%")

        # Learning Intensity Chart
        st.subheader("Weekly Learning Intensity")
        weekly_data = tracker.df.groupby('week')['words_learned'].sum()
        st.bar_chart(weekly_data)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Categories",
                      insights['vocabulary_diversity']['unique_categories'])
        with col2:
            st.subheader("Top Categories")
            if insights['vocabulary_diversity']['top_categories']:
                for cat, count in insights['vocabulary_diversity']['top_categories'].items():
                    st.text(f"{cat}: {count} entries")
            else:
                st.text("No categories yet")

    with tab3:
        st.subheader("Peak Learning Hours")
        peak_hours = insights['learning_time']['peak_hours']
        if peak_hours:
            st.text(
                f"Top learning hours: {peak_hours[0]}:00 and {peak_hours[1]}:00")

        # Learning Time Distribution
        st.subheader("Learning Time Distribution")
        time_dist = insights['learning_time']['time_distribution']
        st.bar_chart(pd.Series(time_dist))

    # Add New Entry Expander
    with st.expander("Add New Entry"):
        words = st.number_input("Words Learned Today", min_value=1)
        notes = st.text_area("Notes (Optional, comma-separated categories)",
                             placeholder="grammar, vocabulary, idioms")
        if st.button("Save"):
            tracker.add_entry(words, notes)
            st.success("Entry saved!")

    # Cumulative Progress Chart
    st.subheader("Cumulative Vocabulary Progress")
    progress_fig = tracker.plot_progress()
    st.plotly_chart(progress_fig, use_container_width=True)


if __name__ == "__main__":
    main()
