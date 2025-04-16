import streamlit as st
from tracker import FrenchVocabTracker


def main():
    st.set_page_config(page_title="French Learning Dashboard", layout="wide")
    tracker = FrenchVocabTracker()

    st.title("🇫🇷 Data-Driven French Learning Journey")

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
        if prediction['confidence'] > 0.3:
            st.metric("Estimated Fluency Date", prediction['prediction_date'])
        else:
            st.warning(
                f"Prediction Unavailable: {prediction['prediction_date']}")
    if not tracker.df.empty:
        st.plotly_chart(tracker.plot_progress(), use_container_width=True)
    else:
        st.info("📊 Add your first entry to see progress visualizations!")

    with st.expander("Add New Entry"):
        words = st.number_input("Words Learned Today", min_value=1)
        notes = st.text_area("Notes")
        if st.button("Save"):
            tracker.add_entry(words, notes)
            st.success("Entry saved!")

    st.subheader("Learning Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(tracker.df.set_index('date')['words_learned'])
    with col2:
        st.write("Weekly Summary")
        st.dataframe(tracker.df.groupby('week')[
                     'words_learned'].agg(['sum', 'mean']))


if __name__ == "__main__":
    main()
