🐦 BrandPulse AI: Sentiment Analysis Dashboard
This project analyzes social media sentiment (tweets) to determine if a user's opinion is Positive, Negative, or Neutral. The app provides a high-tech Live Inference Engine and a Dynamic Dashboard, making it a perfect tool for brand managers to monitor public perception in real-time.

The backend is built using Dash (Flask) and machine learning with Logistic Regression and Naive Bayes, while the NLP pipeline utilizes NLTK for advanced text processing. The app is deployed live using Render.

🔗 Live App Link
You can access the deployed application here:
    (https://sentiment-analysis-on-twitter-data-nlp-13-evl9.onrender.com)

🚀 Click to Open BrandPulse AI Live App

🧩 Project Overview
1. Problem Statement
In the fast-paced world of social media, brands need to react quickly to customer feedback. Manually reading thousands of tweets is impossible. This project automates sentiment detection to provide instant, actionable insights.

2. Dataset
Source: Sentiment140 (Twitter Dataset)

Scale: 1.6 Million processed tweets.

Features: Raw tweet text, timestamps, and user handles.

Target: Sentiment (Positive / Negative / Neutral)

3. Model & NLP Pipeline
Algorithms: Logistic Regression (Primary), Naive Bayes, and LSTM (Validation).

NLP Preprocessing:

Noise Removal: Cleaning URLs, @mentions, and hashtags using Regex.

Stop-word Removal: Filtering out common words that don't carry sentiment.

Lemmatization: Using NLTK's WordNetLemmatizer to reduce words to their root form (e.g., "running" → "run").

Vectorization: TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into mathematical features.

4. Key Features
Real-Time Inference: Enter any custom text to see how the model classifies it instantly.

Interactive Pie Chart: A "Sentiment Distribution" chart that updates dynamically when a new prediction is made.

Performance Jitter: Live recalculation of Accuracy, Precision, Recall, and F1-Score to simulate a production monitoring environment.

Live Feed Simulation: A sidebar stream displaying real-world processed tweets to mimic a live data firehose.

⚙️ App Features
Custom Input: Type or paste any tweet to analyze its emotional tone.

Dynamic Metrics Table: View live updates for Accuracy, Precision, and F1-Score.

Confusion Matrix: A heatmap visualization showing how well the model distinguishes between classes.

Activity Trend: A line graph showing simulated sentiment volume over the last 24 hours.

🗂️ Project Structure
Plaintext
Sentiment-Analysis-NLP/
├─ app.py                  # Main Dash/Flask application logic
├─ logistic_model.pkl      # Pre-trained Logistic Regression model
├─ tfidf_vectorizer.pkl    # Pre-trained TF-IDF Vectorizer
├─ cleaned_tweets.csv      # Processed dataset for dashboard visuals
├─ requirements.txt        # Python dependencies (NLTK, Scikit-Learn, Dash)
├─ Presentation ppt.pdf   # Project presentation for review

💻 How to Run Locally
Clone the repository:

Bash
git clone https://github.com/NGayathri1234/Sentiment-Analysis-on-Twitter-Data-NLP-.git
cd Sentiment-Analysis-on-Twitter-Data-NLP-
Install dependencies:

Bash
pip install -r requirements.txt
Run the app:

Bash
python app.py
Access the Dashboard:
Open http://127.0.0.1:10000 in your web browser.

🛠️ Tech Stack
Backend: Python, Flask, Gunicorn

NLP & ML: NLTK, Scikit-Learn, Joblib, Pandas, NumPy

Frontend: Dash (Plotly), HTML5, CSS3 (Custom Sidebar Layout)

Deployment: Render (Web Service)

📊 How It Works
User Input: The user types a tweet into the TextArea.

Text Cleaning: The app applies Regex and Lemmatization to normalize the text.

Inference: The TF-IDF vectorizer transforms the text, and the Logistic Regression model predicts the sentiment.

UI Update: The callback function refreshes the Pie Chart, Trend Line, and Metrics Table simultaneously to reflect the new data.

📈 Model Performance
The model is optimized for a balance between speed and accuracy to ensure the dashboard remains responsive.

Accuracy: ~84% (Average for Sentiment140 dataset)

Precision/Recall: High weighted scores to handle neutral sentiment variance.

Visualization: Integrated annotated heatmaps for Confusion Matrix analysis.

👤 Author: N. Hamsalekha 

📧 Email: nhamsalekhahamsa@gmail.com

📞 Phone: +91 8431017029

🔗 GitHub Profile: github.com/NGayathri1234


📄 References
Twitter US Airline Sentiment dataset (Kaggle)

NLTK Documentation (Natural Language Toolkit)

Dash Plotly Documentation

Render Deployment Guides
