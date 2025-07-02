📊 Company Feedback Sentiment Analyzer
Welcome to the Company Feedback Sentiment Analyzer – a Streamlit-powered web app that helps you analyze employee feedback collected from ITViec by performing:

✅ Sentiment Analysis (Positive/Negative)

📌 Topic Modeling (via LDA)

🧠 Clustering for improvement suggestions

🚀 How to Use
🔧 1. Installation (Local)
If you're running the project locally, make sure you have Python installed. Then:
pip install -r requirements.txt
▶️ 2. Run the App
In the terminal, run:
streamlit run test.py

🖥️ Dashboard Overview
After launching the app, the sidebar will guide you through four main sections:

1. 📊 Sentiment Analysis
Select a company from the dropdown list.

View the Positive and Negative Topics extracted via LDA.

Test your own feedback by typing it into the input box – the app will tell you if it's positive or negative!

2. 📌 Information Clustering
View topic clusters extracted from positive and negative feedback.

See automated suggestions for improvement based on common issues.

3. 📃 Project Overview
Learn about the objectives, tools, and steps involved in the project.

Designed for advisors, instructors, or stakeholders.

4. 🛠 Project Process
Go through step-by-step processes: from raw data → preprocessing → modeling → visualization.

Includes EDA, cleaning logs, and word cloud visualizations.

📂 File Structure
📁 final_project/
│
├── test.py                  # Main Streamlit app file
├── Processed_Reviews.xlsx  # Cleaned feedback data
├── requirements.txt        # Python dependencies
├── README.md               # ← You're here!
├── resources/              # Emoji, stopwords, teencode, etc.
│   ├── emojicon.txt
│   ├── english-vnmese.txt
│   ├── teencode.txt
│   ├── wrong-word.txt
│   └── vietnamese-stopwords.txt
└── OIP.jfif                # Optional image/logo shown on dashboard

💡 Notes for Users
📁 Ensure Processed_Reviews.xlsx is present in the same directory as app.py (or update the path).

🧠 The app uses Naive Bayes and LDA for analysis, with preprocessing tailored for Vietnamese text.

🛜 Can be deployed on Streamlit Cloud (just upload all files and click "Deploy").

👨‍💻 Developers
Pham Nhat Minh – mphamm12@gmail.com

Vo Quoc Hung – hung232803@gmail.com

Supervised by: Ms. Khuất Thùy Phương


