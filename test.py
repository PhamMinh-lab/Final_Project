import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# Set page configuration
st.sidebar.title("🔧 Settings")
option = st.sidebar.selectbox("Choose a category", ["Project Overview", "Sentiment Analysis", "Information CLustering", "Project Process"], key="main_selectbox")
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍🎓 Group Members")
st.sidebar.markdown("1. Pham Nhat Minh \nEmail: mphamm12@gmail.com  \n2. Vo Quoc Hung \nEmail: hung232803@gmail.com")

# ----------------- SENTIMENT ANALYSIS -----------------
if option == "Sentiment Analysis":
    st.title("📊 Sentiment Analysis of Feedback in ITViec for Companies")

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_excel("Processed_Reviews.xlsx")

    data = load_data()

    # Map 'positive' -> 1, 'negative' -> 0
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
    data = data.dropna(subset=['sentiment', 'text_clean'])
    data = data[data['text_clean'].astype(str).str.strip() != ""]

    # Preprocessing for model
    # Filter rows with valid text_clean before vectorizing
    valid_data = data.dropna(subset=['text_clean'])
    text_clean = valid_data['text_clean'].astype(str)
    y = valid_data['sentiment'].values  # Make sure y is aligned to X
    ######################################
    # y = np.array(valid_data['sentiment'].tolist())

    ###################################
    
    vectorizer = CountVectorizer(max_df=0.95, min_df=2)
    X_vect = vectorizer.fit_transform(text_clean)

    # Oversample
    oversample = RandomOverSampler(sampling_strategy='auto', random_state=42)

    ##############################################


    #############################################
    # X_resampled, y_resampled = oversample.fit_resample(X_vect, y.to_numpy())


    # Split train/test
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_vect, y, random_state=42)
    nvmodel = MultinomialNB()
    nvmodel.fit(X_train1, y_train1)

    # Company selector
    companies = data[['id', 'Company Name']].drop_duplicates().reset_index(drop=True)
    selected_company_name = st.selectbox("Select a company", companies['Company Name'])
    selected_company_id = companies[companies['Company Name'] == selected_company_name]['id'].values[0]

    # Filter company data
    company_data = data[data['id'] == selected_company_id].reset_index(drop=True)

    def lda_topics(text_data, n_topics=5):
        vectorizer_lda = CountVectorizer(min_df=1)
        doc_term_matrix = vectorizer_lda.fit_transform(text_data)
        lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_model.fit(doc_term_matrix)
        feature_names = vectorizer_lda.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
            topics.append(f"Topic #{topic_idx+1}: {', '.join(top_words)}")
        return topics

    st.subheader("😊 Positive Feedback Topics")
    pos_feedback = company_data['positive_clean'].dropna().astype(str)
    if not pos_feedback.empty:
        for topic in lda_topics(pos_feedback):
            st.markdown(f"- {topic}")
    else:
        st.info("No positive feedback available.")

    st.subheader("😡 Negative Feedback Topics")
    neg_feedback = company_data['negative_clean'].dropna().astype(str)
    if not neg_feedback.empty:
        for topic in lda_topics(neg_feedback):
            st.markdown(f"- {topic}")
    else:
        st.info("No negative feedback available.")

    # User input for custom prediction
    st.subheader("✍️ Test Your Own Feedback")
    user_input = st.text_area("Enter your feedback text:")

    if user_input:
        input_vector = vectorizer.transform([user_input])
        prediction = nvmodel.predict(input_vector)[0]
        if prediction == 1:
            st.success("🟢 Predicted as Positive Feedback")
        else:
            st.error("🔴 Predicted as Negative Feedback")

#----------------------Information CLustering----------------------
elif option == "Information CLustering":
    st.title("📊 Information Clustering for Feedbacks of Companies on ITViec")

    # Load data
    @st.cache_data
    def load_cluster_data():
        return pd.read_excel("Processed_Reviews.xlsx")

    df = load_cluster_data()

    # Preprocess: drop rows with NaN
    df = df.dropna(subset=["positive_clean", "negative_clean", "sentiment", "Company Name"])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1, 2))
    positive_matrix = vectorizer.fit_transform(df['positive_clean'].astype(str))
    negative_matrix = vectorizer.transform(df['negative_clean'].astype(str))

    # LDA topic modeling
    lda_positive = LatentDirichletAllocation(n_components=4, random_state=42)
    lda_positive_matrix = lda_positive.fit_transform(positive_matrix)

    lda_negative = LatentDirichletAllocation(n_components=3, random_state=42)
    lda_negative_matrix = lda_negative.fit_transform(negative_matrix)

    # Clustering
    df['pos_cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(lda_positive_matrix)
    df['neg_cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(lda_negative_matrix)

    # Get topic keywords
    def get_top_words(model, feature_names, n_top_words=10):
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics[topic_idx] = top_words
        return topics

    pos_topics = get_top_words(lda_positive, vectorizer.get_feature_names_out())
    neg_topics = get_top_words(lda_negative, vectorizer.get_feature_names_out())

    # Select company
    st.subheader("🏢 Select a Company")
    companies = df[['id', 'Company Name']].drop_duplicates().reset_index(drop=True)
    selected_company = st.selectbox("Choose a company", companies['Company Name'])
    selected_id = companies[companies['Company Name'] == selected_company]['id'].values[0]

    company_data = df[df['id'] == selected_id]

    st.markdown(f"##  Clustering Analysis for `{selected_company}`")

    # Show POSITIVE clusters
    st.markdown("### 😊 Positive Feedback Clusters:")
    pos_data = company_data[company_data['sentiment'] == 'positive']
    if not pos_data.empty:
        for cluster in pos_data['pos_cluster'].unique():
            st.markdown(f"- **Cluster {cluster}**: {', '.join(pos_topics.get(cluster, []))}")
    else:
        st.info("No positive feedback found.")

    # Show NEGATIVE clusters
    st.markdown("### 😟 Negative Feedback Clusters:")
    neg_data = company_data[company_data['sentiment'] == 'negative']
    if not neg_data.empty:
        for cluster in neg_data['neg_cluster'].unique():
            topic_words = neg_topics.get(cluster, [])
            st.markdown(f"- **Cluster {cluster}**: {', '.join(topic_words)}")

            # Suggest improvement based on topic words
            if topic_words:
                suggestions = []
                for word in topic_words:
                    if word in ["chậm", "delay", "trễ", "lâu"]:
                        suggestions.append("🔧 Improve response time or hiring process.")
                    elif word in ["lương", "chính sách", "phúc lợi"]:
                        suggestions.append("💰 Consider reviewing salary and benefits.")
                    elif word in ["quản lý", "sếp", "boss", "lãnh đạo"]:
                        suggestions.append("🧑‍💼 Enhance leadership communication or training.")
                    elif word in ["môi trường", "áp lực", "stress", "nội bộ"]:
                        suggestions.append("🏢 Improve internal culture or work environment.")
                if suggestions:
                    st.markdown("**🔎 Suggested Improvements:**")
                    for s in set(suggestions):
                        st.markdown(f"- {s}")
    else:
        st.info("No negative feedback found.")
#----------------------Project Overview----------------------
elif option == "Project Overview":
    st.title("📊 Project Process Overview")
    st.image("OIP.jfif")
    st.markdown("""
    This project aims to analyze feedback from companies on ITViec, focusing on sentiment analysis and information clustering.
    
    ### Steps:
    1. **Data Collection**: Gathered feedback data from ITViec.
    2. **Data Preprocessing**: Cleaned and prepared the data for analysis.
    3. **Sentiment Analysis**: Used Naive Bayes classifier to predict sentiment.
    4. **Information Clustering**: Applied LDA and KMeans to cluster feedback into topics.
    
    ### Tools Used:
    - Python libraries: Streamlit, Pandas, Scikit-learn, Imbalanced-learn
    - Machine Learning models: Naive Bayes, LDA, KMeans
    
    ### Giáo viên Hướng Dẫn: Khuất Thùy Phương
    
    
    """)

#----------------------Project Process----------------------
elif option == "Project Process":
    st.title("🔧 Project Process: Feedback Analysis Workflow")

    @st.cache_data
    def load_data():
        return pd.read_excel("Processed_Reviews.xlsx")

    df = load_data()

    # Step 1: Explore Initial Data
    st.header("1️⃣ Explore the Initial Data")
    st.write("**Number of rows:**", df.shape[0])
    st.write("**Number of columns:**", df.shape[1])
    st.write("**Column names:**", list(df.columns))

    # Step 2: Data Cleaning
    st.header("2️⃣ Data Cleaning")
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates().dropna(how="all")
    cleaned_shape = df_cleaned.shape
    st.write(f"Removed {initial_shape[0] - cleaned_shape[0]} duplicate/null rows.")
    st.dataframe(df_cleaned.head())

    # Step 3: EDA
    st.header("3️⃣ Exploratory Data Analysis (EDA)")
    st.write("Distribution of Sentiments")
    st.bar_chart(df_cleaned['sentiment'].value_counts())
    st.write("Top Companies by Feedback Volume")
    top_companies = df_cleaned['Company Name'].value_counts().head(10)
    st.bar_chart(top_companies)

    # Step 4: NLP Preprocessing
    st.header("4️⃣ NLP Preprocessing Steps")
    st.markdown("""
    - 🔤 **Language Translation**: Translate English to Vietnamese using `english-vnmese.txt`
    - 🔡 **Encoding**: Apply UTF-8 encoding
    - 😎 **Remove Noisy Characters**: Remove emojis, teencode using `emojicon.txt` and `teencode.txt`
    - 🧠 **POS Tagging**: Merge meaningful words using `pos_tag`
    - ❌ **Filter Out Wrong/Stop Words**: Use `wrong-word.txt` and `vietnamese-stopwords.txt`
    """)

    # Step 5: Data Visualization
    st.header("5️⃣ Data Visualization: WordCloud")
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Positive Feedback WordCloud")
        text_pos = " ".join(df_cleaned["positive_clean"].dropna().astype(str).tolist())
        wordcloud_pos = WordCloud(width=1000, height=600, background_color="white").generate(text_pos)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wordcloud_pos, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    with col2:
        st.subheader("Negative Feedback WordCloud")
        text_neg = " ".join(df_cleaned["negative_clean"].dropna().astype(str).tolist())
        wordcloud_neg = WordCloud(width=1000, height=600, background_color="white").generate(text_neg)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.imshow(wordcloud_neg, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)

