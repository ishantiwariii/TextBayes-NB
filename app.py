import streamlit as st
import pickle 
import nltk 

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

st.set_page_config(page_title="TextBayes", page_icon="📩", layout="centered")

ps = PorterStemmer()

# ---------- CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a);
}

/* Center everything */
.block-container {
    padding-top: 2rem;
}

/* Logo center */
.logo {
    display: flex;
    justify-content: center;
    margin-bottom: 10px;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Card */
.box {
    background: rgba(30, 41, 59, 0.6);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148,163,184,0.2);
}

/* Input */
.stTextArea textarea {
    background-color: #020617;
    color: white;
    border-radius: 10px;
    border: 1px solid #334155;
}

.stTextArea textarea:focus {
    border: 1px solid #a78bfa;
    box-shadow: 0 0 8px rgba(167,139,250,0.5);
}

/* Button */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
}

/* Result */
.result {
    text-align: center;
    padding: 14px;
    border-radius: 10px;
    font-size: 20px;
    margin-top: 15px;
}

.spam {
    background: #7f1d1d;
    color: #fecaca;
}

.ham {
    background: #064e3b;
    color: #bbf7d0;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="logo">', unsafe_allow_html=True)
st.image("TextBayes_logo.png", width=90)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="title">TextBayes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered SMS & Email Spam Detection</div>', unsafe_allow_html=True)

# ---------- CARD ----------
st.markdown('<div class="box">', unsafe_allow_html=True)

input_sms = st.text_area("✉️ Enter your message", height=120)

# ---------- MODEL ----------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---------- BUTTON ----------
if st.button("🚀 Analyze Message"):
    
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_text = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.markdown('<div class="result spam">🚨 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result ham">✅ Not Spam</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<center style='color:#64748b; margin-top:20px;'>Built with ❤️ using Naive Bayes</center>", unsafe_allow_html=True)