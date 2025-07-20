import streamlit as st
from transformers import pipeline
@st.cache_resource
def load_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment",
        device=-1
    )

classifier=load_pipeline()
# classifier = pipeline("sentiment-analysis", model = "cardiffnlp/twitter-roberta-base-sentiment")

st.title("Sentiment Analysis App")

text = st.text_area("Enter your text here:")

result=classifier(text)
result=result[0]['label']
if st.button("Analyze Sentiment"):
    if result == "LABEL_0":
        st.write("The text is Negative😒")
    elif result== 'LABEL_1':
        st.write("The text is neutral😑")
    elif result == 'LABEL_2':
        st.write("The text is positive😃")
