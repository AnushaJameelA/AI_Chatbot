
import os
import streamlit as st
from chatbot import answer_question, clean_text

def main():
    print("Hello from ai-chatbot!")

st.set_page_config(page_title="AI Q&A Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI-Powered Q&A Chatbot")
st.write("Ask questions based on a given context using Hugging Face API.")


st.sidebar.header("Settings")
debug = st.sidebar.checkbox("Debug Info", value=False)


if not os.getenv("HF_TOKEN"):
    st.warning("⚠ Hugging Face Token not found. Set in environment or Streamlit Secrets.")


context = st.text_area("Enter the context (paragraph or document):", height=200)
question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    if not context.strip() or not question.strip():
        st.error("Please provide both context and question.")
    else:
        with st.spinner("Generating answer..."):
            result = answer_question(question, context)

        if result["ok"]:
            st.success(f"*Answer:* {clean_text(result['answer'])}")
            st.caption(f"Confidence: {result['score']:.2f}")
        else:
            st.error(f"Failed: {result['error']}")
            if debug:
                st.code(f"Status: {result['status_code']}")

if __name__ == "__main__":
    main()
