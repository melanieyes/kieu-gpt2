import streamlit as st
import sys
import torch
import transformers
from transformers import pipeline

# ✅ MUST BE FIRST
st.set_page_config(
    page_title="Truyện Kiều - Vietnamese Poem Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== Debug Info ======================
st.sidebar.markdown("### ℹ️ Debug Info")
st.sidebar.write(f"Python version: `{sys.version}`")
st.sidebar.write(f"PyTorch version: `{torch.__version__}`")
st.sidebar.write(f"Transformers version: `{transformers.__version__}`")

# ====================== Header ======================
st.image("truyen-kieu.jpg", width=350)
st.title("Truyện Kiều - Vietnamese Poem Generator")
st.markdown(
    """
    This app generates verses inspired by **Nguyễn Du**'s *Truyện Kiều*, using a GPT-2 model fine-tuned on the poem.<br>
    Model: <a href="https://huggingface.co/melanieyes/kieu-gpt2" target="_blank">melanieyes/kieu-gpt2</a>
    """,
    unsafe_allow_html=True
)

with st.expander("📜 Instructions"):
    st.write("""
    1. Provide a starting phrase or verse in Vietnamese.
    2. Adjust the generation parameters.
    3. Click **Generate Poem** to create a new poetic verse inspired by Truyện Kiều.
    """)

# ====================== Input Prompt ======================
prompt_input = st.text_area("✍️ Starting Phrase:", "Trăm năm trong cõi người ta\n", height=100)

col1, col_spacer, col2 = st.columns([0.7, 0.05, 1.0])

with col1:
    max_length = st.slider("Max Output Tokens", 10, 200, 75)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.8)
    top_k = st.slider("Top-k", 1, 100, 50)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2)

with col2:
    if st.button("📌 Generate Poem"):
        with st.spinner("⏳ Generating poem..."):
            try:
                generator = pipeline(
                    "text-generation",
                    model="melanieyes/kieu-gpt2",
                    tokenizer="melanieyes/kieu-gpt2",
                    trust_remote_code=True,
                    local_files_only=False
                )

                result = generator(
                    prompt_input,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )

                st.subheader("🌸 Generated Poem:")
                for line in result[0]['generated_text'].split('\n'):
                    st.write(line)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# ====================== Footer ======================
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        Made by <strong>Melanie</strong>, 2025<br>
        Product of <em>Introduction to Artificial Intelligence</em> class<br>
    </div>
    """,
    unsafe_allow_html=True
)
