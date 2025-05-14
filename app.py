import streamlit as st
import torch
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==== Streamlit Config ====
st.set_page_config(
    page_title="Lục Bát Line Generator",
    page_icon="📝",
    layout="wide"
)

# ==== Layout: Two Columns ====
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.image("truyen-kieu.jpg", use_container_width=True)

with right_col:
    with st.container():
        st.title("Lục Bát Line Generator")
        st.markdown("""
        This app generates a single Vietnamese *lục* or *bát* poetic line using a GPT-2 model fine-tuned on the **Truyện Kiều** dataset by Nguyễn Du.<br>
        Model: <a href="https://huggingface.co/melanieyes/melanie-poem-generation" target="_blank">melanieyes/melanie-poem-generation</a>
        """, unsafe_allow_html=True)

        with st.expander("📜 Instructions", expanded=True):
            st.markdown("""
            1. Select line type: **Lục** (6 syllables) or **Bát** (8 syllables).  
            2. Enter a Vietnamese phrase as a prompt.  
            3. Click **Generate Line** to get one valid poetic line.
            """)

        prompt = st.text_input("✍️ Starting Prompt:", "trăm năm trăm cõi người ta")
        mode = st.selectbox("🔢 Select Line Type:", options=["luc", "bat"], format_func=lambda x: "Lục (6 chữ)" if x == "luc" else "Bát (8 chữ)")

    # ==== Load Model from Hugging Face ====
    @st.cache_resource
    def load_model_and_tokenizer():
        model = AutoModelForCausalLM.from_pretrained("melanieyes/melanie-poem-generation")
        tokenizer = AutoTokenizer.from_pretrained("melanieyes/melanie-poem-generation", use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        return model, tokenizer

    model, tokenizer = load_model_and_tokenizer()

    # ==== Tone Utilities ====
    def get_tone_class(syllable):
        syllable = unicodedata.normalize('NFC', syllable.lower())
        for char in syllable[::-1]:
            if char in 'àèìòùỳ':
                return 'bằng'
            elif char in 'áéíóúýảẻỉỏủỷãẽĩõũỹạẹịọụỵ':
                return 'trắc'
        return 'bằng'

    def check_bat_tone_rule(line):
        words = line.strip().split()
        if len(words) != 8:
            return False
        expected = ['bằng', 'trắc', 'bằng', 'bằng']
        positions = [1, 3, 5, 7]
        return all(get_tone_class(words[i]) == exp for i, exp in zip(positions, expected))

    # ==== Generation Logic ====
    def generate_single_line(model, tokenizer, prompt, mode="luc", max_attempts=10):
        target_len = 6 if mode == "luc" else 8
        for _ in range(max_attempts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=30,
                temperature=0.9,
                top_k=40,
                top_p=0.95,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            words = decoded.strip().split()
            for i in range(len(words) - target_len + 1):
                candidate = " ".join(words[i:i + target_len])
                if mode == "luc":
                    return candidate
                elif mode == "bat" and check_bat_tone_rule(candidate):
                    return candidate
        return "[FAILED TO GENERATE LINE]"

    # ==== Generate and Display ====
    if st.button("📌 Generate Line"):
        with st.spinner("✨ Generating..."):
            try:
                line = generate_single_line(model, tokenizer, prompt, mode=mode)
                st.subheader("🌸 Generated Poetic Line")
                st.text(line)
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==== Footer ====
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        A project from <em>Introduction to Artificial Intelligence</em>, made by <strong>Melanie</strong>, 2025
    </div>
    """,
    unsafe_allow_html=True
)
