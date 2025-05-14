import streamlit as st
import torch
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==== Streamlit UI ====
st.set_page_config(
    page_title="Lục Bát Generator",
    page_icon="🌸",
    layout="wide"
)

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.image("truyen-kieu.jpg", use_container_width=True)

with right_col:
    st.title("Lục Bát Generator")
    st.markdown("""
    This app generates a **bát line (8 syllables)** that completes your **lục line (6 syllables)** input using a fine-tuned GPT-2 model.<br>
    Model: <a href="https://huggingface.co/melanieyes/melanie-poem-generation" target="_blank">melanieyes/melanie-poem-generation</a>
    """, unsafe_allow_html=True)

    with st.expander("📜 Instructions", expanded=True):
        st.markdown("""
        1. Enter a valid 6-syllable *lục* line.  
        2. Click **Generate** to get a matching *bát* line.
        """)

    col1, col2 = st.columns([3, 1])
    luc_line = col1.text_input("✍️ Lục Line (6 syllables):", "trăng vàng in bóng bên thềm")
    generate_clicked = col2.button("📌 Generate")

# ==== Load model/tokenizer ====
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("melanieyes/melanie-poem-generation")
    tokenizer = AutoTokenizer.from_pretrained("melanieyes/melanie-poem-generation", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ==== Lục Bát tone rule ====
def get_tone_class(syllable):
    syllable = unicodedata.normalize('NFC', syllable.lower())
    for char in syllable[::-1]:
        if char in 'àèìòùỳ' or char in 'aeiouy':
            return 'bằng'
        elif char in 'áéíóúýảẻỉỏủỷãẽĩõũỹạẹịọụỵ':
            return 'trắc'
    return 'bằng'

def check_luc_bat_rule(line6, line8):
    w6 = line6.strip().split()
    w8 = line8.strip().split()
    if len(w6) != 6 or len(w8) != 8:
        return False

    tone6 = [get_tone_class(w) for w in w6]
    tone8 = [get_tone_class(w) for w in w8]

    return (
        tone6[2] == 'bằng' and tone6[4] == 'trắc' and tone6[5] == 'bằng' and
        tone8[2] == 'bằng' and tone8[4] == 'trắc' and tone8[5] == 'bằng' and tone8[6] == 'trắc' and tone8[7] == 'bằng'
    )

# ==== Generation ====
def generate_bat_line(model, tokenizer, luc_line, max_attempts=10):
    for _ in range(max_attempts):
        inputs = tokenizer(luc_line, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=40,
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
        for i in range(len(words) - 8 + 1):
            candidate = " ".join(words[i:i+8])
            if check_luc_bat_rule(luc_line, candidate):
                return candidate
    return "[FAILED]"

# ==== Output ====
if generate_clicked:
    with st.spinner("✨ Generating..."):
        bat_line = generate_bat_line(model, tokenizer, luc_line)
        st.subheader("🌸 Lục Bát Couple:")
        st.text(luc_line.strip())
        st.text(bat_line.strip())
