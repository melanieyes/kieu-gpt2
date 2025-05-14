import streamlit as st
import torch
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==== Page Setup ====
st.set_page_config(
    page_title="Luc Bát Poem Generator",
    page_icon="📝",
    layout="wide"
)

# ==== App Banner ====
st.image("truyen-kieu.jpg", width=400, caption="Illustration from Truyện Kiều")

# ==== Title and Description ====
st.title("Luc Bát Poem Generator")
st.markdown("""
This app generates Vietnamese *lục bát* poems using a GPT-2 model fine-tuned on the **Truyện Kiều** dataset by Nguyễn Du.<br>
Model: <a href="https://huggingface.co/melanieyes/kieu-gpt2" target="_blank">melanieyes/kieu-gpt2</a>
""", unsafe_allow_html=True)

with st.expander("📜 Instructions"):
    st.write("""
    1. Enter a Vietnamese phrase to begin the poem (typically 6–8 syllables).
    2. Click **Generate Poem** to produce 4 lines in *lục bát* style.
    """)

# ==== User Input ====
prompt = st.text_input("✍️ Starting Prompt:", "thương sao cho trọn thì thương")

# ==== Load Model from Hugging Face ====
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("melanieyes/kieu-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("melanieyes/kieu-gpt2", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ==== Tone and Form Functions ====
def get_tone_class(syllable):
    syllable = unicodedata.normalize('NFC', syllable.lower())
    for char in syllable[::-1]:
        if char in 'àèìòùỳ':
            return 'bằng'
        elif char in 'áéíóúýảẻỉỏủỷãẽĩõũỹạẹịọụỵ':
            return 'trắc'
    return 'bằng'

def check_luc_bat_tone_rule(line):
    words = line.strip().split()
    if len(words) != 8:
        return False
    expected = ['bằng', 'trắc', 'bằng', 'bằng']
    positions = [1, 3, 5, 7]
    return all(get_tone_class(words[idx]) == exp for idx, exp in zip(positions, expected))

def split_luc_bat_poem(raw_text, max_lines=3):
    words = raw_text.strip().split()
    lines, i, toggle = [], 0, 6
    while i + toggle <= len(words) and len(lines) < max_lines:
        line = " ".join(words[i:i+toggle])
        if toggle == 8 and not check_luc_bat_tone_rule(line):
            i += 1
            continue
        lines.append(line)
        i += toggle
        toggle = 8 if toggle == 6 else 6
    return lines

def generate_luc_bat_poem(model, tokenizer, prompt, max_lines=3, max_attempts=10):
    for _ in range(max_attempts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            temperature=0.9,
            top_k=40,
            top_p=0.95,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        cleaned = decoded.replace("[SOS]", "").replace("[EOS]", "").replace("[EOL]", "")
        lines = split_luc_bat_poem(cleaned, max_lines=max_lines)
        if len(lines) == max_lines:
            return lines
    return ["[FAILED TO GENERATE]"] * max_lines

# ==== Generate and Display ====
if st.button("📌 Generate Poem"):
    with st.spinner("✨ Generating..."):
        try:
            poem_lines = generate_luc_bat_poem(model, tokenizer, prompt, max_lines=4)
            st.subheader("🌸 Generated Lục Bát Poem")
            st.text("\n".join(poem_lines))
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
