import streamlit as st
import torch
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Streamlit setup ===
st.set_page_config(
    page_title="Lục Bát 4-Line Generator",
    page_icon="📝",
    layout="wide"
)

# === Layout ===
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.image("truyen-kieu.jpg", use_container_width=True)

with right_col:
    st.title("Lục Bát Poem Generator (4 lines)")
    st.markdown("""
    This app generates a **4-line Lục Bát poem** starting from your custom 6-syllable input line.<br>
    Model: <a href="https://huggingface.co/melanieyes/melanie-poem-generation" target="_blank">melanieyes/melanie-poem-generation</a>
    """, unsafe_allow_html=True)

    with st.expander("📜 Instructions", expanded=True):
        st.markdown("""
        1. Enter a valid 6-syllable Lục line.  
        2. The model will generate 3 more lines to complete a 4-line poem:
            - Lục (your input)  
            - Bát (model)  
            - Lục (model)  
            - Bát (model)  
        """)

    luc_input = st.text_input("✍️ Your First Lục Line (6 syllables):", "trăng vàng in bóng bên thềm")

# === Load model and tokenizer ===
@st.cache_resource
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("melanieyes/melanie-poem-generation")
    tokenizer = AutoTokenizer.from_pretrained("melanieyes/melanie-poem-generation", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# === Tone checking logic ===
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

    if tone6[2] != 'bằng' or tone6[4] != 'trắc' or tone6[5] != 'bằng':
        return False
    if tone8[2] != 'bằng' or tone8[4] != 'trắc' or tone8[5] != 'bằng' or tone8[6] != 'trắc' or tone8[7] != 'bằng':
        return False

    return True

# === Generate one line of given length (6 or 8), check tone rule ===
def generate_line(model, tokenizer, prompt, target_len, check_fn=None, max_attempts=10):
    for _ in range(max_attempts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
        for i in range(len(words) - target_len + 1):
            candidate = " ".join(words[i:i + target_len])
            if check_fn is None or check_fn(candidate):
                return candidate
    return "[FAILED]"

# === Full generation logic ===
def generate_luc_bat_poem_4lines(model, tokenizer, input_line6):
    poem = [input_line6.strip()]
    # generate bát line
    line2 = generate_line(model, tokenizer, poem[-1], 8, lambda b: check_luc_bat_rule(poem[-1], b))
    poem.append(line2)

    # generate 2nd lục + bát pair
    line3 = generate_line(model, tokenizer, poem[-1], 6)
    line4 = generate_line(model, tokenizer, line3, 8, lambda b: check_luc_bat_rule(line3, b))
    poem.extend([line3, line4])
    return poem

# === UI interaction ===
if st.button("📌 Generate 4-line Poem"):
    with st.spinner("✨ Generating..."):
        try:
            poem = generate_luc_bat_poem_4lines(model, tokenizer, luc_input)
            st.subheader("🌸 Generated Lục Bát Poem")
            st.text("\n".join(poem))
        except Exception as e:
            st.error(f"❌ Error: {e}")

# === Footer ===
st.markdown("""
<hr>
<div style='text-align: center; color: gray; font-size: 14px;'>
    A project from <em>Introduction to Artificial Intelligence</em>, made by <strong>Melanie</strong>, 2025
</div>
""", unsafe_allow_html=True)
