# 📜 Luc Bát Poem Generator

This web application generates Vietnamese *lục bát* (six-eight) poems using a GPT-2 model fine-tuned on the **Truyện Kiều** dataset by Nguyễn Du. It combines traditional Vietnamese poetic structure with modern natural language generation techniques.

![Truyen Kieu Illustration](truyen-kieu.jpg)

---

## 🚀 Features

- Generate original *lục bát* verses in Vietnamese
- Follows tone rules (bằng/trắc) of traditional Vietnamese poetry
- Input a custom phrase and receive 4 generated lines
- Built with [Streamlit](https://streamlit.io) and Hugging Face 🤗 Transformers

---

## 🧠 Model Info

- **Base model:** [`danghuy1999/gpt2-viwiki`](https://huggingface.co/danghuy1999/gpt2-viwiki)
- **Fine-tuned model:** [`melanieyes/kieu-gpt2`](https://huggingface.co/melanieyes/kieu-gpt2)
- **Training data:** Processed lines from *Truyện Kiều*
- **Architecture:** GPT-2 (Causal LM)

---

## 🖥️ Demo

Try it live on **Streamlit Cloud** or locally:

```bash
# 1. Clone this repository
git clone https://github.com/yourusername/kieu-lucbat-generator.git
cd kieu-lucbat-generator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
