import streamlit as st
import pickle
from keras.models import load_model
from utils import generate_hoax_from_word, extract_spok

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ========================
# Load model dan tokenizer
# ========================
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model('hoax_lstm_model.h5')
    with open('tokenizer_hoax.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
max_seq_len = model.input_shape[1] + 1

# ========================
# UI Streamlit
# ========================
st.set_page_config(page_title="AI Hoax Generator", layout="centered")
st.title("🧠 AI Hoax Generator Bahasa Indonesia")
st.write("Masukkan satu kata awal, dan sistem akan mengarang kalimat hoax tiruan berdasarkan model LSTM.")
st.write("Setelah itu, sistem juga akan menampilkan analisis SPOK-nya.")

seed_word = st.text_input("🔤 Masukkan Kata Awal (contoh: 'mata')", "")

temperature = st.slider("🎚️ Temperatur (kreativitas):", min_value=0.2, max_value=1.5, value=0.8, step=0.1)

if st.button("🚀 Generate Hoax"):
    if seed_word.strip() == "":
        st.warning("Silakan isi kata awal terlebih dahulu.")
    else:
        with st.spinner("Sedang menghasilkan hoax..."):
            generated = generate_hoax_from_word(
                seed_word, tokenizer, model, max_seq_len,
                max_words=15, temperature=temperature
            )
            st.success("✅ Kalimat berhasil dihasilkan!")
            st.subheader("📝 Kalimat Hoax:")
            st.write(generated)

            st.subheader("🔍 Analisis SPOK:")
            spok_results = extract_spok(generated)
            for i, item in enumerate(spok_results):
                st.markdown(f"**Kalimat {i+1}:** {item['kalimat']}")
                st.write({
                    "Subjek": item['subjek'],
                    "Predikat": item['predikat'],
                    "Objek": item['objek'],
                    "Keterangan": item['keterangan']
                })
