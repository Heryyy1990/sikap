import json, re, time
import streamlit as st
import google.generativeai as genai
from src.config import GEMINI_MODEL

def _get_client():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY tidak ditemukan!")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

def safe_generate(model, prompt, retries=3):
    for attempt in range(retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            if "429" in str(e):
                wait = 6 * (attempt + 1)
                st.warning(f"Rate limit. Menunggu {wait}s...")
                time.sleep(wait)
            else:
                raise e
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def extract_surat_inti(teks_surat: str) -> str:
    model = _get_client()
    prompt = f"""Baca surat ini dan tuliskan ESENSI-nya dalam satu kalimat pendek (MAKS 8 kata).
Fokus pada OBJEK yang dimohon, abaikan tujuan akhir.

SURAT:
{teks_surat}

ESENSI:"""
    resp = safe_generate(model, prompt)
    if resp:
        return resp.text.strip()
    return teks_surat[:150] + "..."

@st.cache_data(ttl=3600, show_spinner=False)
def classify_primary_secondary(inti_surat: str, df_codes: str) -> dict:
    """
    ATURAN KERAS:
    - Jika menyangkut TANAH / SERTIFIKAT TANAH → PRIMER 500, SEKUNDER 500.17
    - Abaikan kata seperti 'perpustakaan' jika hanya sebagai tujuan.
    """
    # ---- Rule‑based local untuk mencegah Gemini salah total ----
    keywords_tanah = ['sertifikat tanah','sertifikasi tanah','status tanah','pertanahan']
    if any(k in inti_surat.lower() for k in keywords_tanah):
        return {"primer":"500","sekunder":"500.17","alasan":"Keyword tanah terdeteksi"}

    model = _get_client()
    prompt = f"""Tentukan kode PRIMER (level 1) dan SEKUNDER (level 2) yang PALING TEPAT.
Jika topik adalah TANAH / SERTIFIKAT TANAH, maka PRIMER=500, SEKUNDER=500.17.

INTI SURAT:
{inti_surat}

DAFTAR KODE PRIMER & SEKUNDER:
{df_codes}

Format JSON:
{{"primer":"XXX","sekunder":"XXX.XX","alasan":"..."}}"""
    resp = safe_generate(model, prompt)
    if not resp:
        return {"primer":"500","sekunder":"500.17","alasan":"Fallback pertanahan"}

    text = resp.text.strip()
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except:
        pass
    return {"primer":"500","sekunder":"500.17","alasan":"Fallback pertanahan"}
