import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
API_URL = "http://localhost:8000/query"
LOG_PATH = Path("data/logs/logs.jsonl")

st.set_page_config(layout="wide", page_title="RAG ANEEL")

st.title("⚡ RAG ANEEL")
st.caption("Sistema de busca e resposta sobre atos normativos da ANEEL")

# -------------------------
# CARREGAR LOGS
# -------------------------
if LOG_PATH.exists():
    df_logs = pd.read_json(LOG_PATH, lines=True)
else:
    df_logs = pd.DataFrame()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("🎛️ Consulta")

mapa_tipos = {
    "REH": "Resolução Homologatória",
    "REN": "Resolução Normativa",
    "DSP": "Despacho",
    "PRT": "Portaria"
}

tipo_ato = st.sidebar.selectbox(
    "Tipo de Ato",
    ["Todos"] + list(mapa_tipos.keys()),
    format_func=lambda x: mapa_tipos.get(x, x)
)

anos = st.sidebar.multiselect("Ano", ["2016", "2021", "2022"])
numero = st.sidebar.text_input("Número do ato")
palavra_chave = st.sidebar.text_input("Palavra-chave")
n_resultados = st.sidebar.slider("Top-K", 1, 20, 5)

# -------------------------
# MONTAR FILTROS
# -------------------------
filtros = {}

if tipo_ato != "Todos":
    filtros["tipo_codigo"] = tipo_ato
if anos:
    filtros["ano_fonte"] = anos
if numero:
    filtros["numero"] = numero
if palavra_chave:
    filtros["texto"] = palavra_chave

# -------------------------
# CONSULTA
# -------------------------
st.markdown("## 🔎 Nova Consulta")

pergunta = st.text_input("Digite sua pergunta")

if st.button("Consultar", use_container_width=True):
    if not pergunta:
        st.warning("Digite uma pergunta.")
    else:
        with st.spinner("Consultando..."):
            payload = {
                "pergunta": pergunta,
                "n_resultados": n_resultados,
                "filtros": filtros if filtros else None
            }

            res = requests.post(API_URL, json=payload)
            data = res.json()

        st.session_state["last_response"] = data

# -------------------------
# MOSTRAR RESPOSTA
# -------------------------
if "last_response" in st.session_state:
    data = st.session_state["last_response"]

    st.markdown("## 📄 Resposta")

    if "Não encontrado" in data["resposta"]:
        st.error(data["resposta"])
    else:
        st.success(data["resposta"])

    # métricas
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Modelo", data["modelo"])
    col2.metric("Latência Total", f"{data['latencia_total_ms']} ms")
    col3.metric("Latência LLM", f"{data['latencia_llm_ms']} ms")
    col4.metric("Tokens", data["tokens_resposta"])

    # prompt
    with st.expander("🧠 System Prompt"):
        st.code(data["system_prompt"], language="text")

    # fontes
    st.markdown("### 📚 Fontes")

    for f in data["fontes"]:
        with st.expander(f"{f['tipo_nome']} nº {f['numero']}/{f['ano']}"):
            st.write(f"**Assunto:** {f['assunto']}")
            st.write(f"**Score:** {f['score_final']:.3f}")
            st.write(f["trecho"])

# -------------------------
# DASHBOARD DE LOGS
# -------------------------
st.markdown("---")
st.markdown("## 📊 Análise do Sistema")

if not df_logs.empty:

    # -------------------------
    # FILTROS DOS LOGS
    # -------------------------
    st.markdown("### 🎛️ Filtros de Logs")

    col1, col2, col3 = st.columns(3)

    filtro_fallback = col1.selectbox("Fallback", ["Todos", "Sim", "Não"])
    latencia_max = col2.slider("Latência máxima (ms)", 0, 5000, 2000)
    apenas_erros = col3.checkbox("Apenas respostas ruins")

    df_f = df_logs.copy()

    if filtro_fallback == "Sim":
        df_f = df_f[df_f["fallback"] == 1]
    elif filtro_fallback == "Não":
        df_f = df_f[df_f["fallback"] == 0]

    df_f = df_f[df_f["latency_ms"] <= latencia_max]

    if apenas_erros:
        df_f = df_f[df_f["faithfulness"] == 0]

    # -------------------------
    # MÉTRICAS
    # -------------------------
    st.markdown("### 📈 Métricas")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latência média", f"{df_f['latency_ms'].mean():.0f} ms")
    col2.metric("Tokens médios", f"{df_f['tokens_response'].mean():.0f}")
    col3.metric("Fallback (%)", f"{df_f['fallback'].mean()*100:.1f}%")
    col4.metric("Faithfulness", f"{df_f['faithfulness'].mean():.2f}")

    st.line_chart(df_f["latency_ms"])

    # -------------------------
    # HISTÓRICO INTERATIVO
    # -------------------------
    st.markdown("### 📜 Histórico")

    for i, row in df_f.tail(20).iterrows():
        col1, col2 = st.columns([8, 1])

        with col1:
            st.write(f"**{row['query'][:80]}...**")

        with col2:
            if st.button("Ver", key=f"log_{i}"):
                st.session_state["selected_log"] = row

    # -------------------------
    # DETALHE DO LOG
    # -------------------------
    if "selected_log" in st.session_state:
        row = st.session_state["selected_log"]

        st.markdown("## 🧠 Análise da Execução")

        st.markdown("### Pergunta")
        st.write(row["query"])

        st.markdown("### Resposta")
        st.write(row["response"])

        if row["fallback"]:
            st.error("⚠️ Fallback ativado")

        if row["faithfulness"] == 0:
            st.warning("⚠️ Possível alucinação detectada")

        st.markdown("### Chunks")

        for c in row["chunks"]:
            st.write(
                f"- {c.get('tipo_nome')} nº {c.get('numero')}/{c.get('ano')} "
                f"(score: {c.get('score_final')})"
            )

else:
    st.info("Nenhum log disponível ainda.")