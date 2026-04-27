import streamlit as st
import requests
import pandas as pd
from pathlib import Path

API_URL  = "http://localhost:8000/query"
LOG_PATH = Path("data/logs/logs.jsonl")

st.set_page_config(layout="wide", page_title="RAG ANEEL")

# --- Custom CSS para cores suaves e aparência geral ---
st.markdown("""
<style>
/* Ocultar o icone padrão de alertas dentro das mensagens de chat */
div[data-testid="stChatMessage"] div[data-testid="stAlert"] div[data-baseweb="icon"] {
    display: none !important;
}

/* Sucesso (Checkmark) -> Verde Pastel */
div[data-testid="stAlert"]:has(svg path[d^="M12"]) {
    background-color: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    color: #166534 !important;
    padding: 0.8rem !important;
}
/* Erro (Warning) -> Vermelho Pastel */
div[data-testid="stAlert"]:has(svg path[d^="M12 2C6.48"]) {
    background-color: #fef2f2 !important;
    border: 1px solid #fecaca !important;
    color: #991b1b !important;
    padding: 0.8rem !important;
}

/* Reduzir fonte e espaçamentos dentro dos expansores das métricas para ficar mais clean */
div[data-testid="stExpander"] details summary p {
    font-size: 0.9em;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# --- Inicializa estado ---
if "historico" not in st.session_state:
    st.session_state["historico"] = []      # lista de dicts {pergunta, resposta}
if "n_esclarecimentos" not in st.session_state:
    st.session_state["n_esclarecimentos"] = 0
if "filtros_salvos" not in st.session_state:
    st.session_state["filtros_salvos"] = {
        "tipo_ato": "Todos",
        "anos": [],
        "numero": "",
        "n_resultados": 10
    }
if "processando" not in st.session_state:
    st.session_state["processando"] = False
if "prompt_pendente" not in st.session_state:
    st.session_state["prompt_pendente"] = None

# --- Carrega logs ---
@st.cache_data(ttl=10) # cache curto pra não pesar
def carregar_logs():
    if LOG_PATH.exists():
        try:
            return pd.read_json(LOG_PATH, lines=True)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

df_logs = carregar_logs()

# --- Aba Lateral (Navegação) ---
st.sidebar.title("RAG ANEEL")
menu_opcoes = ["💬 Chat", "⚙️ Ajustes e Filtros", "📊 Métricas e Logs"]
aba_atual = st.sidebar.radio("Navegação", menu_opcoes)

# Captura o input do chat (aparece fixado no fundo da tela principal)
prompt = None
if aba_atual == "💬 Chat":
    prompt = st.chat_input("Pergunte ao assistente sobre os atos normativos...", disabled=st.session_state.get("processando", False))

if prompt:
    st.session_state["prompt_pendente"] = prompt
    st.session_state["processando"] = True
    st.rerun()

# Indicador de histórico no sidebar
if len(st.session_state["historico"]) > 0 or st.session_state.get("prompt_pendente"):
    st.sidebar.warning("⚠️ Iniciar um novo chat apagará as perguntas e respostas atuais.")
    if st.sidebar.button("➕ Novo Chat", use_container_width=True, type="primary"):
        st.session_state["historico"] = []
        st.session_state["n_esclarecimentos"] = 0
        st.session_state["prompt_pendente"] = None
        st.session_state["processando"] = False
        st.rerun()

mapa_tipos = {
    "REH": "Resolucao Homologatoria",
    "REN": "Resolucao Normativa",
    "DSP": "Despacho",
    "PRT": "Portaria",
    "REA": "Resolucao Autorizativa",
}

# ==============================================================================
# ==============================================================================
# TELA 2: AJUSTES E FILTROS
# ==============================================================================
if aba_atual == "⚙️ Ajustes e Filtros":
    st.header("⚙️ Ajustes e Filtros")
    st.markdown("Defina aqui os filtros para refinar a busca no RAG. As alterações são salvas automaticamente.")
    
    fs = st.session_state["filtros_salvos"]
    
    with st.container(border=True):
        st.subheader("Filtros de Busca")
        col1, col2 = st.columns(2)
        with col1:
            tipo_ato = st.selectbox(
                "Tipo de Ato",
                ["Todos"] + list(mapa_tipos.keys()),
                index=(["Todos"] + list(mapa_tipos.keys())).index(fs["tipo_ato"]) if fs["tipo_ato"] in (["Todos"] + list(mapa_tipos.keys())) else 0,
                format_func=lambda x: mapa_tipos.get(x, x),
            )
            anos = st.multiselect("Ano", ["2015", "2016", "2021", "2022", "2023", "2024"], default=fs["anos"])
        with col2:
            numero = st.text_input("Número do ato", value=fs["numero"])
            
    with st.container(border=True):
        st.subheader("Configurações do Algoritmo")
        n_resultados = st.slider("Top-K (chunks)", 1, 20, fs["n_resultados"])
        
    st.session_state["filtros_salvos"] = {
        "tipo_ato": tipo_ato,
        "anos": anos,
        "numero": numero,
        "n_resultados": n_resultados
    }

# ==============================================================================
# TELA 3: MÉTRICAS E LOGS
# ==============================================================================
elif aba_atual == "📊 Métricas e Logs":
    st.header("📊 Análise do Sistema")
    
    if not df_logs.empty:
        st.markdown("### Filtros de Logs")
        col1, col2, col3, col4 = st.columns(4)

        filtro_fallback = col1.selectbox("Fallback", ["Todos", "Sim", "Nao"])
        latencia_max    = col2.slider("Latência máxima (ms)", 0, 10000, 5000)
        apenas_erros    = col3.checkbox("Apenas respostas ruins")
        filtro_tipo     = col4.selectbox("Tipo de pergunta",
                                         ["Todos", "resumo", "busca", "comparacao", "tabela"])

        df_f = df_logs.copy()

        if filtro_fallback == "Sim":
            df_f = df_f[df_f["fallback"] == 1]
        elif filtro_fallback == "Nao":
            df_f = df_f[df_f["fallback"] == 0]
        if "latency_ms" in df_f.columns:
            df_f = df_f[df_f["latency_ms"] <= latencia_max]
        if apenas_erros and "faithfulness" in df_f.columns:
            df_f = df_f[df_f["faithfulness"] == 0]
        if filtro_tipo != "Todos" and "tipo_pergunta" in df_f.columns:
            df_f = df_f[df_f["tipo_pergunta"] == filtro_tipo]

        st.markdown("### Métricas")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Consultas", len(df_f))
        c2.metric("Lat. média",   (f"{df_f['latency_ms'].mean():.0f} ms") if "latency_ms" in df_f.columns and not df_f["latency_ms"].isna().all() else "-")
        c3.metric("Tokens médios",(f"{df_f['tokens_response'].mean():.0f}") if "tokens_response" in df_f.columns and not df_f["tokens_response"].isna().all() else "-")
        c4.metric("Fallback (%)", (f"{df_f['fallback'].mean()*100:.1f}%") if "fallback" in df_f.columns and not df_f["fallback"].isna().all() else "-")
        c5.metric("Faithfulness", (f"{df_f['faithfulness'].mean():.2f}")   if "faithfulness" in df_f.columns and not df_f["faithfulness"].isna().all() else "-")

        if "latency_ms" in df_f.columns:
            st.line_chart(df_f["latency_ms"].reset_index(drop=True))

        if "tipo_pergunta" in df_f.columns:
            st.markdown("### Tipos de pergunta")
            st.bar_chart(df_f["tipo_pergunta"].value_counts())

        st.markdown("### Histórico recente")
        for i, row in df_f.tail(20).iterrows():
            with st.expander(f"[{row.get('tipo_pergunta', '')}] {str(row.get('query', ''))[:80]}..."):
                st.write("**Pergunta:**", row.get("query", ""))
                st.write("**Resposta:**", row.get("response", ""))
                if row.get("fallback"):
                    st.error("Fallback ativado")
                if row.get("faithfulness") == 0:
                    st.warning("Possível problema de faithfulness")
                st.markdown("**Chunks utilizados:**")
                for c in row.get("chunks", []):
                    st.caption(f"- {c.get('tipo','?')} nº {c.get('numero','?')}/{c.get('ano','?')} | score: {c.get('score',0):.3f} | fonte: {c.get('fonte','')}")
    else:
        st.info("Nenhum log disponível ainda. Faça uma consulta para começar.")

# ==============================================================================
# TELA 1: CHAT PRINCIPAL
# ==============================================================================
elif aba_atual == "💬 Chat":
    # Mostra filtros ativos no topo
    fs = st.session_state["filtros_salvos"]
    filtros_ativos = []
    if fs["tipo_ato"] != "Todos": filtros_ativos.append(f"Tipo: {fs['tipo_ato']}")
    if fs["anos"]: filtros_ativos.append(f"Anos: {','.join(fs['anos'])}")
    if fs["numero"]: filtros_ativos.append(f"Nº: {fs['numero']}")
    
    if filtros_ativos:
        st.caption(f"⚙️ **Filtros ativos:** {' | '.join(filtros_ativos)}")
    
    # Tela inicial vazia com nome e frase
    if not st.session_state["historico"] and not st.session_state.get("prompt_pendente"):
        st.markdown("<h1 style='text-align: center; margin-top: 10vh; font-size: 4rem; color: #1f2937;'>ANEEL</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6b7280; font-size: 1.2rem;'>Sistema de busca inteligente sobre os atos normativos da Agência Nacional de Energia Elétrica.</p>", unsafe_allow_html=True)
    else:
        # Renderiza mensagens anteriores do histórico apenas se houver
        for msg in st.session_state["historico"]:
            with st.chat_message("user"):
                st.markdown(msg["pergunta"])
            with st.chat_message("assistant", avatar="⚡"):
                if msg.get("is_error"):
                    st.error(msg["resposta"])
                else:
                    st.success(msg["resposta"])

    # Se teve input pendente, roda a logica de resposta
    if st.session_state.get("prompt_pendente"):
        pergunta_atual = st.session_state["prompt_pendente"]
        
        # 1. Exibe a pergunta do usuário
        with st.chat_message("user"):
            st.markdown(pergunta_atual)
            
        # 2. Prepara filtros e request
        filtros = {}
        if fs["tipo_ato"] != "Todos":
            filtros["tipo_codigo"] = fs["tipo_ato"]
        if fs["anos"]:
            filtros["ano"] = fs["anos"]
        if fs["numero"]:
            filtros["numero"] = fs["numero"]

        payload = {
            "pergunta":     pergunta_atual,
            "n_resultados": fs["n_resultados"],
            "filtros":      filtros if filtros else None,
            "historico":    st.session_state["historico"][-6:] if st.session_state["historico"] else None,
        }

        # 3. Exibe a área do assistente e faz a requisição
        with st.chat_message("assistant", avatar="⚡"):
            with st.spinner("⚡ Pensando..."):
                try:
                    res  = requests.post(API_URL, json=payload, timeout=120)
                    if res.status_code == 200:
                        data = res.json()
                        resposta = data.get("resposta", "Sem resposta.")
                        tipo_resp = data.get("tipo_resposta", "normal")
                        
                        # Exibe a resposta principal
                        is_error = False
                        if tipo_resp == "analitico":
                            st.success(resposta)
                        elif not resposta or "Nao encontrado" in resposta or "Não encontrado" in resposta:
                            st.error(resposta)
                            is_error = True
                        else:
                            st.success(resposta)

                        # Exibe os detalhes (fontes, métricas) em um expansor suave
                        if tipo_resp == "normal":
                            with st.expander("🛠️ Detalhes e Fontes", expanded=False):
                                with st.container(height=300):
                                    c1, c2, c3, c4 = st.columns(4)
                                    c1.metric("Modelo", data.get("modelo", "-"))
                                    c2.metric("Latência", f"{data.get('latencia_total_ms', 0)}ms")
                                    c3.metric("Tokens", data.get("tokens_resposta", 0))
                                    c4.metric("Score Max", f"{max([f.get('score_final', 0) for f in data.get('fontes', [])] + [0]):.2f}")
                                    
                                    st.markdown("**Fontes Utilizadas:**")
                                    fontes = data.get("fontes", [])
                                    if fontes:
                                        for f in fontes:
                                            label = f"{f.get('tipo_nome', 'Doc')} nº {f.get('numero', '?')}/{f.get('ano', '?')}"
                                            st.caption(f"🔹 **{label}** (Score: {f.get('score_final', 0):.3f})")
                                            if f.get("trecho"):
                                                st.markdown(f"> *{f.get('trecho')[:300]}...*")
                                            if f.get("url_pdf"):
                                                st.markdown(f"[Abrir PDF]({f.get('url_pdf')})")
                                            st.divider()
                                    else:
                                        st.caption("Nenhuma fonte específica citada.")

                        # Salva no histórico
                        st.session_state["historico"].append({
                            "pergunta": pergunta_atual,
                            "resposta": resposta,
                            "is_error": is_error,
                            "n_esclarecimentos": 0,
                        })
                        # Limite de histórico
                        st.session_state["historico"] = st.session_state["historico"][-10:]

                    else:
                        st.error(f"Erro na API: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Erro ao conectar com o servidor local: {e}")
                    
        # Limpa o estado pendente e libera o input
        st.session_state["prompt_pendente"] = None
        st.session_state["processando"] = False
        st.rerun()