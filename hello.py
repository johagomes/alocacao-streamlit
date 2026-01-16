import io
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG (REGRAS)
# =========================
OCCUPANCY_M3 = 0.90
OCCUPANCY_KG = 0.90

FLEET_PRIORITY = {
    "KANGU": 1,
    "FF": 2,
    "SPOT": 3,
    "SPOT DPC": 4,  # última prioridade
}

CAPACITY_ROWS = [
    ("Vuc",16,1600),
    ("Van",8,1500),
    ("Médio",25,3500),
    ("Truck",50,12000),
    ("Carreta",90,24000),
    ("Vuc EL",16,1600),
    ("VUC com ajudante",17,1400),
    ("HR",12,1800),
    ("M1 Rental Médio DD*FM",37,3500),
    ("Toco",40,6000),
    ("MELIONE RENTAL VAN",8,2200),
    ("M1 Rental Vuc DD*FM",17,1600),
    ("VUC Dedicado com Ajudante",17,1400),
    ("M1 VUC DD*FF",17,1600),
    ("MeliOne Yellow Pool",8,2200),
    ("Rental Medio FM",37,3500),
    ("Rental VUC FM",17,1600),
    ("VUC Elétrico",16,1600),
    ("Van Frota Fixa - Equipe dupla",8,1500),
    ("Vuc Rental TKS",20,5300),
    ("Utilitários",3,650),
    ("VUC Dedicado FBM 7K",17,1700),
]

# =========================
# HELPERS
# =========================
def norm(s: str) -> str:
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s).strip()
    return s

def vehicle_class(modal: str) -> str:
    m = norm(modal)
    if "CARRETA" in m: return "CARRETA"
    if "TRUCK" in m: return "TRUCK"
    if "TOCO" in m: return "TOCO"
    if "MEDIO" in m: return "MEDIO"
    if "HR" in m: return "HR"
    if "VAN" in m: return "VAN"
    if "VUC" in m: return "VUC"
    return "OUTRO"

cap_df = pd.DataFrame(CAPACITY_ROWS, columns=["perfil","cap_m3","cap_kg"])
cap_df["perfil_norm"] = cap_df["perfil"].map(norm)

def capacity_for_modal(modal: str):
    m = norm(modal)

    exact = cap_df[cap_df["perfil_norm"] == m]
    if len(exact):
        r = exact.iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil

    matches = []
    for _, row in cap_df.iterrows():
        if row.perfil_norm and row.perfil_norm in m:
            matches.append((len(row.perfil_norm), row.cap_m3, row.cap_kg, row.perfil))
    if matches:
        matches.sort(reverse=True)
        _, m3, kg, perfil = matches[0]
        return float(m3), float(kg), perfil

    if "MEDIO" in m:
        r = cap_df[cap_df["perfil_norm"].str.contains("MEDIO")].sort_values("cap_m3", ascending=False).iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil
    if "VUC" in m:
        r = cap_df[cap_df["perfil_norm"] == "VUC"].iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil
    if "VAN" in m:
        r = cap_df[cap_df["perfil_norm"] == "VAN"].iloc[0]
        return float(r.cap_m3), float(r.cap_kg), r.perfil

    return np.nan, np.nan, None

def find_col(df, candidates):
    cols_norm = {norm(c): c for c in df.columns}
    for cand in candidates:
        c = cols_norm.get(norm(cand))
        if c is not None:
            return c
    for cand in candidates:
        cn = norm(cand)
        for k, orig in cols_norm.items():
            if cn in k:
                return orig
    return None

def parse_number_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    x = x.str.replace(".", "", regex=False)
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")

def to_excel_bytes(output_consolidado: pd.DataFrame, saldo_plano: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        output_consolidado.to_excel(writer, sheet_name="output_consolidado", index=False)
        saldo_plano.to_excel(writer, sheet_name="saldo_plano", index=False)
    buf.seek(0)
    return buf.read()

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def fmt_int(n: int) -> str:
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        return str(n)

# =========================
# SEU ALGORITMO (runner)
# =========================
# ⚠️ Aqui está um placeholder simples para manter o app rodando.
# Substitua por: def run_allocation(plan_df, is_df): ... (seu código completo)
def run_allocation(plan_df: pd.DataFrame, is_df: pd.DataFrame):
    # Colunas mínimas (ajuste conforme sua base real)
    col_cluster_p = find_col(plan_df, ["Cluster"])
    col_transp = find_col(plan_df, ["Transportadora", "Carrier", "Transporter"])
    col_modal = find_col(plan_df, ["Modal", "Perfil"])
    col_frota = find_col(plan_df, ["Tipo Frota", "Frota", "Fleet Type"])
    col_avail = find_col(plan_df, ["Disponibilidade de Modais", "Disponibilidade", "Qtd", "Quantidade"])

    col_cluster_i = find_col(is_df, ["CLUSTER", "Cluster"])
    col_hub = find_col(is_df, ["HUB", "Warehouse", "WH", "WAREHOUSE_ID"])
    col_kg = find_col(is_df, ["Peso(kg)", "Peso", "KG", "WEIGHT"])
    col_m3 = find_col(is_df, ["Volume(m³)", "Volume", "M3", "M³", "CUBAGEM"])

    missing_plan = [("Cluster", col_cluster_p), ("Transportadora", col_transp), ("Modal", col_modal), ("Tipo Frota", col_frota), ("Disponibilidade", col_avail)]
    missing_plan = [name for name, col in missing_plan if col is None]
    if missing_plan:
        raise ValueError(f"PlanoRotas: não encontrei as colunas necessárias: {', '.join(missing_plan)}")

    missing_is = [("Cluster", col_cluster_i), ("HUB", col_hub), ("Peso", col_kg), ("Volume", col_m3)]
    missing_is = [name for name, col in missing_is if col is None]
    if missing_is:
        raise ValueError(f"ISs: não encontrei as colunas necessárias: {', '.join(missing_is)}")

    # Exemplinho: apenas retorna duas tabelas coerentes
    plan = plan_df.rename(columns={
        col_cluster_p: "Cluster",
        col_transp: "Transportadora",
        col_modal: "Modal",
        col_frota: "Tipo Frota",
        col_avail: "Disponibilidade",
    }).copy()
    plan["Disponibilidade"] = pd.to_numeric(plan["Disponibilidade"], errors="coerce").fillna(0).astype(int)

    output_consolidado = (
        plan.groupby(["Cluster","Transportadora","Tipo Frota","Modal"], as_index=False)["Disponibilidade"]
            .sum()
            .rename(columns={"Disponibilidade":"Veiculos"})
            .sort_values(["Cluster","Veiculos"], ascending=[True, False])
    )

    saldo_plano = (
        output_consolidado.rename(columns={"Veiculos":"Disponibilidade_Restante"})
            .sort_values(["Cluster","Disponibilidade_Restante"], ascending=[True, False])
    )

    # Diagnóstico (exemplo)
    diag = pd.DataFrame({
        "Check": ["Plano: linhas", "ISs: linhas", "Clusters (Plano)", "Clusters (ISs)"],
        "Valor": [
            len(plan_df),
            len(is_df),
            plan["Cluster"].astype(str).nunique(),
            is_df[col_cluster_i].astype(str).nunique(),
        ]
    })

    return output_consolidado, saldo_plano, diag

# =========================
# UI (Tema / Layout)
# =========================
st.set_page_config(page_title="Alocação • Estilo ML", layout="wide")

# ---- Sidebar controls (modo) ----
with st.sidebar:
    st.markdown("### Aparência")
    mode = st.radio("Tema", ["Auto", "Claro", "Escuro"], index=0, horizontal=True)
    st.divider()

# ---- CSS: Auto + overrides por modo ----
# Auto: segue prefers-color-scheme do navegador
# Claro/Escuro: força variáveis por atributo data-mode no body (via JS)
base_css = """
<style>
  /* Corrige o banner cortado pelo header fixo */
  .block-container { padding-top: 4.8rem !important; padding-bottom: 2rem; max-width: 1200px; }

  header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(8px);
    border-bottom: 1px solid var(--ml-border);
  }

  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFF4A3 0%, var(--ml-bg) 70%);
    border-right: 1px solid var(--ml-border);
  }

  :root {
    --ml-primary: #FFE600;
    --ml-blue: #032D6E;
    --ml-bg: #FFFFFF;
    --ml-card: #FFFFFF;
    --ml-text: #111827;
    --ml-muted: #6B7280;
    --ml-border: #E5E7EB;
    --ml-shadow: 0 10px 24px rgba(17, 24, 39, 0.06);
  }

  /* Auto dark */
  @media (prefers-color-scheme: dark) {
    :root {
      --ml-bg: #0B1220;
      --ml-card: #0F1A2E;
      --ml-text: #E5E7EB;
      --ml-muted: #9CA3AF;
      --ml-border: #22314A;
      --ml-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
    }
    header[data-testid="stHeader"] { background: rgba(11, 18, 32, 0.78); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #2B2A10 0%, var(--ml-bg) 70%); }
  }

  /* Força modo pelo atributo data-mode no body */
  body[data-mode="light"] {
    --ml-bg: #FFFFFF;
    --ml-card: #FFFFFF;
    --ml-text: #111827;
    --ml-muted: #6B7280;
    --ml-border: #E5E7EB;
    --ml-shadow: 0 10px 24px rgba(17, 24, 39, 0.06);
  }
  body[data-mode="dark"] {
    --ml-bg: #0B1220;
    --ml-card: #0F1A2E;
    --ml-text: #E5E7EB;
    --ml-muted: #9CA3AF;
    --ml-border: #22314A;
    --ml-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
  }
  body[data-mode="dark"] header[data-testid="stHeader"] { background: rgba(11, 18, 32, 0.78); }
  body[data-mode="dark"] section[data-testid="stSidebar"] { background: linear-gradient(180deg, #2B2A10 0%, var(--ml-bg) 70%); }

  html, body, [data-testid="stAppViewContainer"] { background: var(--ml-bg) !important; color: var(--ml-text) !important; }

  /* “Logo” custom (não é marca oficial) */
  .ml-title {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.9rem 1rem;
    border-radius: 18px;
    background: linear-gradient(90deg, var(--ml-primary) 0%, #FFF4A3 60%, var(--ml-card) 100%);
    border: 1px solid var(--ml-border);
    margin-top: 0.25rem;
    margin-bottom: 1rem;
  }
  .ml-logo {
    width: 42px; height: 42px;
    border-radius: 999px;
    background: var(--ml-blue);
    display: grid;
    place-items: center;
    box-shadow: 0 8px 18px rgba(3, 45, 110, 0.25);
    flex: 0 0 auto;
  }
  .ml-badge {
    background: rgba(255,255,255,0.92);
    color: #111827;
    font-weight: 950;
    font-size: 12px;
    padding: 5px 10px;
    border-radius: 999px;
    letter-spacing: 0.4px;
    line-height: 1;
  }
  .ml-subtle { color: var(--ml-muted); font-size: 0.95rem; margin-top: -6px; }

  .ml-card {
    background: var(--ml-card);
    border: 1px solid var(--ml-border);
    border-radius: 16px;
    padding: 1rem 1rem 0.8rem 1rem;
    box-shadow: var(--ml-shadow);
  }

  /* botões */
  .stButton > button,
  div[data-testid="stDownloadButton"] > button {
    border-radius: 12px !important;
    border: 1px solid var(--ml-border) !important;
    padding: 0.65rem 1rem !important;
    font-weight: 900 !important;
  }
  .stButton > button[kind="primary"] {
    background: var(--ml-primary) !important;
    color: #111827 !important;
    border: 1px solid #E6D200 !important;
  }

  /* tabelas */
  div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--ml-border); }

  /* abas */
  button[data-baseweb="tab"] {
    font-weight: 900;
  }
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

# ---- JS: aplica modo selecionado (light/dark) ou remove para auto ----
js = f"""
<script>
(function() {{
  const mode = "{mode}";
  const b = document.body;
  if (!b) return;
  if (mode === "Auto") {{
    b.removeAttribute("data-mode");
  }} else if (mode === "Claro") {{
    b.setAttribute("data-mode", "light");
  }} else {{
    b.setAttribute("data-mode", "dark");
  }}
}})();
</script>
"""
st.markdown(js, unsafe_allow_html=True)

# =========================
# HEADER (Logo custom)
# =========================
st.markdown(
    """
    <div class="ml-title">
      <div class="ml-logo">
        <div class="ml-badge">ML</div>
      </div>
      <div>
        <div style="font-size: 1.40rem; font-weight: 950; color: var(--ml-text);">
          Alocação de Veículos por Cluster
        </div>
        <div class="ml-subtle">
          Upload do Plano e ISs • Execute • Visualize • Baixe as saídas
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR (Uploads)
# =========================
with st.sidebar:
    st.markdown("### Upload dos arquivos")
    st.caption("Envie as duas bases em Excel e rode a alocação.")
    plan_file = st.file_uploader("PlanoRotas (Excel)", type=["xlsx"])
    is_file = st.file_uploader("ISsDIa (Excel)", type=["xlsx"])

    st.divider()
    st.markdown("### Parâmetros")
    st.write(f"**OCCUPANCY_M3:** {OCCUPANCY_M3}")
    st.write(f"**OCCUPANCY_KG:** {OCCUPANCY_KG}")
    st.divider()
    st.info("Dica: se der erro de colunas, confira os nomes no Excel (Cluster, Modal, Tipo Frota, HUB, Peso, Volume).")

run = st.button("Rodar alocação", type="primary", disabled=not (plan_file and is_file))

# =========================
# STATE
# =========================
if "output_consolidado" not in st.session_state:
    st.session_state.output_consolidado = pd.DataFrame()
    st.session_state.saldo_plano = pd.DataFrame()
    st.session_state.diag = pd.DataFrame()
    st.session_state.last_error = None

if run:
    try:
        with st.spinner("Lendo arquivos e processando..."):
            plan_df = pd.read_excel(plan_file)
            is_df = pd.read_excel(is_file)
            out, saldo, diag = run_allocation(plan_df, is_df)

        st.session_state.output_consolidado = out
        st.session_state.saldo_plano = saldo
        st.session_state.diag = diag
        st.session_state.last_error = None

        st.success("Processamento concluído!")

    except Exception as e:
        st.session_state.last_error = e
        st.error("Erro ao processar. Veja detalhes na aba Diagnóstico.")

# =========================
# ABAS
# =========================
tab_resultados, tab_downloads, tab_diag = st.tabs(["Resultados", "Downloads", "Diagnóstico"])

# -------- Resultados --------
with tab_resultados:
    out = st.session_state.output_consolidado
    saldo = st.session_state.saldo_plano

    st.markdown('<div class="ml-card">', unsafe_allow_html=True)

    if out.empty and saldo.empty:
        st.info("Faça upload dos 2 arquivos na barra lateral e clique em **Rodar alocação**.")
    else:
        clusters_qtd = out["Cluster"].nunique() if ("Cluster" in out.columns and not out.empty) else 0
        hubs_qtd = out["HUB"].nunique() if ("HUB" in out.columns and not out.empty) else 0
        veic_total = int(out["Veiculos"].sum()) if ("Veiculos" in out.columns and not out.empty) else 0
        saldo_total = int(saldo["Disponibilidade_Restante"].sum()) if ("Disponibilidade_Restante" in saldo.columns and not saldo.empty) else 0

        m1, m2, m3, m4 = st.columns(4, gap="large")
        m1.metric("Clusters", fmt_int(clusters_qtd))
        m2.metric("HUBs", fmt_int(hubs_qtd))
        m3.metric("Veículos alocados", fmt_int(veic_total))
        m4.metric("Saldo total (plano)", fmt_int(saldo_total))

    st.markdown("</div>", unsafe_allow_html=True)

    if not out.empty or not saldo.empty:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.markdown('<div class="ml-card">', unsafe_allow_html=True)
            st.subheader("output_consolidado")
            st.dataframe(out, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="ml-card">', unsafe_allow_html=True)
            st.subheader("saldo_plano")
            st.dataframe(saldo, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -------- Downloads --------
with tab_downloads:
    out = st.session_state.output_consolidado
    saldo = st.session_state.saldo_plano

    st.markdown('<div class="ml-card">', unsafe_allow_html=True)
    st.subheader("Baixar arquivos")

    if out.empty and saldo.empty:
        st.info("Rode a alocação para liberar os downloads.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        excel_bytes = to_excel_bytes(out, saldo)

        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.download_button(
                "⬇️ output_consolidado (CSV)",
                data=to_csv_bytes(out),
                file_name="output_consolidado.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇️ saldo_plano (CSV)",
                data=to_csv_bytes(saldo),
                file_name="saldo_plano.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                "⬇️ Excel completo (2 abas)",
                data=excel_bytes,
                file_name="output_alocacao_por_cluster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.caption("Obs.: se você quiser incluir também Scores/Faltas aqui, eu adiciono mais abas no Excel e botões extras.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------- Diagnóstico --------
with tab_diag:
    st.markdown('<div class="ml-card">', unsafe_allow_html=True)
    st.subheader("Diagnóstico")

    if st.session_state.last_error is not None:
        st.error("Ocorreu um erro no processamento:")
        st.exception(st.session_state.last_error)

    diag = st.session_state.diag
    if diag is not None and not diag.empty:
        st.markdown("#### Checks rápidos")
        st.dataframe(diag, use_container_width=True, hide_index=True)

    st.markdown("#### Dicas comuns")
    st.markdown(
        """
- **Colunas não encontradas**: revise os nomes (ou equivalentes) no Excel.
- **Números com vírgula/ponto**: o app tenta normalizar (1.234,56 → 1234.56).
- **Clusters em comum**: precisa existir interseção de Cluster entre Plano e ISs.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
