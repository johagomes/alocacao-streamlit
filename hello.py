import io
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# CONFIG
# =========================
OCCUPANCY_M3 = 0.90
OCCUPANCY_KG = 0.90

FLEET_PRIORITY = {
    "KANGU": 1,
    "FF": 2,
    "SPOT": 3,
    "SPOT DPC": 4,  # <- última prioridade
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

# =========================
# CAPACIDADES EFETIVAS
# =========================
VUC_BASE_M3_EFF = 16 * OCCUPANCY_M3
VUC_BASE_KG_EFF = 1600 * OCCUPANCY_KG

MEDIO_BASE_M3_EFF = 37 * OCCUPANCY_M3
MEDIO_BASE_KG_EFF = 3500 * OCCUPANCY_KG

def split_oversize_vs_vuc(is_hub: pd.DataFrame):
    overs = is_hub[(is_hub["Peso_kg"] > VUC_BASE_KG_EFF) | (is_hub["Volume_m3"] > VUC_BASE_M3_EFF)]
    rem = is_hub.drop(overs.index)
    return overs, rem

def required_units_by_capacity(sum_kg, sum_m3, cap_kg_eff, cap_m3_eff):
    if cap_kg_eff <= 0 or cap_m3_eff <= 0:
        return 0
    return int(math.ceil(max(sum_kg / cap_kg_eff, sum_m3 / cap_m3_eff)))

# =========================
# SCORE (EXTRAS)
# =========================
def hub_tail_score(is_hub: pd.DataFrame):
    kg = is_hub["Peso_kg"].astype(float)
    m3 = is_hub["Volume_m3"].astype(float)

    overs = (kg > VUC_BASE_KG_EFF) | (m3 > VUC_BASE_M3_EFF)
    fits = ~overs
    df_fit = is_hub[fits].copy()

    thr_kg = 0.75 * VUC_BASE_KG_EFF
    thr_m3 = 0.75 * VUC_BASE_M3_EFF
    heavy = df_fit[(df_fit["Peso_kg"] >= thr_kg) | (df_fit["Volume_m3"] >= thr_m3)]

    heavy_kg = float(heavy["Peso_kg"].sum())
    heavy_m3 = float(heavy["Volume_m3"].sum())

    p95_kg = float(np.nanpercentile(kg, 95)) if len(kg) else 0.0
    p95_m3 = float(np.nanpercentile(m3, 95)) if len(m3) else 0.0

    score = (
        0.55 * max(
            heavy_kg / VUC_BASE_KG_EFF if VUC_BASE_KG_EFF else 0,
            heavy_m3 / VUC_BASE_M3_EFF if VUC_BASE_M3_EFF else 0,
        )
        + 0.45 * (
            0.5 * (p95_kg / VUC_BASE_KG_EFF if VUC_BASE_KG_EFF else 0)
            + 0.5 * (p95_m3 / VUC_BASE_M3_EFF if VUC_BASE_M3_EFF else 0)
        )
    )

    extra_need = int(math.ceil(max(
        heavy_kg / MEDIO_BASE_KG_EFF if MEDIO_BASE_KG_EFF else 0,
        heavy_m3 / MEDIO_BASE_M3_EFF if MEDIO_BASE_M3_EFF else 0
    )))

    return {"score": float(score), "extra_need": int(extra_need), "p95_kg": p95_kg, "p95_m3": p95_m3}

def proportional_split(scores: dict, needs: dict, total_supply: int):
    hubs = [h for h in scores if scores[h] > 0 and needs.get(h, 0) > 0]
    if total_supply <= 0 or not hubs:
        return {}
    tot = sum(scores[h] for h in hubs)
    if tot <= 0:
        return {}

    raw = {h: total_supply * (scores[h] / tot) for h in hubs}
    base = {h: min(needs[h], int(math.floor(raw[h]))) for h in hubs}

    used = sum(base.values())
    rem = total_supply - used

    frac = sorted([(h, raw[h] - math.floor(raw[h])) for h in hubs], key=lambda x: x[1], reverse=True)
    i = 0
    while rem > 0 and frac:
        h = frac[i][0]
        if base[h] < needs[h]:
            base[h] += 1
            rem -= 1
        i = (i + 1) % len(frac)
        if all(base[x] >= needs[x] for x in hubs):
            break
    return base

# =========================
# POOL ALLOCATION
# =========================
def selector_class(cls_name: str):
    return lambda r: r["vehicle_class"] == cls_name

def is_big_vehicle_row(r):
    if pd.isna(r["cap_m3_eff"]) or pd.isna(r["cap_kg_eff"]):
        return False
    return (r["cap_m3_eff"] >= VUC_BASE_M3_EFF) or (r["cap_kg_eff"] >= VUC_BASE_KG_EFF)

def selector_big(r):
    return is_big_vehicle_row(r)

def allocate_one_best(plan_pool: pd.DataFrame, selector_fn):
    eligible = plan_pool[(plan_pool["avail"] > 0)].copy()
    eligible = eligible[eligible.apply(selector_fn, axis=1)].copy()
    if eligible.empty:
        return None, plan_pool

    eligible = eligible.sort_values(
        ["fleet_priority", "cap_m3_eff", "cap_kg_eff", "avail"],
        ascending=[True, False, False, False],
    )
    row = eligible.iloc[0]
    idx = row.name
    plan_pool.loc[idx, "avail"] = int(plan_pool.loc[idx, "avail"]) - 1
    return row, plan_pool

# =========================
# CORE RUNNER
# =========================
def run_allocation(plan_df: pd.DataFrame, is_df: pd.DataFrame):
    # Detecta colunas do Plano
    col_cluster_p = find_col(plan_df, ["Cluster"])
    col_transp = find_col(plan_df, ["Transportadora", "Carrier", "Transporter"])
    col_modal = find_col(plan_df, ["Modal", "Perfil"])
    col_frota = find_col(plan_df, ["Tipo Frota", "Frota", "Fleet Type"])
    col_avail = find_col(plan_df, ["Disponibilidade de Modais", "Disponibilidade", "Qtd", "Quantidade"])

    missing_plan = [("Cluster", col_cluster_p), ("Transportadora", col_transp), ("Modal", col_modal), ("Tipo Frota", col_frota), ("Disponibilidade", col_avail)]
    missing_plan = [name for name, col in missing_plan if col is None]
    if missing_plan:
        raise ValueError(f"PlanoRotas: não encontrei as colunas necessárias: {', '.join(missing_plan)}")

    # Detecta colunas IS
    col_cluster_i = find_col(is_df, ["CLUSTER", "Cluster"])
    col_hub = find_col(is_df, ["HUB", "Warehouse", "WH", "WAREHOUSE_ID"])
    col_kg = find_col(is_df, ["Peso(kg)", "Peso", "KG", "WEIGHT"])
    col_m3 = find_col(is_df, ["Volume(m³)", "Volume", "M3", "M³", "CUBAGEM"])

    missing_is = [("Cluster", col_cluster_i), ("HUB", col_hub), ("Peso", col_kg), ("Volume", col_m3)]
    missing_is = [name for name, col in missing_is if col is None]
    if missing_is:
        raise ValueError(f"ISs: não encontrei as colunas necessárias: {', '.join(missing_is)}")

    plan = plan_df.rename(
        columns={
            col_cluster_p: "Cluster",
            col_transp: "Transportadora",
            col_modal: "Modal",
            col_frota: "Tipo Frota",
            col_avail: "Disponibilidade",
        }
    ).copy()

    plan["Disponibilidade"] = pd.to_numeric(plan["Disponibilidade"], errors="coerce").fillna(0).astype(int)
    plan = plan[plan["Disponibilidade"] > 0].copy()

    plan["cap_m3"], plan["cap_kg"], plan["perfil_cap"] = zip(*plan["Modal"].map(capacity_for_modal))
    plan["cap_m3_eff"] = plan["cap_m3"] * OCCUPANCY_M3
    plan["cap_kg_eff"] = plan["cap_kg"] * OCCUPANCY_KG
    plan["vehicle_class"] = plan["Modal"].map(vehicle_class)
    plan["fleet_priority"] = plan["Tipo Frota"].map(lambda x: FLEET_PRIORITY.get(str(x).upper(), 9))
    plan["avail"] = plan["Disponibilidade"].astype(int)

    isdata = is_df.rename(
        columns={
            col_cluster_i: "Cluster",
            col_hub: "HUB",
            col_kg: "Peso_kg",
            col_m3: "Volume_m3",
        }
    ).copy()

    isdata["Peso_kg"] = parse_number_series(isdata["Peso_kg"])
    isdata["Volume_m3"] = parse_number_series(isdata["Volume_m3"])
    isdata = isdata.dropna(subset=["Cluster", "HUB", "Peso_kg", "Volume_m3"]).copy()

    clusters = sorted(list(set(plan["Cluster"].astype(str)).intersection(set(isdata["Cluster"].astype(str)))))
    if not clusters:
        raise ValueError("Não encontrei clusters em comum entre Plano e ISs.")

    all_outputs, all_saldos, all_scores, all_faltas = [], [], [], []

    for cluster in clusters:
        plan_cluster = plan[plan["Cluster"].astype(str) == str(cluster)].copy()
        is_cluster = isdata[isdata["Cluster"].astype(str) == str(cluster)].copy()
        if plan_cluster.empty or is_cluster.empty:
            continue

        plan_pool = plan_cluster.copy()
        records = []

        # 0) score hubs
        hub_meta = {}
        for hub, df_hub in is_cluster.groupby("HUB"):
            s = hub_tail_score(df_hub)
            hub_meta[hub] = {"score": s["score"], "extra_need": s["extra_need"]}
            all_scores.append({"Cluster": cluster, "HUB": hub, **s})

        hubs_sorted = sorted([(h, hub_meta[h]["score"]) for h in hub_meta], key=lambda x: x[1], reverse=True)

        # 1) demanda por HUB (após remover oversize)
        hub_demand = {}
        for hub, df_hub in is_cluster.groupby("HUB"):
            overs, rem = split_oversize_vs_vuc(df_hub)
            hub_demand[hub] = {
                "rem_kg": float(rem["Peso_kg"].sum()),
                "rem_m3": float(rem["Volume_m3"].sum()),
                "ov_kg": float(overs["Peso_kg"].sum()),
                "ov_m3": float(overs["Volume_m3"].sum()),
            }

        # 2) MIN_MEDIO (obrigatório)
        for hub in sorted(hub_demand.keys()):
            sum_ov_kg = hub_demand[hub]["ov_kg"]
            sum_ov_m3 = hub_demand[hub]["ov_m3"]
            min_medio = required_units_by_capacity(sum_ov_kg, sum_ov_m3, MEDIO_BASE_KG_EFF, MEDIO_BASE_M3_EFF)

            for _ in range(min_medio):
                row, plan_pool = allocate_one_best(plan_pool, selector_class("MEDIO"))
                if row is None:
                    all_faltas.append({"Cluster": cluster, "HUB": hub, "Tipo": "MIN_MEDIO", "Faltou": 1})
                    break
                records.append({
                    "Cluster": cluster, "HUB": hub, "Tipo": "MIN_MEDIO",
                    "Transportadora": row["Transportadora"], "Tipo Frota": row["Tipo Frota"],
                    "Modal": row["Modal"], "Veiculos": 1,
                })

        # 3) EXTRAS (UPGRADE) - abate a demanda
        remaining_big_supply = int(plan_pool[plan_pool.apply(is_big_vehicle_row, axis=1)]["avail"].sum())
        scores = {h: hub_meta[h]["score"] for h, _ in hubs_sorted}
        needs  = {h: max(0, hub_meta[h]["extra_need"]) for h, _ in hubs_sorted}
        extras_by_hub = proportional_split(scores, needs, remaining_big_supply)

        for hub, _ in hubs_sorted:
            extra_units = int(extras_by_hub.get(hub, 0))
            if extra_units <= 0:
                continue

            for _ in range(extra_units):
                if hub_demand[hub]["rem_kg"] <= 1e-6 and hub_demand[hub]["rem_m3"] <= 1e-6:
                    break

                row, plan_pool = allocate_one_best(plan_pool, selector_big)
                if row is None:
                    all_faltas.append({"Cluster": cluster, "HUB": hub, "Tipo": "EXTRA_BIG", "Faltou": 1})
                    break

                records.append({
                    "Cluster": cluster, "HUB": hub, "Tipo": "EXTRA_BIG",
                    "Transportadora": row["Transportadora"], "Tipo Frota": row["Tipo Frota"],
                    "Modal": row["Modal"], "Veiculos": 1,
                })

                hub_demand[hub]["rem_kg"] = max(0.0, hub_demand[hub]["rem_kg"] - float(row["cap_kg_eff"]))
                hub_demand[hub]["rem_m3"] = max(0.0, hub_demand[hub]["rem_m3"] - float(row["cap_m3_eff"]))

        # 4) MIN_FILL (completa o residual)
        for hub in sorted(hub_demand.keys()):
            rem_kg = float(hub_demand[hub]["rem_kg"])
            rem_m3 = float(hub_demand[hub]["rem_m3"])

            while rem_kg > 1e-6 or rem_m3 > 1e-6:
                row, plan_pool = allocate_one_best(plan_pool, lambda r: True)
                if row is None:
                    records.append({
                        "Cluster": cluster, "HUB": hub, "Tipo": "MIN_FILL",
                        "Transportadora": "(SEM OFERTA)", "Tipo Frota": "", "Modal": "(SEM OFERTA)", "Veiculos": 1,
                    })
                    break

                records.append({
                    "Cluster": cluster, "HUB": hub, "Tipo": "MIN_FILL",
                    "Transportadora": row["Transportadora"], "Tipo Frota": row["Tipo Frota"],
                    "Modal": row["Modal"], "Veiculos": 1,
                })

                rem_kg = max(0.0, rem_kg - float(row["cap_kg_eff"]))
                rem_m3 = max(0.0, rem_m3 - float(row["cap_m3_eff"]))

            hub_demand[hub]["rem_kg"] = rem_kg
            hub_demand[hub]["rem_m3"] = rem_m3

        alloc_df = pd.DataFrame(records)

        output = (
            alloc_df.groupby(["Cluster","HUB","Tipo","Transportadora","Tipo Frota","Modal"], as_index=False)["Veiculos"]
            .sum()
            .sort_values(["Cluster","HUB","Tipo","Veiculos"], ascending=[True, True, True, False])
        )

        saldo = (
            plan_pool.groupby(["Cluster","Transportadora","Tipo Frota","Modal"], as_index=False)["avail"]
            .sum()
            .rename(columns={"avail":"Disponibilidade_Restante"})
            .sort_values(["Cluster","Disponibilidade_Restante"], ascending=[True, False])
        )

        all_outputs.append(output)
        all_saldos.append(saldo)

    final_output = pd.concat(all_outputs, ignore_index=True) if all_outputs else pd.DataFrame()
    final_saldo  = pd.concat(all_saldos, ignore_index=True) if all_saldos else pd.DataFrame()

    return final_output, final_saldo

def to_excel_bytes(output_consolidado: pd.DataFrame, saldo_plano: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        output_consolidado.to_excel(writer, sheet_name="output_consolidado", index=False)
        saldo_plano.to_excel(writer, sheet_name="saldo_plano", index=False)
    buf.seek(0)
    return buf.read()

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Alocação por Cluster", layout="wide")

st.title("Alocação de Veículos por Cluster (Plano x ISs)")

with st.sidebar:
    st.header("Upload dos arquivos")
    plan_file = st.file_uploader("PlanoRotas (Excel)", type=["xlsx"])
    is_file = st.file_uploader("ISsDIa (Excel)", type=["xlsx"])

    st.divider()
    st.caption("Config atual:")
    st.write(f"- OCCUPANCY_M3: {OCCUPANCY_M3}")
    st.write(f"- OCCUPANCY_KG: {OCCUPANCY_KG}")

run = st.button("Rodar alocação", type="primary", disabled=not (plan_file and is_file))

if run:
    try:
        with st.spinner("Lendo arquivos e processando..."):
            plan_df = pd.read_excel(plan_file)
            is_df = pd.read_excel(is_file)
            output_consolidado, saldo_plano = run_allocation(plan_df, is_df)

        st.success("Processamento concluído!")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("output_consolidado")
            st.dataframe(output_consolidado, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Baixar output_consolidado (CSV)",
                data=to_csv_bytes(output_consolidado),
                file_name="output_consolidado.csv",
                mime="text/csv",
            )

        with c2:
            st.subheader("saldo_plano")
            st.dataframe(saldo_plano, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Baixar saldo_plano (CSV)",
                data=to_csv_bytes(saldo_plano),
                file_name="saldo_plano.csv",
                mime="text/csv",
            )

        st.divider()
        excel_bytes = to_excel_bytes(output_consolidado, saldo_plano)
        st.download_button(
            "⬇️ Baixar Excel completo (2 abas)",
            data=excel_bytes,
            file_name="output_alocacao_por_cluster.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error("Erro ao processar. Veja detalhes abaixo:")
        st.exception(e)
else:
    st.info("Faça upload dos 2 arquivos na barra lateral e clique em **Rodar alocação**.")
