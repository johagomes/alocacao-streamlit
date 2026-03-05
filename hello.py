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

# =========================
# ✅ NOVO: ORDEM DE DISTRIBUIÇÃO POR HUB (prioridade)
# =========================
HUB_DISTRIBUTION_PRIORITY = {
    "SSP22": 1,
    "SSP12": 2,
    "SSP11": 3,
    "BRSP11": 4,
    "BRSP10": 5,
    "BRSP06": 6,
    "BRRC03": 7,
    "BRRC02": 8,
    "BRRC01": 9,
    "BRSC02": 10,
    "SRJ2": 11,
    "SPR8": 12,
    "SMG2": 13,
    "SMG8": 14,
    "SMG4": 15,
    "BRMG01": 16,
    "BRXMG3": 17,
    "BRXMG2": 18,
    "BRMG02": 19,
    "BRMG03": 20,
    "BRBA01": 21,

    # aliases comuns (se aparecerem sem zero à esquerda, por ex.)
    "BRSP6": 6,
}

def hub_priority(hub) -> int:
    h = str(hub).strip().upper()
    return int(HUB_DISTRIBUTION_PRIORITY.get(h, 999999))

def order_hubs(hubs_iterable):
    """Ordena hubs pela prioridade definida; hubs não mapeados vão pro final (ordem alfabética entre eles)."""
    hubs = list(hubs_iterable)
    return sorted(hubs, key=lambda h: (hub_priority(h), str(h).strip().upper()))

# =========================
# SANITY HELPERS (auto-correção de colunas Modal x Tipo Frota)
# =========================
def _norm_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _fleet_match_pct(s: pd.Series) -> float:
    fleet_set = set([k.upper() for k in FLEET_PRIORITY.keys()]) | {"KANGU"}
    if s is None or len(s) == 0:
        return 0.0
    v = _norm_str_series(s)
    return float(v.isin(fleet_set).mean())

def _maybe_swap_modal_frota(plan_df: pd.DataFrame, col_modal: str, col_frota: str):
    """Se detectar que col_modal parece 'Tipo Frota' (FF/SPOT/...) e col_frota parece 'Modal' (VUC/MÉDIO/...), troca."""
    try:
        pct_modal = _fleet_match_pct(plan_df[col_modal])
        pct_frota = _fleet_match_pct(plan_df[col_frota])
    except Exception:
        return col_modal, col_frota

    # se modal parece frota e frota NÃO parece frota -> swap
    if pct_modal > 0.65 and pct_frota < 0.35:
        return col_frota, col_modal
    return col_modal, col_frota

# =========================
# CAPACIDADES BASE (você pode ajustar conforme necessidade)
# =========================
VUC_BASE_KG = 1800
VUC_BASE_M3 = 10.0

MEDIO_BASE_KG = 5000
MEDIO_BASE_M3 = 16.0

TRUCK_BASE_KG = 9000
TRUCK_BASE_M3 = 30.0

CARRETA_BASE_KG = 14000
CARRETA_BASE_M3 = 55.0

# efetivos (com ocupação)
VUC_BASE_KG_EFF = VUC_BASE_KG * OCCUPANCY_KG
VUC_BASE_M3_EFF = VUC_BASE_M3 * OCCUPANCY_M3

MEDIO_BASE_KG_EFF = MEDIO_BASE_KG * OCCUPANCY_KG
MEDIO_BASE_M3_EFF = MEDIO_BASE_M3 * OCCUPANCY_M3

TRUCK_BASE_KG_EFF = TRUCK_BASE_KG * OCCUPANCY_KG
TRUCK_BASE_M3_EFF = TRUCK_BASE_M3 * OCCUPANCY_M3

CARRETA_BASE_KG_EFF = CARRETA_BASE_KG * OCCUPANCY_KG
CARRETA_BASE_M3_EFF = CARRETA_BASE_M3 * OCCUPANCY_M3

# =========================
# HELPERS: NORMALIZAÇÃO E CLASSIFICAÇÃO
# =========================
def norm(s):
    return str(s).strip().upper()

def is_big_vehicle(modal: str) -> bool:
    m = norm(modal)
    return ("TRUCK" in m) or ("CARRETA" in m)

def is_medio(modal: str) -> bool:
    m = norm(modal)
    return ("MEDIO" in m) or ("MÉDIO" in m)

def is_vuc(modal: str) -> bool:
    m = norm(modal)
    return ("VUC" in m)

def selector_class(target_class: str):
    tc = norm(target_class)
    def _sel(r):
        m = norm(r.get("Modal", ""))
        if tc == "MEDIO":
            return is_medio(m)
        if tc == "VUC":
            return is_vuc(m) and (not is_medio(m))
        if tc == "BIG":
            return is_big_vehicle(m)
        return True
    return _sel

def cap_for_modal(modal: str):
    m = norm(modal)
    if is_medio(m):
        return MEDIO_BASE_KG_EFF, MEDIO_BASE_M3_EFF
    if is_big_vehicle(m):
        if "CARRETA" in m:
            return CARRETA_BASE_KG_EFF, CARRETA_BASE_M3_EFF
        return TRUCK_BASE_KG_EFF, TRUCK_BASE_M3_EFF
    if is_vuc(m):
        return VUC_BASE_KG_EFF, VUC_BASE_M3_EFF
    return 0.0, 0.0

def required_units_by_capacity(kg, m3, cap_kg, cap_m3):
    if cap_kg <= 0 or cap_m3 <= 0:
        return 0
    n_kg = kg / cap_kg
    n_m3 = m3 / cap_m3
    return int(math.ceil(max(n_kg, n_m3)))

# =========================
# REGRAS: OVERSIZE (>=16m3 OU >=1800kg -> MÉDIO obrigatório)
# =========================
def split_oversize_vs_vuc(df_hub: pd.DataFrame):
    overs = df_hub[(df_hub["Volume_m3"] >= 16) | (df_hub["Peso_kg"] >= 1800)].copy()
    rem = df_hub.drop(overs.index).copy()
    return overs, rem

# =========================
# SCORE DO HUB (cauda / necessidade extra)
# =========================
def hub_tail_score(df_hub: pd.DataFrame):
    # heurística: % de volume/peso na cauda e necessidade extra
    # (mantém lógica original do seu código)
    total_m3 = float(df_hub["Volume_m3"].sum())
    total_kg = float(df_hub["Peso_kg"].sum())
    if total_m3 <= 0 and total_kg <= 0:
        return {"score": 0.0, "extra_need": 0.0, "tail_m3_pct": 0.0, "tail_kg_pct": 0.0}

    # define cauda como acima de percentil 90 do volume
    thr_m3 = float(df_hub["Volume_m3"].quantile(0.90)) if len(df_hub) > 0 else 0.0
    tail = df_hub[df_hub["Volume_m3"] >= thr_m3].copy()

    tail_m3 = float(tail["Volume_m3"].sum())
    tail_kg = float(tail["Peso_kg"].sum())

    tail_m3_pct = 0.0 if total_m3 <= 0 else tail_m3 / total_m3
    tail_kg_pct = 0.0 if total_kg <= 0 else tail_kg / total_kg

    score = (tail_m3_pct + tail_kg_pct) / 2.0

    # necessidade extra proporcional à cauda
    extra_need = score * 10.0

    return {
        "score": float(score),
        "extra_need": float(extra_need),
        "tail_m3_pct": float(tail_m3_pct),
        "tail_kg_pct": float(tail_kg_pct),
        "thr_m3": float(thr_m3),
    }

# =========================
# SPLIT PROPORCIONAL
# =========================
def proportional_split(scores: dict, needs: dict, total_units: int):
    if total_units <= 0:
        return {k: 0 for k in needs.keys()}

    # peso = score * need (mantém lógica)
    weights = {}
    for k in needs.keys():
        w = float(scores.get(k, 0.0)) * float(needs.get(k, 0.0))
        weights[k] = max(w, 0.0)

    s = sum(weights.values())
    if s <= 0:
        # fallback: split uniforme
        keys = list(needs.keys())
        base = total_units // max(1, len(keys))
        rem = total_units - base * max(1, len(keys))
        out = {k: base for k in keys}
        for i in range(rem):
            out[keys[i % len(keys)]] += 1
        return out

    raw = {k: (weights[k] / s) * total_units for k in weights.keys()}
    flo = {k: int(math.floor(raw[k])) for k in raw.keys()}
    assigned = sum(flo.values())
    remainder = total_units - assigned

    # distribui os restos pelos maiores decimais
    dec = sorted([(k, raw[k] - flo[k]) for k in raw.keys()], key=lambda x: x[1], reverse=True)
    out = dict(flo)
    for i in range(remainder):
        out[dec[i % len(dec)][0]] += 1

    return out

# =========================
# PLAN POOL: PREP
# =========================
def prep_plan_pool(plan_df: pd.DataFrame):
    df = plan_df.copy()

    # auto swap colunas se vierem invertidas
    col_modal = "Modal"
    col_frota = "Tipo Frota"
    if col_modal in df.columns and col_frota in df.columns:
        col_modal, col_frota = _maybe_swap_modal_frota(df, col_modal, col_frota)

        # renomeia para garantir nomes padrão
        if col_modal != "Modal":
            df = df.rename(columns={col_modal: "Modal"})
        if col_frota != "Tipo Frota":
            df = df.rename(columns={col_frota: "Tipo Frota"})

    df["Modal"] = df["Modal"].astype(str)
    df["Tipo Frota"] = df["Tipo Frota"].astype(str)

    # capacidade efetiva por linha
    caps = df["Modal"].apply(lambda m: cap_for_modal(m))
    df["cap_kg_eff"] = caps.apply(lambda x: x[0])
    df["cap_m3_eff"] = caps.apply(lambda x: x[1])

    # prioridade frota
    df["fleet_prio"] = df["Tipo Frota"].apply(lambda f: FLEET_PRIORITY.get(norm(f), 999))

    # disponibilidade (default 1)
    if "avail" not in df.columns:
        df["avail"] = 1
    df["avail"] = df["avail"].fillna(0).astype(int)

    return df

# =========================
# ALLOCATE ONE BEST
# =========================
def allocate_one_best(
    plan_pool: pd.DataFrame,
    selector_fn,
    demand_cluster: str,
    demand_hub: str,
    group_key: str,
    tracker: dict,
    group_supply: dict,
):
    if plan_pool is None or len(plan_pool) == 0:
        return None, plan_pool

    df = plan_pool.copy()

    # filtra disponíveis
    df = df[df["avail"] > 0].copy()
    if len(df) == 0:
        return None, plan_pool

    # filtra por classe/modal
    try:
        mask = df.apply(selector_fn, axis=1)
        df = df[mask].copy()
    except Exception:
        pass

    if len(df) == 0:
        return None, plan_pool

    # ordena por prioridade frota e maior capacidade (mantém lógica original)
    df = df.sort_values(["fleet_prio", "cap_m3_eff", "cap_kg_eff"], ascending=[True, False, False])

    pick = df.iloc[0].to_dict()

    # debita do pool original
    idx = df.index[0]
    plan_pool.loc[idx, "avail"] = int(plan_pool.loc[idx, "avail"]) - 1

    # tracking
    tracker.setdefault("allocs", [])
    tracker["allocs"].append({
        "Grupo_Sinergia": group_key,
        "Cluster": demand_cluster,
        "HUB": demand_hub,
        "Modal": pick.get("Modal", ""),
        "Tipo Frota": pick.get("Tipo Frota", ""),
    })

    # supply por grupo
    group_supply.setdefault(group_key, 0)
    group_supply[group_key] += 1

    return pick, plan_pool

# =========================
# BIG VEHICLE ROW CHECK
# =========================
def is_big_vehicle_row(r):
    return is_big_vehicle(r.get("Modal", ""))

# =========================
# ALLOCAÇÃO POR CLUSTER
# =========================
def allocate_for_cluster(
    cluster_name: str,
    group_key: str,
    is_cluster: pd.DataFrame,
    plan_pool: pd.DataFrame,
    group_supply: dict,
    tracker: dict,
    all_scores: list,
    all_faltas: list,
):
    records = []

    # 0) score hubs
    hub_meta = {}
    for hub, df_hub in is_cluster.groupby("HUB"):
        s = hub_tail_score(df_hub)
        hub_meta[hub] = {"score": s["score"], "extra_need": s["extra_need"]}
        all_scores.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, **s})

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

    # 2) MIN_MEDIO (obrigatório) - oversize pela regra nova (>=16m3 OU >=1800kg)
    for hub in order_hubs(hub_demand.keys()):
        sum_ov_kg = hub_demand[hub]["ov_kg"]
        sum_ov_m3 = hub_demand[hub]["ov_m3"]
        min_medio = required_units_by_capacity(sum_ov_kg, sum_ov_m3, MEDIO_BASE_KG_EFF, MEDIO_BASE_M3_EFF)

        for _ in range(min_medio):
            row, plan_pool = allocate_one_best(
                plan_pool,
                selector_class("MEDIO"),
                demand_cluster=cluster_name,
                demand_hub=hub,
                group_key=group_key,
                tracker=tracker,
                group_supply=group_supply,
            )
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "MIN_MEDIO", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "HUB": hub,
                "Tipo": "MIN_MEDIO",
                "Modal": row.get("Modal", ""),
                "Tipo Frota": row.get("Tipo Frota", ""),
                "cap_kg_eff": float(row.get("cap_kg_eff", 0.0)),
                "cap_m3_eff": float(row.get("cap_m3_eff", 0.0)),
            })

    # 3) PRE DISTRIBUIÇÃO DE MÉDIOS (opcional / por score)
    remaining_medio_supply = int(plan_pool[plan_pool.apply(selector_class("MEDIO"), axis=1)]["avail"].sum())
    if remaining_medio_supply > 0:
        hub_medio_needs = {h: required_units_by_capacity(hub_demand[h]["rem_kg"], hub_demand[h]["rem_m3"], VUC_BASE_KG_EFF, VUC_BASE_M3_EFF) for h in hub_demand.keys()}

        # scores_mean: usa score do hub como peso
        hub_mean_scores = {h: float(hub_meta.get(h, {}).get("score", 0.0)) for h in hub_demand.keys()}
        if sum(hub_mean_scores.values()) <= 1e-12:
            scores_mean = {h: 1.0 for h in hub_medio_needs}
        else:
            scores_mean = {h: max(hub_mean_scores.get(h, 0.0), 1e-9) for h in hub_medio_needs}

        medio_by_hub = proportional_split(scores_mean, hub_medio_needs, remaining_medio_supply)

        for hub, _ in sorted(
            medio_by_hub.items(),
            key=lambda kv: (hub_priority(kv[0]), -float(scores_mean.get(kv[0], 0.0)), str(kv[0]).strip().upper()),
        ):
            units = int(medio_by_hub.get(hub, 0))
            if units <= 0:
                continue

            for _ in range(units):
                if hub_demand[hub]["rem_kg"] <= 1e-6 and hub_demand[hub]["rem_m3"] <= 1e-6:
                    break

                row, plan_pool = allocate_one_best(
                    plan_pool,
                    selector_class("MEDIO"),
                    demand_cluster=cluster_name,
                    demand_hub=hub,
                    group_key=group_key,
                    tracker=tracker,
                    group_supply=group_supply,
                )
                if row is None:
                    all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "PRE_MEDIO", "Faltou": 1})
                    break

                records.append({
                    "Grupo_Sinergia": group_key,
                    "Cluster": cluster_name,
                    "HUB": hub,
                    "Tipo": "PRE_MEDIO",
                    "Modal": row.get("Modal", ""),
                    "Tipo Frota": row.get("Tipo Frota", ""),
                    "cap_kg_eff": float(row.get("cap_kg_eff", 0.0)),
                    "cap_m3_eff": float(row.get("cap_m3_eff", 0.0)),
                })

                # debita demanda remanescente
                hub_demand[hub]["rem_kg"] = max(0.0, hub_demand[hub]["rem_kg"] - float(row["cap_kg_eff"]))
                hub_demand[hub]["rem_m3"] = max(0.0, hub_demand[hub]["rem_m3"] - float(row["cap_m3_eff"]))

    # EXTRA_BIG (UPGRADE): aloca BIG onde há cauda/need extra
    remaining_big_supply = int(plan_pool[plan_pool.apply(is_big_vehicle_row, axis=1)]["avail"].sum())
    scores = {h: hub_meta[h]["score"] for h, _ in hubs_sorted}
    needs  = {h: max(0, hub_meta[h]["extra_need"]) for h, _ in hubs_sorted}
    extras_by_hub = proportional_split(scores, needs, remaining_big_supply)

    for hub, _ in sorted(
        hubs_sorted,
        key=lambda x: (hub_priority(x[0]), -float(x[1]), str(x[0]).strip().upper()),
    ):
        extra_units = int(extras_by_hub.get(hub, 0))
        if extra_units <= 0:
            continue

        for _ in range(extra_units):
            if hub_demand[hub]["rem_kg"] <= 1e-6 and hub_demand[hub]["rem_m3"] <= 1e-6:
                break

            row, plan_pool = allocate_one_best(
                plan_pool,
                selector_class("BIG"),
                demand_cluster=cluster_name,
                demand_hub=hub,
                group_key=group_key,
                tracker=tracker,
                group_supply=group_supply,
            )
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "EXTRA_BIG", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "HUB": hub,
                "Tipo": "EXTRA_BIG",
                "Modal": row.get("Modal", ""),
                "Tipo Frota": row.get("Tipo Frota", ""),
                "cap_kg_eff": float(row.get("cap_kg_eff", 0.0)),
                "cap_m3_eff": float(row.get("cap_m3_eff", 0.0)),
            })

            hub_demand[hub]["rem_kg"] = max(0.0, hub_demand[hub]["rem_kg"] - float(row["cap_kg_eff"]))
            hub_demand[hub]["rem_m3"] = max(0.0, hub_demand[hub]["rem_m3"] - float(row["cap_m3_eff"]))

    # 4) MIN_FILL
    for hub in order_hubs(hub_demand.keys()):
        rem_kg = float(hub_demand[hub]["rem_kg"])
        rem_m3 = float(hub_demand[hub]["rem_m3"])

        while rem_kg > 1e-6 or rem_m3 > 1e-6:
            row, plan_pool = allocate_one_best(
                plan_pool,
                lambda r: True,
                demand_cluster=cluster_name,
                demand_hub=hub,
                group_key=group_key,
                tracker=tracker,
                group_supply=group_supply,
            )
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "MIN_FILL", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "HUB": hub,
                "Tipo": "MIN_FILL",
                "Modal": row.get("Modal", ""),
                "Tipo Frota": row.get("Tipo Frota", ""),
                "cap_kg_eff": float(row.get("cap_kg_eff", 0.0)),
                "cap_m3_eff": float(row.get("cap_m3_eff", 0.0)),
            })

            rem_kg = max(0.0, rem_kg - float(row["cap_kg_eff"]))
            rem_m3 = max(0.0, rem_m3 - float(row["cap_m3_eff"]))

        hub_demand[hub]["rem_kg"] = rem_kg
        hub_demand[hub]["rem_m3"] = rem_m3

    return records, plan_pool

# =========================
# STREAMLIT APP
# =========================
def main():
    st.title("Alocação de Modal - Distribuição por Cluster")

    st.write("Faça upload das bases necessárias e rode a alocação.")

    is_file = st.file_uploader("Upload base IS (First Mile / demanda) - CSV/Excel", type=["csv", "xlsx"])
    plan_file = st.file_uploader("Upload base Plano (pool de veículos) - CSV/Excel", type=["csv", "xlsx"])

    if is_file is None or plan_file is None:
        st.info("Envie os dois arquivos para continuar.")
        return

    def read_file(f):
        name = f.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(f)
        return pd.read_excel(f)

    is_df = read_file(is_file)
    plan_df = read_file(plan_file)

    st.subheader("Pré-visualização IS")
    st.dataframe(is_df.head(30))

    st.subheader("Pré-visualização Plano")
    st.dataframe(plan_df.head(30))

    # Ajuste nomes esperados (mantém lógica original)
    # Espera colunas: HUB, Cluster, Volume_m3, Peso_kg (na IS)
    # E no plano: Modal, Tipo Frota, avail (opcional)
    req_is = {"HUB", "Cluster", "Volume_m3", "Peso_kg"}
    if not req_is.issubset(set(is_df.columns)):
        st.error(f"IS precisa ter colunas: {sorted(list(req_is))}")
        return

    if "Modal" not in plan_df.columns or ("Tipo Frota" not in plan_df.columns and "Tipo_Frota" not in plan_df.columns):
        st.error("Plano precisa ter colunas: Modal e Tipo Frota (ou Tipo_Frota)")
        return

    if "Tipo_Frota" in plan_df.columns and "Tipo Frota" not in plan_df.columns:
        plan_df = plan_df.rename(columns={"Tipo_Frota": "Tipo Frota"})

    plan_pool = prep_plan_pool(plan_df)

    tracker = {}
    group_supply = {}
    all_scores = []
    all_faltas = []
    all_records = []

    # agrupamento por cluster
    for cluster_name, df_cluster in is_df.groupby("Cluster"):
        group_key = str(cluster_name)

        recs, plan_pool = allocate_for_cluster(
            cluster_name=cluster_name,
            group_key=group_key,
            is_cluster=df_cluster,
            plan_pool=plan_pool,
            group_supply=group_supply,
            tracker=tracker,
            all_scores=all_scores,
            all_faltas=all_faltas,
        )
        all_records.extend(recs)

    result_df = pd.DataFrame(all_records)
    faltas_df = pd.DataFrame(all_faltas)
    scores_df = pd.DataFrame(all_scores)

    st.subheader("Resultado de Alocação")
    st.dataframe(result_df)

    st.subheader("Faltas")
    st.dataframe(faltas_df)

    st.subheader("Scores (Hub Tail)")
    st.dataframe(scores_df)

    # Download
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False, sheet_name="allocs")
        faltas_df.to_excel(writer, index=False, sheet_name="faltas")
        scores_df.to_excel(writer, index=False, sheet_name="scores")
        plan_pool.to_excel(writer, index=False, sheet_name="plan_pool_final")

    st.download_button(
        "Baixar Excel (allocs/faltas/scores/plan_pool_final)",
        data=out.getvalue(),
        file_name="resultado_alocacao.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    main()
