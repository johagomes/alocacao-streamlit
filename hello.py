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

def cluster_synergy_key(cluster: str) -> str:
    """
    Regra de sinergia:
    - Todo cluster contém '.'
    - Se o "prefixo antes do ponto" se repete, então esses clusters podem compartilhar rotas.
      Ex: 'CLUSTER 1.1 OESTE' e 'CLUSTER 1.2 SUDOESTE' => chave 'CLUSTER 1'
    """
    s = str(cluster).strip()
    if "." not in s:
        return s
    # captura tudo antes do primeiro ponto, preservando espaçamento original
    return s.split(".", 1)[0].strip()

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

def allocate_one_best(
    plan_pool: pd.DataFrame,
    selector_fn,
    demand_cluster: str | None = None,
    group_key: str | None = None,
    tracker: dict | None = None,
    group_supply: dict | None = None,
):
    """Seleciona 1 veículo respeitando:
    - prioridade de frota (Kangu -> FF -> Spot -> Spot DPC)
    - preferências de capacidade já existentes (cap_m3_eff/cap_kg_eff)
    - Kangu NÃO pode fazer sinergia entre clusters
    - NOVA REGRA: uso proporcional entre Transportadoras dentro da mesma região (grupo de sinergia),
      aplicado PARA TODOS OS MODAIS (vehicle_class).
        * A escolha do modal/porte (vehicle_class) segue a lógica original (capacidade).
        * Depois disso, a Transportadora dentro desse modal é escolhida proporcionalmente ao saldo inicial
          do bucket (group_key, vehicle_class, fleet_priority).
    """
    eligible = plan_pool[(plan_pool["avail"] > 0)].copy()
    eligible = eligible[eligible.apply(selector_fn, axis=1)].copy()

    # Regra: Kangu NÃO pode fazer sinergia entre clusters.
    if demand_cluster is not None and not eligible.empty:
        kangu_mask = eligible["Tipo Frota"].astype(str).str.upper().str.strip().eq("KANGU")
        eligible = pd.concat(
            [
                eligible[~kangu_mask],
                eligible[kangu_mask & (eligible["Cluster"].astype(str) == str(demand_cluster))],
            ],
            axis=0,
        )

    if eligible.empty:
        return None, plan_pool

    # Inicializa estruturas
    if tracker is None:
        tracker = {}
    if group_supply is None:
        group_supply = {}

    # 1) Primeiro, decide QUAL modal/porte (vehicle_class) usaria, seguindo a lógica original
    base_sorted = eligible.sort_values(
        ["fleet_priority", "cap_m3_eff", "cap_kg_eff", "avail"],
        ascending=[True, False, False, False],
    )
    base_row = base_sorted.iloc[0]
    fp_target = int(base_row.get("fleet_priority", 9))
    vc_target = str(base_row.get("vehicle_class", ""))

    # 2) Dentro do mesmo (fleet_priority + vehicle_class), aplica proporcionalidade por transportadora
    bucket = eligible[
        (eligible["fleet_priority"].astype(int) == fp_target)
        & (eligible["vehicle_class"].astype(str) == vc_target)
    ].copy()

    # fallback (não deveria acontecer)
    if bucket.empty:
        bucket = eligible.copy()

    gk = str(group_key) if group_key is not None else ""

    def usage_ratio_row(r):
        vc = str(r.get("vehicle_class", ""))
        fp = int(r.get("fleet_priority", 9))
        tr = str(r.get("Transportadora", ""))
        denom = float(group_supply.get((gk, vc, fp, tr), 0.0))
        if denom <= 0:
            denom = float(r.get("init_avail", 1)) if float(r.get("init_avail", 0) or 0) > 0 else 1.0
        used = float(tracker.get((gk, vc, fp, tr), 0))
        return used / denom

    bucket["_usage_ratio"] = bucket.apply(usage_ratio_row, axis=1)
    bucket["_used_abs"] = bucket.apply(
        lambda r: float(tracker.get((gk, str(r.get("vehicle_class","")), int(r.get("fleet_priority",9)), str(r.get("Transportadora",""))), 0)),
        axis=1
    )

    # Ordenação final:
    # - menor ratio (para distribuir proporcional)
    # - depois capacidade (mantém “maiores para ISs maiores” dentro do modal escolhido)
    bucket = bucket.sort_values(
        ["_usage_ratio", "cap_m3_eff", "cap_kg_eff", "_used_abs", "avail"],
        ascending=[True, False, False, True, False],
    )

    row = bucket.iloc[0]
    idx = row.name
    plan_pool.loc[idx, "avail"] = int(plan_pool.loc[idx, "avail"]) - 1

    # Atualiza tracker do bucket escolhido
    vc = str(row.get("vehicle_class", ""))
    fp = int(row.get("fleet_priority", 9))
    tr = str(row.get("Transportadora", ""))
    tracker[(gk, vc, fp, tr)] = int(tracker.get((gk, vc, fp, tr), 0)) + 1

    return row, plan_pool


def cluster_demand_score(df_cluster: pd.DataFrame) -> float:
    """
    Heurística só para ordenar clusters dentro do mesmo grupo de sinergia:
    - quanto maior a demanda (em "unidades VUC"), mais cedo aloca (reduz risco de faltar no fim)
    """
    sum_kg = float(df_cluster["Peso_kg"].sum())
    sum_m3 = float(df_cluster["Volume_m3"].sum())
    if VUC_BASE_KG_EFF <= 0 or VUC_BASE_M3_EFF <= 0:
        return sum_kg + sum_m3
    return float(max(sum_kg / VUC_BASE_KG_EFF, sum_m3 / VUC_BASE_M3_EFF))

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
    """
    Roda exatamente o mesmo algoritmo atual, mas:
    - usa um plan_pool "compartilhado" (sinergia) quando aplicável
    - registra também o cluster de origem da oferta (Cluster_Oferta)
    """
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

    # 2) MIN_MEDIO (obrigatório) - para oversize > perfil VUC
    for hub in sorted(hub_demand.keys()):
        sum_ov_kg = hub_demand[hub]["ov_kg"]
        sum_ov_m3 = hub_demand[hub]["ov_m3"]
        min_medio = required_units_by_capacity(sum_ov_kg, sum_ov_m3, MEDIO_BASE_KG_EFF, MEDIO_BASE_M3_EFF)

        for _ in range(min_medio):
            row, plan_pool = allocate_one_best(plan_pool, selector_class("MEDIO"), demand_cluster=cluster_name, group_key=group_key, tracker=tracker, group_supply=group_supply)
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "MIN_MEDIO", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "Cluster_Oferta": row["Cluster"],
                "HUB": hub,
                "Tipo": "MIN_MEDIO",
                "Transportadora": row["Transportadora"],
                "Tipo Frota": row["Tipo Frota"],
                "Modal": row["Modal"],
                "Veiculos": 1,
            })

    # 3) EXTRAS (UPGRADE) - redistribui veículos maiores, abatendo demanda residual
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

            row, plan_pool = allocate_one_best(plan_pool, selector_big, demand_cluster=cluster_name, group_key=group_key, tracker=tracker, group_supply=group_supply)
            if row is None:
                all_faltas.append({"Grupo_Sinergia": group_key, "Cluster": cluster_name, "HUB": hub, "Tipo": "EXTRA_BIG", "Faltou": 1})
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "Cluster_Oferta": row["Cluster"],
                "HUB": hub,
                "Tipo": "EXTRA_BIG",
                "Transportadora": row["Transportadora"],
                "Tipo Frota": row["Tipo Frota"],
                "Modal": row["Modal"],
                "Veiculos": 1,
            })

            hub_demand[hub]["rem_kg"] = max(0.0, hub_demand[hub]["rem_kg"] - float(row["cap_kg_eff"]))
            hub_demand[hub]["rem_m3"] = max(0.0, hub_demand[hub]["rem_m3"] - float(row["cap_m3_eff"]))

    # 4) MIN_FILL (completa residual) - respeita prioridades Kangu -> FF -> Spot -> Spot DPC
    for hub in sorted(hub_demand.keys()):
        rem_kg = float(hub_demand[hub]["rem_kg"])
        rem_m3 = float(hub_demand[hub]["rem_m3"])

        while rem_kg > 1e-6 or rem_m3 > 1e-6:
            row, plan_pool = allocate_one_best(plan_pool, lambda r: True, demand_cluster=cluster_name, group_key=group_key, tracker=tracker, group_supply=group_supply)
            if row is None:
                records.append({
                    "Grupo_Sinergia": group_key,
                    "Cluster": cluster_name,
                    "Cluster_Oferta": "",
                    "HUB": hub,
                    "Tipo": "MIN_FILL",
                    "Transportadora": "(SEM OFERTA)",
                    "Tipo Frota": "",
                    "Modal": "(SEM OFERTA)",
                    "Veiculos": 1,
                })
                break

            records.append({
                "Grupo_Sinergia": group_key,
                "Cluster": cluster_name,
                "Cluster_Oferta": row["Cluster"],
                "HUB": hub,
                "Tipo": "MIN_FILL",
                "Transportadora": row["Transportadora"],
                "Tipo Frota": row["Tipo Frota"],
                "Modal": row["Modal"],
                "Veiculos": 1,
            })

            rem_kg = max(0.0, rem_kg - float(row["cap_kg_eff"]))
            rem_m3 = max(0.0, rem_m3 - float(row["cap_m3_eff"]))

        hub_demand[hub]["rem_kg"] = rem_kg
        hub_demand[hub]["rem_m3"] = rem_m3

    return records, plan_pool

# =========================
# CORE RUNNER
# =========================
def run_allocation(plan_df: pd.DataFrame, is_df: pd.DataFrame, enable_synergy: bool = True, return_debug: bool = False):
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
    plan["init_avail"] = plan["avail"]

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

    # Apenas clusters que existem nos dois lados (igualdade exata)
    common_clusters = sorted(list(set(plan["Cluster"].astype(str)).intersection(set(isdata["Cluster"].astype(str)))))
    if not common_clusters:
        raise ValueError("Não encontrei clusters em comum entre Plano e ISs.")

    # =========================
    # SINERGIA POR PREFIXO (antes do ponto)
    # =========================
    if enable_synergy:
        plan["Grupo_Sinergia"] = plan["Cluster"].map(cluster_synergy_key)
        isdata["Grupo_Sinergia"] = isdata["Cluster"].map(cluster_synergy_key)
    else:
        plan["Grupo_Sinergia"] = plan["Cluster"].astype(str)
        isdata["Grupo_Sinergia"] = isdata["Cluster"].astype(str)

    # Grupos só com clusters em comum
    groups = {}
    for c in common_clusters:
        g = cluster_synergy_key(c) if enable_synergy else str(c)
        groups.setdefault(g, []).append(str(c))

    all_allocs, all_saldos, all_scores, all_faltas = [], [], [], []
    tracker = {}

    # Processa por grupo (pool compartilhado dentro do grupo)
    for group_key, member_clusters in sorted(groups.items(), key=lambda x: x[0]):
        plan_pool = plan[plan["Cluster"].astype(str).isin(member_clusters)].copy()

        # base de oferta inicial por transportadora (para distribuição proporcional)
        # chave: (group_key, vehicle_class, fleet_priority, Transportadora) -> init_avail_total
        group_supply = (
            plan_pool.groupby(["vehicle_class", "fleet_priority", "Transportadora"], as_index=False)["init_avail"]
            .sum()
        )
        group_supply = {(
            str(group_key),
            str(r["vehicle_class"]),
            int(r["fleet_priority"]),
            str(r["Transportadora"]),
        ): float(r["init_avail"]) for _, r in group_supply.iterrows()}

        # ordem de alocação dos clusters dentro do grupo (maior demanda primeiro)
        demand_clusters = []
        for c in member_clusters:
            df_c = isdata[isdata["Cluster"].astype(str) == str(c)].copy()
            if df_c.empty:
                continue
            demand_clusters.append((c, cluster_demand_score(df_c)))
        demand_clusters.sort(key=lambda x: x[1], reverse=True)

        # roda cada cluster (demanda) usando o MESMO pool
        for cluster_name, _score in demand_clusters:
            is_cluster = isdata[isdata["Cluster"].astype(str) == str(cluster_name)].copy()
            if is_cluster.empty or plan_pool.empty:
                continue

            records, plan_pool = allocate_for_cluster(
                cluster_name=str(cluster_name),
                group_key=str(group_key),
                is_cluster=is_cluster,
                plan_pool=plan_pool,
                group_supply=group_supply,
                tracker=tracker,
                all_scores=all_scores,
                all_faltas=all_faltas,
            )

            alloc_df = pd.DataFrame(records)

            if alloc_df.empty:
                continue

            # Guardamos o detalhado (debug) para permitir análises (sinergia, proporcionalidade etc.)
            all_allocs.append(alloc_df)

        # saldo do pool ao final do grupo (por cluster de oferta original)
        if not plan_pool.empty:
            saldo = (
                plan_pool.groupby(["Grupo_Sinergia","Cluster","Transportadora","Tipo Frota","Modal"], as_index=False)["avail"]
                .sum()
                .rename(columns={"avail":"Disponibilidade_Restante"})
                .sort_values(["Grupo_Sinergia","Cluster","Disponibilidade_Restante"], ascending=[True, True, False])
            )
            all_saldos.append(saldo)

    debug_alloc = pd.concat(all_allocs, ignore_index=True) if all_allocs else pd.DataFrame()
    saldo_debug  = pd.concat(all_saldos, ignore_index=True) if all_saldos else pd.DataFrame()

    # =========================
    # OUTPUT FINAL (somente colunas pedidas)
    # =========================
    if debug_alloc.empty:
        final_output = pd.DataFrame(columns=["Cluster","HUB","Transportadora","Tipo Frota","Modal","Veiculos"])
    else:
        final_output = (
            debug_alloc.groupby(["Cluster","HUB","Transportadora","Tipo Frota","Modal"], as_index=False)["Veiculos"]
            .sum()
            .sort_values(["Cluster","HUB","Tipo Frota","Transportadora","Modal"], ascending=[True, True, True, True, True])
        )

    if saldo_debug.empty:
        final_saldo = pd.DataFrame(columns=["Cluster","Transportadora","Tipo Frota","Modal","Disponibilidade_Restante"])
    else:
        final_saldo = (
            saldo_debug.groupby(["Cluster","Transportadora","Tipo Frota","Modal"], as_index=False)["Disponibilidade_Restante"]
            .sum()
            .sort_values(["Cluster","Tipo Frota","Transportadora","Modal"], ascending=[True, True, True, True])
        )

    if return_debug:
        # plan_common (oferta) é útil para análises de distribuição
        plan_common = plan[plan["Cluster"].astype(str).isin(common_clusters)].copy()
        return final_output, final_saldo, debug_alloc, saldo_debug, plan_common

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

# =========================
# ANALYTICS
# =========================
def _safe_pct(num, den):
    return float(num) / float(den) if den and den != 0 else 0.0

def build_analyses(output_final: pd.DataFrame, saldo_final: pd.DataFrame, debug_alloc: pd.DataFrame, plan_common: pd.DataFrame) -> dict:
    """
    Gera análises de distribuição de frota entre:
    - Clusters / HUBs
    - Transportadoras
    - Tipo Frota
    - Modal (exato) e Classe (VUC/VAN/MEDIO/TRUCK/CARRETA/...)
    - Sinergia (emprestimos entre clusters dentro do mesmo Grupo_Sinergia)
    - Diagnóstico de proporcionalidade (por bucket)
    """
    analyses = {}

    # -------------------------
    # Normalizações / colunas base
    # -------------------------
    if plan_common is None or plan_common.empty:
        plan_common = pd.DataFrame(columns=["Cluster","Grupo_Sinergia","Transportadora","Tipo Frota","Modal","Disponibilidade"])

    if "Grupo_Sinergia" not in plan_common.columns:
        plan_common["Grupo_Sinergia"] = plan_common["Cluster"].map(cluster_synergy_key)

    if "vehicle_class" not in plan_common.columns:
        plan_common["vehicle_class"] = plan_common["Modal"].map(vehicle_class)

    # Output usado (remove SEM OFERTA)
    used_rows = output_final.copy() if output_final is not None else pd.DataFrame()
    if not used_rows.empty:
        used_rows = used_rows[~used_rows["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True)].copy()
        used_rows["Grupo_Sinergia"] = used_rows["Cluster"].map(cluster_synergy_key)
        used_rows["vehicle_class"] = used_rows["Modal"].map(vehicle_class)

    # Linhas de falta (SEM OFERTA)
    falta_rows = output_final.copy() if output_final is not None else pd.DataFrame()
    if not falta_rows.empty:
        falta_rows = falta_rows[falta_rows["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True)].copy()

    # -------------------------
    # Helpers de comparação Oferta vs Uso vs Saldo
    # -------------------------
    def _merge_offer_use_saldo(group_cols, offer_cols=None):
        offer = plan_common.groupby(group_cols, as_index=False)["Disponibilidade"].sum().rename(columns={"Disponibilidade":"Oferta"})

        if used_rows.empty:
            used = pd.DataFrame(columns=group_cols + ["Usado"])
        else:
            used = used_rows.groupby(group_cols, as_index=False)["Veiculos"].sum().rename(columns={"Veiculos":"Usado"})

        if saldo_final is None or saldo_final.empty:
            saldo = pd.DataFrame(columns=group_cols + ["Saldo"])
        else:
            # saldo_final tem: Cluster, Transportadora, Tipo Frota, Modal, Disponibilidade_Restante
            # vamos reconstruir vehicle_class e Grupo_Sinergia se necessário para agrupar
            s = saldo_final.copy()
            if "Grupo_Sinergia" not in s.columns:
                s["Grupo_Sinergia"] = s["Cluster"].map(cluster_synergy_key)
            if "vehicle_class" not in s.columns and "Modal" in s.columns:
                s["vehicle_class"] = s["Modal"].map(vehicle_class)

            # tenta usar coluna Disponibilidade_Restante; se não existir, cria
            if "Disponibilidade_Restante" in s.columns:
                s["_SaldoBase"] = s["Disponibilidade_Restante"]
            elif "Saldo" in s.columns:
                s["_SaldoBase"] = s["Saldo"]
            else:
                s["_SaldoBase"] = 0

            saldo = s.groupby(group_cols, as_index=False)["_SaldoBase"].sum().rename(columns={"_SaldoBase":"Saldo"})

        df = offer.merge(used, on=group_cols, how="outer").merge(saldo, on=group_cols, how="outer").fillna(0)
        df["Utilizacao_%"] = df.apply(lambda r: _safe_pct(r.get("Usado",0), r.get("Oferta",0)), axis=1)
        return df

    def _add_shares(df, total_cols):
        totals = df.groupby(total_cols, as_index=False)[["Oferta","Usado"]].sum().rename(columns={"Oferta":"Oferta_Total","Usado":"Usado_Total"})
        out = df.merge(totals, on=total_cols, how="left")
        out["Oferta_%"] = out.apply(lambda r: _safe_pct(r["Oferta"], r["Oferta_Total"]), axis=1)
        out["Uso_%"] = out.apply(lambda r: _safe_pct(r["Usado"], r["Usado_Total"]), axis=1)
        out["Delta_pp"] = (out["Uso_%"] - out["Oferta_%"]) * 100
        return out

    # -------------------------
    # Resumos macro
    # -------------------------
    analyses["Resumo_Frota"] = _merge_offer_use_saldo(["Tipo Frota"]).sort_values("Tipo Frota", ascending=True)
    analyses["Resumo_Classe"] = _merge_offer_use_saldo(["vehicle_class"]).sort_values("vehicle_class", ascending=True)
    analyses["Resumo_Modal"] = _merge_offer_use_saldo(["Modal"]).sort_values("Oferta", ascending=False)

    # -------------------------
    # Uso / Saldo por Cluster e HUB
    # -------------------------
    analyses["Uso_Cluster_Frota"] = (
        used_rows.groupby(["Cluster","Tipo Frota"], as_index=False)["Veiculos"].sum().sort_values(["Cluster","Tipo Frota"])
        if not used_rows.empty else pd.DataFrame(columns=["Cluster","Tipo Frota","Veiculos"])
    )

    analyses["Saldo_Cluster_Frota"] = (
        (saldo_final.copy().assign(Grupo_Sinergia=lambda d: d["Cluster"].map(cluster_synergy_key))
         .groupby(["Cluster","Tipo Frota"], as_index=False)["Disponibilidade_Restante"].sum()
         .rename(columns={"Disponibilidade_Restante":"Saldo"})
         .sort_values(["Cluster","Tipo Frota"]))
        if saldo_final is not None and not saldo_final.empty else pd.DataFrame(columns=["Cluster","Tipo Frota","Saldo"])
    )

    analyses["Uso_HUB_Frota"] = (
        used_rows.groupby(["HUB","Tipo Frota"], as_index=False)["Veiculos"].sum().sort_values(["HUB","Tipo Frota"])
        if not used_rows.empty else pd.DataFrame(columns=["HUB","Tipo Frota","Veiculos"])
    )

    analyses["Uso_Cluster_Modal"] = (
        used_rows.groupby(["Cluster","Modal"], as_index=False)["Veiculos"].sum().sort_values(["Cluster","Veiculos"], ascending=[True,False])
        if not used_rows.empty else pd.DataFrame(columns=["Cluster","Modal","Veiculos"])
    )

    analyses["Uso_HUB_Modal"] = (
        used_rows.groupby(["HUB","Modal"], as_index=False)["Veiculos"].sum().sort_values(["HUB","Veiculos"], ascending=[True,False])
        if not used_rows.empty else pd.DataFrame(columns=["HUB","Modal","Veiculos"])
    )

    # -------------------------
    # Distribuição por Transportadora (macro)
    # -------------------------
    dist_car = _merge_offer_use_saldo(["Transportadora"]).sort_values("Oferta", ascending=False)

    tot_of = float(dist_car["Oferta"].sum()) if "Oferta" in dist_car.columns else 0.0
    tot_us = float(dist_car["Usado"].sum()) if "Usado" in dist_car.columns else 0.0
    dist_car["Oferta_%"] = dist_car["Oferta"].apply(lambda x: _safe_pct(x, tot_of))
    dist_car["Uso_%"] = dist_car["Usado"].apply(lambda x: _safe_pct(x, tot_us))
    dist_car["Delta_pp"] = (dist_car["Uso_%"] - dist_car["Oferta_%"]) * 100
    analyses["Distribuicao_Transportadora"] = dist_car

    # Transportadora x Tipo Frota
    dist_car_frota = _merge_offer_use_saldo(["Tipo Frota","Transportadora"])
    dist_car_frota = _add_shares(dist_car_frota, ["Tipo Frota"]).sort_values(["Tipo Frota","Oferta"], ascending=[True,False])
    analyses["Distribuicao_Transportadora_Frota"] = dist_car_frota

    # Transportadora x Tipo Frota x Classe
    dist_car_bucket = _merge_offer_use_saldo(["Tipo Frota","vehicle_class","Transportadora"])
    dist_car_bucket = _add_shares(dist_car_bucket, ["Tipo Frota","vehicle_class"]).sort_values(["Tipo Frota","vehicle_class","Oferta"], ascending=[True,True,False])
    analyses["Distribuicao_Transportadora_Classe"] = dist_car_bucket

    # Cluster x Transportadora (concentração)
    analyses["Uso_Cluster_Transportadora"] = (
        used_rows.groupby(["Cluster","Transportadora"], as_index=False)["Veiculos"].sum().sort_values(["Cluster","Veiculos"], ascending=[True,False])
        if not used_rows.empty else pd.DataFrame(columns=["Cluster","Transportadora","Veiculos"])
    )

    analyses["Uso_HUB_Transportadora"] = (
        used_rows.groupby(["HUB","Transportadora"], as_index=False)["Veiculos"].sum().sort_values(["HUB","Veiculos"], ascending=[True,False])
        if not used_rows.empty else pd.DataFrame(columns=["HUB","Transportadora","Veiculos"])
    )

    # -------------------------
    # Mix (shares) por HUB e por Cluster
    # -------------------------
    if not used_rows.empty:
        mix_hub = used_rows.groupby(["HUB","Tipo Frota"], as_index=False)["Veiculos"].sum()
        mix_hub_tot = mix_hub.groupby(["HUB"], as_index=False)["Veiculos"].sum().rename(columns={"Veiculos":"Total"})
        mix_hub = mix_hub.merge(mix_hub_tot, on="HUB", how="left")
        mix_hub["Share_%"] = mix_hub.apply(lambda r: _safe_pct(r["Veiculos"], r["Total"]), axis=1)
        analyses["Mix_Frota_por_HUB"] = mix_hub.sort_values(["HUB","Share_%"], ascending=[True,False])

        mix_cluster = used_rows.groupby(["Cluster","Tipo Frota"], as_index=False)["Veiculos"].sum()
        mix_cluster_tot = mix_cluster.groupby(["Cluster"], as_index=False)["Veiculos"].sum().rename(columns={"Veiculos":"Total"})
        mix_cluster = mix_cluster.merge(mix_cluster_tot, on="Cluster", how="left")
        mix_cluster["Share_%"] = mix_cluster.apply(lambda r: _safe_pct(r["Veiculos"], r["Total"]), axis=1)
        analyses["Mix_Frota_por_Cluster"] = mix_cluster.sort_values(["Cluster","Share_%"], ascending=[True,False])
    else:
        analyses["Mix_Frota_por_HUB"] = pd.DataFrame(columns=["HUB","Tipo Frota","Veiculos","Total","Share_%"])
        analyses["Mix_Frota_por_Cluster"] = pd.DataFrame(columns=["Cluster","Tipo Frota","Veiculos","Total","Share_%"])

    # -------------------------
    # Sinergia (emprestimos) usando debug_alloc (que mantém Cluster_Oferta)
    # -------------------------
    sinergia = pd.DataFrame(columns=["Grupo_Sinergia","Cluster","Cluster_Oferta","Tipo Frota","vehicle_class","Veiculos"])
    if debug_alloc is not None and not debug_alloc.empty:
        tmp = debug_alloc.copy()
        tmp["vehicle_class"] = tmp["Modal"].map(vehicle_class)
        # emprestimo = cluster_oferta diferente do cluster da demanda
        tmp = tmp[(tmp["Cluster_Oferta"].astype(str) != tmp["Cluster"].astype(str))].copy()
        # Kangu não empresta
        tmp = tmp[tmp["Tipo Frota"].astype(str).str.upper().str.strip() != "KANGU"]
        if not tmp.empty:
            sinergia = (
                tmp.groupby(["Grupo_Sinergia","Cluster","Cluster_Oferta","Tipo Frota","vehicle_class"], as_index=False)["Veiculos"]
                .sum()
                .sort_values(["Grupo_Sinergia","Cluster","Veiculos"], ascending=[True,True,False])
            )
    analyses["Sinergia_Emprestimos"] = sinergia

    # -------------------------
    # Diagnóstico de proporcionalidade por bucket (região + frota + classe + transportadora)
    # -------------------------
    # Oferta no bucket (plano)
    oferta_bucket = (
        plan_common.groupby(["Grupo_Sinergia","Tipo Frota","vehicle_class","Transportadora"], as_index=False)["Disponibilidade"]
        .sum()
        .rename(columns={"Disponibilidade":"Oferta"})
    )

    # Uso no bucket (debug_alloc mantém Grupo_Sinergia)
    if debug_alloc is None or debug_alloc.empty:
        uso_bucket = pd.DataFrame(columns=["Grupo_Sinergia","Tipo Frota","vehicle_class","Transportadora","Usado"])
    else:
        da = debug_alloc.copy()
        da = da[~da["Transportadora"].astype(str).str.contains(r"\(SEM OFERTA\)", regex=True)].copy()
        da["vehicle_class"] = da["Modal"].map(vehicle_class)
        uso_bucket = (
            da.groupby(["Grupo_Sinergia","Tipo Frota","vehicle_class","Transportadora"], as_index=False)["Veiculos"]
            .sum()
            .rename(columns={"Veiculos":"Usado"})
        )

    bucket = oferta_bucket.merge(uso_bucket, on=["Grupo_Sinergia","Tipo Frota","vehicle_class","Transportadora"], how="outer").fillna(0)
    # shares e delta por bucket (região + frota + classe)
    bucket = _add_shares(bucket, ["Grupo_Sinergia","Tipo Frota","vehicle_class"])
    bucket = bucket.sort_values(["Grupo_Sinergia","Tipo Frota","vehicle_class","Oferta"], ascending=[True,True,True,False])
    analyses["Proporcionalidade_Bucket"] = bucket

    # -------------------------
    # Falta de oferta (onde apareceu SEM OFERTA)
    # -------------------------
    if falta_rows is not None and not falta_rows.empty:
        analyses["Falta_Oferta_por_Cluster_HUB"] = (
            falta_rows.groupby(["Cluster","HUB"], as_index=False)["Veiculos"].sum().sort_values(["Veiculos"], ascending=False)
        )
    else:
        analyses["Falta_Oferta_por_Cluster_HUB"] = pd.DataFrame(columns=["Cluster","HUB","Veiculos"])

    return analyses

# STREAMLIT UI
# =========================

st.set_page_config(page_title="Alocação por Cluster", layout="wide")

st.title("Distribuição de Rotas Routing")

# -------------------------
# SESSION STATE
# -------------------------
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False
    st.session_state.output_consolidado = None
    st.session_state.saldo_plano = None
    st.session_state.analyses = None
    st.session_state.debug_alloc = None
    st.session_state.plan_common = None

# -------------------------
# SIDEBAR (inputs + navegação)
# -------------------------
with st.sidebar:
    st.header("Arquivos")

    plan_file = st.file_uploader("PlanoRotas (Excel)", type=["xlsx"], key="plan_upl")
    is_file = st.file_uploader("ISsDIa (Excel)", type=["xlsx"], key="is_upl")

    st.divider()
    st.header("Regras / Opções")
    st.write("Sinergia: **sempre ativa** (Kangu não faz sinergia).")

    st.divider()
    page = st.radio("Interface", ["Output", "Análises"], index=0, key="page")

    run = st.button("Rodar alocação", type="primary", disabled=not (plan_file and is_file))

# -------------------------
# PROCESSAMENTO (quando clicar)
# -------------------------
if run:
    try:
        with st.spinner("Lendo arquivos e processando..."):
            plan_df = pd.read_excel(plan_file)
            is_df = pd.read_excel(is_file)

            # roda com debug para alimentar análises e rastreios (sinergia, proporcionalidade etc.)
            output_consolidado, saldo_plano, debug_alloc, saldo_debug, plan_common = run_allocation(
                plan_df, is_df, enable_synergy=True, return_debug=True
            )

            analyses = build_analyses(
                output_final=output_consolidado,
                saldo_final=saldo_plano,
                debug_alloc=debug_alloc,
                plan_common=plan_common,
            )

            st.session_state.output_consolidado = output_consolidado
            st.session_state.saldo_plano = saldo_plano
            st.session_state.debug_alloc = debug_alloc
            st.session_state.plan_common = plan_common
            st.session_state.analyses = analyses
            st.session_state.result_ready = True

        st.success("Processamento concluído!")

    except Exception as e:
        st.session_state.result_ready = False
        st.error("Erro ao processar. Veja detalhes abaixo:")
        st.exception(e)

# -------------------------
# UI (duas interfaces)
# -------------------------
if not st.session_state.result_ready:
    st.info("Faça upload dos 2 arquivos na barra lateral e clique em **Rodar alocação**.")
else:
    output_consolidado = st.session_state.output_consolidado
    saldo_plano = st.session_state.saldo_plano
    analyses = st.session_state.analyses or {}

    if page == "Output":
        st.subheader("Output consolidado")

        st.dataframe(output_consolidado, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "⬇️ Baixar Output (CSV)",
                data=to_csv_bytes(output_consolidado),
                file_name="output_consolidado.csv",
                mime="text/csv",
            )

        with col2:
            st.download_button(
                "⬇️ Baixar Saldo (CSV)",
                data=to_csv_bytes(saldo_plano),
                file_name="saldo_plano.csv",
                mime="text/csv",
            )

        with col3:
            excel_bytes = to_excel_bytes_multi(
                {"output_consolidado": output_consolidado, "saldo_plano": saldo_plano}
            )
            st.download_button(
                "⬇️ Baixar Excel (Output + Saldo)",
                data=excel_bytes,
                file_name="output_alocacao_por_cluster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.divider()
        st.subheader("Saldo do plano")
        st.dataframe(saldo_plano, use_container_width=True, hide_index=True)

    else:
        st.subheader("Análises de distribuição")

        # KPI rápidos (derivados das análises, quando disponíveis)
        if "Resumo_Frota" in analyses and isinstance(analyses["Resumo_Frota"], pd.DataFrame) and not analyses["Resumo_Frota"].empty:
            rf = analyses["Resumo_Frota"].copy()
            total_oferta = float(rf["Oferta"].sum()) if "Oferta" in rf.columns else 0.0
            total_usado = float(rf["Usado"].sum()) if "Usado" in rf.columns else 0.0
            total_saldo = float(rf["Saldo"].sum()) if "Saldo" in rf.columns else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Oferta total (plano)", f"{int(total_oferta)}")
            c2.metric("Usado total", f"{int(total_usado)}")
            c3.metric("Saldo total", f"{int(total_saldo)}")

        # Seletor de tabela
        keys = list(analyses.keys())
        if not keys:
            st.warning("Nenhuma análise foi gerada.")
        else:
            default_key = "Distribuicao_Transportadora" if "Distribuicao_Transportadora" in keys else keys[0]
            pick = st.selectbox("Escolha uma análise", options=keys, index=keys.index(default_key))

            df = analyses.get(pick)
            if isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.download_button(
                    f"⬇️ Baixar '{pick}' (CSV)",
                    data=to_csv_bytes(df),
                    file_name=f"{pick}.csv",
                    mime="text/csv",
                )
            else:
                st.write(df)

            st.divider()
            st.subheader("Download das análises (Excel)")

            sheets = {"output_consolidado": output_consolidado, "saldo_plano": saldo_plano, **analyses}
            excel_bytes = to_excel_bytes_multi(sheets)
            st.download_button(
                "⬇️ Baixar Excel (Output + Saldo + Análises)",
                data=excel_bytes,
                file_name="analises_distribuicao_frota.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

