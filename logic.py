import re, io
import pandas as pd
import pdfplumber

# =========================
# CONFIG
# =========================
TOL_DEFAULT = 0.10
DEDUP_CASES_ACROSS_INVOICES = True

BIG_MAX_RATIO = 1.25
BIG_MIN_RATIO = 0.80
TINY_GUARD_KG = 1.0
TINY_MAX_RATIO = 2.0
MAX_ABS_CHANGE_KG = 80.0
CUSHION_KG = 0.5
MIN_CHANGE_KG = 0.30

LBS_TO_KG = 0.45359237
KG_TO_LBS = 1.0 / LBS_TO_KG

PACKING_LIST_RE = re.compile(
    r"(P\s*A\s*C\s*K\s*I\s*N\s*G\s+L\s*I\s*S\s*T)|"
    r"(PACKING\s*LIST)|"
    r"(LISTA\s+DE\s+EMPAQUE)|"
    r"(LISTA\s+DE\s+EMBALAGEM)",
    re.IGNORECASE
)

# =========================
# Bands
# =========================
def invoice_allowed_band(inv_total, tol=0.10):
    return inv_total * (1 - tol), inv_total * (1 + tol)

def target_band_for_new_invoice_from_gr(gr_total, tol=0.10):
    return gr_total / (1 + tol), gr_total / (1 - tol)

# =========================
# PDF helpers
# =========================
def pdf_bytes_to_lines(pdf_bytes):
    lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for ln in txt.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines

def is_gr(lines):
    u = " ".join(lines).upper()
    return ("WAREHOUSE RECEIPT" in u) or ("ORDGR" in u)

# =========================
# Invoice number
# =========================
def normalize_invoice_no(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def extract_invoice_number(lines):
    text = "\n".join(lines).upper()
    m = re.search(r"INVOICE\s+NUMBER\s*[:\-]?\s*([A-Z0-9\s\-]{6,})", text)
    if m:
        return normalize_invoice_no(m.group(1))
    return None

# =========================
# CONTAINER NUMBER (robusto)
# =========================
def normalize_container_no(s: str) -> str:
    s = (s or "").upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def extract_container_number(lines):
    for i, ln in enumerate(lines):
        u = ln.upper()
        if "CONTAINER NO" in u or "CNTR NO" in u:
            # mismo renglón
            m = re.search(r"(AF\d{8,}|[A-Z]{2}\d{6,})", u)
            if m:
                return normalize_container_no(m.group(1))
            # siguiente renglón
            if i + 1 < len(lines):
                nxt = lines[i + 1].upper()
                cands = re.findall(r"\b[A-Z]{2}\d{6,}\b", nxt)
                if cands:
                    return normalize_container_no(cands[-1])
    return None

# =========================
# PACKING LIST
# =========================
def extract_packing_list_block(lines, invoice_name):
    start = None
    for i, l in enumerate(lines):
        if PACKING_LIST_RE.search(l):
            start = i
            break
    if start is None:
        raise ValueError(f"{invoice_name}: PACKING LIST no encontrado")

    block = []
    for l in lines[start:]:
        u = l.upper()
        if "TOTALS" in u:
            break
        block.append(l)
    return block

def parse_invoice_packing_list(lines, invoice_name, invoice_no):
    block = extract_packing_list_block(lines, invoice_name)

    rows = []
    for i in range(len(block)):
        nums = re.findall(r"\d+(?:\.\d+)?", block[i])
        if len(nums) >= 3:
            case_no = nums[0]
            gross_lbs = float(nums[1])
            gross_kg = float(nums[2]) if float(nums[2]) < 5000 else gross_lbs * LBS_TO_KG
            rows.append({
                "INVOICE_FILE": invoice_name,
                "INVOICE_NO": invoice_no or invoice_name,
                "CASE_NO": case_no,
                "CAT_WEIGHT_LBS": gross_lbs,
                "CAT_WEIGHT_KG": gross_kg
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"{invoice_name}: no se extrajeron piezas")

    df["CASE_OCC"] = df.groupby(["INVOICE_FILE", "CASE_NO"]).cumcount() + 1
    df["PIECE_ID"] = df["INVOICE_FILE"] + "|" + df["CASE_NO"] + "|" + df["CASE_OCC"].astype(str)
    return df.reset_index(drop=True)

# =========================
# DEDUPE
# =========================
def collapse_duplicates_across_invoices(inv_df):
    idx = inv_df.groupby("CASE_NO")["CAT_WEIGHT_KG"].idxmax()
    base = inv_df.loc[idx].copy()
    base["PIECE_ID"] = base["CASE_NO"].astype(str)
    base["CASE_OCC"] = 1
    return base.reset_index(drop=True)

# =========================
# GR
# =========================
def parse_gr(lines):
    weights = []
    for ln in lines:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*KGM", ln.upper()):
            weights.append(float(m.group(1)))
    if not weights:
        raise ValueError("GR: no se detectaron pesos")
    return sum(weights), weights

# =========================
# MATCHING (NUEVO – 2 FASES)
# =========================
def match_gr_to_invoice_by_similarity(inv_df, gr_pieces,
                                      abs_tol_kg=0.30,
                                      rel_tol=0.010):

    inv_sorted = inv_df.sort_values("CAT_WEIGHT_KG").reset_index(drop=True)
    inv_list = [
        (inv_sorted.loc[i, "PIECE_ID"], float(inv_sorted.loc[i, "CAT_WEIGHT_KG"]))
        for i in range(len(inv_sorted))
    ]
    gr_sorted = sorted([float(x) for x in gr_pieces])

    used_inv, used_gr = set(), set()
    mapping = {}

    # FASE 1 – match fuerte
    candidates = []
    for i, (pid, iw) in enumerate(inv_list):
        for j, gw in enumerate(gr_sorted):
            abs_d = abs(iw - gw)
            rel_d = abs_d / max(1e-9, (iw + gw) / 2)
            if abs_d <= abs_tol_kg or rel_d <= rel_tol:
                score = abs_d + rel_d
                candidates.append((score, i, j, pid, gw))

    candidates.sort(key=lambda x: x[0])
    for _, i, j, pid, gw in candidates:
        if i in used_inv or j in used_gr:
            continue
        mapping[pid] = gw
        used_inv.add(i)
        used_gr.add(j)

    # FASE 2 – resto por orden
    rem_inv = [(i, pid) for i, (pid, _) in enumerate(inv_list) if i not in used_inv]
    rem_gr = [gr_sorted[j] for j in range(len(gr_sorted)) if j not in used_gr]

    for k in range(min(len(rem_inv), len(rem_gr))):
        _, pid = rem_inv[k]
        mapping[pid] = rem_gr[k]

    return mapping, len(inv_sorted), len(gr_sorted)

# =========================
# AJUSTE
# =========================
def gr_guided_adjust_auto(inv_df, gr_map, target_low_total, target_high_total):
    cur_total = inv_df["CAT_WEIGHT_KG"].sum()
    if target_low_total <= cur_total <= target_high_total:
        return {}

    need_up = cur_total < target_low_total
    target = target_low_total if need_up else target_high_total
    gap = abs(target - cur_total)

    updates = {}
    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        cur = r["CAT_WEIGHT_KG"]
        gr = gr_map.get(pid)
        if gr is None:
            continue
        delta = min(gap, abs(gr - cur))
        if delta >= MIN_CHANGE_KG:
            new_kg = cur + delta if need_up else cur - delta
            updates[pid] = round(min(new_kg, gr), 2)
            gap -= delta
        if gap <= 0:
            break
    return updates

# =========================
# OUTPUT TABLES
# =========================
def build_cat_tables(inv_df, updates):
    rows_full, rows_adj = [], []

    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        old_kg = r["CAT_WEIGHT_KG"]
        old_lbs = r["CAT_WEIGHT_LBS"]
        new_kg = updates.get(pid, old_kg)
        new_lbs = new_kg * KG_TO_LBS

        rows_full.append({
            "CASE/BOX": r["CASE_NO"],
            "NEW WEIGHT lbs": round(new_lbs, 2),
            "NEW WEIGHT kgs": round(new_kg, 2),
        })

        if pid in updates:
            rows_adj.append({
                "CASE/BOX": r["CASE_NO"],
                "CAT WEIGHT lbs": round(old_lbs, 2),
                "NEW WEIGHT lbs": round(new_lbs, 2),
                "NEW WEIGHT kgs": round(new_kg, 2),
                "INVOICE #": r["INVOICE_NO"],
            })

    return pd.DataFrame(rows_full), pd.DataFrame(rows_adj)

def build_validation(inv_df, gr_map, updates):
    rows = []
    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        rows.append({
            "CASE/BOX": r["CASE_NO"],
            "INVOICE ORIGINAL KG": r["CAT_WEIGHT_KG"],
            "GR KG (matched)": gr_map.get(pid),
            "NEW KG": updates.get(pid, r["CAT_WEIGHT_KG"]),
        })
    return pd.DataFrame(rows)

# =========================
# RUNNER (app.py compatible)
# =========================
def run_analysis(uploaded, tol=TOL_DEFAULT):

    invoices = []
    gr_lines = None
    gr_file = None

    for fname, fbytes in uploaded.items():
        lines = pdf_bytes_to_lines(fbytes)
        if is_gr(lines) and gr_lines is None:
            gr_lines = lines
            gr_file = fname
        else:
            inv_no = extract_invoice_number(lines)
            cntr = extract_container_number(lines)
            invoices.append((fname, lines, inv_no, cntr))

    gr_total, gr_pieces = parse_gr(gr_lines)

    inv_dfs = [
        parse_invoice_packing_list(lines, name, inv_no)
        for name, lines, inv_no, _ in invoices
    ]
    inv_df = pd.concat(inv_dfs, ignore_index=True)

    if DEDUP_CASES_ACROSS_INVOICES:
        inv_df = collapse_duplicates_across_invoices(inv_df)

    inv_total = inv_df["CAT_WEIGHT_KG"].sum()
    allowed_low, allowed_high = invoice_allowed_band(inv_total, tol)
    in_before = allowed_low <= gr_total <= allowed_high

    target_low, target_high = target_band_for_new_invoice_from_gr(gr_total, tol)
    gr_map, inv_n, gr_n = match_gr_to_invoice_by_similarity(inv_df, gr_pieces)

    updates = {}
    if not in_before:
        updates = gr_guided_adjust_auto(inv_df, gr_map, target_low, target_high)

    df_full, df_adj = build_cat_tables(inv_df, updates)
    new_total = df_full["NEW WEIGHT kgs"].sum()

    allowed_low_after, allowed_high_after = invoice_allowed_band(new_total, tol)
    in_after = allowed_low_after <= gr_total <= allowed_high_after

    # Containers por invoice
    inv_to_cntr = {}
    for _, _, inv_no, cntr in invoices:
        if cntr:
            inv_to_cntr.setdefault(inv_no or "UNKNOWN_INVOICE", set()).add(cntr)

    containers_detected = (
        " | ".join(f"{k}: {', '.join(v)}" for k, v in inv_to_cntr.items())
        if inv_to_cntr else "N/A"
    )

    summary = pd.DataFrame([{
        "GR file": gr_file,
        "Invoices files": ", ".join([x[0] for x in invoices]),
        "Invoice numbers detected": ", ".join([x[2] for x in invoices if x[2]]),
        "Container numbers detected": containers_detected,
        "Invoice total (kg)": round(inv_total, 2),
        "GR total (kg)": round(gr_total, 2),
        "Allowed low (kg)": round(allowed_low, 2),
        "Allowed high (kg)": round(allowed_high, 2),
        "Pieces detected": len(inv_df),
        "GR pieces extracted": gr_n,
        "Pieces changed": len(df_adj),
        "New total (kg)": round(new_total, 2),
        "In tolerance BEFORE": in_before,
        "In tolerance AFTER": in_after,
    }])

    validation_df = build_validation(inv_df, gr_map, updates)

    return summary, df_full, df_adj, validation_df
