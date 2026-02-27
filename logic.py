# logic.py
import re
import io
import pandas as pd
import pdfplumber

# =========================
# CONFIG
# =========================
TOL_DEFAULT = 0.10

# If a CASE/PACKAGE appears in 2 invoices, treat it as 1 physical piece
DEDUP_CASES_ACROSS_INVOICES = True

# Anti-crazy guards (GR still rules when a match exists)
BIG_MAX_RATIO = 1.25
BIG_MIN_RATIO = 0.80
TINY_GUARD_KG = 1.0
TINY_MAX_RATIO = 2.0
MAX_ABS_CHANGE_KG = 80.0
CUSHION_KG = 0.5  # kept for compatibility; the new adjustment targets the boundary directly
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
    # Allowed band based on the INVOICE TOTAL
    return inv_total * (1 - tol), inv_total * (1 + tol)

def target_band_for_new_invoice_from_gr(gr_total, tol=0.10):
    # Target band for the NEW invoice total derived from GR
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

def is_invoice(lines):
    return any(PACKING_LIST_RE.search(l) for l in lines)

def is_gr(lines):
    u = " ".join(lines).upper()
    return ("WAREHOUSE RECEIPT" in u) or ("ORDGR" in u) or re.search(r"\b[A-Z]{3}GR\d{6,}\b", u) is not None

# =========================
# Invoice number extraction (robust + normalizes spaces)
# Supports: "M 8N 000588" -> "M8N000588"
# =========================
def normalize_invoice_no(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "").replace("-", "")
    return s

def extract_invoice_number(lines):
    text = "\n".join(lines)
    u = text.upper()

    # Strong patterns
    strong_patterns = [
        # INVOICE NUMBER: M 8N 000588  (3 tokens)
        r"\bINVOICE\s+NUMBER\s*[:\-]?\s*([A-Z0-9]{1,5}\s+[A-Z0-9]{1,5}\s+\d{4,10})\b",
        # INVOICE NUMBER: M8N 000588  (2 tokens)
        r"\bINVOICE\s+NUMBER\s*[:\-]?\s*([A-Z0-9]{2,12}\s+\d{4,10})\b",
        # INVOICE NUMBER: ZDE 044871 / M 85 028110
        r"\bINVOICE\s+NUMBER\s*[:\-]?\s*([A-Z]{1,5}\s*\d{1,3}\s*\d{4,10})\b",
        r"\bINVOICE\s+NUMBER\s*[:\-]?\s*([A-Z]{1,5}\s*\d{4,12})\b",
        # Variants
        r"\bINVOICE\s+NO\.?\s*[:\-]?\s*([A-Z0-9]{1,5}\s+[A-Z0-9]{1,5}\s+\d{4,10})\b",
        r"\bINVOICE\s+NO\.?\s*[:\-]?\s*([A-Z0-9]{2,12}\s+\d{4,10})\b",
        r"\bCOMMERCIAL\s+INVOICE\s+NUMBER\s*[:\-]?\s*([A-Z0-9]{1,5}\s+[A-Z0-9]{1,5}\s+\d{4,10})\b",
        r"\bFACTURA\s*(?:NO\.|NRO|NUMERO|#)?\s*[:\-]?\s*([A-Z0-9]{1,5}\s+[A-Z0-9]{1,5}\s+\d{4,10})\b",
        r"\bN[ÚU]MERO\s+DE\s+FACTURA\s*[:\-]?\s*([A-Z0-9]{1,5}\s+[A-Z0-9]{1,5}\s+\d{4,10})\b",
    ]

    for pat in strong_patterns:
        m = re.search(pat, u, flags=re.IGNORECASE)
        if m:
            return normalize_invoice_no(m.group(1))

    # Asterisk patterns
    m2 = re.search(r"\*{3,}\s*([A-Z0-9]{1,5}\s+[A-Z0-9]{1,5}\s+\d{4,10})\s*\*{3,}", u)
    if m2:
        return normalize_invoice_no(m2.group(1))
    m3 = re.search(r"\*{3,}\s*([A-Z0-9]{2,12}\s+\d{4,10})\s*\*{3,}", u)
    if m3:
        return normalize_invoice_no(m3.group(1))

    # Fallback (avoid CAPL)
    if "INVOICE NUMBER" not in u and "INVOICE NO" not in u:
        candidates = re.findall(r"\b(?!CAPL\b)[A-Z]{1,5}\s*\d{5,12}\b", u)
        if candidates:
            return normalize_invoice_no(candidates[0])

    return None

# =========================
# CONTAINER NUM extraction (summary-only)
# - Prefer searching inside PACKING LIST block (where "CONTAINER NUM:" often appears)
# - Must NOT swallow "CNTR" at the end (e.g., "...010526CNTR")
# - Supports: "CONTAINER NUM: 2309 2/13/26 OB"
# =========================
def normalize_container_no(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9_\-\/\s]", " ", s)   # allow _ - /
    s = re.sub(r"\s+", " ", s).strip()

    # Remove trailing junk tokens if they got captured
    s = re.sub(r"(?:\s|^)(CNTR|CNTRNO|CNTR\.?|CONTAINER|NUM|NO)\s*$", "", s).strip()
    s = re.sub(r"(CNTR|CNTRNO)$", "", s).strip()

    s = s.replace(" ", "")
    return s

def extract_container_number(lines):
    # 1) Prefer searching inside PACKING LIST block
    block = None
    try:
        start = None
        for i, l in enumerate(lines):
            if PACKING_LIST_RE.search(l):
                start = i
                break
        if start is not None:
            tmp = []
            for l in lines[start:]:
                u = l.upper().strip()
                if "INVOICE TOTALS" in u or u.startswith("TOTALS:"):
                    break
                tmp.append(l)
            if tmp:
                block = tmp
    except Exception:
        block = None

    search_lines = block if block else lines

    patterns = [
        r"\bCONTAINER\s*NUM\b\s*[:\-]?\s*(?P<val>[A-Z0-9_\-\/ ]{3,50})",
        r"\bCNTR\s*NO\b\s*[:\-]?\s*(?P<val>[A-Z0-9_\-\/ ]{3,50})",
    ]

    text = "\n".join(search_lines).upper()

    # Try CONTAINER NUM first
    m = re.search(patterns[0], text, flags=re.IGNORECASE)
    if m:
        val = m.group("val")
        val = re.split(
            r"\bCNTR\s*SEAL\b|\bSEAL\b|\bINVOICE\b|\bPACKING\b|\bLIST\b",
            val,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0].strip()
        val = re.sub(r"\bCNTR\b\s*$", "", val, flags=re.IGNORECASE).strip()
        val = re.sub(r"CNTR$", "", val, flags=re.IGNORECASE).strip()
        return normalize_container_no(val) or None

    # Else try CNTR NO
    m2 = re.search(patterns[1], text, flags=re.IGNORECASE)
    if m2:
        val = m2.group("val")
        val = re.split(
            r"\bCNTR\s*SEAL\b|\bSEAL\b|\bINVOICE\b|\bPACKING\b|\bLIST\b",
            val,
            maxsplit=1,
            flags=re.IGNORECASE
        )[0].strip()
        val = re.sub(r"\bCNTR\b\s*$", "", val, flags=re.IGNORECASE).strip()
        val = re.sub(r"CNTR$", "", val, flags=re.IGNORECASE).strip()
        return normalize_container_no(val) or None

    return None

# =========================
# Extract PACKING LIST block
# =========================
def extract_packing_list_block(lines, invoice_name):
    start = None
    for i, l in enumerate(lines):
        if PACKING_LIST_RE.search(l):
            start = i
            break
    if start is None:
        raise ValueError(f"Invoice '{invoice_name}': PACKING LIST not found.")

    block = []
    for l in lines[start:]:
        u = l.upper().strip()
        if "INVOICE TOTALS" in u or u.startswith("TOTALS:"):
            break
        block.append(l)

    if not block:
        raise ValueError(f"Invoice '{invoice_name}': PACKING LIST block is empty or malformed.")
    return block

# =========================
# Invoice parser (CASE / PACKAGE) + stores the extracted INVOICE_NO
# =========================
def parse_invoice_packing_list(lines, invoice_name, invoice_no):
    block = extract_packing_list_block(lines, invoice_name)

    # Stitch lines when the kg line is separated
    stitched = []
    i = 0
    while i < len(block):
        cur = block[i]
        if i + 1 < len(block):
            nxt = block[i + 1]
            if (re.search(r"\b\d{4,}\b", cur) or "PACKAGE" in cur.upper()) and len(re.findall(r"\d+\.\d+", cur)) < 2:
                stitched.append(cur + " " + nxt)
                i += 2
                continue
        stitched.append(cur)
        i += 1
    block = stitched

    piece_re_src = re.compile(
        r"^(?P<src>[A-Z]{3})\s+(?P<id>\d{6,})\s+.+?\s+(?P<grosslbs>\d+\.\d+)\s+(?P<netlbs>\d+\.\d+)\b",
        re.IGNORECASE
    )
    piece_re_case = re.compile(
        r"^(?P<id>\d{6,})\s+.+?\s+(?P<grosslbs>\d+\.\d+)\s+(?P<netlbs>\d+\.\d+)\b",
        re.IGNORECASE
    )
    piece_re_pkg = re.compile(
        r"^(?P<id>\d{4,})\s+.+?\s+(?P<grosslbs>\d+(?:\.\d+)?)\s+(?P<netlbs>\d+(?:\.\d+)?)\b",
        re.IGNORECASE
    )

    def find_kg_line(idx, lookahead=4):
        for j in range(1, lookahead + 1):
            if idx + j >= len(block):
                return None
            s = block[idx + j].strip()
            if re.match(r"^\d+(\.\d+)?\s", s):
                return s
        return None

    rows = []
    i = 0
    while i < len(block):
        line = block[i].strip()
        u = line.upper()

        if u.startswith("SRC ") or u.startswith("--------") or ("KILOS" in u and ("CASE" in u or "PACKAGE" in u)):
            i += 1
            continue

        m = piece_re_src.match(line) or piece_re_case.match(line) or piece_re_pkg.match(line)
        if not m:
            i += 1
            continue

        piece_id = str(m.group("id"))
        gross_lbs = float(m.group("grosslbs"))

        gross_kg = None
        kg_line = find_kg_line(i, lookahead=4)
        if kg_line:
            nums = re.findall(r"\d+(?:\.\d+)?", kg_line)
            if nums:
                gross_kg = float(nums[0])
        if gross_kg is None:
            gross_kg = gross_lbs * LBS_TO_KG

        rows.append({
            "INVOICE_FILE": invoice_name,
            "INVOICE_NO": (invoice_no or invoice_name),
            "CASE_NO": piece_id,
            "CAT_WEIGHT_LBS": gross_lbs,
            "CAT_WEIGHT_KG": gross_kg
        })

        if kg_line and (i + 1 < len(block)) and (kg_line.strip() == block[i + 1].strip()):
            i += 2
        else:
            i += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Invoice '{invoice_name}': No pieces could be extracted from PACKING LIST.")

    df["CASE_OCC"] = df.groupby(["INVOICE_FILE", "CASE_NO"]).cumcount() + 1
    df["PIECE_ID"] = df["INVOICE_FILE"].astype(str) + "|" + df["CASE_NO"].astype(str) + "|" + df["CASE_OCC"].astype(str)
    return df.reset_index(drop=True)

# =========================
# Dedupe across invoices (same physical piece by CASE_NO)
# =========================
def collapse_duplicates_across_invoices(inv_df):
    inv_df = inv_df.copy().reset_index(drop=True)

    idx = inv_df.groupby("CASE_NO")["CAT_WEIGHT_KG"].idxmax()
    base = inv_df.loc[idx].copy()

    files = inv_df.groupby("CASE_NO")["INVOICE_FILE"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()
    invnos = inv_df.groupby("CASE_NO")["INVOICE_NO"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()

    base = base.merge(files, on="CASE_NO", how="left")
    base = base.merge(invnos, on="CASE_NO", how="left", suffixes=("", "_ALL"))

    base["CASE_OCC"] = 1
    base["PIECE_ID"] = base["CASE_NO"].astype(str)
    return base.sort_values("CASE_NO").reset_index(drop=True)

# =========================
# GR parser (KGM)
# =========================
def parse_gr(lines):
    start = None
    for i, l in enumerate(lines):
        u = l.upper()
        if u.startswith("PACKAGE ID") or ("PACKAGE ID" in u and "WGT" in u):
            start = i
            break

    gr_pieces = []

    if start is not None:
        window = lines[start:start + 800]
        for l in window:
            u = l.upper()
            if "HANDLING" in u or "CHARGES" in u or "SERVICE" in u:
                break
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*KGM\b", u):
                gr_pieces.append(float(m.group(1)))
            for m in re.finditer(r"(\d+(?:\.\d+)?)KGM\b", u.replace(" ", "")):
                gr_pieces.append(float(m.group(1)))
        if gr_pieces:
            return float(sum(gr_pieces)), gr_pieces

    for l in lines:
        u = l.upper()
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*KGM\b", u):
            gr_pieces.append(float(m.group(1)))
        for m in re.finditer(r"(\d+(?:\.\d+)?)KGM\b", u.replace(" ", "")):
            gr_pieces.append(float(m.group(1)))

    if gr_pieces:
        return float(sum(gr_pieces)), gr_pieces

    raise ValueError("GR: Could not extract KGM weights from GR.")

# =========================
# Similarity matching (sorted-to-sorted)
# =========================
def match_gr_to_invoice_by_similarity(inv_df, gr_pieces):
    inv_sorted = inv_df.sort_values("CAT_WEIGHT_KG").reset_index(drop=True)
    gr_sorted = sorted(gr_pieces)
    n = min(len(inv_sorted), len(gr_sorted))
    mapping = {}
    for i in range(n):
        mapping[inv_sorted.loc[i, "PIECE_ID"]] = float(gr_sorted[i])
    return mapping, len(inv_sorted), len(gr_sorted)

# =========================
# GR-guided adjustment (target band derived from GR)
# NEW behavior: aim for minimal margin (band boundary), expand to more pieces if needed.
# =========================
def gr_guided_adjust_auto(inv_df, gr_map, target_low_total, target_high_total):
    """
    Adjusts per-piece invoice weights so the *invoice total* lands inside the target band derived from GR:
        target_low_total <= NEW_INVOICE_TOTAL <= target_high_total

    Key principles:
    1) Minimal-margin targeting:
       - If invoice total is too LOW, we target as close as possible to target_low_total (not the middle of the band).
       - If invoice total is too HIGH, we target as close as possible to target_high_total.
       This reduces the risk of landing outside tolerance due to rounding/caps.

    2) Safety guards:
       - Never increase a piece above its matched GR weight (when available).
       - Caps are enforced to avoid extreme changes, but we will expand to more pieces if needed
         to actually reach the band.

    Returns:
        updates: dict {PIECE_ID: new_kg_rounded_to_2dp}
    """
    inv = inv_df.copy().reset_index(drop=True)
    n = len(inv)
    cur_total = float(inv["CAT_WEIGHT_KG"].sum())

    # Already within target band: no changes
    if target_low_total <= cur_total <= target_high_total:
        return {}

    # Decide direction and the exact boundary target (minimal margin)
    if cur_total < target_low_total:
        need_up = True
        target = float(target_low_total)  # minimal margin: go to the lower boundary
        gap = target - cur_total
    else:
        need_up = False
        target = float(target_high_total)  # minimal margin: go to the upper boundary
        gap = cur_total - target

    # If the gap is tiny, ignore
    if gap < MIN_CHANGE_KG:
        return {}

    def allowed_delta(cur, gr):
        """
        Maximum delta allowed for this piece in the needed direction.
        If GR is present:
          - for increase: cannot exceed (gr - cur)
          - for decrease: cannot go below gr (so max decrease is (cur - gr))
        If GR is missing:
          - use ratio guards based on size.
        """
        if cur <= 0:
            return 0.0

        if gr is not None and isinstance(gr, (int, float)) and gr > 0:
            if need_up:
                return max(0.0, float(gr) - cur)
            return max(0.0, cur - float(gr))

        # No GR match: apply ratio-based guards
        if cur < TINY_GUARD_KG:
            if need_up:
                return cur * (TINY_MAX_RATIO - 1.0)
            return cur * (1.0 - (1.0 / TINY_MAX_RATIO))

        if need_up:
            return cur * (BIG_MAX_RATIO - 1.0)
        return cur * (1.0 - BIG_MIN_RATIO)

    # Build candidate list with caps
    cands = []
    for _, r in inv.iterrows():
        pid = r["PIECE_ID"]
        cur = float(r["CAT_WEIGHT_KG"])
        gr = gr_map.get(pid, None)

        cap = min(MAX_ABS_CHANGE_KG, allowed_delta(cur, gr))
        if cap < MIN_CHANGE_KG:
            continue

        # Score: prefer pieces that can cover the remaining gap and have higher cap
        score = (1000 * min(cap, gap)) + (50 * cap) + cur
        cands.append((score, pid, cur, gr, cap))

    cands.sort(key=lambda x: x[0], reverse=True)
    if not cands:
        return {}

    # Iterative expansion:
    # Start with a conservative limit; if we still can't reach the band, expand up to all candidates.
    base_limit = max(3, min(20, int(round(0.50 * n))))
    max_limit = min(len(cands), max(base_limit, n))  # allow expansion up to n (or all candidates)

    def apply_updates(limit):
        updates_local = {}
        remaining = gap

        for _, pid, cur, gr, cap in cands[:limit]:
            if remaining <= 1e-9:
                break

            delta = min(remaining, cap)
            if delta < MIN_CHANGE_KG:
                continue

            new_kg = (cur + delta) if need_up else max(0.01, cur - delta)

            # Never exceed matched GR (if exists)
            if gr is not None and isinstance(gr, (int, float)) and gr > 0:
                new_kg = min(new_kg, float(gr)) if need_up else max(new_kg, float(gr))

            # Round to 2 decimals (what CAT typically needs)
            new_kg = round(new_kg, 2)

            # If rounding killed the move, skip
            if abs(new_kg - cur) < MIN_CHANGE_KG:
                continue

            updates_local[pid] = new_kg
            remaining -= abs(new_kg - cur)

        return updates_local

    def total_after(updates_local):
        if not updates_local:
            return cur_total
        new_vals = inv["CAT_WEIGHT_KG"].copy()
        pid_to_idx = {inv.loc[i, "PIECE_ID"]: i for i in range(len(inv))}
        for pid, newkg in updates_local.items():
            idx = pid_to_idx.get(pid)
            if idx is not None:
                new_vals.iloc[idx] = float(newkg)
        return float(new_vals.sum())

    limit = base_limit
    best_updates = apply_updates(limit)
    best_total = total_after(best_updates)

    # Expand until we land inside the target band or we can't improve
    while not (target_low_total <= best_total <= target_high_total) and limit < max_limit:
        limit = min(max_limit, max(limit + 5, int(round(limit * 1.25))))
        candidate_updates = apply_updates(limit)
        candidate_total = total_after(candidate_updates)

        # Keep the candidate that moves closer to the band boundary in the right direction
        if need_up:
            better = candidate_total >= best_total
        else:
            better = candidate_total <= best_total

        if better:
            best_updates, best_total = candidate_updates, candidate_total

        # Safety: if no updates found at all, stop.
        if not best_updates:
            break

    # Final safeguard:
    # If still outside due to rounding, try micro-nudge on a changed piece (when possible).
    if best_updates and not (target_low_total <= best_total <= target_high_total):
        pid_to_row = {r["PIECE_ID"]: r for _, r in inv.iterrows()}
        for pid in list(best_updates.keys())[::-1]:
            r = pid_to_row[pid]
            cur = float(r["CAT_WEIGHT_KG"])
            gr = gr_map.get(pid, None)
            proposed = float(best_updates[pid])

            for step in (0.01, 0.02, 0.05):
                if need_up:
                    cand = round(proposed + step, 2)
                    if gr is not None and isinstance(gr, (int, float)) and gr > 0:
                        cand = min(cand, float(gr))
                    if cand - cur < MIN_CHANGE_KG:
                        continue
                else:
                    cand = round(max(0.01, proposed - step), 2)
                    if gr is not None and isinstance(gr, (int, float)) and gr > 0:
                        cand = max(cand, float(gr))
                    if cur - cand < MIN_CHANGE_KG:
                        continue

                tmp = dict(best_updates)
                tmp[pid] = cand
                tmp_total = total_after(tmp)
                if target_low_total <= tmp_total <= target_high_total:
                    best_updates = tmp
                    best_total = tmp_total
                    break

            if target_low_total <= best_total <= target_high_total:
                break

    return best_updates

# =========================
# Output tables
# =========================
def build_cat_tables(inv_df, updates):
    rows_full = []
    rows_adj = []

    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        old_lbs = float(r["CAT_WEIGHT_LBS"])
        old_kg = float(r["CAT_WEIGHT_KG"])

        if pid in updates:
            new_kg = float(updates[pid])
            new_lbs = new_kg * KG_TO_LBS
            changed = True
        else:
            new_kg = old_kg
            new_lbs = old_lbs
            changed = False

        rows_full.append({
            "CASE/BOX": r["CASE_NO"],
            "NEW WEIGHT lbs": round(new_lbs, 2),
            "NEW WEIGHT kgs": round(new_kg, 2),
        })

        if changed:
            rows_adj.append({
                "CASE/BOX": r["CASE_NO"],
                "CAT WEIGHT lbs": round(old_lbs, 2),
                "NEW WEIGHT lbs": round(new_lbs, 2),
                "NEW WEIGHT kgs": round(new_kg, 2),
                "NEW LENGTH": "N/A",
                "NEW WIDTH": "N/A",
                "NEW HEIGHT": "N/A",
                "INVOICE #": r.get("INVOICE_NO", r.get("INVOICE_FILE", "")),
            })

    df_full = pd.DataFrame(rows_full)
    df_adj = pd.DataFrame(rows_adj)
    return df_full, df_adj

def build_validation(inv_df, gr_map, updates):
    rows = []
    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        inv_kg = float(r["CAT_WEIGHT_KG"])
        gr_kg = gr_map.get(pid, None)
        new_kg = float(updates.get(pid, inv_kg))
        rows.append({
            "CASE/BOX": r["CASE_NO"],
            "INVOICE ORIGINAL KG": round(inv_kg, 2),
            "GR KG (matched)": (round(gr_kg, 2) if gr_kg is not None else None),
            "NEW KG": round(new_kg, 2),
        })
    return pd.DataFrame(rows)

# =========================
# RUNNER
# =========================
def run_analysis(uploaded, tol=TOL_DEFAULT):
    if len(uploaded) < 2:
        raise ValueError("Upload at least 2 PDFs: 1 GR + 1 or more Invoices.")

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
            cntr_no = extract_container_number(lines)
            invoices.append((fname, lines, inv_no, cntr_no))

    if gr_lines is None:
        raise ValueError("GR not found.")
    if not invoices:
        raise ValueError("No invoice with PACKING LIST was found.")

    gr_total, gr_pieces = parse_gr(gr_lines)

    inv_dfs = [parse_invoice_packing_list(lines, name, inv_no) for name, lines, inv_no, _cntr in invoices]
    inv_df = pd.concat(inv_dfs, ignore_index=True)

    if DEDUP_CASES_ACROSS_INVOICES:
        inv_df = collapse_duplicates_across_invoices(inv_df)

    inv_total = float(inv_df["CAT_WEIGHT_KG"].sum())

    allowed_low_before, allowed_high_before = invoice_allowed_band(inv_total, tol)
    in_before = (allowed_low_before <= gr_total <= allowed_high_before)

    target_low_total, target_high_total = target_band_for_new_invoice_from_gr(gr_total, tol)

    gr_map, inv_n, gr_n = match_gr_to_invoice_by_similarity(inv_df, gr_pieces)

    updates = {}
    if not in_before:
        updates = gr_guided_adjust_auto(inv_df, gr_map, target_low_total, target_high_total)

    df_full, df_adj = build_cat_tables(inv_df, updates)
    new_total = float(df_full["NEW WEIGHT kgs"].sum())

    allowed_low_after, allowed_high_after = invoice_allowed_band(new_total, tol)
    in_after = (allowed_low_after <= gr_total <= allowed_high_after)

    validation_df = build_validation(inv_df, gr_map, updates)

    containers_detected = (
        " | ".join([
            f"{(x[2] or 'UNKNOWN_INVOICE')}: {x[3]}"
            for x in invoices
            if x[3]
        ])
        if any(x[3] for x in invoices)
        else "N/A"
    )

    summary = pd.DataFrame([{
        "GR file": gr_file,
        "Invoices files": ", ".join([x[0] for x in invoices]),
        "Invoice numbers detected": ", ".join([x[2] for x in invoices if x[2]]) if any(x[2] for x in invoices) else "N/A",
        "Container numbers detected": containers_detected,

        "Invoice total (kg)": round(inv_total, 2),
        "GR total (kg)": round(gr_total, 2),

        "Allowed low (kg)": round(allowed_low_before, 2),
        "Allowed high (kg)": round(allowed_high_before, 2),

        "Target NEW Invoice low (kg) [GR/1.10]": round(target_low_total, 2),
        "Target NEW Invoice high (kg) [GR/0.90]": round(target_high_total, 2),

        "Pieces detected": int(len(inv_df)),
        "GR pieces extracted": int(gr_n),

        "Pieces changed": int(len(df_adj)),
        "New total (kg)": round(new_total, 2),

        "Allowed low after (kg)": round(allowed_low_after, 2),
        "Allowed high after (kg)": round(allowed_high_after, 2),

        "In tolerance BEFORE": bool(in_before),
        "In tolerance AFTER": bool(in_after),
    }])

    return summary, df_full, df_adj, validation_df
