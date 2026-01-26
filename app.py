import pandas as pd
import streamlit as st

from logic import (
    invoice_allowed_band,
    target_band_for_new_invoice_from_gr,
    run_analysis,
)

st.set_page_config(page_title="Weight Discrepancy Checker", layout="wide")

# =========================
# Header
# =========================
st.title("📦 Weight Discrepancy Checker")

# =========================
# Pre-check Calculator
# =========================
st.subheader("🧮 Pre-check Calculator")
st.caption(
    "Pre-check calculator used to determine whether a weight discrepancy exists based on the ±10% tolerance"
    " rule, before uploading any documents"
)

col1, col2 = st.columns(2)
with col1:
    gr_val = st.number_input("GR (kg)", min_value=0.0, value=0.0, step=0.1)
with col2:
    inv_val = st.number_input("Invoice (kg)", min_value=0.0, value=0.0, step=0.1)

calc_btn = st.button("Calculate")

if calc_btn:
    if gr_val <= 0 or inv_val <= 0:
        st.error("⚠️ Please enter values greater than 0 for both GR and Invoice.")
    else:
        # Allowed band based on Invoice total (±10%)
        low_allowed, high_allowed = invoice_allowed_band(inv_val, tol=0.10)
        in_tol = (low_allowed <= gr_val <= high_allowed)

        # Target band for NEW invoice total derived from GR
        target_low, target_high = target_band_for_new_invoice_from_gr(gr_val, tol=0.10)

        st.markdown("**Allowed Band (±10% around Invoice total):**")
        df_main = pd.DataFrame([{
            "-10.00% (LOW)": round(low_allowed, 3),
            "Commercial invoice weight -->": round(inv_val, 2),
            "+10.00% (HIGH)": round(high_allowed, 3),
            "enter GXD weight here -->": round(gr_val, 2),
        }])
        st.dataframe(df_main, use_container_width=True)

        st.markdown("**Target Band for the NEW Invoice total (derived from GR):**")
        df_target = pd.DataFrame([{
            "Target NEW Invoice LOW (GR/1.10)": round(target_low, 3),
            "Target NEW Invoice HIGH (GR/0.90)": round(target_high, 3),
        }])
        st.dataframe(df_target, use_container_width=True)

        if in_tol:
            st.success("✅ No weight discrepancy detected. You do not need to upload documents.")
        else:
            st.warning(
                "⚠️ Weight discrepancy detected. "
                "Upload the PDFs below to generate an adjustment proposal."
            )

st.divider()

# =========================
# Upload + Run Analysis
# =========================
st.subheader("📤 Upload Shipment PDFs")
st.caption(
    "Upload at least 2 PDFs: 1 GR and 1 or more Invoice files. "
    "Then click **Run Analysis**."
)

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

run_btn = st.button("🔎 Run Analysis")

if run_btn:
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("⚠️ You must upload at least 2 PDFs: 1 GR + 1 or more Invoices.")
    else:
        uploaded = {f.name: f.read() for f in uploaded_files}

        try:
            with st.spinner("Analyzing PDFs..."):
                summary, df_full, df_adjusted, validation_df = run_analysis(uploaded, tol=0.10)

            st.success("✅ Analysis completed")

            # -------------------------
            # Shipment Summary
            # -------------------------
            st.subheader("📊 Shipment Summary")
            st.caption(
                "High-level shipment results: totals, allowed ranges, target band, "
                "and tolerance status (BEFORE/AFTER)."
            )
            st.dataframe(summary, use_container_width=True)

            # -------------------------
            # All pieces table (total validation)
            # -------------------------
            st.subheader("📦 All Pieces Weight Summary (Used for Total Validation)")
            st.caption(
                "Consolidated view of all shipment pieces, including adjusted and non-adjusted cases. "
                "This table is used to verify the total weight and confirm that the shipment no longer "
                "has a weight discrepancy."
            )
            st.dataframe(df_full, use_container_width=True)

            if "NEW WEIGHT lbs" in df_full.columns:
                st.write(f"🔹 Total NEW WEIGHT (lbs): {round(df_full['NEW WEIGHT lbs'].sum(), 2)}")
            if "NEW WEIGHT kgs" in df_full.columns:
                st.write(f"🔹 Total NEW WEIGHT (kgs): {round(df_full['NEW WEIGHT kgs'].sum(), 2)}")

            # -------------------------
            # Adjusted pieces only
            # -------------------------
            st.subheader("📦 Adjusted Pieces Only (CAT)")
            st.caption(
                "Only the cases that were modified to bring the shipment back within tolerance. "
                "Use this table to prepare CAT tickets."
            )
            st.dataframe(df_adjusted, use_container_width=True)

            # -------------------------
            # Validation table
            # -------------------------
            if validation_df is not None:
                st.subheader("📊 Validation – Invoice vs GR vs New Weight")
                st.caption(
                    "Case-level traceability: original invoice weight vs matched GR weight vs proposed new weight."
                )
                st.dataframe(validation_df, use_container_width=True)

        except Exception as e:
            st.error(
                "❌ The app encountered an error while processing the PDFs. "
                "Please verify that you uploaded 1 GR and at least 1 Invoice with a PACKING LIST."
            )
            st.exception(e)





