import io
import os
import re
import json
import zipfile
import calendar
import warnings
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# Page Configuration
st.set_page_config(page_title="FinCrime Unit Analysis", layout="wide", page_icon="🛡️")

# ============================================================
# REPOSITORIES & CONSTANTS
# ============================================================

RULES_REPOSITORY_RAW = {
    "1000829": "ALERT ACQ - Average Fraud Score Increase",
    "1000828": "ACQ - Average Fraud Score increase",
    "1000827": "ALERT ACQ -Decline spike per rule trigger 1h",
    "1000826": "ALERT ACQ -Decline spike per merchant ID",
    "1000818": "ALERT ACQ Decline spike per rule trigger 1d",
    "1000817": "ACQ -Decline spike per merchant ID",
    "1000765": "MDM - Critical Approval Rate Drop(DRC-ACQ)",
    "1000683": "ACQ - Email domain / Card Cluster",
    "1000682": "ACQ - IP ISP / BIN Cluster",
    "1000681": "ACQ Card vs IP country mismatch",
    "1000675": "ACQ Email is disposable",
    "1000674": "ACQ Card type is CORPORATE/BUSINESS",
    "1000669": "ACQ -20+ CVV failures from the same IP/device within 30 min",
    "1000668": "ACQ - 2+ CVV failures on the same card hash within 1h",
    "1000660": "ACQ - Geo Shift",
    "1000659": "ACQ - Deposit Escalation (Online gambling / Sweepstakes)",
    "1000658": "ACQ - Deposit Declines (Online Gambling / Sweepstakes)",
    "1000650": "ACQ - Micro-Deposits",
    "1000648": "ACQ -Failed 3DS Attempt Velocity",
    "1000643": "ACQ - High risk Response Codes",
    "1000642": "ACQ - Response Code Card Block",
    "1000522": "ACQ Different cardholder names on Payer",
    "1000520": "ACQ Repeated currency switching within 1h",
    "1000519": "ACQ Suspected VoIP phone number",
    "1000518": "ACQ Shared phone number - multi-account threshold",
    "1000517": "ACQ Multiple shipping addresses per Payer",
    "1000516": "ACQ Repeated round-amounts",
    "1000515": "ACQ Multi-account dormant burst",
    "1000514": "ACQ Shipping address contains mail forwarding\\virtual storage",
    "1000513": "ACQ Unusually high-value first purchase for a payer",
    "1000507": "ACQ >= 3 diff GIFT cards used within 15 minutes",
    "1000506": "ACQ High amount vs average",
    "1000504": "ACQ Public proxy",
    "1000500": "ACQ 90 days of inactivity+ unusually high purchase amount",
    "1000499": "ACQ 90 days of inactivity+ unusually high purchase volume",
    "1000495": "ACQ Reuse of billing address accross different Payers",
    "1000494": "ACQ Reuse of payment card accross different Payers",
    "1000493": "TEST - ACQ Multiple Payers from same IP",
    "1000490": "ACQ High order frequency in 5 minutes",
    "1000489": "ACQ Billing vs shipping country mismatch",
    "1000488": "ACQ IP vs billing country\\shipping country mismatch",
    "1000487": "ACQ shipping country risk-based scoring",
    "1000486": "ACQ Card country risk-based scoring",
    "1000485": "ACQ IP Country risk scoring",
    "1000484": "ACQ Prohibited/restricted card-issuer countries",
    "1000483": "ACQ Prohibited/restricted IP country",
    "1000477": "ACQ Payer used more than 3 unique IP addresses in 1 day",
    "1000476": "ACQ Payer used more than 3 unique IP addresses in 10 min",
    "1000474": "ACQ Multiple IP countries per Payer in 1 week",
    "1000473": "ACQ Multiple IP countries per Payer in 1 day",
    "1000472": "ACQ Multiple IP countries per Payer in 10 min",
    "1000471": "ACQ Multiple cards per Payer within 1 day",
    "1000470": "ACQ Multiple cards per Payer within 1 h",
    "1000469": "ACQ Multiple card-issuing countries per Payer 1d",
    "1000468": "ACQ Multiple card-issuing countries per Payer 1h",
    "1000467": "ACQ Multiple card-issuing countries per Payer 1w",
    "1000466": "ACQ Multiple cards per Payer within 1 week",
    "1000464": "ACQ BIN IP Concentration",
    "1000463": "ACQ BIN Card testing - 20+ cards same BIN in 1 minute",
    "1000825": "ALERT ACQ - Fraud Score Threshold Exceeded",
    "1000824": "ALERT ACQ - Unusual Velocity Spike",
    "1000823": "ACQ - Decline Rate Threshold Exceeded",
    "1000819": "ACQ - High Risk Transaction Velocity",
    "1000679": "To delete/modify - Deprecated Rule",
    "1000677": "ACQ Micro-Deposits Pattern",
    "1000676": "ACQ Micro-Deposits Failure",
    "1000666": "Duplicate Card Pattern",
    "1000661": "ACQ - Card Prepaid Risk",
    "1000521": "ACQ User Full Name Invalid",
    "1000512": "ACQ Reuse of Device Fingerprint",
    "1000511": "ACQ Reuse of IP Address",
    "1000510": "ACQ 90 days of inactivity",
    "1000509": "ACQ Gift card/Prepaid Velocity",
    "1000508": "ACQ Repeated Payer Velocity",
    "1000505": "ACQ Same IP Multiple Payers",
    "1000498": "ACQ New Payer Velocity",
    "1000497": "ACQ Blacklist Check Failed",
    "1000492": "ACQ Device Risk High",
    "1000491": "ACQ Same Card Multiple Accounts",
    "1000482": "ACQ Multiple Failed Attempts",
    "1000481": "ACQ Multiple Unique BINs per IP",
    "1000465": "ACQ Temp Block Triggered",
}
RULES_REPOSITORY = {str(k).strip(): str(v).strip() for k, v in RULES_REPOSITORY_RAW.items()}

EXCLUDED_DOMAINS = ['whitetech.com', 'paydo.com', 'paymentop.com']
RESTRICTED_COUNTRIES = [
    'AF', 'BY', 'BI', 'CF', 'CU', 'CD', 'GN', 'GW', 'HT', 'IQ', 'IR', 'LB', 'LY',
    'ML', 'MM', 'NI', 'KP', 'PS', 'RU', 'SO', 'SS', 'SD', 'SY', 'TW', 'US', 'VE',
    'EH', 'YE', 'ZW'
]
CLOSED_ACCOUNTS_MANAGERS = [
    'Maryna Krasheninnikova', 'Serhii Kryhin', 'Oksana Moskalenko', 'Yevheniia Linichenko'
]
ANALYZED_ASSIGNEES = [
    "Maryna Krasheninnikova", "Oksana Moskalenko", "Yevheniia Linichenko", "Serhii Kryhin", "Anti-Fraud"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_data(uploaded_file, sep=None):
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, sep=sep or ";")
            if df.shape[1] <= 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=",")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = df.columns.astype(str).str.strip()
    return df

def fetch_eur_rates():
    try:
        url = "https://api.frankfurter.app/latest?from=EUR&to=USD,GBP,CAD"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode())
            ecb_rates = payload.get('rates', {})
            rates = {'EUR': 1.0}
            for curr in ('USD', 'GBP', 'CAD'):
                if curr in ecb_rates:
                    rates[curr] = 1.0 / float(ecb_rates[curr])
            return rates
    except Exception:
        return {'EUR': 1.0, 'USD': 0.92, 'GBP': 1.17, 'CAD': 0.68}

def clean_rule_id(rule_id):
    if pd.isna(rule_id) or rule_id is None:
        return ""
    rid_str = str(rule_id).strip().strip("\"'")
    if rid_str.endswith(".0"):
        rid_str = rid_str[:-2]
    return rid_str

def get_rule_name(rule_id):
    rid_str = clean_rule_id(rule_id)
    if not rid_str:
        return "Unknown Rule"
    if rid_str in RULES_REPOSITORY:
        return f"{rid_str} - {RULES_REPOSITORY[rid_str]}"
    return rid_str

def parse_rule_ids(value):
    if value is None or pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = list(value)
    else:
        s = str(value).strip()
        if s.lower() in {"", "nan", "none", "null"}:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                values = parsed
            else:
                values = [parsed]
        except Exception:
            s = s.strip("[](){}")
            values = s.replace(";", ",").split(",")

    result = []
    for item in values:
        rule_id = clean_rule_id(item)
        if rule_id and rule_id.lower() not in {"nan", "none", "null"}:
            if rule_id not in result:
                result.append(rule_id)
    return result

def relative_change(current, baseline):
    if baseline <= 0:
        return np.nan
    return (current / baseline - 1) * 100

def label_consecutive_declines(data, timeframe_hours=0.25):
    result = data.copy()
    result['date'] = pd.to_datetime(result['date'], utc=True, errors='coerce')
    result = result.sort_values(['email/raw_email', 'date']).copy()
    result['consecutive_decline'] = False
    for _, group in result[result['email/raw_email'].notna()].groupby('email/raw_email', sort=False):
        last_decline_time = None
        for idx in group.index:
            state = result.at[idx, 'state']
            current_time = result.at[idx, 'date']
            if pd.isna(current_time):
                continue
            if state == 'DECLINE':
                is_consecutive = (
                    last_decline_time is not None and
                    (current_time - last_decline_time).total_seconds() <= timeframe_hours * 3600
                )
                result.at[idx, 'consecutive_decline'] = is_consecutive
                last_decline_time = current_time
            elif state == 'APPROVE':
                last_decline_time = None
    return result

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ FinCrime Unit Analysis")
st.sidebar.markdown("---")

module = st.sidebar.selectbox(
    "Select Analytics Module:",
    [
        "1. Case & Alert Monitoring",
        "2. Acquiring Monitoring",
        "3. Monthly Decline Report (Waterfall)",
        "4. Top-Up Declines Monitoring"
    ]
)

st.sidebar.markdown("---")

# ============================================================
# MODULE 1: CASE & ALERT MONITORING
# ============================================================
if module == "1. Case & Alert Monitoring":
    sub_module = st.sidebar.radio("Sub-module:", ["Alert Analytics Dashboard", "Daily Case Monitoring Dashboard"])
    
    if sub_module == "Alert Analytics Dashboard":
        st.title("🚨 Alert Analytics Dashboard")
        up_file = st.file_uploader("Upload Alert File (CSV/XLSX):", type=["csv", "xlsx"])
        
        if up_file:
            df = load_data(up_file)
            required_columns = ["alert_id", "alert_trigger_name", "alert_created_at", "alert_status", "alert_assignee"]
            if not all(col in df.columns for col in required_columns):
                st.error(f"Missing required columns: {[c for c in required_columns if c not in df.columns]}")
            else:
                if ANALYZED_ASSIGNEES:
                    df = df[df["alert_assignee"].astype(str).str.strip().isin(ANALYZED_ASSIGNEES)].copy()
                
                df["alert_created_at"] = pd.to_datetime(df["alert_created_at"], errors="coerce")
                df = df.dropna(subset=["alert_created_at"]).copy()
                df["alert_date"] = df["alert_created_at"].dt.date
                
                def classify_alert_group(trigger):
                    trigger = str(trigger).strip()
                    if trigger.startswith("Policy Breach"): return "Policy Breach"
                    if trigger.startswith("IoCs"): return "IoCs"
                    if trigger.startswith("CP"): return "CP"
                    if trigger.startswith("Synthetic and Coordinated Activity"): return "Synthetic and Coordinated Activity"
                    return "Other"

                df["alert_group"] = df["alert_trigger_name"].apply(classify_alert_group)
                df = df.drop_duplicates(subset=["alert_id"]).copy()

                assignees = ["ALL"] + sorted(df["alert_assignee"].dropna().unique().tolist())
                sel_assignee = st.selectbox("Assignee Filter:", assignees)
                
                if sel_assignee != "ALL":
                    df = df[df["alert_assignee"] == sel_assignee]
                    
                daily = df.groupby("alert_date")["alert_id"].nunique().reset_index(name="Alerts")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Period Alerts", f"{df['alert_id'].nunique():,}")
                col2.metric("Active Days", f"{len(daily):,}")
                col3.metric("Unique Triggers", f"{df['alert_trigger_name'].nunique():,}")
                
                fig = px.line(daily, x="alert_date", y="Alerts", title="Daily Alert Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    top5 = df.groupby("alert_trigger_name")["alert_id"].nunique().nlargest(5).reset_index(name="Count")
                    fig_top = px.bar(top5, x="Count", y="alert_trigger_name", orientation="h", title="Top 5 Alert Triggers")
                    st.plotly_chart(fig_top, use_container_width=True)
                with col_b:
                    grp_counts = df.groupby("alert_group")["alert_id"].nunique().reset_index(name="Count")
                    fig_grp = px.pie(grp_counts, values="Count", names="alert_group", title="Alerts by Group")
                    st.plotly_chart(fig_grp, use_container_width=True)

                daily_summary = df.groupby(["alert_date", "alert_group"])["alert_id"].nunique().reset_index(name="Alerts")
                daily_summary = daily_summary.pivot(index="alert_date", columns="alert_group", values="Alerts").fillna(0).reset_index()
                
                st.subheader("Daily Summary Table")
                st.dataframe(daily_summary, use_container_width=True)

                output_alerts = io.BytesIO()
                with pd.ExcelWriter(output_alerts, engine="xlsxwriter") as writer:
                    daily_summary.to_excel(writer, sheet_name="Daily Summary", index=False)
                    top5.to_excel(writer, sheet_name="Top 5 Triggers", index=False)
                st.download_button(
                    label="📥 Download Alert Report (Excel)",
                    data=output_alerts.getvalue(),
                    file_name="Alert_Analytics_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    elif sub_module == "Daily Case Monitoring Dashboard":
        st.title("📋 Daily Case Monitoring Dashboard")
        up_file = st.file_uploader("Upload Case File (CSV/XLSX):", type=["csv", "xlsx"])
        if up_file:
            df = load_data(up_file)
            if ANALYZED_ASSIGNEES and "case_assignee" in df.columns:
                df = df[df["case_assignee"].astype(str).str.strip().isin(ANALYZED_ASSIGNEES)].copy()

            df["case_created_at"] = pd.to_datetime(df["case_created_at"], errors="coerce")
            df = df.dropna(subset=["case_created_at"]).copy()
            df["case_date"] = df["case_created_at"].dt.date
            
            def classify_case_group(title):
                title = str(title).strip()
                if title.startswith("Policy Breach"): return "Policy Breach"
                if title.startswith("IoCs"): return "IoCs"
                if title.startswith("CP"): return "CP"
                if title.startswith("Synthetic and Coordinated Activity"): return "Synthetic and Coordinated Activity"
                if title.startswith("ACQ_"): return "ACQ"
                return "Other"

            if "case_title" in df.columns:
                df["case_group"] = df["case_title"].apply(classify_case_group)
            df = df.drop_duplicates(subset=["case_id"]).copy()

            st.metric("Total Cases", f"{df['case_id'].nunique():,}")
            daily_case = df.groupby("case_date")["case_id"].nunique().reset_index(name="Cases")
            fig_c = px.line(daily_case, x="case_date", y="Cases", title="Daily Case Trend", markers=True)
            st.plotly_chart(fig_c, use_container_width=True)
            
            col_x, col_y = st.columns(2)
            with col_x:
                if "case_group" in df.columns:
                    grp_c = df.groupby("case_group")["case_id"].nunique().reset_index(name="Count")
                    fig_cg = px.bar(grp_c, x="case_group", y="Count", title="Cases by Group")
                    st.plotly_chart(fig_cg, use_container_width=True)
            with col_y:
                if "case_status" in df.columns:
                    st_c = df.groupby("case_status")["case_id"].nunique().reset_index(name="Count")
                    fig_cs = px.pie(st_c, values="Count", names="case_status", title="Cases by Status")
                    st.plotly_chart(fig_cs, use_container_width=True)

            output_cases = io.BytesIO()
            with pd.ExcelWriter(output_cases, engine="xlsxwriter") as writer:
                daily_case.to_excel(writer, sheet_name="Daily Summary", index=False)
            st.download_button(
                label="📥 Download Case Report (Excel)",
                data=output_cases.getvalue(),
                file_name="Case_Monitoring_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# ============================================================
# MODULE 2: ACQUIRING MONITORING
# ============================================================
elif module == "2. Acquiring Monitoring":
    sub_module = st.sidebar.radio(
        "Acquiring Section:", 
        [
            "2.1 Spike Monitoring (Top 10 Merchants)", 
            "2.2 Single Merchant Historical Analysis"
        ]
    )
    
    if sub_module == "2.1 Spike Monitoring (Top 10 Merchants)":
        st.title("📉 Acquiring Merchant Decline Spike Monitoring")
        st.write("Detect decline rate spikes across merchants (last 7 days vs 21+ active days baseline) and run deep analysis.")
        
        # Local Module Constants
        MONITORING_DAYS = 7
        MIN_BASELINE_ACTIVE_DAYS = 21
        DECLINE_THRESHOLD = 0.06
        TOP_MERCHANTS_LIMIT = 10
        TOP_RULES_LIMIT = 10
        TOP_EMAILS_LIMIT = 10
        MIN_SPIKE_TRANSACTIONS = 10

        DATE_COLUMN = "date"
        STATE_COLUMN = "state"
        MERCHANT_COLUMN = "merchant_id"
        MERCHANT_NAME_COLUMN = "user_fullname"
        TRANSACTION_ID_COLUMN = "transaction_id"
        RULES_COLUMN = "applied_rule_ids"
        EMAIL_COLUMN = "email/email"
        DECLINE_STATE = "DECLINE"

        GENERAL_REPORT_FILE = "SEON_Top10_Merchant_Decline_Spikes.xlsx"
        DEEP_REPORT_FILE = "SEON_Selected_Merchants_Deep_Analysis.xlsx"
        CHARTS_ZIP_FILE = "Selected_Merchant_Rule_Charts.zip"

        up_file = st.file_uploader("Upload SEON Acquiring Export (Excel/CSV):", type=["xlsx", "csv"], key="m21_upload")
        
        if up_file:
            if st.button("🚀 Run Spike Detection", type="primary", key="m21_run_screening"):
                with st.spinner("Analyzing baseline and detecting decline spikes..."):
                    df = load_data(up_file)
                    
                    req_cols = [DATE_COLUMN, STATE_COLUMN, MERCHANT_COLUMN, TRANSACTION_ID_COLUMN, RULES_COLUMN, EMAIL_COLUMN, MERCHANT_NAME_COLUMN]
                    missing = [c for c in req_cols if c not in df.columns]
                    if missing:
                        st.error(f"Missing required columns in uploaded file: {', '.join(missing)}")
                        st.stop()
                        
                    card_col = "bin_country_code"
                    if card_col not in df.columns:
                        candidates = [c for c in df.columns if c.lower() in ["bin_country_code", "bin_country", "country", "card_country"]]
                        card_col = candidates[0] if candidates else None

                    ip_col = "ip_country"
                    if ip_col not in df.columns:
                        candidates_ip = [c for c in df.columns if c.lower() in ["ip_country", "ipcountry"]]
                        ip_col = candidates_ip[0] if candidates_ip else None

                    # Clean and normalize data
                    df_clean = df.copy()
                    df_clean["_date_day"] = pd.to_datetime(df_clean[DATE_COLUMN].astype(str).str[:10], errors="coerce").dt.normalize()
                    df_clean[STATE_COLUMN] = df_clean[STATE_COLUMN].fillna("").astype(str).str.strip().str.upper()
                    df_clean[MERCHANT_COLUMN] = df_clean[MERCHANT_COLUMN].fillna("").astype(str).str.strip()
                    df_clean[MERCHANT_NAME_COLUMN] = df_clean[MERCHANT_NAME_COLUMN].fillna("").astype(str).str.strip()
                    df_clean[TRANSACTION_ID_COLUMN] = df_clean[TRANSACTION_ID_COLUMN].fillna("").astype(str).str.strip()

                    df_clean = df_clean[df_clean["_date_day"].notna() & df_clean[MERCHANT_COLUMN].ne("")].copy()
                    
                    df_clean["_transaction_key"] = df_clean[TRANSACTION_ID_COLUMN]
                    missing_tx = df_clean["_transaction_key"].eq("")
                    df_clean.loc[missing_tx, "_transaction_key"] = [f"_missing_tx_{i}" for i in df_clean.index[missing_tx]]

                    df_clean = df_clean.drop_duplicates(subset=["_transaction_key"], keep="first").copy()
                    df_clean = df_clean[df_clean[STATE_COLUMN].isin(["APPROVE", "DECLINE"])].copy()

                    latest_date = df_clean["_date_day"].max()
                    monitoring_start = latest_date - pd.Timedelta(days=MONITORING_DAYS - 1)

                    monitoring_df = df_clean[(df_clean["_date_day"] >= monitoring_start) & (df_clean["_date_day"] <= latest_date)].copy()
                    baseline_df = df_clean[df_clean["_date_day"] < monitoring_start].copy()

                    if monitoring_df.empty or baseline_df.empty:
                        st.error("Insufficient historical or monitoring data to compute baseline.")
                        st.stop()

                    # Calculate Baseline metrics per merchant
                    baseline_stats = baseline_df.groupby(MERCHANT_COLUMN).agg(
                        baseline_transactions=("_transaction_key", "nunique"),
                        baseline_declines=(STATE_COLUMN, lambda s: (s == DECLINE_STATE).sum()),
                        baseline_active_days=("_date_day", "nunique"),
                    ).reset_index()
                    baseline_stats["baseline_decline_rate"] = baseline_stats["baseline_declines"] / baseline_stats["baseline_transactions"]

                    eligible_merchants = baseline_stats[baseline_stats["baseline_active_days"] >= MIN_BASELINE_ACTIVE_DAYS].copy()
                    eligible_ids = set(eligible_merchants[MERCHANT_COLUMN])

                    monitoring_eligible = monitoring_df[monitoring_df[MERCHANT_COLUMN].isin(eligible_ids)].copy()

                    daily_stats = monitoring_eligible.groupby([MERCHANT_COLUMN, "_date_day"]).agg(
                        total_transactions=("_transaction_key", "nunique"),
                        declined_transactions=(STATE_COLUMN, lambda s: (s == DECLINE_STATE).sum()),
                    ).reset_index()
                    daily_stats["decline_rate"] = daily_stats["declined_transactions"] / daily_stats["total_transactions"]

                    daily_stats = daily_stats.merge(
                        eligible_merchants[[MERCHANT_COLUMN, "baseline_transactions", "baseline_declines", "baseline_active_days", "baseline_decline_rate"]],
                        on=MERCHANT_COLUMN,
                        how="left"
                    )

                    # Identify spike days
                    spikes = daily_stats[
                        (daily_stats["total_transactions"] >= MIN_SPIKE_TRANSACTIONS) &
                        (daily_stats["decline_rate"] > DECLINE_THRESHOLD) &
                        (daily_stats["decline_rate"] > daily_stats["baseline_decline_rate"])
                    ].copy()

                    if spikes.empty:
                        st.warning("No merchants met the spike criteria (> 6% decline rate and above baseline).")
                        st.stop()

                    spikes["increase_pp"] = (spikes["decline_rate"] - spikes["baseline_decline_rate"]) * 100
                    spikes["relative_increase"] = np.where(
                        spikes["baseline_decline_rate"] > 0,
                        (spikes["decline_rate"] - spikes["baseline_decline_rate"]) / spikes["baseline_decline_rate"],
                        np.inf
                    )
                    spikes["relative_increase_pct"] = spikes["relative_increase"] * 100
                    spikes["spike_date"] = spikes["_date_day"]

                    # Rank Top 10 Merchants by max relative increase
                    rank_rows = []
                    for merchant_id, merchant_spikes in spikes.groupby(MERCHANT_COLUMN):
                        merchant_spikes = merchant_spikes.sort_values(["relative_increase", "increase_pp", "decline_rate"], ascending=False)
                        max_spike = merchant_spikes.iloc[0]

                        name_vals = monitoring_eligible.loc[monitoring_eligible[MERCHANT_COLUMN] == merchant_id, MERCHANT_NAME_COLUMN].dropna().astype(str)
                        name_vals = name_vals[name_vals.str.strip().ne("")]
                        merchant_name = name_vals.mode().iloc[0] if not name_vals.empty else ""

                        rank_rows.append({
                            MERCHANT_COLUMN: merchant_id,
                            MERCHANT_NAME_COLUMN: merchant_name,
                            "baseline_decline_rate": max_spike["baseline_decline_rate"],
                            "baseline_active_days": int(max_spike["baseline_active_days"]),
                            "max_spike_date": max_spike["spike_date"],
                            "max_spike_decline_rate": max_spike["decline_rate"],
                            "max_spike_increase_pp": max_spike["increase_pp"],
                            "max_relative_increase": max_spike["relative_increase"],
                            "max_relative_increase_pct": max_spike["relative_increase_pct"],
                            "spike_days_count": int(len(merchant_spikes)),
                            "total_spike_declines": int(merchant_spikes["declined_transactions"].sum()),
                        })

                    top_merchants = pd.DataFrame(rank_rows).sort_values(
                        ["max_relative_increase", "max_spike_increase_pp", "max_spike_decline_rate"],
                        ascending=False
                    ).head(TOP_MERCHANTS_LIMIT).reset_index(drop=True)

                    top_merchants.insert(0, "rank", range(1, len(top_merchants) + 1))
                    top_merchant_ids = set(top_merchants[MERCHANT_COLUMN].astype(str))
                    top_spikes = spikes[spikes[MERCHANT_COLUMN].astype(str).isin(top_merchant_ids)].copy()

                    # Merge rank & user_fullname into top_spikes to prevent KeyError
                    top_spikes = top_spikes.merge(
                        top_merchants[["rank", MERCHANT_COLUMN, MERCHANT_NAME_COLUMN]],
                        on=MERCHANT_COLUMN,
                        how="left"
                    )

                    # Store state for Streamlit UI continuity
                    st.session_state["m21_processed"] = True
                    st.session_state["m21_df_clean"] = df_clean
                    st.session_state["m21_top_merchants"] = top_merchants
                    st.session_state["m21_top_spikes"] = top_spikes
                    st.session_state["m21_card_col"] = card_col
                    st.session_state["m21_ip_col"] = ip_col
                    st.session_state["m21_eligible_ids"] = eligible_ids
                    st.session_state["m21_latest_date"] = latest_date
                    st.session_state["m21_monitoring_start"] = monitoring_start

        if st.session_state.get("m21_processed", False):
            top_merchants = st.session_state["m21_top_merchants"]
            top_spikes = st.session_state["m21_top_spikes"]
            df_clean = st.session_state["m21_df_clean"]
            card_col = st.session_state["m21_card_col"]
            ip_col = st.session_state["m21_ip_col"]
            eligible_ids = st.session_state["m21_eligible_ids"]
            latest_date = st.session_state["m21_latest_date"]
            monitoring_start = st.session_state["m21_monitoring_start"]

            st.subheader("🏆 Top 10 Spiking Merchants Summary")
            st.dataframe(top_merchants, use_container_width=True)

            # Generate and Render Merchant Timeline Charts
            st.subheader("📈 Merchant Decline Rate Charts (Historical & Monitoring Window)")
            all_daily_stats = (
                df_clean[df_clean[MERCHANT_COLUMN].isin(eligible_ids)]
                .groupby([MERCHANT_COLUMN, "_date_day"])
                .agg(
                    total_transactions=("_transaction_key", "nunique"),
                    declined_transactions=(STATE_COLUMN, lambda s: (s == DECLINE_STATE).sum()),
                )
                .reset_index()
            )
            all_daily_stats["decline_rate"] = all_daily_stats["declined_transactions"] / all_daily_stats["total_transactions"]
            min_date = df_clean["_date_day"].min()
            full_dates = pd.date_range(start=min_date, end=latest_date, freq="D")

            for _, merchant_row in top_merchants.iterrows():
                merchant_id = str(merchant_row[MERCHANT_COLUMN])
                merchant_name = str(merchant_row[MERCHANT_NAME_COLUMN])
                merchant_daily = all_daily_stats[all_daily_stats[MERCHANT_COLUMN].astype(str) == merchant_id].copy()
                merchant_spikes = top_spikes[top_spikes[MERCHANT_COLUMN].astype(str) == merchant_id].copy()
                baseline_rate = float(merchant_row["baseline_decline_rate"])

                merchant_daily = merchant_daily.set_index("_date_day")
                chart_rates = merchant_daily["decline_rate"].reindex(full_dates)

                fig, ax = plt.subplots(figsize=(12, 4.5))
                ax.axvspan(
                    monitoring_start - pd.Timedelta(hours=12),
                    latest_date + pd.Timedelta(hours=12),
                    color="#ffcccc",
                    alpha=0.35,
                    label=f"Monitoring Window (Last {MONITORING_DAYS} Days)",
                )
                ax.plot(full_dates, chart_rates * 100, marker="o", markersize=4, linewidth=1.8, color="#1f77b4", label="Daily decline rate")
                ax.axhline(baseline_rate * 100, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_rate * 100:.2f}%)")
                ax.axhline(DECLINE_THRESHOLD * 100, color="#d62728", linestyle=":", linewidth=1.5, label=f"Threshold ({DECLINE_THRESHOLD * 100:.0f}%)")

                if not merchant_spikes.empty:
                    ax.scatter(merchant_spikes["spike_date"], merchant_spikes["decline_rate"] * 100, color="#d62728", s=90, zorder=5, label="Detected spike")
                    for _, spike_row in merchant_spikes.iterrows():
                        ax.annotate(
                            f'{spike_row["decline_rate"] * 100:.1f}%',
                            (spike_row["spike_date"], spike_row["decline_rate"] * 100),
                            textcoords="offset points",
                            xytext=(0, 8),
                            ha="center",
                            fontsize=8,
                            weight="bold",
                            color="#a70000",
                        )

                ax.set_title(f"Historical Decline Rate & 7-Day Spike Monitoring — Merchant: {merchant_name} | ID: {merchant_id}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Decline rate (%)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper left")
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            top_merchants_export = top_merchants.copy()
            top_merchants_export["baseline_decline_rate"] *= 100
            top_merchants_export["max_spike_decline_rate"] *= 100

            spike_dates_export = top_spikes[[
                "rank", MERCHANT_COLUMN, MERCHANT_NAME_COLUMN, "spike_date",
                "total_transactions", "declined_transactions", "decline_rate",
                "baseline_decline_rate", "increase_pp", "relative_increase_pct"
            ]].copy()
            spike_dates_export["decline_rate"] *= 100
            spike_dates_export["baseline_decline_rate"] *= 100

            out_general = io.BytesIO()
            with pd.ExcelWriter(out_general, engine="xlsxwriter") as writer:
                top_merchants_export.to_excel(writer, sheet_name="Top 10 Merchants", index=False)
                spike_dates_export.to_excel(writer, sheet_name="Spike Dates", index=False)

            st.download_button(
                label="📥 Download Top 10 Spikes General Report (Excel)",
                data=out_general.getvalue(),
                file_name=GENERAL_REPORT_FILE,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.markdown("---")
            st.subheader("🔬 Interactive Merchant Deep Analysis Selection")
            
            merchant_options = [
                f"Rank {row['rank']} | ID: {row[MERCHANT_COLUMN]} | Name: {row[MERCHANT_NAME_COLUMN]}" 
                for _, row in top_merchants.iterrows()
            ]
            
            selected_options = st.multiselect(
                "Select Merchant(s) for Deep Analysis:",
                options=merchant_options,
                default=[]
            )

            if selected_options:
                selected_ids = [opt.split("| ID: ")[1].split(" | Name:")[0].strip() for opt in selected_options]
                
                if st.button("⚡ Run Deep Analysis for Selected Merchants", type="primary", key="m21_run_deep"):
                    with st.spinner("Processing deep analytics, rule triggers, top emails, and geo matrices..."):
                        rule_results, coverage_results, email_results, selected_spike_rows = [], [], [], []
                        geo_figures = []
                        zip_buffer = io.BytesIO()

                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for sel_id in selected_ids:
                                m_rows = top_merchants[top_merchants[MERCHANT_COLUMN].astype(str) == sel_id]
                                if m_rows.empty:
                                    continue
                                m_row = m_rows.iloc[0]
                                m_name = str(m_row[MERCHANT_NAME_COLUMN])
                                m_rank = int(m_row["rank"])

                                m_spikes = top_spikes[top_spikes[MERCHANT_COLUMN].astype(str) == sel_id].copy()
                                if m_spikes.empty: 
                                    continue

                                spike_dates = pd.to_datetime(m_spikes["spike_date"]).dt.normalize().dropna().unique()
                                spike_dates_sorted = sorted(pd.Timestamp(d).strftime("%Y-%m-%d") for d in spike_dates)
                                spike_dates_label = ", ".join(spike_dates_sorted)

                                for sd in spike_dates_sorted:
                                    selected_spike_rows.append({
                                        "rank": m_rank,
                                        MERCHANT_COLUMN: sel_id,
                                        MERCHANT_NAME_COLUMN: m_name,
                                        "spike_date": sd
                                    })

                                m_spike_declines = df_clean[
                                    (df_clean[MERCHANT_COLUMN].astype(str) == sel_id) &
                                    (df_clean[STATE_COLUMN] == DECLINE_STATE) &
                                    (df_clean["_date_day"].isin(spike_dates))
                                ].copy()

                                total_declines = m_spike_declines["_transaction_key"].nunique()
                                if total_declines == 0:
                                    continue

                                # 1. Declined Transaction Coverage
                                m_spike_declines["_parsed_rules"] = m_spike_declines[RULES_COLUMN].apply(parse_rule_ids)
                                m_spike_declines["_has_rule"] = m_spike_declines["_parsed_rules"].apply(len) > 0

                                decl_with_rules = m_spike_declines.loc[m_spike_declines["_has_rule"], "_transaction_key"].nunique()
                                decl_no_rules = m_spike_declines.loc[~m_spike_declines["_has_rule"], "_transaction_key"].nunique()

                                coverage_results.append({
                                    "rank": m_rank,
                                    MERCHANT_COLUMN: sel_id,
                                    MERCHANT_NAME_COLUMN: m_name,
                                    "spike_dates": spike_dates_label,
                                    "total_declined_transactions": total_declines,
                                    "declined_transactions_with_rules": decl_with_rules,
                                    "declined_transactions_without_rules": decl_no_rules,
                                    "with_rules_pct": decl_with_rules / total_declines * 100 if total_declines else 0,
                                    "without_rules_pct": decl_no_rules / total_declines * 100 if total_declines else 0
                                })

                                # 2. Top 10 Emails
                                email_data = m_spike_declines.copy()
                                email_data["email_for_analysis"] = email_data[EMAIL_COLUMN].fillna("").astype(str).str.strip().replace("", "[Missing email]")
                                email_counts = email_data.groupby("email_for_analysis")["_transaction_key"].nunique().sort_values(ascending=False).head(TOP_EMAILS_LIMIT).reset_index(name="declined_transactions")
                                email_counts.insert(0, "rank", m_rank)
                                email_counts[MERCHANT_COLUMN] = sel_id
                                email_counts[MERCHANT_NAME_COLUMN] = m_name
                                email_counts["spike_dates"] = spike_dates_label
                                email_results.append(email_counts)

                                # 3. Top 10 Rules
                                rule_data = m_spike_declines[["_transaction_key", "_parsed_rules"]].explode("_parsed_rules").dropna()
                                rule_data = rule_data[rule_data["_parsed_rules"].astype(str).str.strip().ne("")].drop_duplicates()

                                safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", sel_id)

                                if not rule_data.empty:
                                    rule_counts = rule_data.groupby("_parsed_rules")["_transaction_key"].nunique().sort_values(ascending=False).head(TOP_RULES_LIMIT).reset_index(name="declined_transactions_with_rule")
                                    rule_counts["share_of_declines_pct"] = rule_counts["declined_transactions_with_rule"] / total_declines * 100 if total_declines else 0
                                    rule_counts = rule_counts.rename(columns={"_parsed_rules": "rule_id"})
                                    rule_counts["rule_name"] = rule_counts["rule_id"].apply(get_rule_name)
                                    rule_counts.insert(0, "rank", m_rank)
                                    rule_counts[MERCHANT_COLUMN] = sel_id
                                    rule_counts[MERCHANT_NAME_COLUMN] = m_name
                                    rule_counts["spike_dates"] = spike_dates_label
                                    rule_results.append(rule_counts)

                                    # Generate Top Rules Chart for ZIP Export
                                    fig, ax = plt.subplots(figsize=(10, 5))
                                    chart_data = rule_counts.sort_values("declined_transactions_with_rule", ascending=True)
                                    ax.barh(chart_data["rule_name"].astype(str), chart_data["declined_transactions_with_rule"], color="#2b5c8f")
                                    ax.set_title(f"Top {TOP_RULES_LIMIT} Rules — Merchant: {m_name} ({sel_id})\nDates: {spike_dates_label}")
                                    plt.tight_layout()

                                    img_buf = io.BytesIO()
                                    plt.savefig(img_buf, format="png", dpi=150)
                                    plt.close(fig)
                                    zip_file.writestr(f"merchant_{safe_id}_top_rules.png", img_buf.getvalue())

                                # 4. Generate Card Country vs IP Country Heatmap for Display and ZIP Export
                                if card_col and ip_col and card_col in m_spike_declines.columns and ip_col in m_spike_declines.columns:
                                    ct = pd.crosstab(
                                        m_spike_declines[card_col].replace("", np.nan).fillna("Unknown"),
                                        m_spike_declines[ip_col].replace("", np.nan).fillna("Unknown")
                                    )

                                    if not ct.empty:
                                        top_cards = m_spike_declines[card_col].replace("", np.nan).fillna("Unknown").value_counts().head(10).index
                                        top_ips = m_spike_declines[ip_col].replace("", np.nan).fillna("Unknown").value_counts().head(10).index
                                        ct_filtered = ct.loc[ct.index.isin(top_cards), ct.columns.isin(top_ips)]

                                        if not ct_filtered.empty:
                                            fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
                                            sns.heatmap(ct_filtered, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax_hm)
                                            ax_hm.set_title(f"Card Country vs IP Country Heatmap (Spike Declines)\nMerchant: {m_name} | ID: {sel_id}")
                                            ax_hm.set_xlabel("IP Country")
                                            ax_hm.set_ylabel("Card Country")
                                            plt.tight_layout()

                                            geo_buf = io.BytesIO()
                                            plt.savefig(geo_buf, format="png", dpi=150)
                                            zip_file.writestr(f"merchant_{safe_id}_geo_matrix.png", geo_buf.getvalue())
                                            
                                            # Keep figure for direct rendering in Streamlit page
                                            geo_figures.append((m_name, sel_id, fig_hm))

                        zip_buffer.seek(0)

                        deep_spike_df = pd.DataFrame(selected_spike_rows)
                        deep_coverage_df = pd.DataFrame(coverage_results)
                        deep_rules_df = pd.concat(rule_results, ignore_index=True) if rule_results else pd.DataFrame()
                        deep_emails_df = pd.concat(email_results, ignore_index=True) if email_results else pd.DataFrame()

                        st.success("Deep Analysis Completed Successfully!")

                        if not deep_rules_df.empty:
                            st.write("#### 📊 Top Triggered Rules Preview")
                            st.dataframe(deep_rules_df, use_container_width=True)

                        if not deep_emails_df.empty:
                            st.write("#### ✉️ Top Declined Payer Emails Preview")
                            st.dataframe(deep_emails_df, use_container_width=True)

                        # Render Card Country vs IP Country Heatmaps directly on Streamlit Page
                        if geo_figures:
                            st.write("#### 🌐 Card Country vs IP Country Heatmaps")
                            for m_name_f, m_id_f, fig_f in geo_figures:
                                st.pyplot(fig_f)
                                plt.close(fig_f)

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            output_deep = io.BytesIO()
                            with pd.ExcelWriter(output_deep, engine="xlsxwriter") as writer:
                                if not deep_spike_df.empty:
                                    deep_spike_df.to_excel(writer, sheet_name="Selected Spike Dates", index=False)
                                if not deep_rules_df.empty: 
                                    deep_rules_df.to_excel(writer, sheet_name="Top Rules per Merchant", index=False)
                                if not deep_coverage_df.empty: 
                                    deep_coverage_df.to_excel(writer, sheet_name="Declined Transaction Coverage", index=False)
                                if not deep_emails_df.empty: 
                                    deep_emails_df.to_excel(writer, sheet_name="Top Declined Emails", index=False)

                            st.download_button(
                                label="📥 Download Selected Deep Analysis Excel",
                                data=output_deep.getvalue(),
                                file_name=DEEP_REPORT_FILE,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        with col_dl2:
                            st.download_button(
                                label="📦 Download Rule & Geo Charts ZIP",
                                data=zip_buffer.getvalue(),
                                file_name=CHARTS_ZIP_FILE,
                                mime="application/zip"
                            )
                            
    elif sub_module == "2.2 Single Merchant Historical Analysis":
        st.title("🔍 Single Merchant Deep Analysis")
        st.write("Historical decline trends, country distributions, top rule triggers, and comprehensive email profiling.")
        
        up_file = st.file_uploader("Upload SEON Acquiring Export (XLSX/CSV):", type=["xlsx", "csv"], key="m22_upload")
        if up_file:
            df = load_data(up_file)
            
            # Auto-detect key columns
            CARD_COUNTRY_COL = "bin_country_code" if "bin_country_code" in df.columns else ("card_country" if "card_country" in df.columns else "country")
            IP_COUNTRY_COL = "ip_country" if "ip_country" in df.columns else "ipcountry"
            EMAIL_COL = "email/email" if "email/email" in df.columns else ("email/raw_email" if "email/raw_email" in df.columns else "email")
            
            # Strict card_hash column mapping
            CARD_HASH_COL = "card_hash" if "card_hash" in df.columns else None

            CARDHOLDER_COL = "card_fullname" if "card_fullname" in df.columns else "user_fullname"
            STATUS_3D_COL = "status_3d" if "status_3d" in df.columns else "3d_status"
            SCORE_COL = "fraud_score" if "fraud_score" in df.columns else "score"
            TAGS_COL = "tags" if "tags" in df.columns else "tags"
            
            df["_date_dt"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None)
            df["_date_day"] = df["_date_dt"].dt.normalize()
            
            df = df.dropna(subset=["merchant_id", "_date_day"]).copy()
            df["merchant_id"] = df["merchant_id"].astype(str).str.strip()
            df["state_clean"] = df["state"].fillna("").astype(str).str.strip().str.upper()
            
            # Standard State-Based Flags for Baseline/Overall Metrics
            df["is_decline"] = df["state_clean"].eq("DECLINE")
            df["is_approve"] = df["state_clean"].eq("APPROVE")
            
            unique_merchants = sorted(df["merchant_id"].unique())
            selected_m = st.selectbox("Select Merchant ID to Analyze:", unique_merchants, key="m22_select_merchant")
            
            merchant_df = df[df["merchant_id"] == selected_m].copy()
            
            # --- 1. HISTORICAL DECLINE TREND ---
            daily = merchant_df.groupby("_date_day").agg(
                total_transactions=("state_clean", "size"),
                declined_transactions=("is_decline", "sum"),
                approved_transactions=("is_approve", "sum")
            ).reset_index().sort_values("_date_day")
            
            daily["decline_rate_pct"] = (daily["declined_transactions"] / daily["total_transactions"]) * 100
            active_daily = daily[daily["total_transactions"] > 0].copy()
            decline_baseline = float(active_daily["decline_rate_pct"].mean()) if not active_daily.empty else 0.0
            
            daily["decline_change_pct"] = daily["decline_rate_pct"].apply(lambda x: relative_change(x, decline_baseline))
            daily["decline_spike"] = (daily["decline_rate_pct"] >= 6.0) & (daily["decline_change_pct"] >= 50.0)
            
            fig_trend = px.line(daily, x="_date_day", y="decline_rate_pct", title=f"Historical Decline Rate — Merchant: {selected_m}")
            fig_trend.add_hline(y=decline_baseline, line_dash="dash", line_color="green", annotation_text=f"Baseline ({decline_baseline:.2f}%)")
            fig_trend.add_hline(y=6.0, line_dash="dot", line_color="red", annotation_text="Min 6%")
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # --- 2. COUNTRY DISTRIBUTION PIE CHARTS ---
            st.subheader("🥧 Country Distributions (Pie Charts)")
            col_pie1, col_pie2 = st.columns(2)
            
            with col_pie1:
                if CARD_COUNTRY_COL in merchant_df.columns:
                    card_c_counts = merchant_df[CARD_COUNTRY_COL].fillna("Unknown").astype(str).str.upper().value_counts().reset_index()
                    card_c_counts.columns = ["Card Country", "Transactions"]
                    fig_card_pie = px.pie(
                        card_c_counts, 
                        names="Card Country", 
                        values="Transactions", 
                        title="💳 Card Country Breakdown",
                        hole=0.3
                    )
                    fig_card_pie.update_traces(textposition='inside', textinfo='percent+label+value')
                    st.plotly_chart(fig_card_pie, use_container_width=True)
                else:
                    st.info("Card Country column not found for Pie Chart.")
                    
            with col_pie2:
                if IP_COUNTRY_COL in merchant_df.columns:
                    ip_c_counts = merchant_df[IP_COUNTRY_COL].fillna("Unknown").astype(str).str.upper().value_counts().reset_index()
                    ip_c_counts.columns = ["IP Country", "Transactions"]
                    fig_ip_pie = px.pie(
                        ip_c_counts, 
                        names="IP Country", 
                        values="Transactions", 
                        title="🌐 IP Country Breakdown",
                        hole=0.3
                    )
                    fig_ip_pie.update_traces(textposition='inside', textinfo='percent+label+value')
                    st.plotly_chart(fig_ip_pie, use_container_width=True)
                else:
                    st.info("IP Country column not found for Pie Chart.")

            # --- 3. TOP RULE TRIGGER RATIO SPIKES (TOP 5 ON DECLINES) ---
            st.subheader("📌 Top 5 Triggered Rules on Declines")
            rule_source = merchant_df[["_date_day", "is_decline", "applied_rule_ids"]].copy() if "applied_rule_ids" in merchant_df.columns else pd.DataFrame()
            if not rule_source.empty:
                rule_source["rule_id"] = rule_source["applied_rule_ids"].apply(parse_rule_ids)
                rule_long = rule_source.explode("rule_id").dropna(subset=["rule_id"]).copy()
                
                rule_long_declines = rule_long[rule_long["is_decline"]].copy()
                if not rule_long_declines.empty:
                    rule_daily = rule_long_declines.groupby(["_date_day", "rule_id"]).agg(
                        rule_declined_count=("is_decline", "size")
                    ).reset_index()
                    
                    rule_daily["rule_name"] = rule_daily["rule_id"].apply(get_rule_name)
                    
                    rule_summary = (
                        rule_daily.groupby(["rule_id", "rule_name"])["rule_declined_count"]
                        .sum()
                        .nlargest(5)
                        .reset_index()
                        .rename(columns={"rule_declined_count": "declined_triggers_count"})
                    )
                    st.dataframe(rule_summary, use_container_width=True)
                else:
                    st.info("No rule triggers found on declined transactions for this merchant.")

            # --- 4. HEATMAPS: OVERALL VS DECLINES ONLY ---
            if CARD_COUNTRY_COL in merchant_df.columns and IP_COUNTRY_COL in merchant_df.columns:
                st.subheader("🌐 Card Country vs IP Country Heatmaps")
                col_hm1, col_hm2 = st.columns(2)
                
                with col_hm1:
                    ct_all = pd.crosstab(merchant_df[CARD_COUNTRY_COL].fillna("Unknown"), merchant_df[IP_COUNTRY_COL].fillna("Unknown"))
                    top_cards_all = merchant_df[CARD_COUNTRY_COL].fillna("Unknown").value_counts().head(10).index
                    top_ips_all = merchant_df[IP_COUNTRY_COL].fillna("Unknown").value_counts().head(10).index
                    ct_filtered_all = ct_all.loc[ct_all.index.isin(top_cards_all), ct_all.columns.isin(top_ips_all)]
                    
                    if not ct_filtered_all.empty:
                        fig_hm_all = px.imshow(ct_filtered_all, text_auto=True, color_continuous_scale="YlGnBu", title="Overall Matrix (All Transactions)")
                        st.plotly_chart(fig_hm_all, use_container_width=True)

                with col_hm2:
                    merchant_declines_df = merchant_df[merchant_df["is_decline"]].copy()
                    if not merchant_declines_df.empty:
                        ct_dec = pd.crosstab(merchant_declines_df[CARD_COUNTRY_COL].fillna("Unknown"), merchant_declines_df[IP_COUNTRY_COL].fillna("Unknown"))
                        top_cards_dec = merchant_declines_df[CARD_COUNTRY_COL].fillna("Unknown").value_counts().head(10).index
                        top_ips_dec = merchant_declines_df[IP_COUNTRY_COL].fillna("Unknown").value_counts().head(10).index
                        ct_filtered_dec = ct_dec.loc[ct_dec.index.isin(top_cards_dec), ct_dec.columns.isin(top_ips_dec)]
                        
                        if not ct_filtered_dec.empty:
                            fig_hm_dec = px.imshow(ct_filtered_dec, text_auto=True, color_continuous_scale="Reds", title="Declines Only Matrix")
                            st.plotly_chart(fig_hm_dec, use_container_width=True)

            # --- 5. ENHANCED EMAIL DEEP ANALYTICS ---
            st.subheader("✉️ Enhanced Email Analytics (Declines & Failed Profiling)")
            
            if EMAIL_COL in merchant_df.columns:
                m_decline_emails = set(
                    merchant_df[merchant_df["is_decline"]][EMAIL_COL].dropna().astype(str).str.strip().unique()
                )
                
                if m_decline_emails:
                    card_email_map = {}
                    if CARD_HASH_COL and CARD_HASH_COL in merchant_df.columns:
                        card_email_map = (
                            merchant_df.groupby(CARD_HASH_COL)[EMAIL_COL]
                            .apply(lambda x: sorted(set(x.dropna().astype(str).str.strip())))
                            .to_dict()
                        )

                    email_rows = []
                    for email_val in sorted(m_decline_emails):
                        e_df = merchant_df[merchant_df[EMAIL_COL].astype(str).str.strip() == email_val].copy()

                        total_tx = len(e_df)
                        
                        # 1. Success check: Tag must explicitly contain "RC 00"
                        if TAGS_COL in e_df.columns:
                            has_rc00_series = e_df[TAGS_COL].fillna("").astype(str).str.upper().str.contains(r"RC\s*00", regex=True)
                        else:
                            has_rc00_series = e_df["state_clean"].eq("APPROVE")
                            
                        successful_tx = int(has_rc00_series.sum())
                        
                        # 2. Direct Anti-Fraud Declines (State == DECLINE)
                        direct_declines_tx = int(e_df["state_clean"].eq("DECLINE").sum())
                        
                        # 3. Failed Transactions: No RC 00 tag AND state is not direct DECLINE
                        failed_tx = int((~has_rc00_series & ~e_df["state_clean"].eq("DECLINE")).sum())

                        # IP Countries, Card Countries, Cardholder Names
                        ip_countries = sorted(e_df[IP_COUNTRY_COL].dropna().astype(str).str.upper().unique()) if IP_COUNTRY_COL in e_df.columns else []
                        card_countries = sorted(e_df[CARD_COUNTRY_COL].dropna().astype(str).str.upper().unique()) if CARD_COUNTRY_COL in e_df.columns else []
                        cardholders = sorted(e_df[CARDHOLDER_COL].dropna().astype(str).unique()) if CARDHOLDER_COL in e_df.columns else []

                        # Strict card_hash mapping for Unique Cards & Shared Cards
                        if CARD_HASH_COL and CARD_HASH_COL in e_df.columns:
                            cards_used = sorted(e_df[CARD_HASH_COL].dropna().astype(str).str.strip().unique())
                            cards_used = [c for c in cards_used if c.lower() not in ["nan", "none", "null", ""]]
                            unique_cards_count = len(cards_used)
                            
                            shared_cards_count = 0
                            shared_emails = set()
                            for card in cards_used:
                                other_emails = set(card_email_map.get(card, []))
                                other_emails.discard(email_val)
                                if len(other_emails) > 0:
                                    shared_cards_count += 1
                                    shared_emails.update(other_emails)

                            has_shared_cards = "Y" if shared_cards_count > 0 else "N"
                            shared_with_str = ", ".join(sorted(shared_emails)) if shared_emails else "None"
                        else:
                            unique_cards_count = "No card_hash column"
                            has_shared_cards = "N/A"
                            shared_cards_count = 0
                            shared_with_str = "No card_hash column"

                        # 3DS Failures Count (status_3d == 'N' and fraud_score < 30)
                        failed_3ds_count = 0
                        if STATUS_3D_COL in e_df.columns and SCORE_COL in e_df.columns:
                            failed_3ds_count = int(len(
                                e_df[
                                    (e_df[STATUS_3D_COL].fillna("").astype(str).str.upper() == "N") &
                                    (pd.to_numeric(e_df[SCORE_COL], errors="coerce") < 30)
                                ]
                            ))

                        email_rows.append({
                            "Email": email_val,
                            "Total Tx": total_tx,
                            "Successful (RC 00 Tag)": successful_tx,
                            "Direct Declines": direct_declines_tx,
                            "Failed Tx (No RC 00 Tag)": failed_tx,
                            "3DS Failed Count": failed_3ds_count,
                            "Unique Cards (card_hash)": unique_cards_count,
                            "Card Countries": ", ".join(card_countries) if card_countries else "Unknown",
                            "IP Countries": ", ".join(ip_countries) if ip_countries else "Unknown",
                            "Cardholder Names": ", ".join(cardholders) if cardholders else "Unknown",
                            "Has Shared Cards": has_shared_cards,
                            "Shared Cards Count": shared_cards_count,
                            "Shared With Emails": shared_with_str
                        })

                    emails_analytics_df = pd.DataFrame(email_rows).sort_values(["Direct Declines", "Failed Tx (No RC 00 Tag)"], ascending=False)
                    
                    # Highlight Shared Cards & 3DS Failures in UI
                    def style_email_table(df_to_style):
                        def highlight_rows(row):
                            styles = [''] * len(row)
                            if row.get("Has Shared Cards") == "Y":
                                return ['background-color: #ffe6e6'] * len(row)
                            return styles

                        styled = df_to_style.style.apply(highlight_rows, axis=1)
                        if "3DS Failed Count" in df_to_style.columns:
                            styled = styled.map(
                                lambda val: 'background-color: #fff3cd; font-weight: bold;' if isinstance(val, (int, float)) and val > 0 else '',
                                subset=["3DS Failed Count"]
                            )
                        return styled

                    st.dataframe(style_email_table(emails_analytics_df), use_container_width=True)
                else:
                    st.info("No emails with decline transactions found for this merchant.")

            # --- EXPORT REPORT ---
            output_merchant = io.BytesIO()
            with pd.ExcelWriter(output_merchant, engine="xlsxwriter") as writer:
                daily.to_excel(writer, sheet_name="Daily Merchant Metrics", index=False)
                if 'rule_summary' in locals():
                    rule_summary.to_excel(writer, sheet_name="Top 5 Rule Spikes Declines", index=False)
                if 'emails_analytics_df' in locals() and not emails_analytics_df.empty:
                    emails_analytics_df.to_excel(writer, sheet_name="Email Deep Profiling", index=False)

            st.download_button(
                label=f"📥 Download Deep Analysis Report ({selected_m})",
                data=output_merchant.getvalue(),
                file_name=f"SEON_Merchant_Analysis_{selected_m}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
# ============================================================
# MODULE 3: MONTHLY TOP-UP DECLINE REPORT
# ============================================================
elif module == "3. Monthly Decline Report (Waterfall)":
    st.title("📊 Monthly Decline Waterfall & Metabase Volume")
    st.write("Complete monthly analytics pipeline matching the exact reference Google Colab logic.")
    
    # Select Month & Year
    col_y, col_m = st.columns(2)
    with col_y:
        ANALYSIS_YEAR = st.number_input("Select Analysis Year:", min_value=2020, max_value=2030, value=2026)
    with col_m:
        ANALYSIS_MONTH = st.selectbox("Select Analysis Month:", list(range(1, 13)), index=8, format_func=lambda m: calendar.month_name[m])
    
    month_label = f"{calendar.month_name[ANALYSIS_MONTH]} {ANALYSIS_YEAR}"
    st.markdown("---")
    
    # --- PHASE 1: UPLOAD SEON TRANSACTIONS & SEON BLACKLIST ---
    st.subheader("1️⃣ STEP 1 & 2: Upload SEON Exports")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        f1 = st.file_uploader("Upload 1/3: SEON Transactions Export:", type=["xlsx", "csv"], key="m3_step1_f1")
    with col_f2:
        f2 = st.file_uploader("Upload 2/3: SEON Blacklist Export (CSV ;):", type=["csv", "xlsx"], key="m3_step1_f2")

    if f1 is not None and f2 is not None:
        if st.button("🚀 Run Step 1 & 2 (Waterfall & Clean User IDs)", type="primary"):
            with st.spinner("Processing SEON Transactions Waterfall & Blacklist..."):
                transactions = load_data(f1)
                blacklist = load_data(f2, sep=';')
                
                transactions.columns = transactions.columns.astype(str).str.strip()
                blacklist.columns = blacklist.columns.astype(str).str.strip()

                # --- STEP 1: TRANSACTIONS WATERFALL ---
                required_tx_cols = [
                    'action_type', 'payment_provider', 'date', 'state', 'email/domain',
                    'bin_country_code', 'user_country', 'merchant_country', 'ip_country',
                    'transaction_amount_eur', 'email/raw_email'
                ]
                missing_tx = [c for c in required_tx_cols if c not in transactions.columns]
                if missing_tx:
                    st.error(f"SEON transactions export missing required columns: {', '.join(missing_tx)}")
                    st.stop()

                transactions['payment_provider'] = pd.to_numeric(transactions['payment_provider'], errors='coerce')
                transactions['transaction_amount_eur'] = pd.to_numeric(transactions['transaction_amount_eur'], errors='coerce').fillna(0)
                transactions['state'] = transactions['state'].astype(str).str.strip().str.upper()
                transactions['action_type'] = transactions['action_type'].astype(str).str.strip().str.lower()
                transactions['date'] = pd.to_datetime(transactions['date'], utc=True, errors='coerce')

                # Filter scope: purchases, excluding provider 521268264, for selected month
                tx_scope = transactions[
                    (transactions['action_type'] == 'purchase') &
                    (transactions['payment_provider'] != 521268264)
                ].copy()
                
                tx_scope = tx_scope[(tx_scope['date'].dt.year == ANALYSIS_YEAR) & (tx_scope['date'].dt.month == ANALYSIS_MONTH)]

                if tx_scope.empty:
                    st.error(f"No purchase transactions found for {month_label} after excluding provider 521268264.")
                    st.stop()

                step_1 = tx_scope[~tx_scope['email/domain'].astype(str).str.strip().str.lower().isin(EXCLUDED_DOMAINS)].copy()
                step_2 = step_1[
                    (~step_1['bin_country_code'].astype(str).str.strip().str.upper().isin(RESTRICTED_COUNTRIES)) &
                    (~step_1['user_country'].astype(str).str.strip().str.upper().isin(RESTRICTED_COUNTRIES)) &
                    (~step_1['merchant_country'].astype(str).str.strip().str.upper().isin(RESTRICTED_COUNTRIES)) &
                    (~step_1['ip_country'].astype(str).str.strip().str.upper().isin(RESTRICTED_COUNTRIES))
                ].copy()
                step_3 = label_consecutive_declines(step_2, timeframe_hours=0.25)
                step_4 = step_3[~step_3['consecutive_decline']].copy()

                def pct(numerator, denominator):
                    return numerator / denominator * 100 if denominator else 0.0

                def decline_count_rate(data):
                    return pct((data['state'] == 'DECLINE').sum(), len(data))

                def decline_amount_rate(data):
                    return pct(data.loc[data['state'] == 'DECLINE', 'transaction_amount_eur'].sum(), data['transaction_amount_eur'].sum())

                raw_c, raw_a = decline_count_rate(step_1), decline_amount_rate(step_1)
                restricted_c, restricted_a = decline_count_rate(step_2), decline_amount_rate(step_2)
                true_reject_c, true_reject_a = decline_count_rate(step_4), decline_amount_rate(step_4)

                restricted_impact_pp = raw_c - restricted_c
                consecutive_impact_pp = restricted_c - true_reject_c

                business_summary = (
                    f"{month_label} raw decline rate was {raw_c:.2f}%, "
                    f"which includes {restricted_impact_pp:.2f}pp from sanctioned/restricted countries and "
                    f"{consecutive_impact_pp:.2f}pp from consecutive-decline deduplication. "
                    f"The remaining true reject rate was {true_reject_c:.2f}%."
                )

                # --- STEP 2: BLACKLIST & NET TRUE POSITIVE MATCHING ---
                required_bl_cols = ['data field', 'value', 'comment', 'last updated', 'last updated by']
                missing_bl = [c for c in required_bl_cols if c not in blacklist.columns]
                if missing_bl:
                    st.error(f"SEON blacklist export missing required columns: {', '.join(missing_bl)}")
                    st.stop()

                # Parse last updated timestamp for month-filtering
                blacklist['last updated_dt'] = pd.to_datetime(blacklist['last updated'], utc=True, errors='coerce', format='mixed')
                
                # Filter Blacklist strictly by selected month for Accounts analysis
                blacklist_month = blacklist[
                    (blacklist['last updated_dt'].dt.year == ANALYSIS_YEAR) & 
                    (blacklist['last updated_dt'].dt.month == ANALYSIS_MONTH)
                ].copy()

                # Match Residual Declines against FULL SEON Blacklist (as in reference script)
                step_4_month = step_4.copy()
                
                def tag_fraud_transactions_ref(data, bl_df_input):
                    res = data.copy()
                    res['is_fraud'] = False
                    if bl_df_input.empty:
                        return res
                    grouped = bl_df_input[['data field', 'value']].dropna().groupby('data field')['value'].apply(
                        lambda vals: set(vals.astype(str).str.strip())
                    )
                    for field, values in grouped.items():
                        if field in res.columns:
                            res.loc[res[field].astype(str).str.strip().isin(values), 'is_fraud'] = True
                    return res

                tagged_month = tag_fraud_transactions_ref(step_4_month, blacklist)
                remaining_declines = tagged_month[tagged_month['state'] == 'DECLINE'].copy()
                fraud_declines = remaining_declines[remaining_declines['is_fraud']].copy()
                net_tp_share = pct(len(fraud_declines), len(remaining_declines))

                # Build Waterfall Reporting Table
                waterfall = pd.DataFrame([
                    {'Description': 'Raw Decline Rate Without Tests accounts', 'Reject Rate (Count)': f'{raw_c:.2f}%'},
                    {'Description': 'Sanctioned/Restricted Countries Filtered Out', 'Reject Rate (Count)': f'{restricted_impact_pp:.2f}%'},
                    {'Description': 'Consecutive Declines Deduplication', 'Reject Rate (Count)': f'{consecutive_impact_pp:.2f}%'},
                    {'Description': 'True Reject Rate', 'Reject Rate (Count)': f'{true_reject_c:.2f}%'},
                    {'Description': 'Net True Positive Reject Rate', 'Reject Rate (Count)': f'{net_tp_share:.2f}%'},
                    {'Description': business_summary, 'Reject Rate (Count)': ''},
                ])

                fraud_summary = pd.DataFrame([
                    {'Metric': 'Residual declines after waterfall', 'Count': len(remaining_declines)},
                    {'Metric': 'Residual declines matching SEON blacklist', 'Count': len(fraud_declines)},
                    {'Metric': 'Net True Positive Reject Rate (% of residual declines)', 'Count': round(net_tp_share, 2)},
                ])

                # --- CLOSED ACCOUNTS & MULTI-ACCOUNTS (MONTHLY SCOPE) ---
                def numeric_ids_ref(values):
                    ids = set()
                    for raw in values.dropna():
                        value = str(raw).strip()
                        if not value or value.lower() in {'nan', 'none', 'null'}:
                            continue
                        if value.endswith('.0') and value[:-2].isdigit():
                            value = value[:-2]
                        try:
                            number = float(value)
                            if number.is_integer() and number >= 0:
                                value = str(int(number))
                        except (TypeError, ValueError, OverflowError):
                            pass
                        if value.isdigit():
                            ids.add(value)
                    return ids

                EMPLOYEE_NAMES = {'maryna krasheninnikova', 'serhii kryhin', 'oksana moskalenko', 'yevheniia linichenko'}
                
                monthly_field = blacklist_month['data field'].astype(str).str.strip().str.lower()
                monthly_comment = blacklist_month['comment'].astype(str).fillna('').str.strip()
                monthly_editor = blacklist_month['last updated by'].astype(str).str.strip().str.lower()

                user_id_rows = blacklist_month[monthly_field == 'user_id'].copy()
                closed_mask = monthly_editor.loc[user_id_rows.index].isin(EMPLOYEE_NAMES)
                closed_acc_df = user_id_rows[closed_mask].copy()
                closed_acc_user_ids = numeric_ids_ref(closed_acc_df['value'])

                multi_mask = monthly_comment.loc[user_id_rows.index].str.contains(r'duplicate|active|multi[\s_-]*acc', case=False, regex=True, na=False)
                multi_acc_df = user_id_rows[multi_mask].copy()
                multi_acc_user_ids = numeric_ids_ref(multi_acc_df['value'])

                clean_user_ids_list = sorted(closed_acc_user_ids - multi_acc_user_ids, key=int)
                result_ids_str = ','.join(clean_user_ids_list)

                account_summary = pd.DataFrame([
                    {'Metric': f'Multi-accounts identified in {month_label}', 'Count': len(multi_acc_user_ids)},
                    {'Metric': f'Closed-account IDs identified in {month_label}', 'Count': len(closed_acc_user_ids)},
                    {'Metric': 'Closed-account IDs excluding multi-accounts', 'Count': len(clean_user_ids_list)},
                ])

                # Save State in Session
                st.session_state['m3_phase1_done'] = True
                st.session_state['m3_waterfall'] = waterfall
                st.session_state['m3_fraud_summary'] = fraud_summary
                st.session_state['m3_account_summary'] = account_summary
                st.session_state['m3_clean_user_ids_list'] = clean_user_ids_list
                st.session_state['m3_result_ids_str'] = result_ids_str

    # --- RENDER RESULTS & METABASE STEP ---
    if st.session_state.get('m3_phase1_done', False):
        st.markdown("---")
        st.subheader(f"STEP 1 RESULTS · REPORTING TABLE — {month_label}")
        st.dataframe(st.session_state['m3_waterfall'], use_container_width=True)

        st.subheader(f"BLACKLIST MATCH SUMMARY — {month_label}")
        st.dataframe(st.session_state['m3_fraud_summary'], use_container_width=True)

        st.subheader(f"STEP 2 RESULTS · CLOSED ACCOUNTS AND MULTI-ACCOUNTS — {month_label}")
        st.dataframe(st.session_state['m3_account_summary'], use_container_width=True)

        st.subheader("📋 USER IDS FOR METABASE QUERY (Copy below):")
        st.code(st.session_state['m3_result_ids_str'] if st.session_state['m3_result_ids_str'] else "No eligible numeric user IDs found.", language="text")

        st.markdown("---")
        st.subheader("2️⃣ STEP 3: Upload Metabase Export (Independent Metabase Query File)")
        f3 = st.file_uploader("Upload 3/3: Metabase Export:", type=["xlsx", "csv"], key="m3_step2_f3")

        if f3 is not None:
            if st.button("💶 Process Metabase Volume & Finalize Report", type="primary"):
                with st.spinner("Processing Metabase Top-Up Volumes & Currency Rates..."):
                    metabase = load_data(f3)
                    
                    required_mb_cols = ['status', 'transaction_type', 'pay_currency', 'pay_amount']
                    missing_mb = [c for c in required_mb_cols if c not in metabase.columns]
                    if missing_mb:
                        st.error(f"Metabase export missing required columns: {', '.join(missing_mb)}")
                        st.stop()

                    rates_to_eur = fetch_eur_rates()

                    metabase_data = metabase.copy()
                    metabase_data['pay_amount'] = pd.to_numeric(metabase_data['pay_amount'], errors='coerce').fillna(0)
                    metabase_data['pay_currency'] = metabase_data['pay_currency'].astype(str).str.strip().str.upper()

                    successful_topups = metabase_data[
                        (metabase_data['status'].astype(str).str.strip() == 'Accepted') &
                        (metabase_data['transaction_type'].astype(str).str.strip() == 'TYPE_TOP_UP')
                    ].copy()

                    topup_summary = successful_topups.groupby('pay_currency', as_index=False)['pay_amount'].sum()
                    topup_summary['rate_to_eur'] = topup_summary['pay_currency'].map(rates_to_eur).fillna(1.0)
                    topup_summary['amount_eur'] = topup_summary['pay_amount'] * topup_summary['rate_to_eur']
                    total_topup_eur = topup_summary['amount_eur'].sum()

                    topup_overview = pd.DataFrame([
                        {'Metric': 'Successful top-up transactions', 'Value': f'{len(successful_topups):,}'},
                        {'Metric': 'Total top-up volume (EUR equivalent)', 'Value': f'{total_topup_eur:,.2f} EUR'},
                    ])
                    
                    st.subheader(f"STEP 3 RESULTS · METABASE TOP-UPS — {month_label}")
                    st.dataframe(topup_overview, use_container_width=True)
                    st.write("#### Currency Breakdown:")
                    st.dataframe(topup_summary.sort_values('amount_eur', ascending=False), use_container_width=True)

                    # --- EXPORT FULL REPORT ---
                    OUTPUT_FILE = f"SEON_Monthly_Analysis_{ANALYSIS_YEAR}_{ANALYSIS_MONTH:02d}.xlsx"
                    output_report = io.BytesIO()
                    with pd.ExcelWriter(output_report, engine="xlsxwriter") as writer:
                        st.session_state['m3_waterfall'].to_excel(writer, sheet_name="Waterfall", index=False)
                        st.session_state['m3_fraud_summary'].to_excel(writer, sheet_name="Blacklist Match", index=False)
                        st.session_state['m3_account_summary'].to_excel(writer, sheet_name="Accounts Summary", index=False)
                        pd.DataFrame({"user_id": st.session_state['m3_clean_user_ids_list']}).to_excel(writer, sheet_name="Eligible User IDs", index=False)
                        topup_overview.to_excel(writer, sheet_name="Topups Summary", index=False)
                        topup_summary.to_excel(writer, sheet_name="Topups by Currency", index=False)

                    st.download_button(
                        label="📥 Download Complete Monthly Analysis Report (Excel)",
                        data=output_report.getvalue(),
                        file_name=OUTPUT_FILE,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
# ============================================================
# MODULE 4: TOP-UP DECLINES MONITORING
# ============================================================
elif module == "4. Top-Up Declines Monitoring":
    st.title("💰 Top-Up Wallet Declines & Route Analytics")
    st.write("Monitoring wallet top-ups, decline reasons, payment methods, geographic routes, top failing users, and peak spike days.")
    
    up_file = st.file_uploader("Upload Top-Up Transaction Export (XLSX/CSV):", type=["xlsx", "csv"], key="m4_upload")
    if up_file:
        df = load_data(up_file)
        
        # Detect key columns
        DATE_COL = "created_date" if "created_date" in df.columns else ("transaction_date" if "transaction_date" in df.columns else "date")
        USER_COL = "user_id" if "user_id" in df.columns else "user"
        STATUS_COL = "status" if "status" in df.columns else "state"
        AMOUNT_COL = "topup_amount" if "topup_amount" in df.columns else ("processing_amount" if "processing_amount" in df.columns else ("converted_amount" if "converted_amount" in df.columns else "transaction_amount_eur"))
        VERIF_TYPE_COL = "verification_type" if "verification_type" in df.columns else ("user_type" if "user_type" in df.columns else "status_type")
        PM_TYPE_COL = "payment_method_type" if "payment_method_type" in df.columns else "payment_method"
        RESIDENCE_COL = "residence_country" if "residence_country" in df.columns else "user_country"
        CARD_COUNTRY_COL = "card_issue_country" if "card_issue_country" in df.columns else ("card_country" if "card_country" in df.columns else "bin_country_code")
        REJECT_REASON_COL = "reject_reason" if "reject_reason" in df.columns else ("response_information" if "response_information" in df.columns else ("applied_rule_ids" if "applied_rule_ids" in df.columns else "rule"))
        TRANSFER_ID_COL = "transfer_id" if "transfer_id" in df.columns else "transaction_id"
        MERCHANT_ID_COL = "merchant_id" if "merchant_id" in df.columns else "admin"

        # Basic processing
        df["_dt"] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True).dt.tz_convert(None)
        df["_date"] = df["_dt"].dt.normalize()
        df["_week"] = df["_dt"].dt.to_period("W").dt.start_time
        
        df["state_clean"] = df[STATUS_COL].fillna("").astype(str).str.strip().str.capitalize()
        df["_is_failed"] = df["state_clean"].str.lower().isin(["failed", "decline", "declined", "error"])
        df["_is_accepted"] = df["state_clean"].str.lower().isin(["accepted", "approve", "approved", "success"])
        
        resp_series = df[REJECT_REASON_COL].fillna("").astype(str).str.lower() if REJECT_REASON_COL in df.columns else pd.Series("", index=df.index)
        df["_is_seon_failed"] = df["_is_failed"] & resp_series.str.contains("security reason", regex=False)
        
        if AMOUNT_COL in df.columns:
            df["_amount"] = pd.to_numeric(df[AMOUNT_COL].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce").fillna(0.0)
        else:
            df["_amount"] = 0.0

        # GLOBAL FILTERS
        st.sidebar.subheader("🎛️ Top-Up Global Filters")
        min_date, max_date = df["_date"].min(), df["_date"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input("Date Range:", [min_date, max_date])
            if len(date_range) == 2:
                df = df[(df["_date"] >= pd.Timestamp(date_range[0])) & (df["_date"] <= pd.Timestamp(date_range[1]))]

        if VERIF_TYPE_COL in df.columns:
            verif_options = ["ALL"] + sorted(df[VERIF_TYPE_COL].dropna().astype(str).unique().tolist())
            sel_verif = st.sidebar.selectbox("Verification Type:", verif_options)
            if sel_verif != "ALL":
                df = df[df[VERIF_TYPE_COL].astype(str) == sel_verif]

        # Key Metrics
        total_tx = len(df)
        accepted_tx = df["_is_accepted"].sum()
        failed_df = df[df["_is_failed"]].copy()
        failed_tx = len(failed_df)
        failed_amount = failed_df["_amount"].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Top-Up Tx", f"{total_tx:,}")
        c2.metric("Accepted Tx", f"{accepted_tx:,}")
        c3.metric("Failed Tx", f"{failed_tx:,}")
        c4.metric("Loss Amount (Failed)", f"{failed_amount:,.2f} EUR")
        
        # --- 1. ORIGINAL OVERALL DECLINE RATE TREND ---
        st.subheader("📈 Overall Decline Rate vs. SEON Decline Rate Trend")
        daily_stats = df.groupby("_date").agg(
            accepted=("_is_accepted", "sum"),
            failed=("_is_failed", "sum"),
            seon_failed=("_is_seon_failed", "sum")
        ).reset_index()
        
        daily_stats["finished"] = daily_stats["accepted"] + daily_stats["failed"]
        daily_stats["overall_decline_rate"] = np.where(daily_stats["finished"] > 0, (daily_stats["failed"] / daily_stats["finished"]) * 100, 0.0)
        daily_stats["seon_decline_rate"] = np.where(daily_stats["finished"] > 0, (daily_stats["seon_failed"] / daily_stats["finished"]) * 100, 0.0)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=daily_stats["_date"], y=daily_stats["overall_decline_rate"], name="Overall Decline Rate (%)", line=dict(color='#e74c3c', width=3)))
        fig_trend.add_trace(go.Scatter(x=daily_stats["_date"], y=daily_stats["seon_decline_rate"], name="SEON Decline Rate (%)", line=dict(color='#2980b9', width=2, dash='dash')))
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # --- 2. ORIGINAL GEO & PAYMENT METHOD BREAKDOWNS ---
        if not failed_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                if RESIDENCE_COL in failed_df.columns:
                    top_user_c = failed_df[RESIDENCE_COL].value_counts().head(8).reset_index()
                    top_user_c.columns = [RESIDENCE_COL, "count"]
                    fig1 = px.bar(top_user_c, x="count", y=RESIDENCE_COL, orientation="h", title="Top User Countries (Declines)")
                    st.plotly_chart(fig1, use_container_width=True)
            with col2:
                if CARD_COUNTRY_COL in failed_df.columns:
                    top_card_c = failed_df[CARD_COUNTRY_COL].value_counts().head(8).reset_index()
                    top_card_c.columns = [CARD_COUNTRY_COL, "count"]
                    fig2 = px.bar(top_card_c, x="count", y=CARD_COUNTRY_COL, orientation="h", title="Top Card Issuer Countries (Declines)")
                    st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if RESIDENCE_COL in failed_df.columns and CARD_COUNTRY_COL in failed_df.columns:
                    spike_geo = failed_df.groupby([RESIDENCE_COL, CARD_COUNTRY_COL]).size().reset_index(name='Failed_Count').sort_values(by='Failed_Count', ascending=False).head(8)
                    spike_geo['route'] = spike_geo[RESIDENCE_COL].astype(str) + ' -> ' + spike_geo[CARD_COUNTRY_COL].astype(str)
                    fig_route = px.bar(spike_geo, x="Failed_Count", y="route", orientation="h", title="Top Routes (User -> Card Country)")
                    st.plotly_chart(fig_route, use_container_width=True)
            with col4:
                if PM_TYPE_COL in failed_df.columns:
                    top_pm = failed_df[PM_TYPE_COL].value_counts().head(8).reset_index()
                    top_pm.columns = ["method", "count"]
                    fig_pm = px.bar(top_pm, x="count", y="method", orientation="h", title="Top Payment Methods (Declines)")
                    st.plotly_chart(fig_pm, use_container_width=True)

        # --- LOCAL TIME SLOT FILTER FOR TOP USERS & REASONS ---
        st.markdown("---")
        st.subheader("🎛️ Time Aggregation for Users & Reasons Analytics")
        time_slot = st.selectbox("Select Time Granularity:", ["Day", "Week"], key="m4_time_slot_local")
        time_col = "_date" if time_slot == "Day" else "_week"

        # --- 3. TOP 10 USERS WITH TOP 2 REASONS HOVER ---
        if not failed_df.empty and USER_COL in failed_df.columns:
            # Helper to calculate Top 2 reasons per user
            def get_top_reasons(user_data):
                if REJECT_REASON_COL in user_data.columns:
                    top_r = user_data[REJECT_REASON_COL].dropna().apply(get_rule_name).value_counts().head(2)
                    return ", ".join([f"{k} ({v})" for k, v in top_r.items()])
                return "N/A"

            col_u1, col_u2 = st.columns(2)
            
            with col_u1:
                st.subheader("👤 Top 10 Users by Declined Amount")
                u_amt_df = failed_df.groupby([USER_COL, time_col]).agg(
                    total_amount=("_amount", "sum"),
                    declines_count=("_amount", "size")
                ).reset_index()
                
                top10_amt_ids = failed_df.groupby(USER_COL)["_amount"].sum().nlargest(10).index
                filtered_u_amt = u_amt_df[u_amt_df[USER_COL].isin(top10_amt_ids)].copy()
                
                # Add Top 2 Reasons metadata
                user_reasons = {u_id: get_top_reasons(failed_df[failed_df[USER_COL] == u_id]) for u_id in top10_amt_ids}
                filtered_u_amt["top_2_reasons"] = filtered_u_amt[USER_COL].map(user_reasons)
                
                fig_u_amt = px.bar(
                    filtered_u_amt,
                    x="total_amount",
                    y=USER_COL,
                    color=time_col,
                    orientation="h",
                    hover_data=["declines_count", "top_2_reasons"],
                    title=f"Top 10 Users by Declined Amount ({time_slot} Slot)",
                    labels={"total_amount": "Total Amount (EUR)", USER_COL: "User ID"}
                )
                fig_u_amt.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_u_amt, use_container_width=True)

            with col_u2:
                st.subheader("🔢 Top 10 Users by Number of Declines")
                u_cnt_df = failed_df.groupby([USER_COL, time_col]).agg(
                    declines_count=("_amount", "size"),
                    total_amount=("_amount", "sum")
                ).reset_index()
                
                top10_cnt_ids = failed_df.groupby(USER_COL).size().nlargest(10).index
                filtered_u_cnt = u_cnt_df[u_cnt_df[USER_COL].isin(top10_cnt_ids)].copy()
                
                user_reasons_cnt = {u_id: get_top_reasons(failed_df[failed_df[USER_COL] == u_id]) for u_id in top10_cnt_ids}
                filtered_u_cnt["top_2_reasons"] = filtered_u_cnt[USER_COL].map(user_reasons_cnt)
                
                fig_u_cnt = px.bar(
                    filtered_u_cnt,
                    x="declines_count",
                    y=USER_COL,
                    color=time_col,
                    orientation="h",
                    hover_data=["total_amount", "top_2_reasons"],
                    title=f"Top 10 Users by Number of Declines ({time_slot} Slot)",
                    labels={"declines_count": "Declines Count", USER_COL: "User ID"}
                )
                fig_u_cnt.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_u_cnt, use_container_width=True)

        # --- 4. TOP 5 DECLINE REASONS ---
        st.markdown("---")
        st.subheader("🚨 Top 5 Decline Reasons (All-Time / Filtered Period)")
        
        if not failed_df.empty and REJECT_REASON_COL in failed_df.columns:
            failed_df["_reason_clean"] = failed_df[REJECT_REASON_COL].apply(get_rule_name)
            
            reasons_summary = (
                failed_df.groupby("_reason_clean")
                .agg(
                    declined_count=("_amount", "size"),
                    total_amount=("_amount", "sum")
                )
                .reset_index()
                .sort_values("declined_count", ascending=False)
            )
            
            top5_reasons_df = reasons_summary.head(5).copy()
            
            fig_reasons = px.bar(
                top5_reasons_df,
                x="declined_count",
                y="_reason_clean",
                color="total_amount",
                color_continuous_scale="Reds",
                orientation="h",
                title="Top 5 Decline Reasons (Bar Color Represents Amount Loss in EUR)",
                labels={
                    "declined_count": "Number of Declines",
                    "_reason_clean": "Decline Reason / Rule",
                    "total_amount": "Total Amount Loss (EUR)"
                }
            )
            fig_reasons.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_reasons, use_container_width=True)
            
            # --- DAILY / WEEKLY DRILL-DOWN TREND FOR DECLINE REASONS ---
            st.subheader(f"📅 Selected Decline Reasons Trend ({time_slot} View)")
            all_reasons = top5_reasons_df["_reason_clean"].tolist()
            selected_reasons = st.multiselect(
                "Select Decline Reason(s) to Analyze Trend:",
                options=reasons_summary["_reason_clean"].tolist(),
                default=all_reasons[:2] if all_reasons else []
            )
            
            if selected_reasons:
                trend_reasons_df = (
                    failed_df[failed_df["_reason_clean"].isin(selected_reasons)]
                    .groupby([time_col, "_reason_clean"])
                    .agg(
                        slot_count=("_amount", "size"),
                        slot_amount=("_amount", "sum")
                    )
                    .reset_index()
                )
                
                fig_reason_trend = px.line(
                    trend_reasons_df,
                    x=time_col,
                    y="slot_count",
                    color="_reason_clean",
                    markers=True,
                    title=f"Trend of Selected Decline Reasons ({time_slot} Slot)",
                    labels={time_col: "Date / Time Slot", "slot_count": "Declines Count", "_reason_clean": "Reason"}
                )
                st.plotly_chart(fig_reason_trend, use_container_width=True)

        # --- 5. ORIGINAL PEAK SPIKE DAY DRILL-DOWN ANALYTICS ---
        st.markdown("---")
        st.subheader("🚨 Peak Spike Day Drill-Down Analytics")
        if not failed_df.empty:
            spike_daily = failed_df.groupby("_date").size().reset_index(name='failed_count')
            peak_spike = spike_daily.sort_values(by=['failed_count', '_date'], ascending=[False, False]).iloc[0]
            spike_date = peak_spike['_date']
            spike_count = peak_spike['failed_count']
            
            st.error(f"Peak Spike Date Detected: **{pd.Timestamp(spike_date).strftime('%Y-%m-%d')}** | Total Declines on Spike: **{spike_count:,}**")
            
            spike_df = failed_df[failed_df['_date'] == spike_date]
            
            p1, p2, p3 = st.columns(3)
            with p1:
                if RESIDENCE_COL in spike_df.columns:
                    uc_spike = spike_df[RESIDENCE_COL].value_counts().head(5).reset_index()
                    uc_spike.columns = [RESIDENCE_COL, "count"]
                    fig_p1 = px.pie(uc_spike, values="count", names=RESIDENCE_COL, title=f"User Country on Spike ({spike_date})")
                    st.plotly_chart(fig_p1, use_container_width=True)
            with p2:
                if CARD_COUNTRY_COL in spike_df.columns:
                    cc_spike = spike_df[CARD_COUNTRY_COL].value_counts().head(5).reset_index()
                    cc_spike.columns = [CARD_COUNTRY_COL, "count"]
                    fig_p2 = px.pie(cc_spike, values="count", names=CARD_COUNTRY_COL, title=f"Card Country on Spike ({spike_date})")
                    st.plotly_chart(fig_p2, use_container_width=True)
            with p3:
                if PM_TYPE_COL in spike_df.columns:
                    pm_spike = spike_df[PM_TYPE_COL].value_counts().head(5).reset_index()
                    pm_spike.columns = [PM_TYPE_COL, "count"]
                    fig_p3 = px.pie(pm_spike, values="count", names=PM_TYPE_COL, title=f"Payment Method on Spike ({spike_date})")
                    st.plotly_chart(fig_p3, use_container_width=True)

        # --- EXPORT REPORT ---
        st.markdown("---")
        st.subheader("📄 Top-Up Declined Transactions Detail Table")
        
        export_cols = [
            c for c in [
                MERCHANT_ID_COL, USER_COL, VERIF_TYPE_COL, PM_TYPE_COL, TRANSFER_ID_COL,
                DATE_COL, AMOUNT_COL, STATUS_COL, REJECT_REASON_COL, RESIDENCE_COL, CARD_COUNTRY_COL
            ] if c in df.columns
        ]
        
        export_df = failed_df[export_cols].copy() if not failed_df.empty else pd.DataFrame()
        st.dataframe(export_df, use_container_width=True)

        if not export_df.empty:
            out_topup = io.BytesIO()
            with pd.ExcelWriter(out_topup, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, sheet_name="Top-Up Declines", index=False)
                if 'top5_reasons_df' in locals():
                    top5_reasons_df.to_excel(writer, sheet_name="Top 5 Reasons", index=False)
                if 'filtered_u_amt' in locals():
                    filtered_u_amt.to_excel(writer, sheet_name="Top Users by Amount", index=False)
                if 'filtered_u_cnt' in locals():
                    filtered_u_cnt.to_excel(writer, sheet_name="Top Users by Count", index=False)

            st.download_button(
                label="📥 Export Top-Up Analytics Report (Excel)",
                data=out_topup.getvalue(),
                file_name="TopUp_Declined_Analytics_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
