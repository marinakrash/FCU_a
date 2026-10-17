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
        st.write("Full pipeline for detecting merchant decline spikes, deep rule coverage analysis, and generating Excel/ZIP reports.")
        
        up_file = st.file_uploader("Upload SEON Acquiring Export (Excel/CSV):", type=["xlsx", "csv"])
        if up_file:
            df = load_data(up_file)
            st.success(f"Loaded records from file: {len(df):,}")
            
            # --- 1. SETTINGS & COLUMN NAMES ---
            MONITORING_DAYS = 7
            MIN_BASELINE_ACTIVE_DAYS = 21
            DECLINE_THRESHOLD = 0.06
            TOP_MERCHANTS_LIMIT = 10
            TOP_RULES_LIMIT = 10
            TOP_EMAILS_LIMIT = 10

            DATE_COLUMN = "date"
            STATE_COLUMN = "state"
            MERCHANT_COLUMN = "merchant_id"
            MERCHANT_NAME_COLUMN = "user_fullname"
            TRANSACTION_ID_COLUMN = "transaction_id"
            RULES_COLUMN = "applied_rule_ids"
            EMAIL_COLUMN = "email/email"
            DECLINE_STATE = "DECLINE"

            # Validate Required Columns
            required_cols = [DATE_COLUMN, STATE_COLUMN, MERCHANT_COLUMN, TRANSACTION_ID_COLUMN, RULES_COLUMN, EMAIL_COLUMN, MERCHANT_NAME_COLUMN]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns in upload: {missing_cols}")
            else:
                # --- 2. CLEAN DATA ---
                df_clean = df.copy()
                df_clean["_date_day"] = pd.to_datetime(df_clean[DATE_COLUMN].astype(str).str[:10], errors="coerce").dt.normalize()
                df_clean[STATE_COLUMN] = df_clean[STATE_COLUMN].fillna("").astype(str).str.strip().str.upper()
                df_clean[MERCHANT_COLUMN] = df_clean[MERCHANT_COLUMN].fillna("").astype(str).str.strip()
                df_clean[MERCHANT_NAME_COLUMN] = df_clean[MERCHANT_NAME_COLUMN].fillna("").astype(str).str.strip()
                df_clean[TRANSACTION_ID_COLUMN] = df_clean[TRANSACTION_ID_COLUMN].fillna("").astype(str).str.strip()

                df_clean = df_clean[df_clean["_date_day"].notna() & df_clean[MERCHANT_COLUMN].ne("")].copy()

                df_clean["_transaction_key"] = df_clean[TRANSACTION_ID_COLUMN]
                missing_tx_id = df_clean["_transaction_key"].eq("")
                df_clean.loc[missing_tx_id, "_transaction_key"] = [f"_missing_tx_{i}" for i in df_clean.index[missing_tx_id]]

                df_clean = df_clean.drop_duplicates(subset=["_transaction_key"], keep="first").copy()
                df_clean = df_clean[df_clean[STATE_COLUMN].isin(["APPROVE", "DECLINE"])].copy()

                latest_date = df_clean["_date_day"].max()
                monitoring_start = latest_date - pd.Timedelta(days=MONITORING_DAYS - 1)
                baseline_end = monitoring_start - pd.Timedelta(days=1)

                monitoring_df = df_clean[(df_clean["_date_day"] >= monitoring_start) & (df_clean["_date_day"] <= latest_date)].copy()
                baseline_df = df_clean[df_clean["_date_day"] < monitoring_start].copy()

                st.info(f"📅 **Monitoring Window:** {monitoring_start.date()} to {latest_date.date()} | **Baseline Period Ends:** {baseline_end.date()}")

                # --- 3. CALCULATE BASELINE & DAILY STATS ---
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
                daily_stats = daily_stats.merge(eligible_merchants[[MERCHANT_COLUMN, "baseline_transactions", "baseline_declines", "baseline_active_days", "baseline_decline_rate"]], on=MERCHANT_COLUMN, how="left")

                # --- 4. IDENTIFY SPIKES & RANK TOP 10 ---
                spikes = daily_stats[(daily_stats["decline_rate"] > DECLINE_THRESHOLD) & (daily_stats["decline_rate"] > daily_stats["baseline_decline_rate"])].copy()

                if spikes.empty:
                    st.warning("No spikes detected with decline rate > 6% and strictly above merchant baseline.")
                else:
                    spikes["increase_pp"] = (spikes["decline_rate"] - spikes["baseline_decline_rate"]) * 100
                    spikes["relative_increase"] = np.where(
                        spikes["baseline_decline_rate"] > 0,
                        (spikes["decline_rate"] - spikes["baseline_decline_rate"]) / spikes["baseline_decline_rate"],
                        np.inf
                    )
                    spikes["relative_increase_pct"] = spikes["relative_increase"] * 100
                    spikes["spike_date"] = spikes["_date_day"]

                    rank_rows = []
                    for merchant_id, merchant_spikes in spikes.groupby(MERCHANT_COLUMN):
                        merchant_spikes = merchant_spikes.sort_values(["relative_increase", "increase_pp", "decline_rate"], ascending=False)
                        max_spike = merchant_spikes.iloc[0]

                        name_values = monitoring_eligible.loc[monitoring_eligible[MERCHANT_COLUMN] == merchant_id, MERCHANT_NAME_COLUMN].dropna().astype(str)
                        name_values = name_values[name_values.str.strip().ne("")]
                        merchant_name = name_values.mode().iloc[0] if not name_values.empty else ""

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

                    top_merchants = pd.DataFrame(rank_rows).sort_values(["max_relative_increase", "max_spike_increase_pp"], ascending=False).head(TOP_MERCHANTS_LIMIT).reset_index(drop=True)
                    top_merchants.insert(0, "rank", range(1, len(top_merchants) + 1))

                    top_merchant_ids = set(top_merchants[MERCHANT_COLUMN].astype(str))
                    top_spikes = spikes[spikes[MERCHANT_COLUMN].astype(str).isin(top_merchant_ids)].copy()

                    rank_map = dict(zip(top_merchants[MERCHANT_COLUMN].astype(str), top_merchants["rank"]))
                    name_map = dict(zip(top_merchants[MERCHANT_COLUMN].astype(str), top_merchants[MERCHANT_NAME_COLUMN]))
                    top_spikes["rank"] = top_spikes[MERCHANT_COLUMN].astype(str).map(rank_map)
                    top_spikes[MERCHANT_NAME_COLUMN] = top_spikes[MERCHANT_COLUMN].astype(str).map(name_map)

                    st.subheader("🏆 Top 10 Spiking Merchants Summary")
                    st.dataframe(top_merchants, use_container_width=True)

                    # --- EXPORT GENERAL REPORT ---
                    top_merchants_export = top_merchants.copy()
                    top_merchants_export["baseline_decline_rate"] *= 100
                    top_merchants_export["max_spike_decline_rate"] *= 100
                    top_merchants_export = top_merchants_export.rename(columns={"baseline_decline_rate": "baseline_decline_rate_pct", "max_spike_decline_rate": "max_spike_decline_rate_pct"})

                    spike_dates_export = top_spikes[["rank", MERCHANT_COLUMN, MERCHANT_NAME_COLUMN, "spike_date", "total_transactions", "declined_transactions", "decline_rate", "baseline_decline_rate", "increase_pp", "relative_increase_pct"]].copy()
                    spike_dates_export["decline_rate"] *= 100
                    spike_dates_export["baseline_decline_rate"] *= 100
                    spike_dates_export = spike_dates_export.rename(columns={"spike_date": "date", "decline_rate": "decline_rate_pct", "baseline_decline_rate": "baseline_decline_rate_pct", "increase_pp": "increase_percentage_points"}).sort_values(["rank", "date"])

                    output_gen = io.BytesIO()
                    with pd.ExcelWriter(output_gen, engine="xlsxwriter") as writer:
                        top_merchants_export.to_excel(writer, sheet_name="Top 10 Merchants", index=False)
                        spike_dates_export.to_excel(writer, sheet_name="Spike Dates", index=False)

                    st.download_button(
                        label="📥 Download General Report (SEON_Top10_Merchant_Decline_Spikes.xlsx)",
                        data=output_gen.getvalue(),
                        file_name="SEON_Top10_Merchant_Decline_Spikes.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    # --- TIMELINE CHARTS ---
                    st.subheader("📈 Historical Timeline Charts (Top 10 Merchants)")
                    min_date = df_clean["_date_day"].min()
                    full_dates = pd.date_range(start=min_date, end=latest_date, freq="D")

                    all_daily_stats = df_clean[df_clean[MERCHANT_COLUMN].isin(eligible_ids)].groupby([MERCHANT_COLUMN, "_date_day"]).agg(
                        total_transactions=("_transaction_key", "nunique"),
                        declined_transactions=(STATE_COLUMN, lambda s: (s == DECLINE_STATE).sum()),
                    ).reset_index()
                    all_daily_stats["decline_rate"] = all_daily_stats["declined_transactions"] / all_daily_stats["total_transactions"]

                    for _, merchant_row in top_merchants.iterrows():
                        m_id = str(merchant_row[MERCHANT_COLUMN])
                        m_name = str(merchant_row[MERCHANT_NAME_COLUMN])
                        m_daily = all_daily_stats[all_daily_stats[MERCHANT_COLUMN].astype(str) == m_id].copy().set_index("_date_day")
                        chart_rates = m_daily["decline_rate"].reindex(full_dates)

                        fig = px.line(x=full_dates, y=chart_rates * 100, title=f"Merchant: {m_name} | ID: {m_id}")
                        fig.add_vrect(x0=monitoring_start, x1=latest_date, fillcolor="rgba(255,0,0,0.15)", line_width=0, annotation_text="Monitoring Window")
                        fig.add_hline(y=float(merchant_row["baseline_decline_rate"]) * 100, line_dash="dash", line_color="green", annotation_text=f"Baseline ({float(merchant_row['baseline_decline_rate']) * 100:.2f}%)")
                        fig.add_hline(y=6.0, line_dash="dot", line_color="red", annotation_text="Threshold 6%")
                        st.plotly_chart(fig, use_container_width=True)

                    # --- DEEP ANALYSIS SECTION FOR SELECTED MERCHANTS ---
                    st.markdown("---")
                    st.subheader("🔬 Selected Merchants Deep Analysis")
                    
                    available_options = top_merchants[MERCHANT_COLUMN].astype(str).tolist()
                    selected_m_ids = st.multiselect(
                        "Select Merchant IDs for Deep Analysis:", 
                        options=available_options, 
                        default=available_options[:1] if available_options else []
                    )

                    if selected_m_ids:
                        with st.spinner("Processing deep analytics and generating rule charts..."):
                            rule_results, coverage_results, email_results, selected_spike_rows = [], [], [], []
                            zip_buffer = io.BytesIO()

                            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                                for sel_id in selected_m_ids:
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

                                    # Coverage Analysis
                                    m_spike_declines["_parsed_rules"] = m_spike_declines[RULES_COLUMN].apply(parse_rule_ids)
                                    m_spike_declines["_has_rule"] = m_spike_declines["_parsed_rules"].apply(len) > 0

                                    decl_with_rules = m_spike_declines.loc[m_spike_declines["_has_rule"], "_transaction_key"].nunique()
                                    decl_no_rules = m_spike_declines.loc[~m_spike_declines["_has_rule"], "_transaction_key"].nunique()

                                    coverage_results.append({
                                        "rank": m_rank, MERCHANT_COLUMN: sel_id, MERCHANT_NAME_COLUMN: m_name,
                                        "spike_dates": spike_dates_label, "total_declined_transactions": total_declines,
                                        "declined_transactions_with_rules": decl_with_rules, "declined_transactions_without_rules": decl_no_rules,
                                        "with_rules_pct": decl_with_rules / total_declines * 100 if total_declines else 0,
                                        "without_rules_pct": decl_no_rules / total_declines * 100 if total_declines else 0
                                    })

                                    # Top Emails
                                    email_data = m_spike_declines.copy()
                                    email_data["email_for_analysis"] = email_data[EMAIL_COLUMN].fillna("").astype(str).str.strip().replace("", "[Missing email]")
                                    email_counts = email_data.groupby("email_for_analysis")["_transaction_key"].nunique().sort_values(ascending=False).head(TOP_EMAILS_LIMIT).reset_index(name="declined_transactions")
                                    email_counts.insert(0, "rank", m_rank)
                                    email_counts[MERCHANT_COLUMN] = sel_id
                                    email_counts[MERCHANT_NAME_COLUMN] = m_name
                                    email_counts["spike_dates"] = spike_dates_label
                                    email_results.append(email_counts)

                                    # Top Rules
                                    rule_data = m_spike_declines[["_transaction_key", "_parsed_rules"]].explode("_parsed_rules").dropna()
                                    rule_data = rule_data[rule_data["_parsed_rules"].astype(str).str.strip().ne("")].drop_duplicates()

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

                                        # Generate Chart for ZIP
                                        fig, ax = plt.subplots(figsize=(10, 5))
                                        chart_data = rule_counts.sort_values("declined_transactions_with_rule", ascending=True)
                                        ax.barh(chart_data["rule_name"].astype(str), chart_data["declined_transactions_with_rule"], color="#2b5c8f")
                                        ax.set_title(f"Top {TOP_RULES_LIMIT} Rules — Merchant: {m_name} ({sel_id})\nDates: {spike_dates_label}")
                                        plt.tight_layout()

                                        img_buf = io.BytesIO()
                                        plt.savefig(img_buf, format="png", dpi=150)
                                        plt.close(fig)
                                        
                                        zip_file.writestr(f"merchant_{sel_id}_top_rules.png", img_buf.getvalue())

                            zip_buffer.seek(0)

                            # Deep Analysis Tables
                            deep_spike_df = pd.DataFrame(selected_spike_rows)
                            deep_coverage_df = pd.DataFrame(coverage_results)
                            deep_rules_df = pd.concat(rule_results, ignore_index=True) if rule_results else pd.DataFrame()
                            deep_emails_df = pd.concat(email_results, ignore_index=True) if email_results else pd.DataFrame()

                            st.write("#### Deep Analysis Summary Preview")
                            if not deep_rules_df.empty: 
                                st.dataframe(deep_rules_df, use_container_width=True)

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
                                    label="📥 Download Deep Analysis Excel",
                                    data=output_deep.getvalue(),
                                    file_name="SEON_Selected_Merchants_Deep_Analysis.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            with col_dl2:
                                st.download_button(
                                    label="📦 Download Rule Charts ZIP",
                                    data=zip_buffer.getvalue(),
                                    file_name="Selected_Merchant_Rule_Charts.zip",
                                    mime="application/zip"
                                )
                            
    elif sub_module == "2.2 Single Merchant Historical Analysis":
        st.title("🔍 Single Merchant Deep Analysis")
        up_file = st.file_uploader("Upload SEON Acquiring Export:", type=["xlsx", "csv"])
        if up_file:
            df = load_data(up_file)
            
            CARD_COUNTRY_COL = "bin_country_code" if "bin_country_code" in df.columns else "card_country"
            IP_COUNTRY_COL = "ip_country"
            EMAIL_COL = "email/email"
            
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
            df = df.dropna(subset=["merchant_id", "date"]).copy()
            df["merchant_id"] = df["merchant_id"].astype(str).str.strip()
            df["state"] = df["state"].astype(str).str.strip().str.upper()
            
            unique_merchants = sorted(df["merchant_id"].unique())
            selected_m = st.selectbox("Select Merchant ID to Analyze:", unique_merchants)
            
            merchant_df = df[df["merchant_id"] == selected_m].copy()
            merchant_df["is_decline"] = merchant_df["state"].eq("DECLINE")
            merchant_df["is_approve"] = merchant_df["state"].eq("APPROVE")
            
            daily = merchant_df.groupby("date").agg(
                total_transactions=("state", "size"),
                declined_transactions=("is_decline", "sum"),
                approved_transactions=("is_approve", "sum")
            ).reset_index().sort_values("date")
            
            daily["decline_rate_pct"] = (daily["declined_transactions"] / daily["total_transactions"]) * 100
            active_daily = daily[daily["total_transactions"] > 0].copy()
            decline_baseline = float(active_daily["decline_rate_pct"].mean()) if not active_daily.empty else 0.0
            
            daily["decline_change_pct"] = daily["decline_rate_pct"].apply(lambda x: relative_change(x, decline_baseline))
            daily["decline_spike"] = (daily["decline_rate_pct"] >= 6.0) & (daily["decline_change_pct"] >= 50.0)
            
            fig = px.line(daily, x="date", y="decline_rate_pct", title=f"Historical Decline Rate — Merchant: {selected_m}")
            fig.add_hline(y=decline_baseline, line_dash="dash", line_color="green", annotation_text=f"Baseline ({decline_baseline:.2f}%)")
            fig.add_hline(y=6.0, line_dash="dot", line_color="red", annotation_text="Min 6%")
            st.plotly_chart(fig, use_container_width=True)
            
            # --- TOP RULE TRIGGER RATIO SPIKES (TOP 5) ---
            st.subheader("📌 Top 5 Rule Trigger Ratio Spikes")
            rule_source = merchant_df[["date", "state", "applied_rule_ids"]].copy() if "applied_rule_ids" in merchant_df.columns else pd.DataFrame()
            if not rule_source.empty:
                rule_source["rule_id"] = rule_source["applied_rule_ids"].apply(parse_rule_ids)
                rule_long = rule_source.explode("rule_id").dropna(subset=["rule_id"]).copy()
                rule_long["is_decline"] = rule_long["state"].eq("DECLINE")
                
                rule_daily = rule_long.groupby(["date", "rule_id"]).agg(
                    rule_trigger_count=("state", "size"),
                    rule_declined_count=("is_decline", "sum")
                ).reset_index()
                
                rule_daily["rule_name"] = rule_daily["rule_id"].apply(get_rule_name)
                rule_summary = rule_daily.groupby(["rule_id", "rule_name"])["rule_trigger_count"].sum().nlargest(5).reset_index()
                st.dataframe(rule_summary, use_container_width=True)
            
            # --- HEATMAP: CARD VS IP COUNTRY ---
            if CARD_COUNTRY_COL in merchant_df.columns and IP_COUNTRY_COL in merchant_df.columns:
                st.subheader("🌐 Card Country vs IP Country Matrix (Top 10)")
                ct = pd.crosstab(merchant_df[CARD_COUNTRY_COL].fillna("Unknown"), merchant_df[IP_COUNTRY_COL].fillna("Unknown"))
                top_cards = merchant_df[CARD_COUNTRY_COL].fillna("Unknown").value_counts().head(10).index
                top_ips = merchant_df[IP_COUNTRY_COL].fillna("Unknown").value_counts().head(10).index
                ct_filtered = ct.loc[ct.index.isin(top_cards), ct.columns.isin(top_ips)]
                
                fig_hm = px.imshow(ct_filtered, text_auto=True, color_continuous_scale="YlGnBu", title="Country Cross-Analysis Matrix")
                st.plotly_chart(fig_hm, use_container_width=True)
            
            # --- TOP PAYER EMAILS DURING LATEST DECLINE SPIKE ---
            spike_dates = daily.loc[daily["decline_spike"], "date"].dropna()
            if not spike_dates.empty:
                st.subheader("✉️ Top Payer Emails During Latest Spike")
                latest_spike = spike_dates.max()
                spike_tx = merchant_df[(merchant_df["date"] == latest_spike) & (merchant_df["state"] == "DECLINE")]
                if EMAIL_COL in spike_tx.columns:
                    top_emails = spike_tx[EMAIL_COL].value_counts().head(10).reset_index()
                    top_emails.columns = ["Email", "Declined Transactions"]
                    st.dataframe(top_emails, use_container_width=True)

            output_merchant = io.BytesIO()
            with pd.ExcelWriter(output_merchant, engine="xlsxwriter") as writer:
                daily.to_excel(writer, sheet_name="Daily Merchant Metrics", index=False)
                if 'rule_summary' in locals():
                    rule_summary.to_excel(writer, sheet_name="Top 5 Rule Spikes", index=False)
            st.download_button(
                label=f"📥 Download Deep Analysis Report ({selected_m})",
                data=output_merchant.getvalue(),
                file_name=f"SEON_Analysis_{selected_m}.xlsx",
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
    st.write("Monitoring wallet top-ups, decline reasons, payment methods, geographic routes, and peak spike days.")
    
    up_file = st.file_uploader("Upload Top-Up Transaction Export (XLSX/CSV):", type=["xlsx", "csv"])
    if up_file:
        df = load_data(up_file)
        
        df['datetime_parsed'] = pd.to_datetime(df['transaction_date'], errors='coerce') if 'transaction_date' in df else pd.to_datetime(df['date'], errors='coerce')
        df['date_only'] = df['datetime_parsed'].dt.date
        
        df['state_clean'] = df['state'].astype(str).str.strip().str.capitalize()
        df['is_failed'] = df['state_clean'] == 'Failed'
        df['is_accepted'] = df['state_clean'] == 'Accepted'
        
        resp_series = df['response_information'].astype(str).str.lower() if 'response_information' in df else pd.Series("", index=df.index)
        df['is_seon_failed'] = df['is_failed'] & resp_series.str.contains('security reason', regex=False)
        
        if 'converted_amount' in df.columns:
            df['amount_num'] = pd.to_numeric(df['converted_amount'].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce').fillna(0.0)
        else:
            df['amount_num'] = 0.0

        total_tx = len(df)
        accepted_tx = df['is_accepted'].sum()
        failed_tx = df['is_failed'].sum()
        seon_failed_tx = df['is_seon_failed'].sum()
        failed_amount = df[df['is_failed']]['amount_num'].sum()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Top-Up Tx", f"{total_tx:,}")
        c2.metric("Accepted Tx", f"{accepted_tx:,}")
        c3.metric("Failed Tx", f"{failed_tx:,}")
        c4.metric("Loss Amount (Failed)", f"{failed_amount:,.2f} EUR")
        
        st.subheader("📈 Overall Decline Rate vs. SEON Decline Rate Trend")
        daily_stats = df.groupby('date_only').agg(
            accepted=('is_accepted', 'sum'),
            failed=('is_failed', 'sum'),
            seon_failed=('is_seon_failed', 'sum')
        ).reset_index()
        
        daily_stats['finished'] = daily_stats['accepted'] + daily_stats['failed']
        daily_stats['overall_decline_rate'] = np.where(daily_stats['finished'] > 0, (daily_stats['failed'] / daily_stats['finished']) * 100, 0.0)
        daily_stats['seon_decline_rate'] = np.where(daily_stats['finished'] > 0, (daily_stats['seon_failed'] / daily_stats['finished']) * 100, 0.0)
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=daily_stats['date_only'], y=daily_stats['overall_decline_rate'], name="Overall Decline Rate (%)", line=dict(color='#e74c3c', width=3)))
        fig_trend.add_trace(go.Scatter(x=daily_stats['date_only'], y=daily_stats['seon_decline_rate'], name="SEON Decline Rate (%)", line=dict(color='#2980b9', width=2, dash='dash')))
        st.plotly_chart(fig_trend, use_container_width=True)
        
        failed_df = df[df['is_failed']]
        
        if not failed_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                if 'user_country' in failed_df.columns:
                    top_user_c = failed_df['user_country'].value_counts().head(8).reset_index()
                    top_user_c.columns = ['user_country', 'count']
                    fig1 = px.bar(top_user_c, x="count", y="user_country", orientation="h", title="Top User Countries (Declines)")
                    st.plotly_chart(fig1, use_container_width=True)
            with col2:
                if 'card_country' in failed_df.columns:
                    top_card_c = failed_df['card_country'].value_counts().head(8).reset_index()
                    top_card_c.columns = ['card_country', 'count']
                    fig2 = px.bar(top_card_c, x="count", y="card_country", orientation="h", title="Top Card Issuer Countries (Declines)")
                    st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if 'user_country' in failed_df.columns and 'card_country' in failed_df.columns:
                    spike_geo = failed_df.groupby(['user_country', 'card_country']).size().reset_index(name='Failed_Count').sort_values(by='Failed_Count', ascending=False).head(8)
                    spike_geo['route'] = spike_geo['user_country'] + ' -> ' + spike_geo['card_country']
                    fig_route = px.bar(spike_geo, x="Failed_Count", y="route", orientation="h", title="Top Routes (User -> Card Country)")
                    st.plotly_chart(fig_route, use_container_width=True)
            with col4:
                pm_col = 'payment_method_name' if 'payment_method_name' in failed_df.columns else 'payment_method_title'
                if pm_col in failed_df.columns:
                    top_pm = failed_df[pm_col].value_counts().head(8).reset_index()
                    top_pm.columns = ['method', 'count']
                    fig_pm = px.bar(top_pm, x="count", y="method", orientation="h", title="Top Payment Methods (Declines)")
                    st.plotly_chart(fig_pm, use_container_width=True)

            st.markdown("---")
            st.subheader("🚨 Peak Spike Day Drill-Down Analytics")
            spike_daily = failed_df.groupby('date_only').size().reset_index(name='failed_count')
            peak_spike = spike_daily.sort_values(by=['failed_count', 'date_only'], ascending=[False, False]).iloc[0]
            spike_date = peak_spike['date_only']
            spike_count = peak_spike['failed_count']
            
            st.error(f"Peak Spike Date Detected: **{spike_date}** | Total Declines on Spike: **{spike_count:,}**")
            
            spike_df = failed_df[failed_df['date_only'] == spike_date]
            
            p1, p2, p3 = st.columns(3)
            with p1:
                if 'user_country' in spike_df.columns:
                    uc_spike = spike_df['user_country'].value_counts().head(5).reset_index()
                    fig_p1 = px.pie(uc_spike, values="count", names="user_country", title=f"User Country on Spike ({spike_date})")
                    st.plotly_chart(fig_p1, use_container_width=True)
            with p2:
                if 'card_country' in spike_df.columns:
                    cc_spike = spike_df['card_country'].value_counts().head(5).reset_index()
                    fig_p2 = px.pie(cc_spike, values="count", names="card_country", title=f"Card Country on Spike ({spike_date})")
                    st.plotly_chart(fig_p2, use_container_width=True)
            with p3:
                pm_col_title = 'payment_method_title' if 'payment_method_title' in spike_df.columns else 'payment_method_name'
                if pm_col_title in spike_df.columns:
                    pm_spike = spike_df[pm_col_title].value_counts().head(5).reset_index()
                    fig_p3 = px.pie(pm_spike, values="count", names=pm_col_title, title=f"Payment Method on Spike ({spike_date})")
                    st.plotly_chart(fig_p3, use_container_width=True)

        output_topup = io.BytesIO()
        with pd.ExcelWriter(output_topup, engine="xlsxwriter") as writer:
            daily_stats.to_excel(writer, sheet_name="Daily Trends", index=False)
            if 'spike_geo' in locals():
                spike_geo.to_excel(writer, sheet_name="Top Geo Routes", index=False)
        st.download_button(
            label="📥 Download Top-Up Monitoring Analytics Report (Excel)",
            data=output_topup.getvalue(),
            file_name="TopUp_Monitoring_Analytics_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
