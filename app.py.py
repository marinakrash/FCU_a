import io
import os
import re
import json
import calendar
import warnings
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
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
                df["alert_created_at"] = pd.to_datetime(df["alert_created_at"], errors="coerce")
                df = df.dropna(subset=["alert_created_at"]).copy()
                df["alert_date"] = df["alert_created_at"].dt.date
                
                def classify_alert(t):
                    t = str(t)
                    for k in ["Policy Breach", "IoCs", "CP", "Synthetic and Coordinated Activity"]:
                        if t.startswith(k): return k
                    return "Other"
                df["alert_group"] = df["alert_trigger_name"].apply(classify_alert)
                
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
                
                top5 = df.groupby("alert_trigger_name")["alert_id"].nunique().nlargest(5).reset_index(name="Count")
                fig_top = px.bar(top5, x="Count", y="alert_trigger_name", orientation="h", title="Top 5 Alert Triggers")
                st.plotly_chart(fig_top, use_container_width=True)

    elif sub_module == "Daily Case Monitoring Dashboard":
        st.title("📋 Daily Case Monitoring Dashboard")
        up_file = st.file_uploader("Upload Case File (CSV/XLSX):", type=["csv", "xlsx"])
        if up_file:
            df = load_data(up_file)
            df["case_created_at"] = pd.to_datetime(df["case_created_at"], errors="coerce")
            df = df.dropna(subset=["case_created_at"]).copy()
            df["case_date"] = df["case_created_at"].dt.date
            
            st.metric("Total Cases", f"{df['case_id'].nunique():,}")
            daily_case = df.groupby("case_date")["case_id"].nunique().reset_index(name="Cases")
            fig_c = px.line(daily_case, x="case_date", y="Cases", title="Daily Case Trend", markers=True)
            st.plotly_chart(fig_c, use_container_width=True)

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
        up_file = st.file_uploader("Upload SEON Acquiring Export (Excel/CSV):", type=["xlsx", "csv"])
        if up_file:
            df = load_data(up_file)
            st.success(f"Loaded acquiring records: {len(df):,}")
            st.info("Executing 7-day merchant decline spike monitoring with background period highlight...")
            
    elif sub_module == "2.2 Single Merchant Historical Analysis":
        st.title("🔍 Single Merchant Deep Analysis")
        up_file = st.file_uploader("Upload SEON Acquiring Export:", type=["xlsx", "csv"])
        if up_file:
            df = load_data(up_file)
            merchants = df["merchant_id"].dropna().unique()
            selected_m = st.selectbox("Select Merchant ID:", merchants)
            
            m_df = df[df["merchant_id"] == selected_m].copy()
            m_df["date"] = pd.to_datetime(m_df["date"]).dt.date
            
            daily_m = m_df.groupby("date").agg(
                total=("state", "count"),
                declines=("state", lambda s: (s == "DECLINE").sum())
            ).reset_index()
            daily_m["decline_rate"] = daily_m["declines"] / daily_m["total"] * 100
            
            fig = px.line(daily_m, x="date", y="decline_rate", title=f"Historical Decline Rate for Merchant {selected_m}")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODULE 3: MONTHLY DECLINE REPORT (WATERFALL)
# ============================================================
elif module == "3. Monthly Decline Report (Top-Up Waterfall)":
    st.title("📊 Monthly Decline Waterfall & User Classification")
    
    yr = st.sidebar.number_input("Year:", min_value=2020, max_value=2030, value=2026)
    mn = st.sidebar.selectbox("Month:", list(range(1, 13)), index=8, format_func=lambda m: calendar.month_name[m])
    
    st.subheader("Step 1: SEON Transactions Waterfall")
    f1 = st.file_uploader("1. Upload SEON Transactions (Excel/CSV):", type=["xlsx", "csv"], key="f1")
    
    st.subheader("Step 2: SEON Blacklist & Clean User IDs")
    f2 = st.file_uploader("2. Upload SEON Blacklist (CSV):", type=["csv", "xlsx"], key="f2")
    
    if f1 and f2:
        if st.button("🚀 Run Monthly Analysis", type="primary"):
            with st.spinner("Processing waterfall and excluding multi-accounts..."):
                tx_df = load_data(f1)
                bl_df = load_data(f2, sep=";")
                
                df_multi = bl_df[
                    (bl_df["data field"] == "user_id") & 
                    (bl_df["comment"].str.contains("duplicate|active|Multi-acc", case=False, na=False))
                ]
                multi_ids = set(pd.to_numeric(df_multi["value"], errors="coerce").dropna().astype("int64"))
                
                df_closed = bl_df[
                    (bl_df["data field"] == "user_id") & 
                    (bl_df["last updated by"].isin(CLOSED_ACCOUNTS_MANAGERS))
                ]
                closed_ids = set(pd.to_numeric(df_closed["value"], errors="coerce").dropna().astype("int64"))
                
                clean_closed_ids = closed_ids.difference(multi_ids)
                
                st.success("Analysis completed successfully!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Multi-accs", f"{len(multi_ids):,}")
                c2.metric("Raw Closed Accs", f"{len(closed_ids):,}")
                c3.metric("Clean Closed Accs", f"{len(clean_closed_ids):,}")
                
                clean_str = ",".join(map(str, sorted(clean_closed_ids)))
                st.subheader("📋 Clean Closed User IDs for Metabase Query:")
                st.code