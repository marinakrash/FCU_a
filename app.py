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
        s = s.strip("[](){}")
        values = s.replace(";", ",").split(",")

    result = []
    for item in values:
        rule_id = clean_rule_id(item)
        if rule_id and rule_id.lower() not in {"nan", "none", "null"}:
            if rule_id not in result:
                result.append(rule_id)
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
                
                col_a, col_b = st.columns(2)
                with col_a:
                    top5 = df.groupby("alert_trigger_name")["alert_id"].nunique().nlargest(5).reset_index(name="Count")
                    fig_top = px.bar(top5, x="Count", y="alert_trigger_name", orientation="h", title="Top 5 Alert Triggers")
                    st.plotly_chart(fig_top, use_container_width=True)
                with col_b:
                    grp_counts = df.groupby("alert_group")["alert_id"].nunique().reset_index(name="Count")
                    fig_grp = px.pie(grp_counts, values="Count", names="alert_group", title="Alerts by Group")
                    st.plotly_chart(fig_grp, use_container_width=True)

                # EXPORT BUTTON FOR ALERTS
                output_alerts = io.BytesIO()
                with pd.ExcelWriter(output_alerts, engine="xlsxwriter") as writer:
                    daily.to_excel(writer, sheet_name="Daily Summary", index=False)
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
            df["case_created_at"] = pd.to_datetime(df["case_created_at"], errors="coerce")
            df = df.dropna(subset=["case_created_at"]).copy()
            df["case_date"] = df["case_created_at"].dt.date
            
            def classify_case(t):
                t = str(t)
                for k in ["Policy Breach", "IoCs", "CP", "Synthetic and Coordinated Activity", "ACQ_"]:
                    if t.startswith(k): return k
                return "Other"
            
            if "case_title" in df.columns:
                df["case_group"] = df["case_title"].apply(classify_case)
            
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

            # EXPORT BUTTON FOR CASES
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
        up_file = st.file_uploader("Upload SEON Acquiring Export (Excel/CSV):", type=["xlsx", "csv"])
        if up_file:
            df = load_data(up_file)
            st.success(f"Loaded acquiring records: {len(df):,}")
            
            df["_date_day"] = pd.to_datetime(df["date"].astype(str).str[:10], errors="coerce").dt.normalize()
            df["state"] = df["state"].fillna("").astype(str).str.strip().str.upper()
            df = df[df["state"].isin(["APPROVE", "DECLINE"])].copy()
            
            latest_date = df["_date_day"].max()
            monitoring_start = latest_date - pd.Timedelta(days=6)
            
            daily_m = df.groupby(["merchant_id", "_date_day"]).agg(
                total=("transaction_id", "nunique"),
                declines=("state", lambda s: (s == "DECLINE").sum())
            ).reset_index()
            daily_m["decline_rate"] = daily_m["declines"] / daily_m["total"] * 100
            
            merchants = daily_m["merchant_id"].unique()[:10]
            for m_id in merchants:
                m_data = daily_m[daily_m["merchant_id"] == m_id]
                fig = px.line(m_data, x="_date_day", y="decline_rate", title=f"Merchant {m_id} - Decline Rate Timeline")
                fig.add_vrect(
                    x0=monitoring_start, x1=latest_date,
                    fillcolor="rgba(255,0,0,0.15)", line_width=0,
                    annotation_text="Monitoring Window (Last 7 Days)"
                )
                st.plotly_chart(fig, use_container_width=True)

            # EXPORT BUTTON FOR SPIKES REPORT
            output_spikes = io.BytesIO()
            with pd.ExcelWriter(output_spikes, engine="xlsxwriter") as writer:
                daily_m.to_excel(writer, sheet_name="Daily Merchants Spike", index=False)
            st.download_button(
                label="📥 Download Top 10 Merchant Spikes Report (Excel)",
                data=output_spikes.getvalue(),
                file_name="SEON_Top10_Merchant_Decline_Spikes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
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
            
            # --- CARD vs IP COUNTRY CROSS MATRIX ---
            st.subheader("🌐 Card Country vs IP Country Heatmap (Top 10)")
            card_col = "bin_country_code" if "bin_country_code" in m_df.columns else "card_country"
            ip_col = "ip_country"
            
            if card_col in m_df.columns and ip_col in m_df.columns:
                ct = pd.crosstab(
                    m_df[card_col].fillna("Unknown"),
                    m_df[ip_col].fillna("Unknown")
                )
                top_cards = m_df[card_col].fillna("Unknown").value_counts().head(10).index
                top_ips = m_df[ip_col].fillna("Unknown").value_counts().head(10).index
                ct_filtered = ct.loc[ct.index.isin(top_cards), ct.columns.isin(top_ips)]
                
                fig_hm = px.imshow(
                    ct_filtered, 
                    labels=dict(x="IP Country", y="Card Country", color="Transactions"),
                    text_auto=True,
                    color_continuous_scale="YlGnBu"
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            
            # --- TOP RULES FOR MERCHANT ---
            st.subheader("📌 Top Applied Rules Breakdown")
            if "applied_rule_ids" in m_df.columns:
                m_df["parsed_rules"] = m_df["applied_rule_ids"].apply(parse_rule_ids)
                rules_exploded = m_df.explode("parsed_rules").dropna(subset=["parsed_rules"])
                rules_exploded["rule_name"] = rules_exploded["parsed_rules"].apply(get_rule_name)
                
                rule_counts = rules_exploded["rule_name"].value_counts().head(10).reset_index()
                rule_counts.columns = ["Rule Name", "Trigger Count"]
                
                fig_rules = px.bar(rule_counts, x="Trigger Count", y="Rule Name", orientation="h", title="Top 10 Applied Rules")
                st.plotly_chart(fig_rules, use_container_width=True)

            # EXPORT BUTTON FOR SINGLE MERCHANT DEEP ANALYSIS
            output_merchant = io.BytesIO()
            with pd.ExcelWriter(output_merchant, engine="xlsxwriter") as writer:
                daily_m.to_excel(writer, sheet_name="Daily Metrics", index=False)
                if 'rule_counts' in locals():
                    rule_counts.to_excel(writer, sheet_name="Top Rules", index=False)
            st.download_button(
                label=f"📥 Download Deep Analysis Report for Merchant {selected_m} (Excel)",
                data=output_merchant.getvalue(),
                file_name=f"SEON_Single_Merchant_Analysis_{selected_m}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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
                st.code(clean_str, language="text")

                # EXPORT BUTTON FOR MONTHLY WATERFALL
                output_waterfall = io.BytesIO()
                with pd.ExcelWriter(output_waterfall, engine="xlsxwriter") as writer:
                    pd.DataFrame({"Clean_Closed_User_IDs": sorted(clean_closed_ids)}).to_excel(writer, sheet_name="Clean Closed User IDs", index=False)
                    pd.DataFrame({"Multi_Account_User_IDs": sorted(multi_ids)}).to_excel(writer, sheet_name="Multi-Accounts", index=False)
                st.download_button(
                    label="📥 Download Monthly Classification Summary (Excel)",
                    data=output_waterfall.getvalue(),
                    file_name=f"Monthly_Decline_User_Classification_{yr}_{mn}.xlsx",
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
        
        # --- CHART 1: OVERALL vs SEON DECLINE RATE TREND ---
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
            # --- CHARTS 2 & 3: USER COUNTRY & CARD COUNTRY ---
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

            # --- CHARTS 4 & 5: ROUTE ANALYTICS & PAYMENT METHODS ---
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

            # --- SECTION 6: PEAK SPIKE DAY DRILL-DOWN (3 PIE CHARTS) ---
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

        # EXPORT BUTTON FOR TOP-UP MONITORING
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
