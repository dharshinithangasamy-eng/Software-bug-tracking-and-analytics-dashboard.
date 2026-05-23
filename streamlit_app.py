import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytics import run_all_analytics, load_data

st.set_page_config(
    page_title="BugScope — Bug Tracking Analytics",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
    .metric-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; }
    .metric-value { font-size: 2.5rem; font-weight: 800; color: #58a6ff; }
    .metric-label { color: #8b949e; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 2px; }
    .critical-badge { background: #ff4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .high-badge { background: #ff8800; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .medium-badge { background: #ffcc00; color: black; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    .low-badge { background: #00cc44; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    code { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    df = load_data()
    analytics = run_all_analytics()
    return df, analytics

with st.spinner("Loading analytics engine..."):
    df, analytics = get_data()

# Sidebar
with st.sidebar:
    st.markdown("## 🐛 BugScope")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Dashboard", "🤖 ML Insights", "🔍 Bug Explorer", "📈 Trends"])
    st.markdown("---")
    
    st.markdown("**Filters**")
    severity_filter = st.multiselect("Severity", ["Critical","High","Medium","Low"], default=["Critical","High","Medium","Low"])
    status_filter = st.multiselect("Status", ["Open","In Progress","Resolved","Closed","Reopened"], default=["Open","In Progress","Resolved","Closed","Reopened"])
    component_filter = st.multiselect("Component", df['component'].unique().tolist(), default=df['component'].unique().tolist())

filtered_df = df[
    df['severity'].isin(severity_filter) &
    df['status'].isin(status_filter) &
    df['component'].isin(component_filter)
]

s = analytics['summary']

# ─── DASHBOARD ───────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("Bug Tracking Dashboard")
    st.markdown(f"*Analyzing **{len(filtered_df)}** bugs across {filtered_df['component'].nunique()} components*")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Bugs", s['total'])
    with col2:
        st.metric("Open", s['open'], delta=f"-{s['resolution_rate']}% resolved")
    with col3:
        st.metric("Critical", s['critical'], delta="⚠️ Needs attention")
    with col4:
        st.metric("Resolution Rate", f"{s['resolution_rate']}%")
    with col5:
        st.metric("Avg Resolution", f"{s['avg_resolution_days']}d")
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Bugs by Component")
        comp_data = filtered_df.groupby('component').size().reset_index(name='count').sort_values('count', ascending=True)
        st.bar_chart(comp_data.set_index('component')['count'])
    
    with col_b:
        st.subheader("Status Distribution")
        status_data = filtered_df.groupby('status').size().reset_index(name='count')
        st.bar_chart(status_data.set_index('status')['count'])
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Severity Breakdown")
        sev_data = filtered_df.groupby('severity').size().reset_index(name='count')
        st.bar_chart(sev_data.set_index('severity')['count'])
    
    with col_d:
        st.subheader("Top Assignees by Bug Count")
        assignee_data = filtered_df.groupby('assignee').size().reset_index(name='count').sort_values('count', ascending=False).head(6)
        st.bar_chart(assignee_data.set_index('assignee')['count'])
    
    st.subheader("Recent Bugs")
    recent = filtered_df.sort_values('created_at', ascending=False).head(10)[
        ['id','title','component','severity','status','assignee','created_at']
    ]
    st.dataframe(recent, use_container_width=True)

# ─── ML INSIGHTS ─────────────────────────────────────────────────────────────
elif page == "🤖 ML Insights":
    st.title("Machine Learning Insights")
    st.markdown("*Powered by scikit-learn — Random Forest, Gradient Boosting & KMeans*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Severity Predictor")
        ml_sev = analytics['ml_severity']
        st.metric("Model Accuracy", f"{ml_sev['accuracy']}%")
        st.caption(f"**{ml_sev['model']}** trained on {ml_sev['train_size']} bugs")
        st.markdown("**Feature Importances:**")
        fi_df = pd.DataFrame(list(ml_sev['feature_importances'].items()), columns=['Feature', 'Importance'])
        st.bar_chart(fi_df.set_index('Feature')['Importance'])
    
    with col2:
        st.markdown("### ⏱️ Resolution Time Predictor")
        ml_res = analytics['ml_resolution']
        st.metric("Mean Abs Error", f"{ml_res['mae_days']} days")
        st.metric("Avg Resolution", f"{ml_res['avg_resolution_days']} days")
        st.caption(f"**{ml_res['model']}**")
        st.markdown("**By Severity:**")
        res_df = pd.DataFrame(list(ml_res['by_severity'].items()), columns=['Severity', 'Days'])
        st.bar_chart(res_df.set_index('Severity')['Days'])
    
    with col3:
        st.markdown("### 🔵 Bug Clusters")
        ml_clust = analytics['ml_clustering']
        st.caption(f"**{ml_clust['model']}** — {ml_clust['n_clusters']} clusters identified")
        clust_df = pd.DataFrame(list(ml_clust['cluster_distribution'].items()), columns=['Cluster', 'Count'])
        st.dataframe(clust_df, use_container_width=True)
        st.bar_chart(clust_df.set_index('Cluster')['Count'])
    
    st.markdown("---")
    st.subheader("📋 Model Summary")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.info(f"**Random Forest Classifier**\nTask: Severity Prediction\nAccuracy: {ml_sev['accuracy']}%\nEstimators: 100")
    with col_m2:
        st.success(f"**Gradient Boosting Regressor**\nTask: Resolution Time\nMAE: {ml_res['mae_days']} days\nEstimators: 100")
    with col_m3:
        st.warning(f"**KMeans Clustering**\nTask: Bug Grouping\nClusters: 4\nMethod: Euclidean Distance")

# ─── BUG EXPLORER ────────────────────────────────────────────────────────────
elif page == "🔍 Bug Explorer":
    st.title("Bug Explorer")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("Search bugs...", placeholder="Type to search title, ID, component...")
    with col2:
        sort_by = st.selectbox("Sort by", ["created_at", "severity", "votes", "comments"])
    
    display_df = filtered_df.copy()
    if search:
        mask = (
            display_df['id'].str.contains(search, case=False) |
            display_df['title'].str.contains(search, case=False) |
            display_df['component'].str.contains(search, case=False)
        )
        display_df = display_df[mask]
    
    display_df = display_df.sort_values(sort_by, ascending=False)
    
    st.markdown(f"*Showing {len(display_df)} bugs*")
    st.dataframe(
        display_df[['id','title','component','severity','status','assignee','environment','votes','comments','created_at']],
        use_container_width=True,
        height=500
    )
    
    if st.button("📥 Export to CSV"):
        csv = display_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "bugs_export.csv", "text/csv")

# ─── TRENDS ──────────────────────────────────────────────────────────────────
elif page == "📈 Trends":
    st.title("Bug Trends & Analysis")
    
    monthly = filtered_df.groupby(filtered_df['created_at'].dt.to_period('M')).size().reset_index(name='count')
    monthly['month'] = monthly['created_at'].astype(str)
    
    st.subheader("Monthly Bug Count")
    st.line_chart(monthly.set_index('month')['count'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Resolution Time by Component")
        res_comp = filtered_df.dropna(subset=['resolution_days']).groupby('component')['resolution_days'].mean().sort_values()
        st.bar_chart(res_comp)
    
    with col2:
        st.subheader("Regression Bugs by Component")
        reg_data = filtered_df[filtered_df['is_regression']].groupby('component').size()
        st.bar_chart(reg_data)
    
    st.subheader("Environment Distribution")
    env_data = filtered_df.groupby(['environment', 'severity']).size().unstack(fill_value=0)
    st.bar_chart(env_data)

st.markdown("---")
st.markdown("<center><small>BugScope Analytics • Built with Streamlit + scikit-learn</small></center>", unsafe_allow_html=True)
