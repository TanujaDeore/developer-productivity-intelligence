# dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys

sys.path.append("src")
from database import (
    load_data_to_db,
    get_commit_activity_by_day,
    get_commit_activity_by_hour,
    get_top_contributors_by_commits,
    get_pr_merge_time_analysis,
    get_code_churn_analysis,
    get_pr_size_distribution,
    get_hourly_productivity_score
)

st.set_page_config(
    page_title="Developer Productivity Intelligence",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Developer Productivity Intelligence")
st.markdown("**Real-time behavioral analytics on microsoft/vscode contributors**")
st.markdown("---")

@st.cache_data
def load_all_data():
    load_data_to_db()
    return {
        "commits_by_day": get_commit_activity_by_day(),
        "commits_by_hour": get_commit_activity_by_hour(),
        "top_contributors": get_top_contributors_by_commits(),
        "pr_merge_times": get_pr_merge_time_analysis(),
        "code_churn": get_code_churn_analysis(),
        "pr_size_dist": get_pr_size_distribution(),
        "hourly_productivity": get_hourly_productivity_score()
    }

data = load_all_data()

commits_df = pd.read_csv("data/raw/commits.csv")
pr_df = pd.read_csv("data/raw/pull_requests.csv")
contributors_df = pd.read_csv("data/raw/contributors.csv")

merged_prs = pr_df[pr_df["merged"] == True]
avg_merge_time = merged_prs["time_to_merge_hours"].mean()
avg_review_time = merged_prs["first_review_hours"].dropna().mean()

st.subheader("📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Commits", f"{len(commits_df)}")
col2.metric("Pull Requests", f"{len(pr_df)}")
col3.metric("Contributors", f"{len(contributors_df)}")
col4.metric("Avg Merge Time", f"{avg_merge_time:.1f}h")
col5.metric("Avg First Review", f"{avg_review_time:.2f}h")

st.markdown("---")

st.subheader("📅 Commit Activity Patterns")
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        data["commits_by_day"],
        x="day_of_week",
        y="total_commits",
        color="total_commits",
        color_continuous_scale="Blues",
        title="Commits by Day of Week",
        labels={"total_commits": "Total Commits", "day_of_week": "Day"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    hourly = data["hourly_productivity"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour_of_day"],
        y=hourly["commits"],
        name="Commits",
        marker_color="lightblue"
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour_of_day"],
        y=hourly["rolling_avg_commits"],
        mode="lines+markers",
        name="Rolling Avg",
        line=dict(color="red", width=2)
    ))
    fig.update_layout(
        title="Commits by Hour of Day (with Rolling Average)",
        xaxis_title="Hour",
        yaxis_title="Commits"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🏆 Top Contributors")
col1, col2 = st.columns(2)

with col1:
    contrib = data["top_contributors"]
    fig = px.bar(
        contrib,
        x="total_commits",
        y="author",
        orientation="h",
        color="total_commits",
        color_continuous_scale="Greens",
        title="Top 10 Contributors by Commit Count",
        labels={"total_commits": "Commits", "author": "Developer"}
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        contrib,
        x="total_commits",
        y="avg_changes_per_commit",
        size="total_additions",
        color="commit_rank",
        hover_name="author",
        title="Commit Volume vs Average Change Size",
        labels={
            "total_commits": "Total Commits",
            "avg_changes_per_commit": "Avg Changes per Commit"
        },
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔀 Pull Request Analysis")
col1, col2 = st.columns(2)

with col1:
    pr_merge = data["pr_merge_times"]
    fig = px.bar(
        pr_merge,
        x="author",
        y="avg_merge_hours",
        color="avg_merge_hours",
        color_continuous_scale="RdYlGn_r",
        title="Average PR Merge Time by Developer (hours)",
        labels={"avg_merge_hours": "Avg Merge Hours", "author": "Developer"}
    )
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    pr_size = data["pr_size_dist"]
    fig = px.scatter(
        pr_size,
        x="pr_size",
        y="avg_merge_hours",
        size="count",
        color="avg_review_comments",
        title="PR Size vs Merge Time (bubble = volume)",
        labels={
            "avg_merge_hours": "Avg Merge Hours",
            "pr_size": "PR Size",
            "avg_review_comments": "Avg Review Comments"
        },
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔄 Code Churn Analysis")
col1, col2 = st.columns(2)

with col1:
    churn = data["code_churn"]
    fig = px.bar(
        churn,
        x="author",
        y=["total_additions", "total_deletions"],
        title="Code Additions vs Deletions by Developer",
        labels={"value": "Lines of Code", "author": "Developer"},
        barmode="group",
        color_discrete_map={
            "total_additions": "green",
            "total_deletions": "red"
        }
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        churn,
        x="total_churn",
        y="deletion_ratio_pct",
        size="commit_count",
        hover_name="author",
        color="churn_rank",
        title="Total Churn vs Deletion Ratio %",
        labels={
            "total_churn": "Total Code Churn",
            "deletion_ratio_pct": "Deletion Ratio %"
        },
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Raw Data Explorer")
tab1, tab2, tab3 = st.tabs(["Commits", "Pull Requests", "Contributors"])

with tab1:
    st.dataframe(commits_df, use_container_width=True)
with tab2:
    st.dataframe(pr_df, use_container_width=True)
with tab3:
    st.dataframe(contributors_df, use_container_width=True)

st.markdown("---")
st.markdown("Built with Python · SQLite · Plotly · Streamlit · GitHub API")