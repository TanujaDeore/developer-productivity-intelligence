# src/database.py
# Loads CSV data into SQLite database and runs SQL analytics

import sqlite3
import pandas as pd
import os

DB_PATH = "data/productivity.db"

def create_connection():
    """Create SQLite database connection"""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def load_data_to_db():
    """Load all CSVs into SQLite database"""
    print("Loading data into database...")
    conn = create_connection()
    
    commits_df = pd.read_csv("data/raw/commits.csv")
    commits_df.to_sql("commits", conn, if_exists="replace", index=False)
    print(f"Loaded {len(commits_df)} commits")
    
    pr_df = pd.read_csv("data/raw/pull_requests.csv")
    pr_df.to_sql("pull_requests", conn, if_exists="replace", index=False)
    print(f"Loaded {len(pr_df)} pull requests")
    
    contributors_df = pd.read_csv("data/raw/contributors.csv")
    contributors_df.to_sql("contributors", conn, if_exists="replace", index=False)
    print(f"Loaded {len(contributors_df)} contributors")
    
    conn.commit()
    conn.close()
    print("Database ready!")


def run_query(query):
    """Run any SQL query and return a dataframe"""
    conn = create_connection()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_commit_activity_by_day():
    query = """
        SELECT 
            day_of_week,
            COUNT(*) as total_commits,
            ROUND(AVG(additions), 1) as avg_additions,
            ROUND(AVG(deletions), 1) as avg_deletions,
            ROUND(AVG(total_changes), 1) as avg_changes
        FROM commits
        GROUP BY day_of_week
        ORDER BY total_commits DESC
    """
    return run_query(query)


def get_commit_activity_by_hour():
    query = """
        SELECT 
            hour_of_day,
            COUNT(*) as total_commits,
            ROUND(AVG(total_changes), 1) as avg_changes
        FROM commits
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """
    return run_query(query)


def get_top_contributors_by_commits():
    query = """
        SELECT 
            author_login,
            author,
            COUNT(*) as total_commits,
            SUM(additions) as total_additions,
            SUM(deletions) as total_deletions,
            ROUND(AVG(total_changes), 1) as avg_changes_per_commit,
            RANK() OVER (ORDER BY COUNT(*) DESC) as commit_rank
        FROM commits
        GROUP BY author_login, author
        ORDER BY total_commits DESC
        LIMIT 10
    """
    return run_query(query)


def get_pr_merge_time_analysis():
    query = """
        WITH pr_stats AS (
            SELECT 
                author,
                pr_number,
                title,
                time_to_merge_hours,
                first_review_hours,
                additions,
                deletions,
                changed_files,
                review_comments,
                NTILE(4) OVER (ORDER BY time_to_merge_hours) as merge_time_quartile
            FROM pull_requests
            WHERE merged = 1
              AND time_to_merge_hours IS NOT NULL
        )
        SELECT 
            author,
            COUNT(*) as total_prs,
            ROUND(AVG(time_to_merge_hours), 2) as avg_merge_hours,
            ROUND(MIN(time_to_merge_hours), 2) as fastest_merge_hours,
            ROUND(MAX(time_to_merge_hours), 2) as slowest_merge_hours,
            ROUND(AVG(first_review_hours), 2) as avg_first_review_hours,
            ROUND(AVG(changed_files), 1) as avg_files_per_pr,
            ROUND(AVG(review_comments), 1) as avg_review_comments
        FROM pr_stats
        GROUP BY author
        ORDER BY avg_merge_hours ASC
        LIMIT 15
    """
    return run_query(query)


def get_code_churn_analysis():
    query = """
        WITH developer_stats AS (
            SELECT 
                author_login,
                author,
                COUNT(*) as commit_count,
                SUM(additions) as total_additions,
                SUM(deletions) as total_deletions,
                SUM(additions + deletions) as total_churn
            FROM commits
            GROUP BY author_login, author
        ),
        churn_ranked AS (
            SELECT 
                *,
                ROUND(
                    CAST(total_deletions AS FLOAT) / 
                    NULLIF(total_additions, 0) * 100, 1
                ) as deletion_ratio_pct,
                RANK() OVER (ORDER BY total_churn DESC) as churn_rank
            FROM developer_stats
        )
        SELECT *
        FROM churn_ranked
        ORDER BY total_churn DESC
        LIMIT 10
    """
    return run_query(query)


def get_pr_size_distribution():
    query = """
        SELECT 
            CASE 
                WHEN changed_files <= 2 THEN 'XS (1-2 files)'
                WHEN changed_files <= 5 THEN 'S (3-5 files)'
                WHEN changed_files <= 10 THEN 'M (6-10 files)'
                WHEN changed_files <= 20 THEN 'L (11-20 files)'
                ELSE 'XL (20+ files)'
            END as pr_size,
            COUNT(*) as count,
            ROUND(AVG(time_to_merge_hours), 2) as avg_merge_hours,
            ROUND(AVG(review_comments), 1) as avg_review_comments
        FROM pull_requests
        WHERE merged = 1
        GROUP BY pr_size
        ORDER BY count DESC
    """
    return run_query(query)


def get_hourly_productivity_score():
    query = """
        WITH hourly_stats AS (
            SELECT 
                hour_of_day,
                COUNT(*) as commits,
                ROUND(AVG(total_changes), 1) as avg_changes
            FROM commits
            GROUP BY hour_of_day
        )
        SELECT 
            hour_of_day,
            commits,
            avg_changes,
            ROUND(
                AVG(commits) OVER (
                    ORDER BY hour_of_day 
                    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
                ), 1
            ) as rolling_avg_commits,
            SUM(commits) OVER (ORDER BY hour_of_day) as cumulative_commits
        FROM hourly_stats
        ORDER BY hour_of_day
    """
    return run_query(query)


if __name__ == "__main__":
    load_data_to_db()
    
    print("\n--- Commit Activity by Day ---")
    print(get_commit_activity_by_day().to_string())
    
    print("\n--- Top Contributors ---")
    print(get_top_contributors_by_commits().to_string())
    
    print("\n--- PR Merge Time Analysis ---")
    print(get_pr_merge_time_analysis().to_string())
    
    print("\n--- Code Churn Analysis ---")
    print(get_code_churn_analysis().to_string())
    
    print("\n--- PR Size Distribution ---")
    print(get_pr_size_distribution().to_string())
    
    print("\n--- Hourly Productivity Score ---")
    print(get_hourly_productivity_score().to_string())