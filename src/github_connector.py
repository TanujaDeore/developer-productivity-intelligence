# src/github_connector.py
# Pulls real developer activity data from a public repository

import os
from github import Github, Auth
from dotenv import load_dotenv
import pandas as pd
import time

# Load environment variables
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("TARGET_REPO_OWNER", "microsoft")
REPO_NAME = os.getenv("TARGET_REPO_NAME", "vscode")


def connect_to_github():
    """Connect to GitHub API"""
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    
    user = g.get_user()
    print(f"Connected as: {user.login}")
    
    # Get target repo
    repo = g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")
    print(f"Target repo: {repo.full_name}")
    print(f"Stars: {repo.stargazers_count:,}")
    print(f"Open issues: {repo.open_issues_count:,}")
    
    return g, repo


def get_recent_commits(repo, max_commits=200):
    """Fetch recent commits from the repository"""
    print(f"\nFetching last {max_commits} commits...")
    
    commits_data = []
    count = 0
    
    for commit in repo.get_commits():
        if count >= max_commits:
            break
        
        try:
            commits_data.append({
                "sha": commit.sha[:7],
                "author": commit.commit.author.name,
                "author_login": commit.author.login if commit.author else "unknown",
                "message": commit.commit.message[:150].replace('\n', ' '),
                "date": commit.commit.author.date,
                "hour_of_day": commit.commit.author.date.hour,
                "day_of_week": commit.commit.author.date.strftime("%A"),
                "additions": commit.stats.additions,
                "deletions": commit.stats.deletions,
                "total_changes": commit.stats.total,
                "files_changed": commit.stats.total
            })
            count += 1
            
            if count % 20 == 0:
                print(f"  Fetched {count} commits...")
            
            time.sleep(0.05)
            
        except Exception as e:
            print(f"  Skipping commit: {e}")
            count += 1
            continue
    
    df = pd.DataFrame(commits_data)
    print(f"Total commits fetched: {len(df)}")
    return df


def get_recent_pull_requests(repo, max_prs=100):
    """Fetch recent pull requests"""
    print(f"\nFetching last {max_prs} pull requests...")
    
    pr_data = []
    count = 0
    
    for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
        if count >= max_prs:
            break
        
        try:
            # Calculate time to merge
            time_to_merge_hours = None
            if pr.merged_at and pr.created_at:
                delta = pr.merged_at - pr.created_at
                time_to_merge_hours = round(delta.total_seconds() / 3600, 2)
            
            # Calculate review turnaround
            review_comments = list(pr.get_review_comments())
            first_review_hours = None
            if review_comments and pr.created_at:
                first_review = min(review_comments, key=lambda x: x.created_at)
                delta = first_review.created_at - pr.created_at
                first_review_hours = round(delta.total_seconds() / 3600, 2)
            
            pr_data.append({
                "pr_number": pr.number,
                "title": pr.title[:100],
                "author": pr.user.login,
                "state": pr.state,
                "merged": pr.merged,
                "created_at": pr.created_at,
                "merged_at": pr.merged_at,
                "closed_at": pr.closed_at,
                "time_to_merge_hours": time_to_merge_hours,
                "first_review_hours": first_review_hours,
                "comments": pr.comments,
                "review_comments": pr.review_comments,
                "commits_in_pr": pr.commits,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changed_files": pr.changed_files,
                "label_count": len(list(pr.get_labels()))
            })
            count += 1
            
            if count % 20 == 0:
                print(f"  Fetched {count} PRs...")
            
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  Skipping PR: {e}")
            count += 1
            continue
    
    df = pd.DataFrame(pr_data)
    print(f"Total PRs fetched: {len(df)}")
    return df


def get_contributor_stats(repo, max_contributors=30):
    """Fetch contributor activity stats"""
    print(f"\nFetching contributor stats...")
    
    contributors_data = []
    count = 0
    
    for contributor in repo.get_contributors():
        if count >= max_contributors:
            break
        
        try:
            contributors_data.append({
                "login": contributor.login,
                "total_commits": contributor.contributions,
                "followers": contributor.followers,
                "public_repos": contributor.public_repos,
                "account_age_days": (
                    pd.Timestamp.now(tz='UTC') - 
                    contributor.created_at
                ).days
            })
            count += 1
            time.sleep(0.05)
            
        except Exception as e:
            print(f"  Skipping contributor: {e}")
            count += 1
            continue
    
    df = pd.DataFrame(contributors_data)
    print(f"Total contributors fetched: {len(df)}")
    return df


def save_data(df, filename):
    """Save dataframe to CSV"""
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Saved → {filepath} ({len(df)} rows)")


def run_pipeline():
    """Run the full data collection pipeline"""
    print("=" * 50)
    print("Developer Productivity Intelligence")
    print(f"Collecting data from: {REPO_OWNER}/{REPO_NAME}")
    print("=" * 50)
    
    # Connect
    g, repo = connect_to_github()
    
    # Collect commits
    commits_df = get_recent_commits(repo, max_commits=200)
    save_data(commits_df, "commits.csv")
    
    # Collect PRs
    pr_df = get_recent_pull_requests(repo, max_prs=100)
    save_data(pr_df, "pull_requests.csv")
    
    # Collect contributors
    contributors_df = get_contributor_stats(repo, max_contributors=30)
    save_data(contributors_df, "contributors.csv")
    
    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print(f"Commits collected:      {len(commits_df)}")
    print(f"Pull requests collected:{len(pr_df)}")
    print(f"Contributors collected: {len(contributors_df)}")
    print("\nData saved in data/raw/ folder")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()