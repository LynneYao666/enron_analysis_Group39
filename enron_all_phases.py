import pandas as pd
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import os

# NLP & ML Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("📥 Downloading VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
except ImportError:
    print("⚠️  Warning: nltk not available. Sentiment analysis will be skipped.")
    SentimentIntensityAnalyzer = None





# =====================================================
# PHASE 0 — LOAD DATA
# =====================================================
print("="*70)
print("PHASE 0 — LOAD DATA")
print("="*70)

# Create output directory if it doesn't exist
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
print(f"✅ Output directory: {output_dir}/")

csv_path = "emails.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Error: {csv_path} not found. Please ensure the dataset is present.")

df = pd.read_csv(csv_path)
print(f"✅ CSV loaded, rows: {len(df)}")

# Expecting columns: file, message
if not all(col in df.columns for col in ["file", "message"]):
    raise ValueError("Error: CSV must contain 'file' and 'message' columns")

# =====================================================
# PHASE 1 — PARSE & NETWORK FILTERING
# =====================================================
print("\n" + "="*70)
print("PHASE 1 — PARSE & NETWORK FILTERING")
print("="*70)

def parse_fields(raw_text):
    # Extract basic headers using regex
    def extract(pattern):
        match = re.search(pattern, raw_text, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else None

    from_field = extract(r"^From:\s*(.*)$")
    to_field = extract(r"^To:\s*(.*)$")
    subject_field = extract(r"^Subject:\s*(.*)$")
    date_field = extract(r"^Date:\s*(.*)$")

    # Extract body (after first blank line)
    parts = re.split(r"\n\s*\n", raw_text, 1)
    body = parts[1] if len(parts) > 1 else ""

    return pd.Series({
        "From": from_field,
        "To": to_field,
        "Subject": subject_field,
        "Date": date_field,
        "Message": body
    })

print("⏳ Parsing emails (this may take a moment)...")
parsed = df["message"].apply(parse_fields)
emails = parsed.copy()

# Drop empty senders/recipients and cleanup
emails = emails.dropna(subset=["From", "To"])
emails["From"] = emails["From"].str.lower().str.strip()
emails["To"] = emails["To"].str.lower().str.strip()
emails["Message"] = emails["Message"].astype(str)

print(f"✅ Parsed emails: {len(emails)}")

# --- Build Communication Graph ---
print("🔗 Building communication graph...")
records = []
for _, row in emails.iterrows():
    tos = str(row["To"]).split(",")
    for t in tos:
        t = t.strip()
        if t:
            records.append({"from": row["From"], "to": t})

edges_df = pd.DataFrame(records)
G = nx.from_pandas_edgelist(edges_df, "from", "to", create_using=nx.DiGraph())
print(f"📊 Graph nodes: {len(G.nodes())}, edges: {len(G.edges())}")

# --- Centrality Analysis ---
print("🔍 Calculating centrality...")
# Approximate centrality for speed
centrality = nx.betweenness_centrality_subset(
    G,
    sources=list(G.nodes())[:300],
    targets=list(G.nodes())[:300],
    normalized=True
)

top_people = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
print("\n🏆 Top 15 Central People:")
for p, c in top_people:
    print(f"  {p:<40} {c:.5f}")

important_people = [p for p, _ in top_people]
filtered = emails[emails["From"].isin(important_people) | emails["To"].isin(important_people)].copy()

filtered.to_csv(os.path.join(output_dir, "phase1_filtered_emails.csv"), index=False)
print(f"✅ Saved phase1_filtered_emails.csv ({len(filtered)} emails)")

# =====================================================
# PHASE 2 — KEYWORD DETECTION
# =====================================================
print("\n" + "="*70)
print("PHASE 2 — KEYWORD DETECTION")
print("="*70)

# Clean text
filtered["clean"] = (
    filtered["Message"]
    .str.lower()
    .str.replace(r"[^a-z\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

insider_terms = [
    "naked calls",
    "naked options",
    "uncovered calls",
    "short calls",
    "sell calls",
    "buy calls",
    "sell puts",
    "buy puts",
    "hedge my options",
    "hedge my position",
    "unexercised vested options",
    "exercise my options",
    "exercising options",

    # Enron / ENE equity specifically
    "enron stock",
    "enron shares",
    "enron options",
    "ene stock",
    "ene shares",
    "ene options",
    "ene equity",

    # Questionable trading behaviour / timing
    "front run",
    "front-running",
    "front running",
    "ahead of the announcement",
    "before earnings",
    "before the earnings release",
    "before it's announced",
    "before its announced",
    "before it is announced",

    # Explicit MNPI / insider language
    "insider trading",
    "inside information",
    "insider information",
    "inside info",
    "material nonpublic information",
    "material non-public information",
    "mnpi",
    "nonpublic information",
    "non-public information",
    "selective disclosure",
    "stock tip",
    "tippee",
    "tipper",

    # Contextual but noisier terms (remove if too many false positives)
    "market rumor",
    "quiet period",
    "blackout period",
    "trading blackout"
]

def score(text):
    return sum(term in text for term in insider_terms)

filtered["keyword_score"] = filtered["clean"].apply(score)
suspicious = filtered[filtered["keyword_score"] > 0].copy()
print(f"⚠️  Emails with insider keywords: {len(suspicious)}")

# =====================================================
# PHASE 3 — TIME FILTERING & SCORING
# =====================================================
print("\n" + "="*70)
print("PHASE 3 — TIME FILTER & ADVANCED SCORING")
print("="*70)

# 1. Handle Dates and Timezones
suspicious["Date"] = pd.to_datetime(suspicious["Date"], errors="coerce", utc=True)

def remove_timezone(series):
    """Convert timezone-aware datetime series to naive"""
    try:
        if hasattr(series.dtype, 'tz') and series.dtype.tz is not None:
            return series.apply(lambda x: x.replace(tzinfo=None) if pd.notna(x) else x)
        return series
    except Exception:
        return series

suspicious["Date"] = remove_timezone(suspicious["Date"])

# 2. Crisis Window Filter (April - Dec 2001)
start_date = pd.Timestamp("2001-04-01")
end_date = pd.Timestamp("2001-12-31")
date_filtered = suspicious[
    suspicious["Date"].between(start_date, end_date, inclusive="both")
].copy()
print(f"📅 Emails in crisis window (Apr-Dec 2001): {len(date_filtered)}")

# 3. Off-Hours Filter (Before 6 AM or After 8 PM)
# Using .dt.hour. 0-23 scale.
date_filtered["hour"] = date_filtered["Date"].dt.hour
off_hours_filtered = date_filtered[
    (date_filtered["hour"] < 6) | (date_filtered["hour"] > 20)
].copy()
print(f"🌙 Emails sent off-hours (< 6 AM or > 8 PM): {len(off_hours_filtered)}")

# If off-hours filter is too aggressive and returns 0, fallback to date_filtered
if len(off_hours_filtered) == 0:
    print("⚠️  Warning: Off-hours filter resulted in 0 emails. Falling back to date-filtered emails.")
    analysis_set = date_filtered.copy()
else:
    analysis_set = off_hours_filtered.copy()

# 4. Sentiment Analysis (VADER)
print("📊 Calculating sentiment scores...")
if SentimentIntensityAnalyzer is not None:
    sia = SentimentIntensityAnalyzer()
    analysis_set["sentiment"] = analysis_set["clean"].apply(
        lambda x: sia.polarity_scores(x)['compound']
    )
else:
    analysis_set["sentiment"] = 0.0

# 5. Advanced Scoring
# Formula: risk_score = keyword_score + (-5 * sentiment)
# Logic: High keywords + Negative sentiment (panic/anger) = Higher Suspicion
analysis_set["risk_score"] = analysis_set["keyword_score"] + (-5 * analysis_set["sentiment"])

# 6. Select Top 100
analysis_set_sorted = analysis_set.sort_values("risk_score", ascending=False)
top100 = analysis_set_sorted.head(100) if len(analysis_set_sorted) >= 100 else analysis_set_sorted.copy()

top100.to_csv(os.path.join(output_dir, "top100_suspicious_emails.csv"), index=False)
print(f"✅ Saved top 100 suspicious emails -> {output_dir}/top100_suspicious_emails.csv")

# =====================================================
# PHASE 4 — EVALUATION & ITERATION
# =====================================================
print("\n" + "="*70)
print("PHASE 4 — EVALUATION & ITERATION")
print("="*70)

# Stop-list for benign financial jargon
stop_list = [
    "quarterly report", "annual report", "meeting agenda", 
    "conference call", "public announcement", "press release",
    "regulatory filing", "compliance", "audit committee", "SEC"
]

def refine_keyword_score(text, stop_list, base_score):
    """Refined scoring that penalizes stop-list terms"""
    # Penalize if stop-list terms are present (likely false positives)
    penalty = sum(1 for stop_term in stop_list if stop_term in text)
    # Reduce the existing risk score (reduced penalty from 2.0 to 1.0 to be less aggressive)
    return max(0, base_score - (penalty * 1.0)) # Penalize moderately

print("🔍 Applying stop-list refinement...")
top100["refined_score"] = top100.apply(
    lambda row: refine_keyword_score(row["clean"], stop_list, row["risk_score"]), axis=1
)

# Re-rank (with reduced penalty, emails with refined_score > 0 should pass)
top100_refined = top100.sort_values("refined_score", ascending=False).head(100)

# Save refined
output_filename = "top100_refined_suspicious_emails.csv"
top100_refined.to_csv(os.path.join(output_dir, output_filename), index=False)
print(f"✅ Saved refined list -> {output_dir}/{output_filename} ({len(top100_refined)} emails)")

# Select Top 5 Refined Suspicious Emails
print("\n" + "="*70)
print("TOP 5 REFINED SUSPICIOUS EMAILS")
print("="*70)

if len(top100_refined) > 0:
    top5_refined = top100_refined.sort_values("refined_score", ascending=False).head(5)
    top5_refined.to_csv(os.path.join(output_dir, "top5_refined_suspicious_emails.csv"), index=False)
    print(f"\n✅ Top 5 refined suspicious emails saved -> {output_dir}/top5_refined_suspicious_emails.csv\n")
    
    for idx, (_, row) in enumerate(top5_refined.iterrows(), 1):
        print("-" * 70)
        print(f"#{idx} | Refined Score: {row['refined_score']:.2f} | Risk Score: {row['risk_score']:.2f}")
        print("-" * 70)
        print(f"From:    {row['From']}")
        print(f"To:      {row['To']}")
        print(f"Subject: {row['Subject']}")
        print(f"Date:    {row['Date']}")
        print(f"Keyword Score: {row['keyword_score']} | Sentiment: {row['sentiment']:.3f}")
        print(f"\nMessage Snippet:")
        message_snippet = str(row['Message'])[:350].replace("\n", " ")
        print(f"{message_snippet}...")
        print()
else:
    print("⚠️  No refined emails available. Using top 5 from unrefined list.")
    top5_unrefined = top100.sort_values("risk_score", ascending=False).head(5)
    top5_unrefined.to_csv(os.path.join(output_dir, "top5_suspicious_emails.csv"), index=False)
    print(f"✅ Top 5 suspicious emails saved -> {output_dir}/top5_suspicious_emails.csv\n")
    
    for idx, (_, row) in enumerate(top5_unrefined.iterrows(), 1):
        print("-" * 70)
        print(f"#{idx} | Risk Score: {row['risk_score']:.2f}")
        print("-" * 70)
        print(f"From:    {row['From']}")
        print(f"To:      {row['To']}")
        print(f"Subject: {row['Subject']}")
        print(f"Date:    {row['Date']}")
        print(f"Keyword Score: {row['keyword_score']} | Sentiment: {row['sentiment']:.3f}")
        print(f"\nMessage Snippet:")
        message_snippet = str(row['Message'])[:350].replace("\n", " ")
        print(f"{message_snippet}...")
        print()

# =====================================================
# PHASE 5 — VISUALIZATION & REPORTING
# =====================================================
print("\n" + "="*70)
print("PHASE 5 — VISUALIZATION & REPORTING")
print("="*70)

final_emails = top100_refined if len(top100_refined) > 0 else top100

# 1. Network Graph
print("📊 Generating Network Graph...")
viz_edges = []
for _, row in final_emails.iterrows():
    tos = str(row["To"]).split(",")
    for t in tos:
        t = t.strip().lower()
        if t:
            viz_edges.append({"from": row["From"], "to": t})

if viz_edges:
    viz_edges_df = pd.DataFrame(viz_edges)
    G_viz = nx.from_pandas_edgelist(viz_edges_df, "from", "to", create_using=nx.DiGraph())
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G_viz, k=0.5, iterations=50, seed=42)
    
    # Node sizes based on degree
    node_degrees = dict(G_viz.degree())
    node_sizes = [v * 100 + 100 for v in node_degrees.values()]
    
    nx.draw_networkx_edges(G_viz, pos, alpha=0.3, width=1.0, arrows=True)
    nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes, node_color="#FF6B6B", alpha=0.8)
    
    # Labels for top nodes
    top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = {node: node.split("@")[0] for node, _ in top_nodes}
    nx.draw_networkx_labels(G_viz, pos, labels, font_size=8, font_weight="bold")
    
    plt.title(f"Network of Top Suspicious Emails ({len(final_emails)} emails)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "network_graph.png"), dpi=300)
    plt.close()
    print(f"✅ Saved {output_dir}/network_graph.png")
else:
    print("⚠️  No edges to visualize.")

# 2. Timeline Chart
print("📊 Generating Timeline Chart...")
if not final_emails.empty:
    final_emails["date_only"] = final_emails["Date"].dt.date
    daily_counts = final_emails.groupby("date_only").size()
    daily_sentiment = final_emails.groupby("date_only")["sentiment"].mean()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Count
    ax1.plot(daily_counts.index, daily_counts.values, marker="o", color="#2E86AB")
    ax1.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color="#2E86AB")
    ax1.set_ylabel("Email Count")
    ax1.set_title("Suspicious Activity Over Time")
    ax1.grid(True, alpha=0.3)
    
    # Sentiment
    ax2.plot(daily_sentiment.index, daily_sentiment.values, marker="s", color="#A23B72")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Avg Sentiment (Negative = Suspicious)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    
    # Key dates
    key_dates = [pd.Timestamp("2001-08-14"), pd.Timestamp("2001-10-16")]
    for date in key_dates:
        if date.date() in daily_counts.index:
            ax1.axvline(x=date.date(), color="red", linestyle="--", alpha=0.6)
            ax2.axvline(x=date.date(), color="red", linestyle="--", alpha=0.6)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timeline_chart.png"), dpi=300)
    plt.close()
    print(f"✅ Saved {output_dir}/timeline_chart.png")