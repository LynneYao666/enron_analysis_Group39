#!/usr/bin/env python3
"""
ENRON INSIDER TRADING DETECTION - IMPROVED VERSION
Aligned with wrongdoing definition focusing on:
1. LJM/Chewco partnerships and accounting fraud
2. Trading coordination language
3. Material nonpublic information
4. Pre-disclosure timing (before Oct 16, 2001)
5. High-risk insiders (Fastow, Skilling, etc.)
"""

import pandas as pd
import numpy as np
import re
import networkx as nx
from datetime import datetime
from dateutil import parser
import os

# =====================================================
# IMPROVED KEYWORD CATEGORIES
# =====================================================

# Category A: Partnership/Accounting Fraud (10 points each)
partnership_fraud_terms = [
    "ljm", "ljm1", "ljm2", "ljm 1", "ljm 2",
    "chewco", 
    "fastow", "kopper", "glisan",
    "special purpose entity", "spe", " spe ", "special purpose vehicle",
    "off balance sheet", "off-balance", "hide loss", "hidden loss",
    "related party", "related-party",
    "raptor", "jedi", "southampton",
    "structured finance", "prepay",
    "mark to market", "fair value",
    "restate", "restatement", "accounting error",
    "overstate earnings", "understate debt", "managed earnings"
]

# Category B: Trading Coordination (8 points each)
trading_terms = [
    "sell stock", "sell shares", "sell my stock",
    "buy stock", "buy shares",
    "dump stock", "unload shares",
    "exercise options", "stock option",
    "trading window", "blackout period",
    "before announcement", "before disclosure", "before public",
    "before this hits", "ahead of earnings",
    "tip off", "heads up", "fyi on",
    "quietly", "discreetly", "without attracting"
]

# Category C: Material Nonpublic Info (5 points each)
material_info_terms = [
    "material nonpublic", "material non-public", "mnpi",
    "confidential financial", "internal numbers", "internal only",
    "actual results", "real numbers", "true picture",
    "not disclosed", "not yet public", "haven't announced",
    "liquidity crisis", "cash flow problem", "cash crunch",
    "debt problem", "credit rating",
    "sec investigation", "sec inquiry", "sec looking",
    "accounting issue", "accounting problem",
    "earnings manipulation", "cookie jar"
]

# Category D: Context Terms (penalty)
routine_business_terms = [
    "quarterly report", "annual report",
    "press release", "public announcement",
    "analyst call", "earnings call",
    "compliance", "disclosure committee",
    "external audit"
]

# =====================================================
# ROLE-BASED CLASSIFICATION
# =====================================================

# High-risk insiders (2.0x multiplier)
high_risk_insiders = [
    'andrew.fastow@enron.com', 'andy.fastow@enron.com',
    'jeffrey.skilling@enron.com', 'jeff.skilling@enron.com',
    'kenneth.lay@enron.com', 'ken.lay@enron.com', 'ken_lay@enron.com',
    'michael.kopper@enron.com',
    'ben.glisan@enron.com',
    'richard.causey@enron.com',
    'jeff.mcmahon@enron.com'
]

# Medium-risk insiders (1.5x multiplier)
medium_risk_insiders = [
    'louise.kitchen@enron.com',
    'greg.whalley@enron.com',
    'rebecca.mark@enron.com',
    'mark.frevert@enron.com',
    'john.arnold@enron.com'
]

# Low-risk roles (0.5x multiplier - NOT likely to have accounting fraud info)
low_risk_roles = [
    'jeff.dasovich@enron.com',
    'richard.shapiro@enron.com',
    'james.steffes@enron.com',
    'steven.kean@enron.com'
]

# =====================================================
# SCORING FUNCTIONS
# =====================================================

def calculate_role_multiplier(sender_email):
    """Adjust score based on sender's role and access to material info"""
    sender = str(sender_email).lower().strip()
    
    if sender in high_risk_insiders:
        return 2.0  # Finance/accounting execs
    elif sender in medium_risk_insiders:
        return 1.5  # Senior operations execs
    elif sender in low_risk_roles:
        return 0.5  # Government relations, PR
    else:
        return 1.0  # Unknown


def calculate_temporal_multiplier(email_date):
    """
    Heavily weight emails sent BEFORE major disclosures.
    Penalize emails sent AFTER disclosures (not insider trading).
    """
    
    # Critical dates
    Q1_EARNINGS = datetime(2001, 4, 17)
    SKILLING_RESIGN = datetime(2001, 8, 14)
    WATKINS_MEMO = datetime(2001, 8, 15)
    Q3_DISCLOSURE = datetime(2001, 10, 16)  # THE major disclosure
    SEC_INVESTIGATION = datetime(2001, 10, 22)
    
    if pd.isna(email_date):
        return 0.5
    
    # Convert to datetime if needed
    if not isinstance(email_date, datetime):
        try:
            email_date = pd.to_datetime(email_date)
        except:
            return 0.5
    
    # AFTER October 16 disclosure = NOT insider trading
    if email_date >= Q3_DISCLOSURE:
        return 0.1  # Heavily penalize
    
    # Between Watkins memo (Aug 15) and Oct 16 = PEAK WINDOW
    if WATKINS_MEMO <= email_date < Q3_DISCLOSURE:
        days_before = (Q3_DISCLOSURE - email_date).days
        if 1 <= days_before <= 7:
            return 5.0  # Week before = extremely suspicious
        elif 8 <= days_before <= 30:
            return 4.0  # Month before = very suspicious  
        else:
            return 3.0  # Aug 15 - Oct 16 window
    
    # Between Skilling resignation and Watkins memo
    if SKILLING_RESIGN <= email_date < WATKINS_MEMO:
        return 2.5
    
    # April - August (problems exist but not yet critical)
    if Q1_EARNINGS <= email_date < SKILLING_RESIGN:
        return 2.0
    
    # Before April 2001
    return 1.0


def score_email_comprehensive(row):
    """
    Comprehensive scoring that aligns with wrongdoing definition
    """
    
    text = str(row.get('clean', '')).lower()
    email_date = row.get('Date')
    sender = row.get('From', '')
    
    # 1. Category-based keyword scoring
    partnership_score = sum(10 for term in partnership_fraud_terms if term in text)
    trading_score = sum(8 for term in trading_terms if term in text)
    material_score = sum(5 for term in material_info_terms if term in text)
    routine_penalty = sum(-3 for term in routine_business_terms if term in text)
    
    keyword_score = partnership_score + trading_score + material_score + routine_penalty
    
    # Require at least some keyword matches
    if keyword_score <= 0:
        return 0
    
    # 2. Temporal multiplier (most important!)
    temporal_mult = calculate_temporal_multiplier(email_date)
    
    # 3. Role multiplier
    role_mult = calculate_role_multiplier(sender)
    
    # 4. Final score
    final_score = keyword_score * temporal_mult * role_mult
    
    return final_score


def get_diverse_top_emails(df, n=10, max_per_sender=2):
    """
    Get top N emails with sender diversity
    """
    
    df_sorted = df[df['final_score'] > 0].sort_values('final_score', ascending=False)
    
    selected = []
    sender_counts = {}
    
    for _, row in df_sorted.iterrows():
        sender = row['From']
        
        if sender_counts.get(sender, 0) < max_per_sender:
            selected.append(row)
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
            
            if len(selected) >= n:
                break
    
    if len(selected) == 0:
        print("⚠️  No emails passed the diversity filter!")
        return pd.DataFrame()
    
    return pd.DataFrame(selected)


# =====================================================
# MAIN PIPELINE
# =====================================================

print("="*70)
print("ENRON INSIDER TRADING DETECTION - IMPROVED VERSION")
print("="*70)

# Load data
csv_path = "emails.csv"
if not os.path.exists(csv_path):
    print(f"❌ Error: {csv_path} not found")
    print("Please run the email extraction script first")
    exit(1)

df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} emails")

# Parse email fields
print("\n⏳ Parsing email headers...")

def parse_fields(raw_text):
    def extract(pattern):
        match = re.search(pattern, raw_text, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    from_field = extract(r"^From:\s*(.*)$")
    to_field = extract(r"^To:\s*(.*)$")
    subject_field = extract(r"^Subject:\s*(.*)$")
    date_field = extract(r"^Date:\s*(.*)$")
    
    parts = re.split(r"\n\s*\n", raw_text, 1)
    body = parts[1] if len(parts) > 1 else ""
    
    return pd.Series({
        "From": from_field,
        "To": to_field,
        "Subject": subject_field,
        "Date": date_field,
        "Message": body
    })

parsed = df["message"].apply(parse_fields)
emails = parsed.copy()

# Clean up
emails = emails.dropna(subset=["From", "To"])
emails["From"] = emails["From"].str.lower().str.strip()
emails["To"] = emails["To"].str.lower().str.strip()
emails["Message"] = emails["Message"].astype(str)

print(f"✅ Parsed {len(emails)} emails")

# Network analysis (optional - keep for centrality info)
print("\n⏳ Building communication network...")
records = []
for _, row in emails.iterrows():
    tos = str(row["To"]).split(",")
    for t in tos:
        t = t.strip()
        if t:
            records.append({"from": row["From"], "to": t})

edges_df = pd.DataFrame(records)
G = nx.from_pandas_edgelist(edges_df, "from", "to", create_using=nx.DiGraph())
print(f"📊 Network: {len(G.nodes())} nodes, {len(G.edges())} edges")

# Calculate centrality for top people
print("🔍 Calculating centrality...")
centrality = nx.betweenness_centrality_subset(
    G,
    sources=list(G.nodes())[:300],
    targets=list(G.nodes())[:300],
    normalized=True
)

top_people = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
important_people = [p for p, _ in top_people]

# Filter to important people OR high-risk insiders
all_important = list(set(important_people + high_risk_insiders + medium_risk_insiders))
filtered = emails[
    emails["From"].isin(all_important) | 
    emails["To"].str.contains('|'.join(high_risk_insiders), case=False, na=False)
]

print(f"✅ Filtered to {len(filtered)} emails from key people")

# Clean text
print("\n⏳ Cleaning text...")
filtered["clean"] = (
    filtered["Message"]
    .str.lower()
    .str.replace(r"[^a-z\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

# Parse dates safely
print("⏳ Parsing dates...")
def safe_parse_date(x):
    try:
        return parser.parse(str(x), ignoretz=True)
    except:
        return None

filtered["Date"] = filtered["Date"].apply(safe_parse_date)
filtered = filtered.dropna(subset=["Date"])

print(f"✅ {len(filtered)} emails with valid dates")

# Filter to crisis window (April - December 2001)
start_date = datetime(2001, 4, 1)
end_date = datetime(2001, 12, 31)
crisis_emails = filtered[
    (filtered["Date"] >= start_date) & 
    (filtered["Date"] <= end_date)
].copy()

print(f"📅 {len(crisis_emails)} emails in crisis window (Apr-Dec 2001)")

# Apply comprehensive scoring
print("\n⏳ Calculating comprehensive scores...")
crisis_emails["final_score"] = crisis_emails.apply(score_email_comprehensive, axis=1)

# Filter to emails with positive scores
scored_emails = crisis_emails[crisis_emails["final_score"] > 0].copy()
print(f"✅ {len(scored_emails)} emails with positive scores")

if len(scored_emails) == 0:
    print("\n❌ No emails found matching the wrongdoing criteria!")
    print("This suggests:")
    print("  1. LJM/Chewco emails may have been deleted")
    print("  2. Sophisticated traders avoided email trails")
    print("  3. Most incriminating communications occurred via phone/in-person")
    exit(0)

# Get diverse top 10
print("\n⏳ Selecting top 10 emails with sender diversity...")
top10 = get_diverse_top_emails(scored_emails, n=10, max_per_sender=2)

if len(top10) == 0:
    print("❌ No diverse emails found. Falling back to top 10 overall...")
    top10 = scored_emails.sort_values("final_score", ascending=False).head(10)

# Save results
top10.to_csv("top10_improved_insider_trading.csv", index=False)

print("\n" + "="*70)
print("✅ TOP 10 SUSPICIOUS EMAILS (IMPROVED METHODOLOGY)")
print("="*70 + "\n")

for idx, (_, row) in enumerate(top10.iterrows(), 1):
    print(f"EMAIL #{idx}")
    print("-" * 70)
    print(f"From:    {row['From']}")
    print(f"To:      {row['To'][:80]}...")
    print(f"Subject: {row['Subject']}")
    print(f"Date:    {row['Date']}")
    print(f"Score:   {row['final_score']:.1f}")
    
    # Show which keywords triggered
    text = row['clean']
    found_partnership = [t for t in partnership_fraud_terms if t in text]
    found_trading = [t for t in trading_terms if t in text]
    found_material = [t for t in material_info_terms if t in text]
    
    if found_partnership:
        print(f"🚨 Partnership terms: {', '.join(found_partnership[:3])}")
    if found_trading:
        print(f"💰 Trading terms: {', '.join(found_trading[:3])}")
    if found_material:
        print(f"🔒 Material info terms: {', '.join(found_material[:3])}")
    
    print(f"Snippet: {row['Message'][:300].replace(chr(10), ' ')}...")
    print()

print("="*70)
print("FILES CREATED:")
print("  - top10_improved_insider_trading.csv")
print("="*70)

# Summary statistics
print("\nSUMMARY STATISTICS:")
print(f"Total emails in dataset: {len(df)}")
print(f"Emails in crisis window: {len(crisis_emails)}")
print(f"Emails with positive scores: {len(scored_emails)}")
print(f"Top scoring emails: {len(top10)}")

if len(scored_emails) > 0:
    print(f"\nScore range: {scored_emails['final_score'].min():.1f} - {scored_emails['final_score'].max():.1f}")
    print(f"Mean score: {scored_emails['final_score'].mean():.1f}")
    print(f"Median score: {scored_emails['final_score'].median():.1f}")

# Date distribution of top results
if len(top10) > 0:
    print("\nDate distribution of top 10:")
    date_counts = top10.groupby(top10['Date'].dt.date).size()
    for date, count in date_counts.items():
        print(f"  {date}: {count} email(s)")
