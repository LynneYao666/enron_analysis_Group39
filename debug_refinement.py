#!/usr/bin/env python3
"""Debug script to see why Phase 4 refinement returns 0 emails"""

import pandas as pd

# Load the top100 file
df = pd.read_csv("top100_suspicious_emails.csv")
print(f"Total emails before refinement: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}\n")

# Stop-list from Phase 4
stop_list = [
    "quarterly report", "annual report", "meeting agenda", 
    "conference call", "public announcement", "press release",
    "regulatory filing", "compliance", "audit committee"
]

print("="*80)
print("ANALYZING EACH EMAIL:")
print("="*80)

for idx, row in df.iterrows():
    print(f"\n--- Email {idx + 1} ---")
    print(f"From: {row['From']}")
    print(f"To: {row['To']}")
    print(f"Subject: {row['Subject']}")
    print(f"Risk Score: {row['risk_score']:.2f}")
    
    # Check for stop-list terms
    text = str(row['clean']).lower()
    found_terms = [term for term in stop_list if term in text]
    
    if found_terms:
        penalty = len(found_terms) * 2.0
        refined_score = max(0, row['risk_score'] - penalty)
        print(f"⚠️  Found stop-list terms: {found_terms}")
        print(f"   Penalty: {penalty:.2f}")
        print(f"   Refined Score: {refined_score:.2f} (FILTERED OUT)")
    else:
        refined_score = row['risk_score']
        print(f"✅ No stop-list terms found")
        print(f"   Refined Score: {refined_score:.2f} (KEPT)")
    
    print(f"Message snippet: {text[:100]}...")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

# Calculate refined scores
def refine_keyword_score(text, stop_list, base_score):
    penalty = sum(1 for stop_term in stop_list if stop_term in str(text).lower())
    return max(0, base_score - (penalty * 2.0))

df["refined_score"] = df.apply(
    lambda row: refine_keyword_score(row["clean"], stop_list, row["risk_score"]), axis=1
)

kept = df[df["refined_score"] > 0]
filtered_out = df[df["refined_score"] == 0]

print(f"Emails kept: {len(kept)}")
print(f"Emails filtered out: {len(filtered_out)}")
print(f"\nAverage risk_score before refinement: {df['risk_score'].mean():.2f}")
print(f"Average refined_score: {df['refined_score'].mean():.2f}")

if len(filtered_out) > 0:
    print(f"\n⚠️  All {len(filtered_out)} emails were filtered out because:")
    print("   - They contain stop-list terms (benign financial jargon)")
    print("   - The penalty (2.0 per term) reduced their scores to 0")
    print("\n💡 Suggestion: Reduce penalty multiplier or make stop-list less aggressive")

