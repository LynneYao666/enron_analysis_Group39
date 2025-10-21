# ---------- Phase 0: Setup and Data Preparation----------
# ----------  Load all Enron emails (simplified) ----------
import os
import pandas as pd

base_dir = os.path.expanduser("~/Desktop/maildir")  # path

rows = []
for root, dirs, files in os.walk(base_dir):
    for name in files:
        if name.startswith("."):  # skip hidden files
            continue
        path = os.path.join(root, name)
        try:
            with open(path, "r", errors="ignore") as f:
                content = f.read()
                rows.append({"path": path, "text": content})
        except Exception as e:
            print("Skipping:", path, "| Reason:", e)

df = pd.DataFrame(rows)
print("✅ Total emails loaded:", len(df))

# Show one example to confirm it worked
if not df.empty:
    print("\nSample email path:", df.iloc[0]["path"])
    print(df.iloc[0]["text"][:400])
# ---------- Phase 1: Parse Email Headers ----------
import re

def parse_email_fields(text):
    fields = {}
    for key in ["From", "To", "Subject", "Date"]:
        match = re.search(rf"^{key}:(.*)$", text, re.MULTILINE)
        fields[key.lower()] = match.group(1).strip() if match else None
    body_split = re.split(r"\n\n", text, 1)
    fields["body"] = body_split[1] if len(body_split) > 1 else ""
    return fields

parsed = df["text"].apply(parse_email_fields)
emails = pd.DataFrame(parsed.tolist())
emails.head()
# ---------- Phase 1: Clean and Normalize ----------

# Drop emails missing sender or recipient
emails = emails.dropna(subset=["from", "to"])

# Convert all to lowercase for consistency
emails["from"] = emails["from"].str.lower().str.strip()
emails["to"] = emails["to"].str.lower().str.strip()

print("Emails with valid sender and recipient:", len(emails))
emails.head()
# ---------- Phase 1: Expand multiple recipients ----------
records = []
for _, row in emails.iterrows():
    for recipient in str(row["to"]).split(","):
        recipient = recipient.strip().lower()
        if recipient:
            records.append({"from": row["from"], "to": recipient})

edges_df = pd.DataFrame(records)
print("Edges created (email pairs):", len(edges_df))
edges_df.head()
# ---------- Phase 1: Build communication network ----------
import networkx as nx

# Build directed graph
G = nx.from_pandas_edgelist(edges_df, "from", "to", create_using=nx.DiGraph())

print(f"Network built. Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

# Compute centrality (how important each person is)
centrality = nx.betweenness_centrality(G)
top_people = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]

print("\nTop 15 most connected people:")
for person, score in top_people:
    print(f"{person:<40} {score:.5f}")
# ---------- Phase 1: Complete ----------
important_people = [p for p, _ in top_people]
filtered_emails = emails[
    emails["from"].isin(important_people) | emails["to"].isin(important_people)
]
filtered_emails.to_csv("phase1_filtered_emails.csv", index=False)

print("Phase 1 complete — filtered dataset saved as phase1_filtered_emails.csv")
