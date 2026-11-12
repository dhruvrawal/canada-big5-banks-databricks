"""Generate a SYNTHETIC retail-banking dataset for Canada's Big 5 banks.
All records are randomly generated (seeded). No real customer data is used."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

banks = pd.DataFrame({
    "bank_id": ["RBC", "TD", "BNS", "BMO", "CIBC"],
    "bank_name": ["Royal Bank of Canada", "Toronto-Dominion Bank", "Scotiabank",
                  "Bank of Montreal", "Canadian Imperial Bank of Commerce"],
    "head_office": ["Toronto", "Toronto", "Toronto", "Toronto", "Toronto"],
})
bank_w = np.array([0.27, 0.26, 0.19, 0.15, 0.13])

prov = {"ON": ("Toronto|Ottawa|Mississauga|Hamilton|London", .38), "QC": ("Montreal|Quebec City|Laval", .22),
        "BC": ("Vancouver|Victoria|Surrey", .14), "AB": ("Calgary|Edmonton", .12),
        "MB": ("Winnipeg", .04), "SK": ("Saskatoon|Regina", .03), "NS": ("Halifax", .03),
        "NB": ("Moncton|Fredericton", .02), "NL": ("St. John's", .01), "PE": ("Charlottetown", .01)}
pcodes = list(prov); pw = np.array([prov[p][1] for p in pcodes]); pw /= pw.sum()

# branches
rows = []
for b in banks.bank_id:
    for i in range(32):
        p = rng.choice(pcodes, p=pw); city = rng.choice(prov[p][0].split("|"))
        rows.append((f"{b}-BR{i+1:03d}", b, city, p, int(rng.integers(2005, 2023))))
branches = pd.DataFrame(rows, columns=["branch_id", "bank_id", "city", "province", "opened_year"])

# customers
N = 20000
cust = pd.DataFrame({
    "customer_id": [f"C{i:06d}" for i in range(1, N + 1)],
    "age": np.clip(rng.normal(44, 15, N), 18, 90).astype(int),
    "province": rng.choice(pcodes, N, p=pw),
    "segment": rng.choice(["Student", "Mass Market", "Affluent", "Small Business"], N, p=[.12, .56, .22, .10]),
    "annual_income": 0,
    "credit_score": 0,
    "primary_bank": rng.choice(banks.bank_id, N, p=bank_w / bank_w.sum()),
    "join_date": pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 3600, N), unit="D"),
})
base = cust.segment.map({"Student": 22000, "Mass Market": 62000, "Affluent": 140000, "Small Business": 95000})
cust["annual_income"] = (base * rng.lognormal(0, .25, N)).round(-2).astype(int)
cust["credit_score"] = np.clip(rng.normal(690, 70, N) + (cust.annual_income / 20000), 300, 900).astype(int)
churn_p = 0.08 + (cust.credit_score < 600) * .06 + (cust.segment == "Student") * .05
cust["churned"] = (rng.random(N) < churn_p).astype(int)

# accounts
acc = []
k = 1
for _, c in cust.iterrows():
    for t in rng.choice(["Chequing", "Savings", "Credit Card", "TFSA", "RRSP"],
                        size=rng.integers(1, 4), replace=False):
        acc.append((f"A{k:07d}", c.customer_id, c.primary_bank, t,
                    round(float(rng.lognormal(8.2 if t != "Credit Card" else 7, 1.1)), 2),
                    rng.choice(branches[branches.bank_id == c.primary_bank].branch_id))); k += 1
accounts = pd.DataFrame(acc, columns=["account_id", "customer_id", "bank_id", "account_type", "balance", "branch_id"])

# transactions
T = 300000
ta = accounts.sample(T, replace=True, random_state=1).reset_index(drop=True)
cat = rng.choice(["Groceries", "Rent/Mortgage", "Restaurants", "Transport", "Utilities",
                  "Online Shopping", "Travel", "Entertainment", "Healthcare", "Transfer"], T,
                 p=[.2, .1, .12, .1, .08, .14, .05, .08, .05, .08])
amt = rng.lognormal(3.6, 1.0, T).round(2)
amt[cat == "Rent/Mortgage"] *= 15; amt[cat == "Travel"] *= 6
date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, T), unit="D")
fraud_p = 0.003 + (amt > 1500) * .02 + (cat == "Online Shopping") * .004
tx = pd.DataFrame({"txn_id": [f"T{i:07d}" for i in range(1, T + 1)], "account_id": ta.account_id,
                   "txn_date": date.date, "category": cat, "channel": rng.choice(["POS", "Online", "ATM", "Mobile"], T, p=[.4, .3, .1, .2]),
                   "amount_cad": amt.round(2), "is_fraud": (rng.random(T) < fraud_p).astype(int)})

# loans
L = 9000
lc = cust.sample(L, random_state=2).reset_index(drop=True)
ltype = rng.choice(["Mortgage", "Auto", "Personal", "Line of Credit"], L, p=[.35, .25, .2, .2])
lamt = np.where(ltype == "Mortgage", rng.normal(420000, 120000, L), rng.normal(22000, 9000, L)).clip(3000).round(-2)
rate = (4.0 + (700 - lc.credit_score) / 120 + rng.normal(0, .4, L)).clip(2.5, 14).round(2)
dp = 0.02 + (lc.credit_score < 600) * .10 + (rate > 8) * .05 + (ltype == "Personal") * .03
loans = pd.DataFrame({"loan_id": [f"L{i:05d}" for i in range(1, L + 1)], "customer_id": lc.customer_id,
                      "bank_id": lc.primary_bank, "loan_type": ltype, "principal_cad": lamt,
                      "interest_rate_pct": rate, "term_months": rng.choice([12, 36, 60, 120, 300], L),
                      "defaulted": (rng.random(L) < dp).astype(int)})

for n, d in [("banks", banks), ("branches", branches), ("customers", cust),
             ("accounts", accounts), ("transactions", tx), ("loans", loans)]:
    d.to_csv(os.path.join(OUT, f"{n}.csv"), index=False); print(n, len(d))
