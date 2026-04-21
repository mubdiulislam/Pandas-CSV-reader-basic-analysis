# Sales Data Analysis — 2024
# Dataset: Bangladesh Regional Sales (500 orders)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load ───────────────────────────────────────────────────

df = pd.read_csv(os.path.join(DATA_DIR, "sales_data.csv"), parse_dates=["order_date"])

print("=" * 55)
print("  SALES DATA — 2024 ANALYSIS")
print("=" * 55)


# ── 2. Inspect ────────────────────────────────────────────────

print("\n-- First 5 rows --")
print(df.head())

print("\n-- Last 5 rows --")
print(df.tail())

# info() gives column types + non-null counts + memory in one shot
print("\n-- Dataset info --")
print(df.info())

print("\n-- Missing values --")
print(df.isnull().sum())

# describe() covers count, mean, std, min, 25%, 50%, 75%, max
print("\n-- Descriptive statistics --")
print(df[["quantity", "unit_price", "total_price", "discount_%"]].describe().round(2))


# ── 3. Add computed columns ───────────────────────────────────

# Label each order by size using pd.cut
df["order_size"] = pd.cut(
    df["total_price"],
    bins=[0, 5000, 50000, float("inf")],
    labels=["Small", "Medium", "Large"]
)

# Flag whether the order had a discount or not
df["discounted"] = np.where(df["discount_%"] > 0, "Yes", "No")

print("\n-- Order size breakdown --")
print(df["order_size"].value_counts().to_string())

print("\n-- Discounted vs full price --")
print(df["discounted"].value_counts().to_string())


# ── 4. Sort ───────────────────────────────────────────────────

print("\n-- Top 5 orders by value --")
top5 = df.sort_values("total_price", ascending=False).head(5)
print(top5[["order_id", "product", "region", "total_price"]].to_string(index=False))


# ── 5. Text filtering with str.contains ───────────────────────

print("\n-- Orders for Laptop or Chair --")
text_filter = df[df["product"].str.contains("Laptop|Chair", case=False)]
print(text_filter[["order_id", "product", "quantity", "total_price"]].head(6).to_string(index=False))


# ── 6. Filter & slice ─────────────────────────────────────────

delivered = df[df["status"] == "Delivered"].copy()
print(f"\n-- Delivered orders: {len(delivered)} / {len(df)}")

# High-value orders
big_orders = (
    df[df["total_price"] > 50_000]
    [["order_id", "order_date", "product", "total_price", "region"]]
    .sort_values("total_price", ascending=False)
    .copy()
)
print(f"\n-- Orders above BDT 50,000: {len(big_orders)}")
print(big_orders.head(5).to_string(index=False))

# Q1 2024
q1 = df[
    (df["order_date"] >= pd.Timestamp("2024-01-01")) &
    (df["order_date"] <= pd.Timestamp("2024-03-31"))
].copy()
print(f"\n-- Q1 2024: {len(q1)} orders  |  Revenue: BDT {q1['total_price'].sum():,.2f}")


# ── 7. Groupby & aggregation ──────────────────────────────────

print("\n-- Revenue by category --")
cat_rev = df.groupby("category")["total_price"].sum().sort_values(ascending=False)
print(cat_rev.apply(lambda x: f"BDT {x:,.2f}").to_string())

print("\n-- Sales rep performance (delivered only) --")
rep_perf = (
    delivered.groupby("sales_rep")["total_price"]
    .agg(orders="count", revenue="sum")
    .sort_values("revenue", ascending=False)
)
rep_display = rep_perf.copy()
rep_display["revenue"] = rep_display["revenue"].apply(lambda x: f"BDT {x:,.2f}")
print(rep_display.to_string())


# ── 8. Concatenate two regions ────────────────────────────────

dhaka      = df[df["region"] == "Dhaka"].copy()
chittagong = df[df["region"] == "Chittagong"].copy()
combined   = pd.concat([dhaka, chittagong])
print(f"\n-- Dhaka + Chittagong combined: {len(combined)} orders")


# ── 9. Iterate over a few rows ────────────────────────────────

print("\n-- Sample order walkthrough --")
for idx, row in df.head(3).iterrows():
    print(f"  {row['order_id']} | {row['product']:15s} | {row['sales_rep']:15s} | BDT {row['total_price']:>10,.2f}")


# ── 10. Charts ────────────────────────────────────────────────

df["month"] = df["order_date"].dt.to_period("M")

monthly_rev = (
    delivered.groupby(df["month"])["total_price"]
    .sum()
    .reset_index()
)
monthly_rev["month"] = monthly_rev["month"].astype(str)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Sales Analysis — 2024", fontsize=14, fontweight="bold")

# Chart 1 — Revenue by category (bar)
cat_rev.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Revenue by Category")
axes[0].set_xlabel("")
axes[0].set_ylabel("Total Revenue (BDT)")
axes[0].tick_params(axis="x", rotation=30)

# Chart 2 — Monthly revenue trend (line)
axes[1].plot(monthly_rev["month"], monthly_rev["total_price"], marker="o", color="darkorange")
axes[1].set_title("Monthly Revenue (Delivered)")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Revenue (BDT)")
axes[1].tick_params(axis="x", rotation=45)

# Chart 3 — Order value distribution (histogram)
axes[2].hist(df["total_price"], bins=30, color="seagreen", edgecolor="white")
axes[2].set_title("Order Value Distribution")
axes[2].set_xlabel("Order Value (BDT)")
axes[2].set_ylabel("Number of Orders")

plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "charts.png")
plt.savefig(chart_path, dpi=150)
plt.show()
print(f"\n[OK] Chart saved -> {chart_path}")


# ── 11. Save results ──────────────────────────────────────────

print("\n" + "=" * 55)
print("  SAVING RESULTS")
print("=" * 55)

delivered_path = os.path.join(OUTPUT_DIR, "delivered_orders.csv")
delivered.to_csv(delivered_path, index=False)
print(f"\n[OK] {delivered_path}  ({len(delivered)} rows)")

big_orders_path = os.path.join(OUTPUT_DIR, "high_value_orders.csv")
big_orders.to_csv(big_orders_path, index=False)
print(f"[OK] {big_orders_path}  ({len(big_orders)} rows)")

xlsx_path = os.path.join(OUTPUT_DIR, "summary_report.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

    df[["quantity", "unit_price", "total_price", "discount_%"]].describe().round(2).to_excel(
        writer, sheet_name="Overall Stats"
    )

    (
        df.groupby("category")
        .agg(
            total_orders=("order_id", "count"),
            total_revenue=("total_price", "sum"),
            avg_order_value=("total_price", "mean"),
            total_units_sold=("quantity", "sum"),
        )
        .round(2)
        .sort_values("total_revenue", ascending=False)
        .to_excel(writer, sheet_name="Category Breakdown")
    )

    monthly_rev.rename(columns={"month": "Month", "total_price": "Revenue (BDT)"}).to_excel(
        writer, sheet_name="Monthly Revenue", index=False
    )

    q1.drop(columns=["month", "order_size", "discounted"], errors="ignore").to_excel(
        writer, sheet_name="Q1 2024", index=False
    )

print(f"[OK] {xlsx_path}  (4 sheets)")

print("\nAll done.\n")