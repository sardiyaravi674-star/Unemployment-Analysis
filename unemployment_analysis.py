import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Unemployment.csv")


plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#2e3044",
    "axes.labelcolor":   "#c9ccd6",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   "#e8eaf0",
    "axes.titlepad":     14,
    "axes.grid":         True,
    "grid.color":        "#2e3044",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.6,
    "xtick.color":       "#8a8fa8",
    "ytick.color":       "#8a8fa8",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#2e3044",
    "legend.labelcolor": "#c9ccd6",
    "legend.fontsize":   9,
    "text.color":        "#c9ccd6",
    "font.family":       "DejaVu Sans",
    "figure.dpi":        110,
})

ACCENT = "#6c63ff"
TEAL   = "#00c9a7"
CORAL  = "#ff6b6b"
AMBER  = "#ffd166"
MUTED  = "#8a8fa8"
RURAL  = "#6c63ff"
URBAN  = "#00c9a7"

def style_spine(ax, left=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#2e3044")
    if left:
        ax.spines["left"].set_color("#2e3044")



print("=" * 55)
print("  Step 1 · Loading & cleaning the dataset")
print("=" * 55)

df = pd.read_csv(r"C:\Users\ravi\Downloads\Unemployment.csv")
df.columns = df.columns.str.strip()
df.rename(columns={
    "Date":                                    "date",
    "Frequency":                               "frequency",
    "Estimated Unemployment Rate (%)":         "unemployment_rate",
    "Estimated Employed":                      "employed",
    "Estimated Labour Participation Rate (%)": "labour_participation",
    "Area":                                    "area",
    "Region":                                  "region",
}, inplace=True)

df["date"] = pd.to_datetime(df["date"].str.strip(), format="%d-%m-%Y")
df.dropna(subset=["unemployment_rate", "region", "area"], inplace=True)
df.reset_index(drop=True, inplace=True)

df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["month_name"] = df["date"].dt.strftime("%b")
df["year_month"] = df["date"].dt.to_period("M")

df["phase"] = df["date"].apply(
    lambda d: "Pre-COVID" if d < pd.Timestamp("2020-03-01") else "Lockdown"
)

print(f"  Rows after cleaning : {len(df)}")
print(f"  Date range          : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"  States / UTs        : {df['region'].nunique()}")
print(f"  Area types          : {sorted(df['area'].unique())}")
print(f"  Missing values left : {df.isnull().sum().sum()}\n")



#  STEP 1 — EXPLORATORY ANALYSIS

print("=" * 55)
print("  Step 2 · Exploratory analysis")
print("=" * 55)

rural = df[df["area"] == "Rural"]["unemployment_rate"]
urban = df[df["area"] == "Urban"]["unemployment_rate"]

print("\n  Unemployment Rate — overall statistics")
for k, v in df["unemployment_rate"].describe().items():
    print(f"    {k:<8}: {v:>7.2f}%")

print(f"\n  Rural  — mean: {rural.mean():.2f}%   median: {rural.median():.2f}%")
print(f"  Urban  — mean: {urban.mean():.2f}%   median: {urban.median():.2f}%")

state_avg = (df.groupby("region")["unemployment_rate"]
               .mean().sort_values(ascending=False))

print("\n  Top 5 highest avg unemployment states:")
for st, val in state_avg.head(5).items():
    print(f"    {st:<22}: {val:.1f}%")
print("\n  Top 5 lowest avg unemployment states:")
for st, val in state_avg.tail(5).items():
    print(f"    {st:<22}: {val:.1f}%\n")



#  STEP 2 — TIME SERIES

print("=" * 55)
print("  Step 3 · Time series & rolling average")
print("=" * 55)

monthly = (df.groupby("year_month")["unemployment_rate"]
             .mean().reset_index())
monthly["date_dt"]  = monthly["year_month"].dt.to_timestamp()
monthly["rolling3"] = monthly["unemployment_rate"].rolling(3, center=True).mean()

peak_row   = monthly.loc[monthly["unemployment_rate"].idxmax()]
trough_row = monthly.loc[monthly["unemployment_rate"].idxmin()]
print(f"  Peak  : {peak_row['year_month']}  →  {peak_row['unemployment_rate']:.1f}%")
print(f"  Trough: {trough_row['year_month']}  →  {trough_row['unemployment_rate']:.1f}%\n")

dates  = monthly["date_dt"]
values = monthly["unemployment_rate"]
roll   = monthly["rolling3"]



#  STEP 3 — COVID IMPACT

print("=" * 55)
print("  Step 4 · COVID-19 impact")
print("=" * 55)

phase_order  = ["Pre-COVID", "Lockdown"]
phase_colors = [TEAL, CORAL]
phase_avg    = df.groupby("phase")["unemployment_rate"].mean().reindex(phase_order)
phase_area   = (df.groupby(["phase", "area"])["unemployment_rate"]
                  .mean().unstack().reindex(phase_order))

for ph in phase_order:
    print(f"  {ph:<12}: {phase_avg[ph]:.2f}%  "
          f"(Rural {phase_area.loc[ph,'Rural']:.1f}%  "
          f"Urban {phase_area.loc[ph,'Urban']:.1f}%)")
print()



#  STEP 4 — SEASONAL & REGIONAL PATTERNS

print("=" * 55)
print("  Step 5 · Seasonal & regional patterns")
print("=" * 55)

month_order  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
seasonal     = (df.groupby("month_name")["unemployment_rate"]
                  .mean().reindex(month_order))
peak_month   = seasonal.idxmax()
trough_month = seasonal.idxmin()
print(f"  Highest avg month : {peak_month}  ({seasonal[peak_month]:.1f}%)")
print(f"  Lowest  avg month : {trough_month}  ({seasonal[trough_month]:.1f}%)")

heatmap_df = (df.groupby(["region", "month_name"])["unemployment_rate"]
                .mean().unstack().reindex(columns=month_order))

corr = df[["unemployment_rate", "labour_participation"]].corr().iloc[0, 1]
print(f"  Correlation (unemployment ↔ labour participation): {corr:.3f}\n")


#  STEP 5 — BUILD & DISPLAY ALL CHARTS

print("=" * 55)
print("  Step 6 · Building all charts ...")
print("=" * 55)


# ── Figure 1 — State-wise bar chart ─────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5.5))
fig1.suptitle("State-wise Average Unemployment Rate",
              fontsize=15, fontweight="bold", color="#e8eaf0")

for ax, data, title, color in [
    (axes1[0], state_avg.head(10), "Top 10 — Highest Unemployment",   CORAL),
    (axes1[1], state_avg.tail(10), "Bottom 10 — Lowest Unemployment",  TEAL),
]:
    bars = ax.barh(data.index, data.values, color=color,
                   alpha=0.85, height=0.6, linewidth=0)
    ax.set_xlabel("Average Unemployment Rate (%)", labelpad=8)
    ax.set_title(title)
    style_spine(ax)
    ax.invert_yaxis()
    for bar, val in zip(bars, data.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8.5, color="#e8eaf0")
    ax.set_xlim(0, data.values.max() * 1.18)

fig1.tight_layout()
print("  ✓  Figure 1 ready — State comparison")


# ── Figure 2 — National time series ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2.suptitle("National Unemployment Rate Over Time (2019 – 2020)",
              fontsize=15, fontweight="bold", color="#e8eaf0")

ax2.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
            alpha=0.18, color=CORAL, label="Lockdown period")
ax2.fill_between(dates, values, alpha=0.15, color=ACCENT)
ax2.plot(dates, values, color=ACCENT, lw=1.8, alpha=0.6, label="Monthly avg")
ax2.plot(dates, roll,   color=TEAL,   lw=2.5,             label="3-month rolling avg")
ax2.annotate(
    f"  Peak: {peak_row['unemployment_rate']:.1f}%\n  ({peak_row['year_month']})",
    xy=(peak_row["date_dt"], peak_row["unemployment_rate"]),
    xytext=(peak_row["date_dt"] - pd.DateOffset(months=3),
            peak_row["unemployment_rate"] - 4),
    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.4),
    fontsize=9, color=CORAL,
)
ax2.set_ylabel("Unemployment Rate (%)", labelpad=8)
ax2.legend(loc="upper left")
style_spine(ax2)
fig2.tight_layout()
print("  ✓  Figure 2 ready — Time series")


# ── Figure 3 — COVID impact ──────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5.5))
fig3.suptitle("COVID-19 Impact on Unemployment",
              fontsize=15, fontweight="bold", color="#e8eaf0")

# overall phase bars
bars3 = axes3[0].bar(phase_avg.index, phase_avg.values,
                     color=phase_colors, width=0.5, linewidth=0, alpha=0.9)
axes3[0].set_ylabel("Avg Unemployment Rate (%)", labelpad=8)
axes3[0].set_title("Overall — by Phase")
style_spine(axes3[0])
for bar, val in zip(bars3, phase_avg.values):
    axes3[0].text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                  f"{val:.1f}%", ha="center", fontsize=10,
                  color="#e8eaf0", fontweight="bold")
axes3[0].set_ylim(0, phase_avg.max() * 1.2)

# Rural vs Urban side-by-side
x   = np.arange(len(phase_order))
w   = 0.35
b_r = axes3[1].bar(x - w/2, phase_area["Rural"], w,
                   color=RURAL, alpha=0.85, label="Rural", linewidth=0)
b_u = axes3[1].bar(x + w/2, phase_area["Urban"], w,
                   color=URBAN, alpha=0.85, label="Urban", linewidth=0)
axes3[1].set_xticks(x)
axes3[1].set_xticklabels(phase_order)
axes3[1].set_ylabel("Avg Unemployment Rate (%)", labelpad=8)
axes3[1].set_title("Rural vs Urban — by Phase")
axes3[1].legend()
style_spine(axes3[1])
for bar in list(b_r) + list(b_u):
    axes3[1].text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.3,
                  f"{bar.get_height():.1f}",
                  ha="center", fontsize=8.5, color="#e8eaf0")
axes3[1].set_ylim(0, phase_area.values.max() * 1.2)

fig3.tight_layout()
print("  ✓  Figure 3 ready — COVID impact")


# ── Figure 4 — Seasonal patterns ────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.5))
fig4.suptitle("Seasonal Trends in Unemployment",
              fontsize=15, fontweight="bold", color="#e8eaf0")

# monthly seasonality line
ax4l = axes4[0]
ax4l.plot(month_order, seasonal.values, color=AMBER, lw=2.5,
          marker="o", markersize=6,
          markerfacecolor="#0f1117", markeredgecolor=AMBER, markeredgewidth=2)
ax4l.fill_between(range(12), seasonal.values, alpha=0.12, color=AMBER)
ax4l.set_xticks(range(12))
ax4l.set_xticklabels(month_order)
ax4l.set_ylabel("Avg Unemployment Rate (%)", labelpad=8)
ax4l.set_title("Monthly Seasonality (all years averaged)")
style_spine(ax4l)
ax4l.annotate(peak_month,
    xy=(month_order.index(peak_month), seasonal[peak_month]),
    xytext=(month_order.index(peak_month) - 1.5, seasonal[peak_month] + 1.2),
    fontsize=9, color=CORAL,
    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))
ax4l.annotate(trough_month,
    xy=(month_order.index(trough_month), seasonal[trough_month]),
    xytext=(month_order.index(trough_month) + 0.3, seasonal[trough_month] + 1.2),
    fontsize=9, color=TEAL,
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.2))

# state × month heatmap
ax4r = axes4[1]
data_vals = heatmap_df.values
vmin = np.nanpercentile(data_vals, 5)
vmax = np.nanpercentile(data_vals, 95)
im = ax4r.imshow(data_vals, aspect="auto", cmap=plt.cm.RdYlGn_r,
                 vmin=vmin, vmax=vmax, interpolation="nearest")
ax4r.set_xticks(range(len(month_order)))
ax4r.set_xticklabels(month_order, rotation=0, fontsize=8)
ax4r.set_yticks(range(len(heatmap_df.index)))
ax4r.set_yticklabels(heatmap_df.index, fontsize=7)
ax4r.set_title("State × Month Heatmap")
ax4r.set_facecolor("#1a1d27")
cbar = fig4.colorbar(im, ax=ax4r, pad=0.02, fraction=0.03)
cbar.ax.tick_params(labelsize=8, colors="#8a8fa8")
cbar.set_label("Unemployment Rate (%)", fontsize=8, color="#8a8fa8")

fig4.tight_layout()
print("  ✓  Figure 4 ready — Seasonal & heatmap")


# ── Figure 5 — Correlation scatter ──────────────────────────
fig5, ax5 = plt.subplots(figsize=(8, 5.5))
fig5.suptitle("Labour Participation vs Unemployment Rate",
              fontsize=15, fontweight="bold", color="#e8eaf0")

colors_area = df["area"].map({"Rural": RURAL, "Urban": URBAN})
ax5.scatter(df["labour_participation"], df["unemployment_rate"],
            c=colors_area, alpha=0.45, s=22, linewidths=0)

m, b_coef = np.polyfit(
    df["labour_participation"].dropna(),
    df.loc[df["labour_participation"].notna(), "unemployment_rate"], 1
)
x_line = np.linspace(df["labour_participation"].min(),
                     df["labour_participation"].max(), 200)
ax5.plot(x_line, m * x_line + b_coef, color=AMBER, lw=2)
ax5.set_xlabel("Labour Participation Rate (%)", labelpad=8)
ax5.set_ylabel("Unemployment Rate (%)", labelpad=8)
ax5.legend(handles=[
    mpatches.Patch(color=RURAL, label="Rural"),
    mpatches.Patch(color=URBAN, label="Urban"),
    plt.Line2D([0], [0], color=AMBER, lw=2, label=f"Trend  (r = {corr:.2f})"),
])
style_spine(ax5)
fig5.tight_layout()
print("  ✓  Figure 5 ready — Correlation scatter")


# ── Figure 6 — Policy Dashboard ─────────────────────────────
fig6 = plt.figure(figsize=(16, 10), facecolor="#0f1117")
fig6.suptitle("India Unemployment Analysis — Policy Dashboard",
              fontsize=17, fontweight="bold", color="#e8eaf0", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig6, hspace=0.42, wspace=0.35)

# Panel A: time series
axA = fig6.add_subplot(gs[0, :2])
axA.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-30"),
            alpha=0.18, color=CORAL)
axA.fill_between(dates, values, alpha=0.12, color=ACCENT)
axA.plot(dates, values, color=ACCENT, lw=1.6, alpha=0.6)
axA.plot(dates, roll,   color=TEAL,   lw=2.4, label="3-month avg")
axA.annotate(f"  Peak {peak_row['unemployment_rate']:.0f}%",
    xy=(peak_row["date_dt"], peak_row["unemployment_rate"]),
    xytext=(peak_row["date_dt"] - pd.DateOffset(months=2),
            peak_row["unemployment_rate"] - 5),
    arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.3),
    fontsize=8.5, color=CORAL)
axA.set_title("A · National Unemployment Trend")
axA.set_ylabel("Rate (%)", labelpad=6)
axA.legend(fontsize=8, loc="upper left")
style_spine(axA)

# Panel B: KPI tiles
axB = fig6.add_subplot(gs[0, 2])
axB.set_facecolor("#0f1117")
axB.axis("off")
kpis = [
    ("Overall mean",        f"{df['unemployment_rate'].mean():.1f}%", ACCENT),
    ("COVID lockdown peak", f"{phase_avg['Lockdown']:.1f}%",          CORAL),
    ("Pre-COVID baseline",  f"{phase_avg['Pre-COVID']:.1f}%",         TEAL),
    ("Urban avg",           f"{urban.mean():.1f}%",                   URBAN),
    ("Rural avg",           f"{rural.mean():.1f}%",                   RURAL),
    ("Participation corr.", f"r = {corr:.2f}",                        AMBER),
]
for i, (label, value, color) in enumerate(kpis):
    yp = 0.90 - i * 0.155
    axB.text(0.05, yp,        label, transform=axB.transAxes,
             fontsize=9,  color=MUTED)
    axB.text(0.05, yp - 0.07, value, transform=axB.transAxes,
             fontsize=14, color=color, fontweight="bold")
axB.set_title("B · Key Metrics")

# Panel C: COVID phase bars
axC = fig6.add_subplot(gs[1, 0])
bars_c = axC.bar(phase_avg.index, phase_avg.values,
                 color=phase_colors, width=0.5, linewidth=0, alpha=0.9)
axC.set_title("C · COVID Phase Averages")
axC.set_ylabel("Rate (%)", labelpad=6)
style_spine(axC)
for bar, val in zip(bars_c, phase_avg.values):
    axC.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
             f"{val:.1f}%", ha="center", fontsize=8.5,
             color="#e8eaf0", fontweight="bold")
axC.set_ylim(0, phase_avg.max() * 1.22)

# Panel D: Top 8 states
axD = fig6.add_subplot(gs[1, 1])
top8   = state_avg.head(8)
bars_d = axD.barh(top8.index, top8.values, color=CORAL,
                  alpha=0.8, height=0.6, linewidth=0)
axD.invert_yaxis()
axD.set_title("D · Highest Unemployment States")
axD.set_xlabel("Avg Rate (%)", labelpad=6)
style_spine(axD, left=False)
axD.spines["left"].set_visible(False)
axD.tick_params(left=False)
for bar, val in zip(bars_d, top8.values):
    axD.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=8, color="#e8eaf0")
axD.set_xlim(0, top8.values.max() * 1.2)

# Panel E: seasonal line
axE = fig6.add_subplot(gs[1, 2])
axE.plot(month_order, seasonal.values, color=AMBER, lw=2.2,
         marker="o", markersize=5,
         markerfacecolor="#0f1117", markeredgecolor=AMBER, markeredgewidth=1.8)
axE.fill_between(range(12), seasonal.values, alpha=0.1, color=AMBER)
axE.set_xticks(range(12))
axE.set_xticklabels(month_order, rotation=45, ha="right", fontsize=8)
axE.set_title("E · Monthly Seasonality")
axE.set_ylabel("Avg Rate (%)", labelpad=6)
style_spine(axE)

fig6.text(0.5, 0.01,
    "Data: CMIE via Kaggle  |  Period: May 2019 – Jun 2020  |  740 observations",
    ha="center", fontsize=8, color=MUTED, style="italic")

fig6.tight_layout(rect=[0, 0.03, 1, 0.97])
print("  ✓  Figure 6 ready — Policy dashboard\n")



print("=" * 55)
print("  Policy Insights Summary")
print("=" * 55)
high_states = state_avg.head(5).index.tolist()
low_states  = state_avg.tail(5).index.tolist()
print(f"""
  1. COVID Shock
     Unemployment jumped from {phase_avg['Pre-COVID']:.1f}% (pre-COVID)
     to {phase_avg['Lockdown']:.1f}% during the lockdown — a rise of
     {phase_avg['Lockdown'] - phase_avg['Pre-COVID']:.1f} percentage points.

  2. Urban-Rural Divide
     Urban spike: {phase_area.loc['Lockdown','Urban']:.1f}%  vs
     Rural spike: {phase_area.loc['Lockdown','Rural']:.1f}%
     Rural structural unemployment needs MGNREGA reinforcement.

  3. States Needing Intervention
     High: {', '.join(high_states)}
     Low : {', '.join(low_states)}

  4. Seasonal Pattern
     Peaks in {peak_month} ({seasonal[peak_month]:.1f}%), dips in {trough_month} ({seasonal[trough_month]:.1f}%).
     Time skill-training programmes to pre-peak months.

  5. Labour Participation
     r = {corr:.2f} (weak link) — unemployment and participation
     are driven by different structural forces per state.
""")

print("  Opening all 6 figures ... (close any window to exit)\n")
plt.show()  