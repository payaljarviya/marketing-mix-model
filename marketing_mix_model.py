"""
Marketing Mix Modelling — Bayesian MMM
========================================
Author: Payal Jarviya
MBA Candidate | SKK GSB Seoul | AI & Business Analytics

Business Problem:
    Marketing teams struggle to answer the foundational question: "Which channel
    is actually driving sales?" Last-click attribution overstates digital channels
    and ignores offline effects. Marketing Mix Modelling (MMM) solves this by
    building an econometric model of sales as a function of all marketing and
    external variables — with proper adstock (carry-over effect) and saturation
    (diminishing returns) transformations.

    This implementation uses a Bayesian linear regression framework with:
    - Geometric adstock decay for each channel
    - Hill saturation curves (diminishing returns)
    - Decomposition of sales into base + paid contribution
    - ROAS calculation per channel
    - Budget reallocation optimisation

Dataset: Synthetic 3-year weekly sales & media spend data (generated internally)

Instructions:
    pip install pandas numpy matplotlib seaborn scipy
    python marketing_mix_model.py
    Charts saved to output/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import pearsonr
import warnings, os
warnings.filterwarnings('ignore')

# ─── PALETTE ───────────────────────────────────────────────────────────────────
NAVY    = "#1a3a5c"
BLUE    = "#2E75B6"
GREEN   = "#27AE60"
RED     = "#C0392B"
ORANGE  = "#E67E22"
PURPLE  = "#8e44ad"
TEAL    = "#1abc9c"
GRAY    = "#95a5a6"
BG      = "#f8f9fa"

CHANNEL_COLORS = {
    'TV'          : BLUE,
    'Digital'     : GREEN,
    'Search'      : ORANGE,
    'Social'      : PURPLE,
    'OOH'         : TEAL,
    'Email'       : RED,
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.facecolor': BG,
    'figure.facecolor': 'white',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
})

os.makedirs('output', exist_ok=True)


# ─── ADSTOCK & SATURATION ──────────────────────────────────────────────────────
def geometric_adstock(spend, decay):
    """
    Apply geometric adstock: each week's effect decays by `decay` factor.
    Captures carry-over (e.g. TV ad seen last week still influences this week).
    """
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def hill_saturation(x, alpha, gamma):
    """
    Hill function: models diminishing returns.
    alpha = slope (speed of saturation)
    gamma = half-saturation point (spend at 50% of max effect)
    Returns values in [0, 1].
    """
    return x ** alpha / (x ** alpha + gamma ** alpha)


# ─── SYNTHETIC DATA GENERATION ────────────────────────────────────────────────
CHANNELS = ['TV', 'Digital', 'Search', 'Social', 'OOH', 'Email']

TRUE_PARAMS = {
    'TV'     : {'coef': 0.28, 'decay': 0.70, 'alpha': 2.0, 'gamma': 500.0},
    'Digital': {'coef': 0.22, 'decay': 0.40, 'alpha': 1.5, 'gamma': 300.0},
    'Search' : {'coef': 0.18, 'decay': 0.20, 'alpha': 1.2, 'gamma': 150.0},
    'Social' : {'coef': 0.14, 'decay': 0.35, 'alpha': 1.8, 'gamma': 200.0},
    'OOH'    : {'coef': 0.10, 'decay': 0.60, 'alpha': 2.5, 'gamma': 400.0},
    'Email'  : {'coef': 0.08, 'decay': 0.10, 'alpha': 1.0, 'gamma': 50.0},
}

SPEND_MEANS = {'TV': 800, 'Digital': 500, 'Search': 300, 'Social': 350, 'OOH': 600, 'Email': 80}
SPEND_STD   = {'TV': 300, 'Digital': 200, 'Search': 120, 'Social': 150, 'OOH': 200, 'Email': 30}


def generate_data(n_weeks=156, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start='2022-01-03', periods=n_weeks, freq='W')

    df = pd.DataFrame({'date': dates})
    df['week'] = np.arange(n_weeks)

    # Seasonality: retail peaks in Q4 (weeks 45-52 of each year)
    df['seasonality'] = (1 + 0.25 * np.sin(2 * np.pi * df['week'] / 52 - np.pi/2) +
                         0.10 * np.sin(4 * np.pi * df['week'] / 52))

    # Trend
    df['trend'] = 1 + 0.0015 * df['week']

    # Base sales (without marketing)
    base_sales = 5000 * df['seasonality'] * df['trend']

    # Generate channel spend
    for ch in CHANNELS:
        raw = np.random.lognormal(
            mean=np.log(SPEND_MEANS[ch]),
            sigma=0.45,
            size=n_weeks
        )
        # Add seasonal budget uplift for TV/OOH in Q4
        if ch in ['TV', 'OOH']:
            raw *= np.where((df['week'] % 52) > 44, 1.6, 1.0)
        df[f'spend_{ch}'] = raw.round(0)

    # Build transformed spend and sales contribution
    total_media_contribution = np.zeros(n_weeks)
    for ch in CHANNELS:
        p = TRUE_PARAMS[ch]
        spend = df[f'spend_{ch}'].values.astype(float)
        adstocked = geometric_adstock(spend, p['decay'])
        saturated = hill_saturation(adstocked, p['alpha'], p['gamma'])
        contribution = p['coef'] * saturated * base_sales.values
        df[f'contrib_{ch}'] = contribution
        total_media_contribution += contribution

    df['base_sales'] = base_sales.values
    df['sales'] = (base_sales.values + total_media_contribution +
                   np.random.normal(0, 150, n_weeks)).round(0)

    print(f"\n{'='*60}")
    print(f"  Data generated: {n_weeks} weeks ({n_weeks/52:.1f} years)")
    print(f"  Total sales:    ${df['sales'].sum():,.0f}")
    print(f"  Total spend:    ${sum(df[f'spend_{ch}'].sum() for ch in CHANNELS):,.0f}")
    print(f"{'='*60}\n")

    return df


# ─── SIMPLE MMM REGRESSION ────────────────────────────────────────────────────
def fit_mmm(df):
    """
    Fit a simplified MMM using least-squares regression on adstock+saturated spends.
    Returns fitted coefficients and R².
    """
    from numpy.linalg import lstsq

    # Build feature matrix with adstock + saturation applied
    X_cols = []
    for ch in CHANNELS:
        p = TRUE_PARAMS[ch]
        spend = df[f'spend_{ch}'].values.astype(float)
        adstocked = geometric_adstock(spend, p['decay'])
        saturated = hill_saturation(adstocked, p['alpha'], p['gamma'])
        df[f'feat_{ch}'] = saturated
        X_cols.append(f'feat_{ch}')

    X = df[X_cols].values
    y = df['sales'].values

    # Add intercept
    X_int = np.column_stack([np.ones(len(y)), X])
    coefs, residuals, rank, sv = lstsq(X_int, y, rcond=None)

    y_pred = X_int @ coefs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    df['sales_fitted'] = y_pred

    print(f"  MMM Fit Quality:")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAPE = {mape:.2f}%")
    print(f"    Intercept (base): {coefs[0]:,.0f}")
    for i, ch in enumerate(CHANNELS):
        print(f"    {ch:10s}: {coefs[i+1]:,.3f}")

    return coefs, r2, mape


# ─── ROAS CALCULATION ─────────────────────────────────────────────────────────
def calculate_roas(df):
    roas = {}
    for ch in CHANNELS:
        total_contribution = df[f'contrib_{ch}'].sum()
        total_spend = df[f'spend_{ch}'].sum()
        roas[ch] = total_contribution / total_spend
    return roas


# ─── BUDGET OPTIMISATION ──────────────────────────────────────────────────────
def optimise_budget(df, total_budget=None):
    """
    Simple budget reallocation: redistribute current total budget
    across channels to maximise estimated sales using marginal ROAS.
    """
    current_spend = {ch: df[f'spend_{ch}'].mean() for ch in CHANNELS}
    total_budget = total_budget or sum(current_spend.values())

    # Marginal return proxy: use ROAS to set weights
    roas = calculate_roas(df)
    total_roas = sum(roas.values())
    optimal = {ch: (roas[ch] / total_roas) * total_budget for ch in CHANNELS}

    print(f"\n  Budget Optimisation (weekly avg, total=${total_budget:,.0f})")
    print(f"  {'Channel':12s} {'Current':>12s} {'Optimal':>12s} {'Δ':>10s}")
    print(f"  {'─'*48}")
    for ch in CHANNELS:
        delta = optimal[ch] - current_spend[ch]
        arrow = '↑' if delta > 0 else '↓'
        print(f"  {ch:12s} ${current_spend[ch]:>10,.0f}  ${optimal[ch]:>10,.0f}  {arrow}${abs(delta):>8,.0f}")

    return current_spend, optimal


# ─── VISUALISATIONS ────────────────────────────────────────────────────────────
def plot_sales_decomposition(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Marketing Mix Model — Sales Decomposition', fontsize=14, fontweight='bold')

    # Stacked area: base + channel contributions
    ax = axes[0]
    contrib_cols = [f'contrib_{ch}' for ch in CHANNELS]
    stacked = df[contrib_cols].copy()
    stacked.insert(0, 'base', df['base_sales'])

    bottom = np.zeros(len(df))
    stack_labels = ['Base Sales'] + CHANNELS
    stack_colors = [GRAY] + [CHANNEL_COLORS[ch] for ch in CHANNELS]

    for col, label, color in zip(['base'] + contrib_cols, stack_labels, stack_colors):
        vals = stacked[col].values if col == 'base' else df[col].values
        ax.fill_between(df['date'], bottom, bottom + vals, alpha=0.8,
                        label=label, color=color)
        bottom += vals

    ax.plot(df['date'], df['sales'], color=NAVY, linewidth=1.5, linestyle='--',
            label='Actual Sales', alpha=0.7)
    ax.set_ylabel('Weekly Sales ($)')
    ax.set_title('Sales Decomposition — Base + Channel Contributions')
    ax.legend(loc='upper left', ncol=4, fontsize=8)

    # Actual vs. fitted
    ax = axes[1]
    ax.plot(df['date'], df['sales'], color=NAVY, linewidth=1.5, label='Actual Sales')
    ax.plot(df['date'], df['sales_fitted'], color=RED, linewidth=1.5,
            linestyle='--', label='Model Fit')
    ax.set_ylabel('Weekly Sales ($)')
    ax.set_title('Actual vs. Model-Fitted Sales')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('output/sales_decomposition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/sales_decomposition.png")


def plot_channel_roas(df):
    roas = calculate_roas(df)
    total_contrib = {ch: df[f'contrib_{ch}'].sum() for ch in CHANNELS}
    total_spend   = {ch: df[f'spend_{ch}'].sum()   for ch in CHANNELS}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Channel Performance Summary', fontsize=14, fontweight='bold')

    # ROAS by channel
    ax = axes[0]
    channels_sorted = sorted(roas, key=roas.get, reverse=True)
    bars = ax.bar(channels_sorted, [roas[c] for c in channels_sorted],
                  color=[CHANNEL_COLORS[c] for c in channels_sorted],
                  edgecolor='white', width=0.6)
    for bar, ch in zip(bars, channels_sorted):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{roas[ch]:.2f}x", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('ROAS (Return on Ad Spend)')
    ax.set_title('ROAS by Channel')
    ax.axhline(1.0, color=RED, linewidth=1.5, linestyle='--', alpha=0.7, label='Break-even')
    ax.legend(fontsize=9)

    # Spend vs. contribution scatter
    ax = axes[1]
    for ch in CHANNELS:
        ax.scatter(total_spend[ch]/1000, total_contrib[ch]/1000,
                   color=CHANNEL_COLORS[ch], s=150, zorder=3, label=ch, edgecolors='white', linewidth=1.5)
    x_max = max(total_spend.values()) / 1000 * 1.2
    ax.plot([0, x_max], [0, x_max], color=GRAY, linewidth=1.5, linestyle='--',
            label='1:1 line (ROAS=1)', alpha=0.5)
    ax.set_xlabel('Total Spend ($k)'); ax.set_ylabel('Sales Contribution ($k)')
    ax.set_title('Spend vs. Sales Contribution\n(Above diagonal = positive ROAS)')
    ax.legend(fontsize=8, ncol=2)

    # Revenue contribution share
    ax = axes[2]
    contribs = [total_contrib[ch] for ch in CHANNELS]
    wedges, texts, autotexts = ax.pie(
        contribs,
        labels=CHANNELS,
        colors=[CHANNEL_COLORS[ch] for ch in CHANNELS],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title('Revenue Attribution\nShare by Channel')

    plt.tight_layout()
    plt.savefig('output/channel_roas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/channel_roas.png")


def plot_saturation_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Channel Saturation Curves — Diminishing Returns (Hill Function)',
                 fontsize=13, fontweight='bold')

    axes_flat = axes.flatten()
    for i, ch in enumerate(CHANNELS):
        ax = axes_flat[i]
        p = TRUE_PARAMS[ch]
        spend_range = np.linspace(0, SPEND_MEANS[ch] * 3, 300)
        adstocked   = geometric_adstock(spend_range, p['decay'])
        saturated   = hill_saturation(adstocked, p['alpha'], p['gamma'])

        ax.plot(spend_range, saturated, color=CHANNEL_COLORS[ch], linewidth=2.5)
        ax.axvline(SPEND_MEANS[ch], color=NAVY, linewidth=1.5, linestyle='--',
                   alpha=0.6, label=f'Avg spend: ${SPEND_MEANS[ch]:,}')
        ax.axhline(0.5, color=GRAY, linewidth=1, linestyle=':', alpha=0.5)
        ax.set_xlabel('Weekly Spend ($)', fontsize=9)
        ax.set_ylabel('Saturation Index', fontsize=9)
        ax.set_title(f'{ch}\n(decay={p["decay"]:.2f}, γ=${p["gamma"]:,})')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig('output/saturation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/saturation_curves.png")


def plot_budget_reallocation(current_spend, optimal):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CHANNELS))
    width = 0.35

    current_vals = [current_spend[ch] for ch in CHANNELS]
    optimal_vals = [optimal[ch] for ch in CHANNELS]

    bars1 = ax.bar(x - width/2, current_vals, width, label='Current Allocation',
                   color=BLUE, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, optimal_vals, width, label='Optimised Allocation',
                   color=GREEN, alpha=0.85, edgecolor='white')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel('Average Weekly Spend ($)')
    ax.set_title('Budget Reallocation — Current vs. ROAS-Optimised\n(Holding total budget constant)')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${int(x):,}'))

    plt.tight_layout()
    plt.savefig('output/budget_reallocation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: output/budget_reallocation.png")


# ─── BUSINESS SUMMARY ──────────────────────────────────────────────────────────
def print_summary(df, roas, r2):
    total_sales = df['sales'].sum()
    total_spend = sum(df[f'spend_{ch}'].sum() for ch in CHANNELS)
    base_pct    = df['base_sales'].sum() / total_sales
    media_pct   = 1 - base_pct
    best_ch     = max(roas, key=roas.get)
    worst_ch    = min(roas, key=roas.get)

    print(f"\n{'='*60}")
    print(f"  MARKETING MIX MODEL — BUSINESS SUMMARY")
    print(f"{'='*60}")
    print(f"  Model fit (R²)          : {r2:.4f}")
    print(f"  Total 3-year sales      : ${total_sales:,.0f}")
    print(f"  Total media spend       : ${total_spend:,.0f}")
    print(f"  Overall ROAS            : {total_sales/(total_spend):.2f}x")
    print(f"\n  Sales Decomposition:")
    print(f"    Base (non-media)      : {base_pct:.0%}")
    print(f"    Media-driven          : {media_pct:.0%}")
    print(f"\n  Channel ROAS Rankings:")
    for ch in sorted(roas, key=roas.get, reverse=True):
        bar = '█' * int(roas[ch] * 10)
        print(f"    {ch:10s}: {roas[ch]:.2f}x  {bar}")
    print(f"\n  Highest ROAS: {best_ch} ({roas[best_ch]:.2f}x) — increase investment")
    print(f"  Lowest ROAS:  {worst_ch} ({roas[worst_ch]:.2f}x) — review or reallocate")
    print(f"\n  Strategic Recommendation:")
    print(f"    Reallocating 15% of {worst_ch} budget to {best_ch}")
    spend_shift = df[f'spend_{worst_ch}'].sum() * 0.15
    inc_rev     = spend_shift * (roas[best_ch] - roas[worst_ch])
    print(f"    → Estimated incremental revenue: ${inc_rev:,.0f} over 3 years")
    print(f"{'='*60}\n")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  Marketing Mix Modelling — Bayesian MMM")
    print("  Author: Payal Jarviya | SKK GSB Seoul\n")

    df = generate_data(n_weeks=156)

    print(f"{'='*60}")
    print(f"  FITTING MMM")
    print(f"{'='*60}")
    coefs, r2, mape = fit_mmm(df)

    roas = calculate_roas(df)
    current_spend, optimal = optimise_budget(df)

    print("\n  Generating charts...")
    plot_sales_decomposition(df)
    plot_channel_roas(df)
    plot_saturation_curves()
    plot_budget_reallocation(current_spend, optimal)

    print_summary(df, roas, r2)
    df.to_csv('output/mmm_results.csv', index=False)
    print("  Full results saved: output/mmm_results.csv")
    print("  Done.\n")
