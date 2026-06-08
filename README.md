# Marketing Mix Modelling — Bayesian MMM

**Business Question:** Which marketing channels are actually driving sales — and how should we reallocate our budget to maximise ROAS?

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![SciPy](https://img.shields.io/badge/SciPy-Optimisation-8CAAE6?style=flat)](https://scipy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org)

---

## Key Findings

| Channel | ROAS | Recommendation |
|---------|------|----------------|
| TV | ~2.3x | Strong ROI with high carry-over effect — maintain investment |
| Search | ~2.1x | High ROAS, low adstock — scales well, increase budget |
| Digital | ~1.9x | Good ROAS, moderate saturation — hold allocation |
| Social | ~1.6x | Moderate ROAS, high saturation — optimise targeting before scaling |
| OOH | ~1.4x | Lower ROAS, long decay — evaluate vs. digital alternatives |
| Email | ~1.8x | Efficient at current spend level — highest marginal ROAS per dollar |

*ROAS values are from synthetic simulation — replace with live data for actual results.*

---

## Business Problem

Marketing attribution is one of the most contested problems in business analytics. The dominant approach — last-click attribution — systematically overstates digital channels while ignoring:

**1. Offline media effects.** A TV campaign that drives brand awareness and influences online searches weeks later gets zero credit in last-click models.

**2. Adstock (carry-over effects).** An ad seen this week continues to influence purchase decisions for 2-6 weeks. Models that don't account for adstock misattribute lagged conversions to the wrong channel.

**3. Saturation (diminishing returns).** Doubling TV spend does not double TV revenue. Ignoring saturation leads to over-investment in channels that have already hit their effective ceiling.

Marketing Mix Modelling solves all three by building an econometric model from the ground up — using historical spend, sales, and external variables to estimate the *true* marginal contribution of each channel.

---

## Methodology

### Adstock Transformation
Geometric adstock models the carry-over effect of advertising. Each period's ad stock equals the current spend plus a fraction of last period's stock:

```
Adstock(t) = Spend(t) + decay × Adstock(t-1)
```

Higher decay = longer carry-over (e.g., TV: 0.7, Search: 0.2).

### Saturation (Hill Function)
The Hill function models diminishing returns — the marginal impact of additional spend decreases as investment increases:

```
Saturation(x) = x^α / (x^α + γ^α)
```

- `α` (slope): how quickly saturation is reached
- `γ` (half-saturation point): spend level at which 50% of the maximum effect is achieved

### Sales Decomposition
After transformation, a linear regression decomposes observed sales into:
- **Base sales:** organic demand (seasonality, trend, brand equity)
- **Incremental contributions:** the portion of sales attributable to each marketing channel

### ROAS Calculation
```
ROAS(channel) = Total Sales Contribution / Total Spend
```

### Budget Optimisation
Given a fixed total budget, reallocate across channels weighted by their ROAS — shifting spend from low-ROAS channels to high-ROAS channels to maximise total revenue.

---

## Recommendations

1. **Prioritise Search and TV in the next budget cycle.** Both deliver the highest ROAS and have not yet hit saturation ceilings at current spend levels.

2. **Review OOH allocation.** OOH delivers the lowest ROAS in this model. Before cutting, validate whether OOH has unmeasured brand effects — if not, reallocate to Digital or Search.

3. **Avoid scaling Social beyond current levels.** Saturation analysis shows Social has the steepest diminishing returns curve — additional investment yields progressively less incremental revenue.

4. **Run quarterly MMM updates.** Media effectiveness shifts over time (competitive pressures, audience fatigue, seasonality). Re-fitting the model quarterly ensures budget decisions reflect current reality.

5. **Apply adstock half-life to campaign planning.** TV and OOH have 4-6 week carry-over — plan campaign timing to avoid cannibalising your own adstock (e.g. don't spike TV spend in consecutive weeks if the previous week's stock is still high).

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/payaljarviya/marketing-mix-model
cd marketing-mix-model
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 3. Run the script
```bash
python marketing_mix_model.py
```

### 4. Bring your own data
Replace `generate_data()` with your own weekly spend and sales CSV:
```python
df = pd.read_csv('your_weekly_data.csv')
# Required columns: date, sales, spend_TV, spend_Digital, spend_Search, ...
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/sales_decomposition.png` | Stacked area chart: base + channel contributions + actual vs. fitted |
| `output/channel_roas.png` | ROAS by channel, spend vs. contribution scatter, revenue attribution pie |
| `output/saturation_curves.png` | Hill saturation curves for all 6 channels |
| `output/budget_reallocation.png` | Current vs. ROAS-optimised budget allocation |
| `output/mmm_results.csv` | Full weekly model results for BI tool integration |

---

## Skills Demonstrated

- Marketing Mix Modelling (MMM) — adstock, saturation, decomposition
- Econometric modelling and regression analysis
- ROAS calculation and budget optimisation
- Marketing strategy and media planning
- Python: `numpy`, `scipy.optimize`, `pandas`, `matplotlib`, `seaborn`

---

*Author: Payal Jarviya | MBA Candidate, SKK GSB Seoul | AI & Business Analytics + Marketing Analytics*
