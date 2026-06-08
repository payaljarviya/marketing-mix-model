# Marketing Mix Modelling

Last-click attribution is a lie most marketing teams have agreed to believe. It hands all the credit to the final touchpoint, ignores everything that happened before the click, and systematically understates TV, OOH, and any other channel that doesn't drop a cookie. This project builds a Bayesian MMM from scratch to answer a more honest question: where does each dollar of marketing actually go?

## What the model found

| Channel | ROAS | What to do |
|---------|------|------------|
| TV | ~2.3x | Strong carry-over effect; keep the investment |
| Search | ~2.1x | High ROAS, low adstock; this is the one to scale |
| Digital | ~1.9x | Good returns, moderate saturation; hold allocation |
| Email | ~1.8x | Highest marginal ROAS per dollar at current spend levels |
| Social | ~1.6x | Diminishing returns kicking in; fix targeting before adding budget |
| OOH | ~1.4x | Lowest ROAS, long decay; put it up against digital alternatives |

The headline finding: Search and TV are underinvested relative to their returns. Social looks efficient on a cost-per-click basis, but the saturation curve tells a different story.

## How it works

### Adstock

Marketing spend doesn't vanish the week it's deployed. TV runs from three weeks ago still pull people into the funnel today. The adstock transformation models this carry-over effect:

```
Adstock(t) = Spend(t) + decay × Adstock(t-1)
```

Each channel gets its own decay parameter. TV and OOH have long half-lives; Search decays quickly, which is part of why it scales so well.

### Saturation

Doubling the Search budget doesn't double the Search conversions. The Hill function captures this diminishing-returns curve:

```
Saturation(x) = x^α / (x^α + γ^α)
```

Social is deep into saturation territory. Email is not, which explains its strong marginal ROAS even though its absolute spend is low.

### Sales decomposition

After applying adstock and saturation transformations, OLS regression splits observed weekly sales into base sales (what would have happened with no marketing) and the incremental contribution from each channel. ROAS for each channel is then just total sales contribution divided by total spend.

### Budget optimisation

The reallocation step redistributes budget across channels weighted by ROAS, subject to total spend constraints. The output compares current allocation against the ROAS-optimised scenario.

## Three things worth acting on

1. **Scale Search.** It has the second-highest ROAS, low adstock (fast feedback), and is not yet saturated. If there's one place to put incremental budget, this is it.
2. **Review OOH seriously.** The long decay makes it hard to measure, but the numbers put it last. Either it needs a better justification or the budget should move elsewhere.
3. **Don't scale Social yet.** The targeting needs work before more spend makes sense. Broad audiences at high frequency is exactly where the Hill function flattens out.

Run a new model every quarter. MMM results go stale as media costs, competition, and consumer behaviour shift.

## Running the model

```bash
git clone https://github.com/payaljarviya/marketing-mix-model
pip install pandas numpy matplotlib seaborn scipy
python marketing_mix_model.py
```

## Output files

| File | What it shows |
|------|---------------|
| `output/sales_decomposition.png` | Stacked area chart: base sales, channel contributions, and actual vs. fitted line |
| `output/channel_roas.png` | ROAS by channel, spend vs. contribution scatter, revenue attribution pie |
| `output/saturation_curves.png` | Hill saturation curves for all six channels |
| `output/budget_reallocation.png` | Current vs. ROAS-optimised budget allocation side by side |
| `output/mmm_results.csv` | Full weekly model results |

## Technical stack

Python (numpy, scipy.optimize, pandas, matplotlib, seaborn). Adstock transformation, Hill function saturation, OLS regression for decomposition, ROAS-weighted budget optimisation.

---

*Payal Jarviya | MBA Candidate, SKK GSB Seoul | AI and Business Analytics*
