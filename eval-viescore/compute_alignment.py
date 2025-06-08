import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

# Functions

def spearman_footrule_list(human_scores, gpt_scores):
    """
    Compute Spearman's footrule distance between human and GPT rankings (4 items).
    """
    human_r = pd.Series(human_scores).rank(method='average')
    gpt_r   = pd.Series(gpt_scores).rank(method='average')
    return float((human_r - gpt_r).abs().sum())


def compute_qwk_list(human_scores, gpt_scores):
    """
    Compute Quadratic Weighted Kappa between human and GPT scores (4 items).
    Returns NaN if undefined.
    """
    try:
        return cohen_kappa_score(human_scores, gpt_scores, weights='quadratic')
    except ValueError:
        return np.nan

# Load the CSV (3 header rows, skip last average row)
df = pd.read_csv("viescore-alignment.csv", header=None)
score_df = df.iloc[3:-1].reset_index(drop=True).astype(int)

# Map each aspect to its human/GPT column pairs (one pair per model)
aspect_to_cols = {}
for col in range(1, df.shape[1]-1, 2):
    aspect = df.iat[1, col]
    aspect_to_cols.setdefault(aspect, []).append((col, col+1))

# Prepare storage for per-example metrics
metrics = {aspect: {'rhos': [], 'footrules': [], 'qwks': [], 'maes': []}
           for aspect in aspect_to_cols}

# Compute per-example metrics for each aspect over 4 models
num_examples = score_df.shape[0]
for idx in range(num_examples):
    for aspect, cols in aspect_to_cols.items():
        h_scores = [int(score_df.iat[idx, h]) for h, _ in cols]
        g_scores = [int(score_df.iat[idx, g]) for _, g in cols]

        # Spearman's rho: only when both human and GPT vary
        if len(set(h_scores)) > 1 and len(set(g_scores)) > 1:
            rho, _ = spearmanr(h_scores, g_scores)
        else:
            rho = np.nan

        # Spearman's footrule distance
        fr = spearman_footrule_list(h_scores, g_scores)

        # Quadratic Weighted Kappa
        qwk = compute_qwk_list(h_scores, g_scores)

        # Mean Absolute Error
        mae = float(np.mean(np.abs(np.array(h_scores) - np.array(g_scores))))

        # Append metrics
        metrics[aspect]['rhos'].append(rho)
        metrics[aspect]['footrules'].append(fr)
        metrics[aspect]['qwks'].append(qwk)
        metrics[aspect]['maes'].append(mae)

# Aggregate per-aspect averages
results = []
for aspect, vals in metrics.items():
    avg_rho = np.nanmean(vals['rhos'])
    avg_fr = np.mean(vals['footrules'])
    avg_qwk = np.nanmean(vals['qwks'])
    avg_mae = np.mean(vals['maes'])
    results.append({
        'Aspect': aspect,
        'Avg-Spearmanρ': round(avg_rho, 3),
        'Avg-Footrule': round(avg_fr, 2),
        'Avg-Quadratic Weighted Kappa': round(avg_qwk, 3),
        'Avg-Mean Absolute Error': round(avg_mae, 3)
    })

# Compute global averages across all aspects and examples
all_rhos = []
all_frs = []
all_qwks = []
all_maes = []
for vals in metrics.values():
    all_rhos.extend(vals['rhos'])
    all_frs.extend(vals['footrules'])
    all_qwks.extend(vals['qwks'])
    all_maes.extend(vals['maes'])

global_avg_rho = np.nanmean(all_rhos)
global_avg_fr = np.mean(all_frs)
global_avg_qwk = np.nanmean(all_qwks)
global_avg_mae = np.mean(all_maes)

results.append({
    'Aspect': 'Global',
    'Avg-Spearmanρ': round(global_avg_rho, 3),
    'Avg-Footrule': round(global_avg_fr, 2),
    'Avg-Quadratic Weighted Kappa': round(global_avg_qwk, 3),
    'Avg-Mean Absolute Error': round(global_avg_mae, 3)
})

# Output results
df_results = pd.DataFrame(results)
print(df_results)
