import polars as pl
import numpy as np
import matplotlib.pyplot as plt

def hitrate_at_k(y_true, y_pred, groups, k=3):
    """
    Calculates HitRate@k.
    """
    df = pl.DataFrame({'group': groups, 'pred': y_pred, 'true': y_true})
    
    return (
        df.filter(pl.col("group").len().over("group") > 10)
        .sort(["group", "pred"], descending=[False, True])
        .group_by("group", maintain_order=True)
        .head(k)
        .group_by("group")
        .agg(pl.col("true").max())
        .select(pl.col("true").mean())
        .item()
    )

def get_feature_importance(model):
    """
    Returns a Polars DataFrame of feature importances.
    """
    xgb_importance = model.get_score(importance_type='gain')
    return pl.DataFrame([{'feature': k, 'importance': v} for k, v in xgb_importance.items()]).sort('importance', descending=True)

def plot_performance_curves(va_df):
    """
    Generates and displays performance plots.
    """
    # Your plotting code from the notebook goes here
    # ...
    print("Plotting performance curves...")
    # This is a simplified version. You would paste your full plotting code here.
    k_values = list(range(1, 11))
    
    def calculate_hitrate_curve(df, k_vals):
        sorted_df = df.sort(["ranker_id", "pred_score"], descending=[False, True])
        return [
            sorted_df.group_by("ranker_id", maintain_order=True).head(k)
            .group_by("ranker_id").agg(pl.col("selected").max().alias("hit"))
            .select(pl.col("hit").mean()).item()
            for k in k_vals
        ]

    hitrates = calculate_hitrate_curve(va_df, k_values)
    
    plt.figure(figsize=(6, 4), dpi=150)
    plt.plot(k_values, hitrates, marker='o')
    plt.title('HitRate@k for All Groups (>10)')
    plt.xlabel('k (top-k predictions)')
    plt.ylabel('HitRate@k')
    plt.grid(True, alpha=0.3)
    plt.show()