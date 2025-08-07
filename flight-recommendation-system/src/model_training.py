import xgboost as xgb
import polars as pl

def train_model(data, feature_cols, cat_features_final, train_height):
    """
    Prepares data and trains the XGBoost ranker model.
    """
    # Convert categorical features to dense ranks for XGBoost
    data_xgb = data.with_columns([(pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int32) for c in cat_features_final])

    # Split data
    n1 = 16487352  # Time-based split point for validation
    n2 = train_height
    
    X = data_xgb.select(feature_cols)
    y = data_xgb.select('selected')
    groups = data_xgb.select('ranker_id')

    data_xgb_tr, data_xgb_va = X[:n1], X[n1:n2]
    y_tr, y_va = y[:n1], y[n1:n2]
    groups_tr, groups_va = groups[:n1], groups[n1:n2]

    group_sizes_tr = groups_tr.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()
    group_sizes_va = groups_va.group_by('ranker_id', maintain_order=True).len()['len'].to_numpy()

    dtrain = xgb.DMatrix(data_xgb_tr, label=y_tr, group=group_sizes_tr, feature_names=feature_cols)
    dval = xgb.DMatrix(data_xgb_va, label=y_va, group=group_sizes_va, feature_names=feature_cols)

    # XGBoost parameters
    xgb_params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'ndcg@3',
        "learning_rate": 0.0226,
        "max_depth": 14,
        "min_child_weight": 2,
        "subsample": 0.88,
        "colsample_bytree": 0.46,
        'seed': 42,
        'n_jobs': -1,
    }

    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=50
    )
    
    return xgb_model