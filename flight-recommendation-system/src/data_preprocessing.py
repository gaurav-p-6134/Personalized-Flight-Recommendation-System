import polars as pl
import numpy as np

def preprocess_data(data_raw, train_df):
    """
    Takes raw concatenated data and training data to perform all feature engineering.
    Returns the processed DataFrame and lists of feature/categorical columns.
    """
    df = data_raw.clone()

    # Efficient duration to minutes converter
    def dur_to_min(col):
        days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
        time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
        hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
        minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
        return (days + hours + minutes).fill_null(0)

    # Process duration columns
    dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]
    if dur_exprs:
        df = df.with_columns(dur_exprs)

    # Precompute marketing carrier columns check
    mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
    mc_exists = [col for col in mc_cols if col in df.columns]
    
    # === Start of Feature Engineering ===
    df = df.with_columns([
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0).then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1)).otherwise(1.0).alias("duration_ratio"),
        (pl.col("legs1_duration").is_null() | (pl.col("legs1_duration") == 0) | pl.col("legs1_segments0_departureFrom_airport_iata").is_null()).cast(pl.Int32).alias("is_one_way"),
        (pl.sum_horizontal(pl.col(col).is_not_null().cast(pl.UInt8) for col in mc_exists) if mc_exists else pl.lit(0)).alias("l0_seg"),
        (pl.col("frequentFlyer").fill_null("").str.count_matches("/") + (pl.col("frequentFlyer").fill_null("") != "").cast(pl.Int32)).alias("n_ff_programs"),
        pl.col("corporateTariffCode").is_not_null().cast(pl.Int32).alias("has_corporate_tariff"),
        (pl.col("pricingInfo_isAccessTP") == 1).cast(pl.Int32).alias("has_access_tp"),
        (pl.col("legs0_segments0_baggageAllowance_quantity").fill_null(0) + pl.col("legs1_segments0_baggageAllowance_quantity").fill_null(0)).alias("baggage_total"),
        (pl.col("miniRules0_monetaryAmount").fill_null(0) + pl.col("miniRules1_monetaryAmount").fill_null(0)).alias("total_fees"),
        pl.col("searchRoute").is_in(["MOWLED/LEDMOW", "LEDMOW/MOWLED", "MOWLED", "LEDMOW", "MOWAER/AERMOW"]).cast(pl.Int32).alias("is_popular_route"),
        pl.mean_horizontal(["legs0_segments0_cabinClass", "legs1_segments0_cabinClass"]).alias("avg_cabin_class"),
        (pl.col("legs0_segments0_cabinClass").fill_null(0) - pl.col("legs1_segments0_cabinClass").fill_null(0)).alias("cabin_class_diff"),
    ])

    # Segment counts
    seg_exprs = []
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
        seg_exprs.append(pl.sum_horizontal(pl.col(c).is_not_null() for c in seg_cols).cast(pl.Int32).alias(f"n_segments_leg{leg}") if seg_cols else pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
    df = df.with_columns(seg_exprs)
    
    df = df.with_columns([
        (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
        (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
        pl.when(pl.col("is_one_way") == 1).then(0).otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
    ])

    # Time features
    time_exprs = []
    for col in ("legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"):
        if col in df.columns:
            dt = pl.col(col).str.to_datetime(strict=False)
            h = dt.dt.hour().fill_null(12)
            time_exprs.extend([
                h.alias(f"{col}_hour"),
                dt.dt.weekday().fill_null(0).alias(f"{col}_weekday"),
                (((h >= 6) & (h <= 9)) | ((h >= 17) & (h <= 20))).cast(pl.Int32).alias(f"{col}_business_time")
            ])
    if time_exprs:
        df = df.with_columns(time_exprs)

    # Ranking and Popularity features
    df = df.with_columns(pl.col("Id").count().over("ranker_id").alias("group_size"))
    
    # Popularity features
    df = df.join(train_df.group_by('legs0_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier0_pop')), on='legs0_segments0_marketingCarrier_code', how='left')
    df = df.join(train_df.group_by('legs1_segments0_marketingCarrier_code').agg(pl.mean('selected').alias('carrier1_pop')), on='legs1_segments0_marketingCarrier_code', how='left')
    
    df = df.with_columns([
        (pl.col('carrier0_pop').fill_null(0.0) * pl.col('carrier1_pop').fill_null(0.0)).alias('carrier_pop_product'),
        pl.col('carrier0_pop').fill_null(0.0),
        pl.col('carrier1_pop').fill_null(0.0),
    ])

    # Price rank features
    df = df.with_columns([
        (pl.col("totalPrice").rank("average").over("ranker_id") / pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
        (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
    ])
    
    # === End of Feature Engineering ===

    # Fill nulls
    data = df.with_columns(
        [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
        [pl.col(c).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
    )

    # Define feature columns
    cat_features = [c for c in df.select(pl.selectors.string()).columns if c not in ['Id', 'ranker_id', 'frequentFlyer']]
    exclude_cols = ['Id', 'ranker_id', 'selected', 'profileId', 'requestDate', 'legs0_departureAt', 'legs0_arrivalAt', 'legs1_departureAt', 'legs1_arrivalAt', 'frequentFlyer', 'pricingInfo_passengerCount']
    
    feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('legs0_segments2') and not col.startswith('legs0_segments3') and not col.startswith('legs1_segments2') and not col.startswith('legs1_segments3')]
    cat_features_final = [col for col in cat_features if col in feature_cols]

    return data, feature_cols, cat_features_final