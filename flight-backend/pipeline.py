import polars as pl
from typing import Dict, Any

# Define the exact categorical features your model expects
CAT_FEATURES = [
    'nationality', 'searchRoute', 'corporateTariffCode',
    'bySelf', 'sex', 'companyID',
    'legs0_segments0_aircraft_code', 'legs0_segments0_arrivalTo_airport_city_iata',
    'legs0_segments0_arrivalTo_airport_iata', 'legs0_segments0_departureFrom_airport_iata',
    'legs0_segments0_marketingCarrier_code', 'legs0_segments0_operatingCarrier_code',
    'legs0_segments0_flightNumber',
    'legs0_segments1_aircraft_code', 'legs0_segments1_arrivalTo_airport_city_iata',
    'legs0_segments1_arrivalTo_airport_iata', 'legs0_segments1_departureFrom_airport_iata',
    'legs0_segments1_marketingCarrier_code', 'legs0_segments1_operatingCarrier_code',
    'legs0_segments1_flightNumber',
    'legs1_segments0_aircraft_code', 'legs1_segments0_arrivalTo_airport_city_iata',
    'legs1_segments0_arrivalTo_airport_iata', 'legs1_segments0_departureFrom_airport_iata',
    'legs1_segments0_marketingCarrier_code', 'legs1_segments0_operatingCarrier_code',
    'legs1_segments0_flightNumber',
    'legs1_segments1_aircraft_code', 'legs1_segments1_arrivalTo_airport_city_iata',
    'legs1_segments1_arrivalTo_airport_iata', 'legs1_segments1_departureFrom_airport_iata',
    'legs1_segments1_marketingCarrier_code', 'legs1_segments1_operatingCarrier_code',
    'legs1_segments1_flightNumber',
]

def dur_to_min(col: pl.Expr) -> pl.Expr:
    """Helper to convert duration strings to minutes."""
    days = col.str.extract(r"^(\d+)\.", 1).cast(pl.Int64).fill_null(0) * 1440
    time_str = pl.when(col.str.contains(r"^\d+\.")).then(col.str.replace(r"^\d+\.", "")).otherwise(col)
    hours = time_str.str.extract(r"^(\d+):", 1).cast(pl.Int64).fill_null(0) * 60
    minutes = time_str.str.extract(r":(\d+):", 1).cast(pl.Int64).fill_null(0)
    return (days + hours + minutes).fill_null(0)

def preprocess_flight_data(
    df: pl.DataFrame, 
    carrier0_map: pl.DataFrame, 
    carrier1_map: pl.DataFrame, 
    rt_freq_map: pl.DataFrame,
    cat_mappings: Dict[str, Any]
) -> pl.DataFrame:
    """Transforms raw flight data into the XGBoost feature matrix."""
    
    # --- FIX: Ensure all expected columns exist before processing ---
    # Start with all categorical features
    expected_cols = CAT_FEATURES.copy()
    
    # Add all other features explicitly referenced in our math operations
    expected_cols.extend([
        "totalPrice", "taxes", "legs0_duration", "legs1_duration",
        "frequentFlyer", "corporateTariffCode", "pricingInfo_isAccessTP",
        "legs0_segments0_baggageAllowance_quantity", "legs1_segments0_baggageAllowance_quantity",
        "miniRules0_monetaryAmount", "miniRules1_monetaryAmount", "searchRoute",
        "legs0_segments0_cabinClass", "legs1_segments0_cabinClass", "isVip",
        "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt"
    ])
    
    # --- FIX: Ensure expected columns exist WITH CORRECT DATA TYPES ---
    # Polars needs to know if an empty column is meant to be text or math
    
    # --- FIX: Ensure expected columns exist WITH CORRECT DATA TYPES ---
    # Polars needs to know if an empty column is meant to be text or math
    
    string_cols = set(CAT_FEATURES + [
        "legs0_duration", "legs1_duration", "frequentFlyer", "corporateTariffCode", 
        "searchRoute", "legs0_departureAt", "legs0_arrivalAt", "legs1_departureAt", "legs1_arrivalAt",
        # Added missing string columns:
        "legs0_segments0_baggageAllowance_weightMeasurementType", "legs0_segments1_baggageAllowance_weightMeasurementType",
        "legs1_segments0_baggageAllowance_weightMeasurementType", "legs1_segments1_baggageAllowance_weightMeasurementType",
        "miniRules0_statusInfos", "miniRules1_statusInfos"
    ] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in range(4)])
    
    numeric_cols = set([
        "totalPrice", "taxes", "legs0_segments0_baggageAllowance_quantity", 
        "legs1_segments0_baggageAllowance_quantity", "miniRules0_monetaryAmount", 
        "miniRules1_monetaryAmount", "legs0_segments0_cabinClass", 
        "legs1_segments0_cabinClass", "isVip", "pricingInfo_isAccessTP",
        # Added missing numeric columns:
        "isAccess3D", "legs0_segments0_seatsAvailable", "legs0_segments1_baggageAllowance_quantity",
        "legs0_segments1_cabinClass", "legs0_segments1_seatsAvailable", "legs1_segments0_seatsAvailable",
        "legs1_segments1_baggageAllowance_quantity", "legs1_segments1_cabinClass", "legs1_segments1_seatsAvailable",
        "miniRules0_percentage", "miniRules1_percentage"
    ])

    missing_exprs = []
    
    for c in string_cols:
        if c not in df.columns:
            # Force standard String casting
            missing_exprs.append(pl.lit(None).cast(pl.String).alias(c))
            
    for c in numeric_cols:
        if c not in df.columns:
            missing_exprs.append(pl.lit(None).cast(pl.Float64).alias(c))

    if missing_exprs:
        df = df.with_columns(missing_exprs)

    # ----------------------------------------------------------------
    # Add segment specific durations
    for l in (0, 1):
        for s in range(4):
            expected_cols.append(f"legs{l}_segments{s}_duration")
            
    # If the incoming JSON omitted any of these keys, add them as explicit nulls
    missing_cols = [pl.lit(None).alias(c) for c in set(expected_cols) if c not in df.columns]
    if missing_cols:
        df = df.with_columns(missing_cols)
    # ----------------------------------------------------------------
    
    # 1. Process Durations (the rest of your existing code continues here...)
    dur_cols = ["legs0_duration", "legs1_duration"] + [f"legs{l}_segments{s}_duration" for l in (0, 1) for s in (0, 1)]
    dur_exprs = [dur_to_min(pl.col(c)).alias(c) for c in dur_cols if c in df.columns]
    if dur_exprs:
        df = df.with_columns(dur_exprs)

    # 2. Base Features & Price Metrics
    mc_cols = [f'legs{l}_segments{s}_marketingCarrier_code' for l in (0, 1) for s in range(4)]
    mc_exists = [col for col in mc_cols if col in df.columns]
    
    df = df.with_columns([
        (pl.col("totalPrice") / (pl.col("taxes") + 1)).alias("price_per_tax"),
        (pl.col("taxes") / (pl.col("totalPrice") + 1)).alias("tax_rate"),
        pl.col("totalPrice").log1p().alias("log_price"),
        (pl.col("legs0_duration").fill_null(0) + pl.col("legs1_duration").fill_null(0)).alias("total_duration"),
        pl.when(pl.col("legs1_duration").fill_null(0) > 0)
            .then(pl.col("legs0_duration") / (pl.col("legs1_duration") + 1))
            .otherwise(1.0).alias("duration_ratio"),
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

    # 3. Segments Logic
    seg_exprs = []
    for leg in (0, 1):
        seg_cols = [f"legs{leg}_segments{s}_duration" for s in range(4) if f"legs{leg}_segments{s}_duration" in df.columns]
        if seg_cols:
            seg_exprs.append(pl.sum_horizontal(pl.col(c).is_not_null()).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
        else:
            seg_exprs.append(pl.lit(0).cast(pl.Int32).alias(f"n_segments_leg{leg}"))
            
    df = df.with_columns(seg_exprs)
    df = df.with_columns([
        (pl.col("n_segments_leg0") + pl.col("n_segments_leg1")).alias("total_segments"),
        (pl.col("n_segments_leg0") == 1).cast(pl.Int32).alias("is_direct_leg0"),
        pl.when(pl.col("is_one_way") == 1).then(0).otherwise((pl.col("n_segments_leg1") == 1).cast(pl.Int32)).alias("is_direct_leg1"),
    ])

    # 4. Derived & Time Features
    df = df.with_columns([
        (pl.col("is_direct_leg0") & pl.col("is_direct_leg1")).cast(pl.Int32).alias("both_direct"),
        ((pl.col("isVip") == 1) | (pl.col("n_ff_programs") > 0)).cast(pl.Int32).alias("is_vip_freq"),
        (pl.col("baggage_total") > 0).cast(pl.Int32).alias("has_baggage"),
        (pl.col("total_fees") > 0).cast(pl.Int32).alias("has_fees"),
        (pl.col("total_fees") / (pl.col("totalPrice") + 1)).alias("fee_rate"),
        pl.col("Id").count().over("ranker_id").alias("group_size"),
    ])
    
    if "legs0_segments0_marketingCarrier_code" in df.columns:
        df = df.with_columns(pl.col("legs0_segments0_marketingCarrier_code").is_in(["SU", "S7", "U6"]).cast(pl.Int32).alias("is_major_carrier"))
    else:
        df = df.with_columns(pl.lit(0).alias("is_major_carrier"))

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

    # 5. Group Ranks
    df = df.with_columns([
        pl.col("group_size").log1p().alias("group_size_log"),
        pl.col("totalPrice").rank().over("ranker_id").alias("price_rank"),
        pl.col("total_duration").rank().over("ranker_id").alias("duration_rank"),
        (pl.col("totalPrice").rank("average").over("ranker_id") / pl.col("totalPrice").count().over("ranker_id")).alias("price_pct_rank"),
        (pl.col("totalPrice") == pl.col("totalPrice").min().over("ranker_id")).cast(pl.Int32).alias("is_cheapest"),
        ((pl.col("totalPrice") - pl.col("totalPrice").median().over("ranker_id")) / (pl.col("totalPrice").std().over("ranker_id") + 1)).alias("price_from_median"),
        (pl.col("l0_seg") == pl.col("l0_seg").min().over("ranker_id")).cast(pl.Int32).alias("is_min_segments"),
    ])

    direct_cheapest = df.filter(pl.col("is_direct_leg0") == 1).group_by("ranker_id").agg(pl.col("totalPrice").min().alias("min_direct"))
    df = df.join(direct_cheapest, on="ranker_id", how="left").with_columns(
        ((pl.col("is_direct_leg0") == 1) & (pl.col("totalPrice") == pl.col("min_direct"))).cast(pl.Int32).fill_null(0).alias("is_direct_cheapest")
    ).drop("min_direct")

    # 6. Joins (Popularity)
    df = df.join(carrier0_map, on='legs0_segments0_marketingCarrier_code', how='left') \
           .join(carrier1_map, on='legs1_segments0_marketingCarrier_code', how='left') \
           .with_columns([pl.col('carrier0_pop').fill_null(0.0), pl.col('carrier1_pop').fill_null(0.0)]) \
           .with_columns((pl.col('carrier0_pop') * pl.col('carrier1_pop')).alias('carrier_pop_product'))

    if all(col in df.columns for col in ["legs0_segments0_departureFrom_airport_iata", "legs0_segments0_arrivalTo_airport_iata", "legs1_segments0_departureFrom_airport_iata", "legs1_segments0_arrivalTo_airport_iata"]):
        df = df.with_columns([
            (pl.col("legs0_segments0_departureFrom_airport_iata") + "_" + 
             pl.col("legs0_segments0_arrivalTo_airport_iata") + "__" + 
             pl.col("legs1_segments0_departureFrom_airport_iata") + "_" + 
             pl.col("legs1_segments0_arrivalTo_airport_iata"))
            .cast(pl.String)  # <-- Forces the new concatenated column back to standard size
            .alias("round_trip_route")
        ])
        df = df.join(rt_freq_map, on="round_trip_route", how="left").with_columns(pl.col("rt_route_count").fill_null(0).alias("round_trip_freq")).drop("round_trip_route")
    else:
        df = df.with_columns(pl.lit(0).alias("round_trip_freq"))


    # 7. Fill Nulls (Type-Safe)
    df = df.with_columns(
        [pl.col(c).fill_null(0) for c in df.select(pl.selectors.numeric()).columns] +
        [pl.col(c).cast(pl.String).fill_null("missing") for c in df.select(pl.selectors.string()).columns]
    )

    # 8. Categorical Encoding via Dictionary Mapping
    encode_exprs = []
    for col in CAT_FEATURES:
        if col in df.columns:
            mapping = cat_mappings.get(col, {})
            # CRITICAL: Force the column to String first, because JSON dictionary keys are strictly strings!
            encode_exprs.append(
                pl.col(col).cast(pl.String).replace(mapping, default=-1).cast(pl.Int32).alias(col)
            )
            
    if encode_exprs:
        df = df.with_columns(encode_exprs)

    return df

