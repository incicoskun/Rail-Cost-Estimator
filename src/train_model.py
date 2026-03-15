import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Modular imports
from src.preprocess import apply_base_features
from src.config import DATA_PROCESSED, FEATURE_ALL, PARAMS, CSV_PATH, ENCODING_PARAMS

warnings.filterwarnings("ignore")

df = pd.read_csv(CSV_PATH)

# Drop rows without target
df = df.dropna(subset=["cost_per_km_2023_musd"]).reset_index(drop=True)

# Sniper Filter: Removing extreme outliers/surface lines mislabeled as heavy rail
outliers = ['Capital Airport Express', 'Tozai Line (Sendai)']
df = df[~df['line'].isin(outliers)].reset_index(drop=True)

# Calculate medians for Production/Inference
train_medians = {
    "num_stations": df["num_stations"].median(),
    "mid_year": ((df["start_year"] + df["end_year"]) / 2).median()
}

# Fill missing values with calculated medians
df["num_stations"] = df["num_stations"].fillna(train_medians["num_stations"])
df = apply_base_features(df)
df["mid_year"] = df["mid_year"].fillna(train_medians["mid_year"])
df["log_cost"] = np.log(df["cost_per_km_2023_musd"])


def bayesian_target_encoding(train, val, target="log_cost", cat="country", m=10):
    global_mean = train[target].mean()
    stats = train.groupby(cat)[target].agg(["mean", "count"])
    # Formula: (count * mean + m * global_mean) / (count + m)
    stats["smooth"] = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
    mapping = stats["smooth"].to_dict()
    
    tr_encoded = train[cat].map(mapping).fillna(global_mean)
    val_encoded = val[cat].map(mapping).fillna(global_mean)
    return tr_encoded, val_encoded, mapping, global_mean

df["cost_quartile"] = pd.qcut(df["log_cost"], q=5, labels=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(df))

print("\n" + "="*115)
print(f"{'STRATIFIED 5-FOLD CROSS-VALIDATION (Target: City-Level)':^115}")
print("="*115)

for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df["cost_quartile"]), 1):
    tr, val = df.iloc[tr_idx].copy(), df.iloc[val_idx].copy()
    
    # Encode Country (Fold-safe)
    tr["country_te"], val["country_te"], _, _ = bayesian_target_encoding(tr, val, cat="country", m=ENCODING_PARAMS["country_m"])
    c_freq = tr["country"].value_counts().to_dict()
    tr["country_freq"] = tr["country"].map(c_freq)
    val["country_freq"] = val["country"].map(c_freq).fillna(1)
    
    # Encode City (Fold-safe & Strategic)
    tr["city_te"], val["city_te"], _, _ = bayesian_target_encoding(tr, val, cat="city", m=ENCODING_PARAMS["city_m"])
    ct_freq = tr["city"].value_counts().to_dict()
    tr["city_freq"] = tr["city"].map(ct_freq)
    val["city_freq"] = val["city"].map(ct_freq).fillna(1)
    
    # Train Model
    model = GradientBoostingRegressor(loss="quantile", alpha=0.5, **PARAMS)
    model.fit(tr[FEATURE_ALL], tr["log_cost"])
    
    # Predict
    oof_preds[val_idx] = model.predict(val[FEATURE_ALL])
    
    # Fold Metrics
    f_r2 = r2_score(val["log_cost"], oof_preds[val_idx])
    f_mae = mean_absolute_error(val["log_cost"], oof_preds[val_idx])
    print(f" Fold {fold} | Validation R2: {f_r2:.4f} | MAE (Log): {f_mae:.4f}")

y_true_orig = np.exp(df["log_cost"])
y_pred_orig = np.exp(oof_preds)
errors = (y_pred_orig - y_true_orig) / y_true_orig
abs_errors = np.abs(errors)

r2_log = r2_score(df["log_cost"], oof_preds)
rmse_log = np.sqrt(mean_squared_error(df["log_cost"], oof_preds))
mdape = np.median(abs_errors) * 100
bias = np.median(errors) * 100
success_30 = np.mean(abs_errors < 0.30) * 100

print("\n" + "="*115)
print(f"{'GLOBAL MODEL PERFORMANCE (OUT-OF-FOLD)':^115}")
print("="*115)
print(f" R2 Score (Log)      : {r2_log:.4f}")
print(f" RMSE (Log)          : {rmse_log:.4f}")
print(f" Median Abs Error %  : {mdape:.2f}% (MdAPE)")
print(f" Success Rate (±30%) : {success_30:.2f}%")
print(f" Model Bias          : {bias:+.2f}%")
print("-" * 115)

res_df = pd.DataFrame({
    'Country': df['country'],
    'True_Log': df['log_cost'], 'Pred_Log': oof_preds,
    'True_Cost': y_true_orig, 'Pred_Cost': y_pred_orig,
    'Pct_Err': errors * 100, 'Abs_Pct_Err': abs_errors * 100
})

def country_metrics(g):
    return pd.Series({
        'Cnt': len(g), 'Med_T': g['True_Cost'].median(), 'Med_P': g['Pred_Cost'].median(),
        'R2': r2_score(g['True_Log'], g['Pred_Log']) if len(g)>1 else 0,
        'MdAPE': g['Abs_Pct_Err'].median(), 'Bias': g['Pct_Err'].median()
    })

stats = res_df.groupby('Country').apply(country_metrics).reset_index()
stats = stats[stats['Cnt'] >= 3].sort_values('Cnt', ascending=False)

print(f" {'Country':<8} | {'Proj.':<5} | {'True(M$)':>10} | {'Pred(M$)':>10} | {'R2':>7} | {'MdAPE(%)':>9} | {'Bias(%)':>9}")
print("-" * 115)

for _, r in stats.iterrows():
    marker = "🟢" if abs(r['Bias']) < 25 else ("🔵" if r['Bias'] < -25 else "🔴")
    print(f" {r['Country']:<8} | {int(r['Cnt']):<5} | {r['Med_T']:>10.1f} | {r['Med_P']:>10.1f} | {r['R2']:>7.3f} | {r['MdAPE']:>8.1f}% | {r['Bias']:>8.1f}% {marker}")


print("\n" + "="*115)
print(f"{'SAVING FINAL ASSETS FOR APP':^115}")
print("="*115)

_, _, country_te_map, g_mean = bayesian_target_encoding(df, df, cat="country", m=ENCODING_PARAMS["country_m"])
_, _, city_te_map, _ = bayesian_target_encoding(df, df, cat="city", m=ENCODING_PARAMS["city_m"])
c_freq_map = df["country"].value_counts().to_dict()
ct_freq_map = df["city"].value_counts().to_dict()

df["country_te"] = df["country"].map(country_te_map).fillna(g_mean)
df["country_freq"] = df["country"].map(c_freq_map).fillna(1)
df["city_te"] = df["city"].map(city_te_map).fillna(g_mean)
df["city_freq"] = df["city"].map(ct_freq_map).fillna(1)

final_model = GradientBoostingRegressor(loss="quantile", alpha=0.5, **PARAMS)
final_model.fit(df[FEATURE_ALL], df["log_cost"])

joblib.dump(final_model, "rail_cost_model.pkl")

memory_package = {
    "country_te_map": country_te_map, 
    "city_te_map": city_te_map,
    "global_mean": g_mean, 
    "country_freq_map": c_freq_map,
    "city_freq_map": ct_freq_map, 
    "train_medians": train_medians
}
joblib.dump(memory_package, "memory_package.pkl")
joblib.dump(FEATURE_ALL, "feature_names.pkl")

print(" [OK] All assets saved.")