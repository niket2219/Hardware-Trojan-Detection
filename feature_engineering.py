import os
import json
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Dict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# ------------------------------
# Configuration
# ------------------------------
OUTPUT_DIR = os.path.join("outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.json")

DATA_FILE = "HEROdata2.xlsx"  # main dataset referenced by current codebase
TARGET_COL = "Label"
CIRCUIT_COL = "Circuit"

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Feature selection thresholds
HIGH_CORR_THRESHOLD = 0.95
LOW_VARIANCE_THRESHOLD = 1e-9


def ensure_dirs() -> None:
	os.makedirs(FIG_DIR, exist_ok=True)
	os.makedirs(DATA_DIR, exist_ok=True)


def save_fig(name: str) -> None:
	path = os.path.join(FIG_DIR, name)
	plt.tight_layout()
	plt.savefig(path, dpi=200, bbox_inches="tight")
	plt.close()


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
	"""Load dataset, keeping consistent with existing pipeline that uses HEROdata2.xlsx."""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Dataset not found at {path}")
	df = pd.read_excel(path)
	return df


def balance_by_circuit_replication(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Replicate rows to balance the ratio between Trojan Free and infected within the same circuit
	category, following logic from preprocess_data.prepare_data.
	"""
	if TARGET_COL not in df.columns or CIRCUIT_COL not in df.columns:
		return df

	trojan_free = df.loc[df[TARGET_COL] == "'Trojan Free'"].reset_index(drop=True)
	for i in range(len(trojan_free)):
		category_substring = str(trojan_free[CIRCUIT_COL][i]).replace("'", "")
		circuit_group = df[df[CIRCUIT_COL].astype(str).str.contains(category_substring, na=False)]
		df1 = circuit_group.iloc[0:1]
		if len(circuit_group) > 1:
			df = pd.concat([df] + [df1] * (len(circuit_group) - 1), ignore_index=True)
	return df


def encode_categoricals_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
	"""Label-encode object columns for compact numeric representation."""
	encoders: Dict[str, LabelEncoder] = {}
	obj_cols = df.select_dtypes(include='object').columns.tolist()
	for col in obj_cols:
		le = LabelEncoder()
		df[col] = le.fit_transform(df[col].astype(str))
		encoders[col] = le
	return df, encoders


def one_hot_encode(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> pd.DataFrame:
	"""One-hot encode categorical columns for linear methods/interpretability."""
	exclude = set(exclude or [])
	cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
	return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def plot_basic_eda(df: pd.DataFrame) -> None:
	"""Generate basic EDA plots and save them to disk."""
	# 1. Target distribution
	if TARGET_COL in df.columns:
		plt.figure(figsize=(6, 4))
		df[TARGET_COL].value_counts().plot(kind='bar', color="#6baed6")
		plt.title("Target distribution")
		plt.xlabel(TARGET_COL)
		plt.ylabel("Count")
		save_fig("target_distribution.png")

	# 2. Missingness heatmap
	plt.figure(figsize=(10, 4))
	sns.heatmap(df.isna(), cbar=False)
	plt.title("Missing values heatmap")
	save_fig("missing_heatmap.png")

	# 3. Numeric distributions
	num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	if num_cols:
		cols = min(4, len(num_cols))
		rows = int(math.ceil(len(num_cols) / cols))
		plt.figure(figsize=(4 * cols, 2.8 * rows))
		for i, col in enumerate(num_cols, 1):
			plt.subplot(rows, cols, i)
			sns.histplot(df[col].dropna(), kde=True, bins=30)
			plt.title(col)
		save_fig("numeric_distributions.png")

	# 4. Correlation heatmap (numeric)
	if len(num_cols) > 1:
		corr = df[num_cols].corr()
		plt.figure(figsize=(10, 8))
		sns.heatmap(corr, cmap="vlag", center=0)
		plt.title("Feature correlation (numeric)")
		save_fig("correlation_heatmap.png")

	# 5. Boxplots by target (if classification)
	if TARGET_COL in df.columns and df[TARGET_COL].nunique() <= 20:
		for col in num_cols[:24]:  # cap to avoid too many figures
			plt.figure(figsize=(5, 3))
			sns.boxplot(data=df, x=TARGET_COL, y=col)
			plt.title(f"{col} by {TARGET_COL}")
			# Sanitize filename: replace invalid characters
			safe_col_name = col.replace("/", "_").replace("\\", "_").replace(" ", "_")
			save_fig(f"box_{safe_col_name}.png")


def low_variance_prune(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
	"""Remove near-constant numeric columns."""
	num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	to_drop: List[str] = []
	for c in num_cols:
		if df[c].var() <= LOW_VARIANCE_THRESHOLD:
			to_drop.append(c)
	if to_drop:
		df = df.drop(columns=to_drop)
	return df, to_drop


def high_correlation_prune(df: pd.DataFrame, threshold: float = HIGH_CORR_THRESHOLD) -> Tuple[pd.DataFrame, List[str]]:
	"""Drop one of highly correlated feature pairs using absolute correlation."""
	num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	if len(num_cols) < 2:
		return df, []
	corr_matrix = df[num_cols].corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
	to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
	if to_drop:
		df = df.drop(columns=to_drop)
	return df, to_drop


def compute_supervised_scores(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
	"""Compute feature relevance scores (ANOVA F, mutual information)."""
	scores = {}
	for name, func in [("f_classif", f_classif), ("mutual_info", mutual_info_classif)]:
		try:
			s = func(x, y)
			# f_classif returns tuple (F-statistics, p-values), extract first element
			if isinstance(s, tuple):
				s = s[0]
			scores[name] = s
		except Exception as e:
			scores[name] = np.full(x.shape[1], np.nan)
	result = pd.DataFrame(scores, index=x.columns)
	result.sort_values(by=list(result.columns), ascending=False, inplace=True)
	return result


def pca_projection(x: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
	pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
	proj = pca.fit_transform(x)
	return proj, pca


def scale_features(x: pd.DataFrame, method: str = "standard") -> Tuple[pd.DataFrame, object]:
	if method == "standard":
		scaler = StandardScaler()
	elif method == "minmax":
		scaler = MinMaxScaler()
	elif method == "yeo-johnson":
		scaler = PowerTransformer(method='yeo-johnson', standardize=True)
	else:
		raise ValueError("Unknown scaling method")
	x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)
	return x_scaled, scaler


def train_baseline_rf(x: pd.DataFrame, y: pd.Series) -> float:
	"""Quick baseline AUC using RandomForest to sanity-check engineered features."""
	try:
		clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
		clf.fit(x, y)
		if y.nunique() == 2:
			proba = clf.predict_proba(x)[:, 1]
			auc = roc_auc_score(y, proba)
			return float(auc)
		else:
			# For multiclass, macro-average one-vs-rest AUC
			proba = clf.predict_proba(x)
			auc = roc_auc_score(y, proba, multi_class='ovr')
			return float(auc)
	except Exception:
		return float("nan")


def main() -> None:
	ensure_dirs()

	# 1) Load
	df_raw = load_data(DATA_FILE)

	# 2) Basic cleaning
	df = df_raw.copy()
	df = df.dropna()  # simple strategy to stay consistent with existing code

	# 3) Balance by circuit replication (mirrors existing preprocessing intent)
	df = balance_by_circuit_replication(df)

	# 4) Preserve original target and circuit
	target_series = df[TARGET_COL] if TARGET_COL in df.columns else pd.Series([], dtype=str)

	# 5) EDA
	plot_basic_eda(df)

	# 6) Encode categoricals to numeric for correlation analysis (label encoding)
	df_encoded, label_encoders = encode_categoricals_label(df.copy())

	# 7) Remove near-constant features
	df_pruned, low_var_dropped = low_variance_prune(df_encoded)

	# 8) Correlation pruning
	df_pruned, high_corr_dropped = high_correlation_prune(df_pruned, threshold=HIGH_CORR_THRESHOLD)

	# 9) Train/test split
	if TARGET_COL in df_pruned.columns:
		y = df_pruned[TARGET_COL]
		x = df_pruned.drop(columns=[TARGET_COL, CIRCUIT_COL], errors='ignore')
	else:
		y = pd.Series([], dtype=int)
		x = df_pruned

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if len(y) else None)

	# 10) Scaling comparisons
	x_train_std, std_scaler = scale_features(x_train, method="standard")
	x_train_minmax, mm_scaler = scale_features(x_train, method="minmax")

	# 11) Supervised feature scores on standardized data
	score_df = compute_supervised_scores(x_train_std, y_train)
	score_df.to_csv(os.path.join(DATA_DIR, "feature_scores.csv"))

	# 12) PCA projection for visualization
	proj, pca_model = pca_projection(x_train_std, n_components=2)
	plt.figure(figsize=(6, 5))
	if y_train.nunique() <= 20:
		sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=y_train.astype(str), palette="tab10", s=25)
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
	else:
		plt.scatter(proj[:, 0], proj[:, 1], s=10, alpha=0.7)
	plt.title("PCA (2D) on Standardized Features")
	save_fig("pca_2d.png")

	# 13) Select top-k features by mutual information
	k = min(20, x_train_std.shape[1])
	topk = score_df.index[:k].tolist()
	x_train_topk = x_train_std[topk]

	# 14) Baseline RF AUC for sanity check
	auc_all = train_baseline_rf(x_train_std, y_train)
	auc_topk = train_baseline_rf(x_train_topk, y_train)

	# 15) Save engineered datasets
	train_full = x_train_std.copy()
	train_full[TARGET_COL] = y_train.values
	train_topk = x_train_topk.copy()
	train_topk[TARGET_COL] = y_train.values

	train_full.to_csv(os.path.join(DATA_DIR, "train_engineered_full.csv"), index=False)
	train_topk.to_csv(os.path.join(DATA_DIR, "train_engineered_topk.csv"), index=False)

	# 16) Save metadata report
	report = {
		"input_file": DATA_FILE,
		"rows_raw": int(len(df_raw)),
		"rows_after_dropna": int(len(df)),
		"low_variance_dropped": low_var_dropped,
		"high_corr_dropped": high_corr_dropped,
		"topk_features": topk,
		"train_shape_full": list(train_full.shape),
		"train_shape_topk": list(train_topk.shape),
		"auc_rf_all_features": auc_all,
		"auc_rf_topk_features": auc_topk,
		"pca_explained_variance_ratio": list(pca_model.explained_variance_ratio_),
	}
	with open(REPORT_PATH, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2)

	print("Feature engineering complete. Outputs saved to:")
	print(f"- Figures: {FIG_DIR}")
	print(f"- Data: {DATA_DIR}")
	print(f"- Report: {REPORT_PATH}")


if __name__ == "__main__":
	main()
