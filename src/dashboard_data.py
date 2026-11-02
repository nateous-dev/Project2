
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import confusion_matrix


# =========================================================
# 프로젝트 경로
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


# =========================================================
# 1. 메타데이터
# =========================================================

def load_metadata():

    metadata_path = MODEL_DIR / "model_metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# 2. CSV 데이터
# =========================================================

def load_customer_risk():

    return pd.read_csv(
        DATA_DIR / "customer_risk.csv",
        encoding="utf-8-sig"
    )


def load_permutation_importance():

    return pd.read_csv(
        DATA_DIR / "permutation_importance.csv",
        encoding="utf-8-sig"
    )


def load_campaign_summary():

    return pd.read_csv(
        DATA_DIR / "campaign_summary.csv",
        encoding="utf-8-sig"
    )


# =========================================================
# 3. KPI
# =========================================================

def get_dashboard_kpi(metadata, customer_risk):

    return {
        "total_customers": len(customer_risk),
        "actual_churn": int(customer_risk["실제이탈"].sum()),
        "final_model": metadata["final_model"],
        "f1_score": metadata["test_metrics"]["f1_score"],
        "threshold": metadata["threshold"]
    }


# =========================================================
# 4. 위험등급별 고객 수
# =========================================================

def get_risk_distribution(customer_risk):

    risk_order = ["고위험", "중위험", "저위험"]

    return (
        customer_risk["위험등급"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
    )


# =========================================================
# 5. 혼동행렬
# =========================================================

def get_confusion_matrix(customer_risk):

    return confusion_matrix(
        customer_risk["실제이탈"],
        customer_risk["이탈예측"],
        labels=[0, 1]
    )


# =========================================================
# 6. Permutation Importance Top 5
# =========================================================

def get_top_importance(importance_df):

    return (
        importance_df
        .sort_values("Importance_Mean", ascending=False)
        .head(5)
        .sort_values("Importance_Mean", ascending=True)
    )
