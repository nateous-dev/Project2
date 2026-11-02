"""Deterministic churn scoring, validation and train-only preprocessing.

No pickled models or opaque scores. The small logistic baseline uses NumPy.
It is a new baseline, not the XGBoost experiment referenced in the old UI.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

NUMERIC = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
           'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
CATEGORICAL = ['Geography', 'Gender']
FEATURES = NUMERIC + CATEGORICAL
GRADES = ['집중관리', '관찰', '일반관리']
COLORS = ['#ef7868', '#e9b44c', '#38b6a0']
ALIASES = {
    '이탈예측확률': ['이탈예측확률', 'predicted_probability', 'churn_probability', 'probability'],
    '실제이탈': ['실제이탈', 'Exited', 'churned'],
    '고객ID': ['고객ID', 'CustomerId', 'customer_id'],
}


def read_csv(raw):
    from io import BytesIO
    for enc in ['utf-8-sig', 'cp949']:
        try:
            df = pd.read_csv(BytesIO(raw), encoding=enc)
            df.columns = df.columns.astype(str).str.strip()
            if df.columns.duplicated().any():
                raise ValueError('공백 제거 후 중복되는 컬럼명이 있습니다.')
            return df
        except UnicodeDecodeError:
            pass
    raise ValueError('UTF-8 또는 CP949 CSV 파일이 필요합니다.')


def normalize(frame):
    df = frame.copy().reset_index(drop=True)
    for dest, aliases in ALIASES.items():
        if dest not in df:
            source = next((c for c in aliases if c in df), None)
            if source:
                df[dest] = df[source]
    # IDs are always strings so search does not interpret regular expressions.
    if '고객ID' not in df:
        df['고객ID'] = [f'ROW-{i + 1:05d}' for i in range(len(df))]
    df['고객ID'] = df['고객ID'].astype('string').fillna('ID 미입력')
    if '실제이탈' in df:
        original = df['실제이탈']
        labels = pd.to_numeric(original, errors='coerce')
        if ((original.notna() & labels.isna()) | (labels.notna() & ~labels.isin([0, 1]))).any():
            raise ValueError('실제이탈 / Exited는 0, 1 또는 빈 값이어야 합니다.')
        df['실제이탈'] = labels
    return df


def probabilities(values):
    """Accept 0..1 numbers or explicit percent strings; never guess a scale."""
    s = values.astype('string').str.strip()
    percent = s.str.endswith('%', na=False)
    p = pd.to_numeric(s.str.replace('%', '', regex=False), errors='coerce').astype(float)
    p = p.where(~percent, p / 100)
    if ((p.notna() & (~np.isfinite(p) | ~p.between(0, 1))) | (s.notna() & p.isna())).any():
        raise ValueError('확률은 0~1 숫자 또는 52% 같은 백분율이어야 합니다. 빈 값은 미산정으로 표시됩니다.')
    return p


def classify(frame, threshold, watch):
    if not (0 < watch < threshold < 1):
        raise ValueError('0 < 관찰 기준 < 집중관리 기준 < 1이어야 합니다.')
    df = normalize(frame)
    if '이탈예측확률' not in df:
        raise ValueError('이탈예측확률 컬럼이 없습니다. 원본 데이터는 모델 학습을 먼저 실행하세요.')
    p = probabilities(df['이탈예측확률'])
    df['이탈예측확률'] = p
    if '위험등급' in df:
        df['기존위험등급'] = df['위험등급']
    df['위험등급'] = np.select([p.ge(threshold), p.ge(watch), p.notna()], GRADES, default='미산정')
    df['이탈예측'] = pd.Series(np.where(p.notna(), p.ge(threshold).astype(float), np.nan), dtype='Float64')
    df['권장조치'] = df['위험등급'].map({
        '집중관리': '우선 상담 · 불편 사유 확인', '관찰': '이용 현황 확인 · 재방문 안내',
        '일반관리': '정기 모니터링', '미산정': '예측 데이터 확인'})
    return df


def filter_customers(df, grades, bounds=(0., 1.), query='', eligible=False):
    mask = df['위험등급'].isin(grades)
    mask &= df['이탈예측확률'].between(*bounds) | df['이탈예측확률'].isna()
    if query:
        mask &= df['고객ID'].str.contains(query, regex=False, na=False)
    if eligible and '실제이탈' in df:
        mask &= df['실제이탈'].ne(1) | df['실제이탈'].isna()
    return df.loc[mask].sort_values('이탈예측확률', ascending=False, na_position='last').copy()


def metrics(y, p, threshold):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
    return dict(accuracy=(tp + tn) / len(y), precision=precision, recall=recall,
                f1_score=f1, roc_auc=auc, brier=float(np.mean((p-y)**2)), tp=tp, fp=fp, fn=fn, tn=tn)


@dataclass
class Baseline:
    medians: pd.Series
    means: np.ndarray
    scales: np.ndarray
    categories: dict
    names: list
    weights: np.ndarray
    intercept: float
    threshold: float
    test_y: np.ndarray
    test_p: np.ndarray
    validation_y: np.ndarray
    validation_p: np.ndarray
    split_sizes: tuple

    def transform(self, data):
        missing = [c for c in FEATURES if c not in data]
        if missing:
            raise ValueError('예측에 필요한 컬럼: ' + ', '.join(missing))
        numeric = validate_features(data)
        x = (numeric.fillna(self.medians).to_numpy(float) - self.means) / self.scales
        for col, cats in self.categories.items():
            s = data[col].astype('string').str.strip().fillna('미입력')
            if not s.isin(cats).all():
                raise ValueError(f'{col}: 학습에 없던 범주가 있습니다. 학습 데이터와 같은 범주를 사용하세요.')
            x = np.column_stack([x, *[(s == c).to_numpy(float) for c in cats[1:]]])
        return x

    def predict(self, data):
        z = np.clip(self.transform(data) @ self.weights + self.intercept, -35, 35)
        return 1 / (1 + np.exp(-z))

    def contributions(self, row):
        # Exact additive contributions in log-odds, not causal explanations.
        return pd.DataFrame({'변수': self.names, '기여도': self.transform(row)[0] * self.weights})


def validate_features(data):
    missing = [c for c in FEATURES if c not in data]
    if missing:
        raise ValueError('필수 입력 컬럼이 없습니다: ' + ', '.join(missing))
    numbers = data[NUMERIC].apply(pd.to_numeric, errors='coerce')
    bad = (data[NUMERIC].notna() & numbers.isna()) | (numbers.notna() & ~np.isfinite(numbers))
    if bad.any().any():
        raise ValueError('숫자 컬럼에 문자 또는 무한대가 있습니다.')
    if (numbers.lt(0)).any().any():
        raise ValueError('고객 입력 값에 음수가 있습니다.')
    for c in ['HasCrCard', 'IsActiveMember']:
        if (numbers[c].notna() & ~numbers[c].isin([0, 1])).any():
            raise ValueError(c + '는 0 또는 1이어야 합니다.')
    for c in ['CreditScore', 'Age', 'Tenure', 'NumOfProducts']:
        if (numbers[c].dropna() % 1 != 0).any():
            raise ValueError(c + '는 정수여야 합니다.')
    return numbers


def train_baseline(frame):
    df = normalize(frame)
    validate_features(df)
    if '실제이탈' not in df or df['실제이탈'].isna().any():
        raise ValueError('학습 데이터에는 결측 없는 Exited 또는 실제이탈(0/1)이 필요합니다.')
    # Deduplicate without ID columns; repeated customer IDs must not cross splits.
    df = df.drop_duplicates(subset=FEATURES + ['실제이탈']).reset_index(drop=True)
    if 'CustomerId' in df and df['CustomerId'].duplicated().any():
        raise ValueError('동일 CustomerId의 여러 기록이 있습니다. 고객당 한 행으로 정리하세요.')
    if frame.columns.isin(['고객ID', 'customer_id']).any() and df['고객ID'].duplicated().any():
        raise ValueError('동일 고객ID의 여러 기록이 있습니다. 고객당 한 행으로 정리하세요.')
    counts = df['실제이탈'].value_counts()
    if len(df) < 100 or len(counts) != 2 or counts.min() < 20:
        raise ValueError('학습에는 최소 100행, 유지·이탈 각 20명 이상이 필요합니다.')
    rng = np.random.default_rng(42)
    parts = [[], [], []]
    for label in [0, 1]:
        idx = rng.permutation(df.index[df['실제이탈'].eq(label)].to_numpy())
        a, b = int(.6 * len(idx)), int(.8 * len(idx))
        for bucket, chunk in zip(parts, [idx[:a], idx[a:b], idx[b:]]):
            bucket.extend(chunk)
    train, val, test = [df.loc[p].copy() for p in parts]
    numeric = validate_features(train)
    medians = numeric.median()
    if medians.isna().any():
        raise ValueError('모든 값이 비어 있는 숫자 컬럼은 학습할 수 없습니다.')
    xnum = numeric.fillna(medians).to_numpy(float)
    means, scales = xnum.mean(axis=0), xnum.std(axis=0)
    scales[scales < 1e-8] = 1
    # Categories are part of the bank schema, not inferred from validation/test.
    cats = {'Geography': ['France', 'Germany', 'Spain', '미입력'],
            'Gender': ['Female', 'Male', '미입력']}
    names = NUMERIC + [f'{c}={v}' for c, vs in cats.items() for v in vs[1:]]
    model = Baseline(medians, means, scales, cats, names, np.zeros(len(names)), 0., .5,
                     np.array([]), np.array([]), np.array([]), np.array([]), tuple(map(len, [train, val, test])))
    x = model.transform(train); y = train['실제이탈'].to_numpy(float)
    bias = float(np.log(y.mean() / (1-y.mean())))
    w = np.zeros(x.shape[1])
    # L2 regularized full-batch convex logistic regression; spectral step bound.
    step = 1 / (.25 * np.linalg.norm(np.column_stack([x, np.ones(len(x))]), 2)**2 / len(x) + .01)
    for _ in range(4000):
        z = np.clip(x @ w + bias, -35, 35)
        error = 1 / (1 + np.exp(-z)) - y
        grad = x.T @ error / len(x) + .01 * w
        gb = error.mean()
        w -= step * grad; bias -= step * gb
        if max(np.max(np.abs(grad)), abs(gb)) < 1e-6:
            break
    model.weights, model.intercept = w, bias
    vp = model.predict(val); vy = val['실제이탈'].to_numpy(int)
    thresholds = np.arange(.1, .901, .01)
    scores = [metrics(vy, vp, t)['f1_score'] for t in thresholds]
    model.threshold = float(round(thresholds[int(np.argmax(scores))], 2))
    model.validation_y, model.validation_p = vy, vp
    model.test_y, model.test_p = test['실제이탈'].to_numpy(int), model.predict(test)
    return model


def demo_data(n=420):
    rng = np.random.default_rng(17)
    df = pd.DataFrame({
        'CustomerId': [f'DEMO-{i+1:04d}' for i in range(n)],
        'CreditScore': rng.integers(400, 850, n), 'Age': rng.integers(22, 76, n),
        'Tenure': rng.integers(0, 11, n), 'Balance': rng.integers(0, 180000, n),
        'NumOfProducts': rng.integers(1, 5, n), 'HasCrCard': rng.integers(0, 2, n),
        'IsActiveMember': rng.integers(0, 2, n), 'EstimatedSalary': rng.integers(15000, 190000, n),
        'Geography': rng.choice(['France', 'Germany', 'Spain'], n),
        'Gender': rng.choice(['Female', 'Male'], n)})
    z = -3 + .045 * (df.Age - 30) + 1.25 * (1 - df.IsActiveMember) + .55 * (df.NumOfProducts >= 3)
    p = 1 / (1 + np.exp(-z))
    df['Exited'] = rng.binomial(1, p)
    df['이탈예측확률'] = p
    return df
