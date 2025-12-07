# tests/test_models.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def _feature_cols(df):
    # keep in sync with Notebook 04
    exclude = {
        "Time", "Sym", "C/P", "Exp",
        "next_spot", "next_return_1d",
        "direction_up", "vol_regime",
        "abs_diff_pct",
    }
    numeric = df.select_dtypes(include=["float64", "int64"]).columns
    return [c for c in numeric if c not in exclude]


def test_targets_balanced(training_df):
    # direction_up should not be crazy imbalanced
    direction_share = training_df["direction_up"].mean()
    assert 0.3 < direction_share < 0.7

    # vol_regime ~ 25% high-vol by design
    vol_share = training_df["vol_regime"].mean()
    assert 0.15 < vol_share < 0.4


def test_random_forest_direction_beats_chance(training_df):
    feats = _feature_cols(training_df)

    X = training_df[feats]
    y = training_df["direction_up"]

    # use a subset for speed
    sample = training_df.sample(n=min(2000, len(training_df)), random_state=42)
    Xs = sample[feats]
    ys = sample["direction_up"]

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.2, random_state=42, stratify=ys
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)

    # random baseline is ~max(class_share)
    assert acc > 0.55