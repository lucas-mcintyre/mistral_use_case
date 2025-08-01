import argparse, json, pathlib, joblib, math
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# IO helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def load_split(split_path):
    print(f"[LOAD] Loading data from {split_path}...")
    texts, labels = [], []
    for i, rec in enumerate(load_jsonl(split_path)):
        if i % 1000 == 0 and i > 0:
            print(f"[LOAD] Loaded {i:,} examples...")
        texts.append(rec["text"])
        labels.append(rec["label"])
    print(f"[LOAD] Finished loading {len(texts):,} examples from {split_path}")
    return texts, labels

# ──────────────────────────────────────────────────────────────────────────────
# Train / Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def train(train_path, val_path, model_dir):
    print("[TRAIN] Starting training process...")
    texts_train, y_train = load_split(train_path)
    texts_val,   y_val   = load_split(val_path)

    print("[TFIDF] Building TF-IDF vectorizer...")
    print(f"[TFIDF] Training examples: {len(texts_train):,}")
    print(f"[TFIDF] Validation examples: {len(texts_val):,}")
    vec = TfidfVectorizer(min_df=3,
                          ngram_range=(1, 2),
                          max_features=200_000,
                          sublinear_tf=True)
    print("[TFIDF] Fitting vectorizer on training data...")
    X_train = vec.fit_transform(texts_train)
    print(f"[TFIDF] Training matrix shape: {X_train.shape}")
    print("[TFIDF] Transforming validation data...")
    X_val   = vec.transform(texts_val)
    print(f"[TFIDF] Validation matrix shape: {X_val.shape}")

    print("[MODEL] Training Logistic Regression classifier...")
    print(f"[MODEL] Number of classes: {len(set(y_train))}")
    clf = LogisticRegression(max_iter=1000,
                             n_jobs=-1,
                             multi_class="multinomial")
    print("[MODEL] Fitting model (this may take a while)...")
    clf.fit(X_train, y_train)
    print("[MODEL] Training completed!")

    print("[SAVE] Creating model directory...")
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    print("[SAVE] Saving TF-IDF vectorizer...")
    joblib.dump(vec, pathlib.Path(model_dir) / "vectorizer.joblib")
    print("[SAVE] Saving trained model...")
    joblib.dump(clf, pathlib.Path(model_dir) / "model.joblib")
    print("[SAVE] Models saved successfully!")

    # validation scores
    print("[EVAL] Computing validation predictions...")
    y_pred = clf.predict(X_val)
    print("[EVAL] Computing validation metrics...")
    f1_micro = f1_score(y_val, y_pred, average="micro")
    f1_macro = f1_score(y_val, y_pred, average="macro")
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }
    print("[SAVE] Saving validation metrics...")
    json.dump(metrics, open(pathlib.Path(model_dir) / "metrics_val.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("[RESULTS] Validation Results:")
    print(json.dumps(metrics, indent=2))
    print("[TRAIN] Training process completed!")
    return vec, clf

def evaluate(test_path, model_dir, conf_top=15):
    print("[EVAL] Starting evaluation process...")
    print("[LOAD] Loading trained models...")
    vec  = joblib.load(pathlib.Path(model_dir) / "vectorizer.joblib")
    clf  = joblib.load(pathlib.Path(model_dir) / "model.joblib")
    print("[LOAD] Models loaded successfully!")
    
    texts_test, y_test = load_split(test_path)
    print("[TFIDF] Transforming test data...")
    X_test = vec.transform(texts_test)
    print(f"[TFIDF] Test matrix shape: {X_test.shape}")
    print("[PRED] Making predictions on test set...")
    y_pred = clf.predict(X_test)
    print(f"[PRED] Predictions completed for {len(y_pred)} examples")

    print("[METRICS] Computing classification report...")
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    print("[SAVE] Saving test report...")
    json.dump(report, open(pathlib.Path(model_dir) / "report_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # confusion matrix for the top-N frequent classes
    print(f"[PLOT] Creating confusion matrix for top {conf_top} classes...")
    top_labels = [l for l, _ in Counter(y_test).most_common(conf_top)]
    print(f"[PLOT] Top classes: {top_labels[:5]}... (showing first 5)")
    idx = [i for i, l in enumerate(y_test) if l in top_labels]
    y_true_top = [y_test[i] for i in idx]
    y_pred_top = [y_pred[i] for i in idx]

    print("[PLOT] Computing confusion matrix...")
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_labels)
    print("[PLOT] Creating matplotlib figure...")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(top_labels)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(top_labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix – top {conf_top} leaves")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    print("[PLOT] Saving confusion matrix plot...")
    fig.savefig(pathlib.Path(model_dir) / f"confusion_top{conf_top}.png", dpi=250)
    print("✅ Saved confusion matrix →", pathlib.Path(model_dir) / f"confusion_top{conf_top}.png")
    print("[EVAL] Evaluation process completed!")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train")
    ap.add_argument("--val")
    ap.add_argument("--test")
    ap.add_argument("--model_dir", default="models/classical")
    ap.add_argument("--mode", choices=["train", "eval"], default="train")
    args = ap.parse_args()

    print(f"[MAIN] Starting classical baseline in {args.mode} mode...")
    print(f"[MAIN] Model directory: {args.model_dir}")
    
    if args.mode == "train":
        print("[MAIN] Running training mode...")
        train(args.train, args.val, args.model_dir)
    else:
        print("[MAIN] Running evaluation mode...")
        evaluate(args.test, args.model_dir)
    
    print("[MAIN] Script completed!")