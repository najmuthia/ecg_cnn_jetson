# =============================================================================
#  ECG PTB-XL CLASSIFICATION — RECORD-LEVEL
#  4 Arsitektur CNN: 1D-CNN, ResNet, Multi-Scale CNN, Attention-Based CNN
#  Input: (1000, 12) — rekaman EKG 10 detik, 12 lead
#
#  Usage:
#    Pilih MODEL_NAME di bawah, lalu jalankan:
#    python ecg_train_record.py
#
#  Output: results/record/<MODEL_NAME>/
# =============================================================================

import os, ast, time, json, random, warnings
import numpy as np
import pandas as pd
import wfdb, psutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
warnings.filterwarnings("ignore")

# =============================================================================
# GPU SETUP
# =============================================================================
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[GPU] Warning: {e}")

# =============================================================================
# PILIH MODEL
# =============================================================================
MODEL_NAME = "CNN"   # "CNN" | "ResNet" | "MultiScale" | "Attention"
assert MODEL_NAME in ["CNN", "ResNet", "MultiScale", "Attention"]

# =============================================================================
# SEEDS
# =============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
PTBXL_DB = "/home/filkom/Documents/skripsi-jj/ptb-xl/ptbxl_database.csv"
SCP_PATH  = "/home/filkom/Documents/skripsi-jj/ptb-xl/scp_statements.csv"
BASE_DIR  = "/home/filkom/Documents/skripsi-jj/ptb-xl/"

# =============================================================================
# KONSTANTA
# =============================================================================
FS            = 100
RECORD_LEN    = 1000        # 10 detik @ 100 Hz
N_LEADS       = 12
INPUT_SHAPE   = (RECORD_LEN, N_LEADS)
VALID_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# =============================================================================
# HYPERPARAMETER — IDENTIK SEMUA MODEL
# =============================================================================
BATCH_SIZE      = 64
MAX_EPOCHS      = 100
LEARNING_RATE   = 3e-4
EARLY_STOP_PAT  = 12
LR_REDUCE_PAT   = 4
LR_REDUCE_FACT  = 0.3
LR_MIN          = 1e-6
L2_STRENGTH     = 1e-3
LABEL_SMOOTHING = 0.10
DROPOUT_RATE    = 0.50
SDROPOUT_RATE   = 0.15

L2_REG = tf.keras.regularizers.l2(L2_STRENGTH)

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
OUT_DIR = os.path.join("results", "record", MODEL_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

PREP_DIR       = os.path.join("results", "preprocessed_record")
os.makedirs(PREP_DIR, exist_ok=True)

X_TRAIN_PATH = os.path.join(PREP_DIR, "X_train.npy")
y_TRAIN_PATH = os.path.join(PREP_DIR, "y_train.npy")
X_VAL_PATH   = os.path.join(PREP_DIR, "X_val.npy")
y_VAL_PATH   = os.path.join(PREP_DIR, "y_val.npy")
X_TEST_PATH  = os.path.join(PREP_DIR, "X_test.npy")
y_TEST_PATH  = os.path.join(PREP_DIR, "y_test.npy")
CLASSES_PATH = os.path.join(PREP_DIR, "classes.npy")
NORM_MEAN    = os.path.join(PREP_DIR, "norm_mean.npy")
NORM_STD     = os.path.join(PREP_DIR, "norm_std.npy")

print(f"\n{'='*60}")
print(f"  MODEL      : {MODEL_NAME}")
print(f"  Input shape: {INPUT_SHAPE}")
print(f"  Output dir : {OUT_DIR}")
print(f"{'='*60}\n")

# =============================================================================
# LOAD METADATA
# =============================================================================
def load_metadata():
    df     = pd.read_csv(PTBXL_DB)
    scp_df = pd.read_csv(SCP_PATH, index_col=0)
    scp_diag = scp_df[scp_df["diagnostic"] == 1]

    def scp_to_superclass(s):
        s = ast.literal_eval(s)
        sup = [scp_diag.loc[c, "diagnostic_class"]
               for c in s if c in scp_diag.index]
        return list(set(sup))

    df["superclass"] = df["scp_codes"].apply(scp_to_superclass)
    df = df[df["superclass"].map(len) == 1]
    df["superclass"] = df["superclass"].map(lambda x: x[0])
    df = df[df["superclass"].isin(VALID_CLASSES)].reset_index(drop=True)

    train_df = df[df["strat_fold"] <= 8].reset_index(drop=True)
    val_df   = df[df["strat_fold"] == 9].reset_index(drop=True)
    test_df  = df[df["strat_fold"] == 10].reset_index(drop=True)

    print(f"[SPLIT] Train:{len(train_df)} | Val:{len(val_df)} | Test:{len(test_df)}")
    return train_df, val_df, test_df

# =============================================================================
# SIGNAL PROCESSING
# =============================================================================
def bandpass_filter(sig):
    nyq  = FS / 2
    b, a = butter(4, [0.5 / nyq, 24 / nyq], btype="band")
    return lfilter(b, a, sig, axis=0)

# =============================================================================
# PREPROCESSING
# =============================================================================
def process_split(df_split):
    X, Y = [], []
    for _, row in df_split.iterrows():
        try:
            rec = wfdb.rdrecord(
                os.path.join(BASE_DIR, row["filename_lr"])
            ).p_signal.astype(np.float32)

            if len(rec) < RECORD_LEN:
                pad = np.zeros((RECORD_LEN - len(rec), N_LEADS), dtype=np.float32)
                rec = np.concatenate([rec, pad], axis=0)
            rec  = rec[:RECORD_LEN]
            filt = bandpass_filter(rec)

            X.append(filt[np.newaxis])
            Y.append(row["superclass"])
        except Exception as e:
            print(f"  [SKIP] ecg_id={row.get('ecg_id','?')}: {e}")

    X_out = np.concatenate(X, axis=0).astype(np.float32)
    print(f"  → shape: {X_out.shape}")
    return X_out, np.array(Y)


def prepare_or_load_data():
    all_paths = [X_TRAIN_PATH, y_TRAIN_PATH, X_VAL_PATH, y_VAL_PATH,
                 X_TEST_PATH, y_TEST_PATH, CLASSES_PATH]

    if all(os.path.exists(p) for p in all_paths):
        print("[DATA] Loading cached dataset...")
        return (
            np.load(X_TRAIN_PATH), np.load(y_TRAIN_PATH),
            np.load(X_VAL_PATH),   np.load(y_VAL_PATH),
            np.load(X_TEST_PATH),  np.load(y_TEST_PATH),
            np.load(CLASSES_PATH, allow_pickle=True),
        )

    print("[DATA] Processing from scratch...")
    train_df, val_df, test_df = load_metadata()

    print("[DATA] Train...")
    X_train, y_str_tr = process_split(train_df)
    print("[DATA] Val...")
    X_val,   y_str_v  = process_split(val_df)
    print("[DATA] Test...")
    X_test,  y_str_te = process_split(test_df)

    le = LabelEncoder()
    le.fit(sorted(VALID_CLASSES))
    y_train = le.transform(y_str_tr)
    y_val   = le.transform(y_str_v)
    y_test  = le.transform(y_str_te)

    np.save(X_TRAIN_PATH, X_train); np.save(y_TRAIN_PATH, y_train)
    np.save(X_VAL_PATH,   X_val);   np.save(y_VAL_PATH,   y_val)
    np.save(X_TEST_PATH,  X_test);  np.save(y_TEST_PATH,  y_test)
    np.save(CLASSES_PATH, le.classes_)
    print(f"[DATA] Saved to {PREP_DIR}")

    return X_train, y_train, X_val, y_val, X_test, y_test, le.classes_


(X_train, y_train, X_val, y_val,
 X_test, y_test, classes_le) = prepare_or_load_data()

num_classes = len(classes_le)
class_names = list(classes_le)
print(f"[DATA] Train:{X_train.shape} Val:{X_val.shape} Test:{X_test.shape}")
print(f"[DATA] Classes: {class_names}")

# =============================================================================
# NORMALIZATION — per-channel z-score dari train saja
# =============================================================================
if os.path.exists(NORM_MEAN) and os.path.exists(NORM_STD):
    print("[NORM] Loading cached params...")
    norm_mean = np.load(NORM_MEAN)
    norm_std  = np.load(NORM_STD)
else:
    print("[NORM] Computing from train set...")
    norm_mean = X_train.mean(axis=(0, 1), keepdims=True)
    norm_std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-8
    np.save(NORM_MEAN, norm_mean)
    np.save(NORM_STD,  norm_std)

X_train = (X_train - norm_mean) / norm_std
X_val   = (X_val   - norm_mean) / norm_std
X_test  = (X_test  - norm_mean) / norm_std
print(f"[NORM] Done. mean≈{X_train.mean():.4f} std≈{X_train.std():.4f}")

# =============================================================================
# AUGMENTASI — hanya pada training
# =============================================================================
def augment_sample(x, y, w):
    noise = tf.random.normal(tf.shape(x), stddev=0.02)
    scale = tf.random.uniform([], 0.90, 1.10)
    return x * scale + noise, y, w

AUTOTUNE = tf.data.AUTOTUNE

def make_train_dataset(X, y_oh, sample_w):
    ds = tf.data.Dataset.from_tensor_slices((
        X.astype(np.float32),
        y_oh.astype(np.float32),
        sample_w.astype(np.float32),
    ))
    ds = ds.shuffle(min(len(X), 15000), seed=SEED)
    ds = ds.map(augment_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

def make_eval_dataset(X, y_oh):
    ds = tf.data.Dataset.from_tensor_slices((
        X.astype(np.float32),
        y_oh.astype(np.float32),
    ))
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# =============================================================================
# CLASS WEIGHTS
# =============================================================================
weights = compute_class_weight("balanced",
                               classes=np.unique(y_train), y=y_train)
class_weights = {i: float(w) for i, w in enumerate(weights)}
print(f"[CLASS WEIGHT] {class_weights}")

# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================
def build_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(32, 7, padding="same", activation="relu",
                        kernel_regularizer=L2_REG)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.SpatialDropout1D(SDROPOUT_RATE)(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(64, 5, padding="same", dilation_rate=2,
                        activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.SpatialDropout1D(SDROPOUT_RATE)(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 3, padding="same", dilation_rate=4,
                        activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="CNN")


def _residual_block(x, filters, dilation_rate=1):
    shortcut = x
    y = layers.Conv1D(filters, 3, padding="same", dilation_rate=dilation_rate,
                      activation="relu", kernel_regularizer=L2_REG)(x)
    y = layers.BatchNormalization()(y)
    y = layers.SpatialDropout1D(SDROPOUT_RATE)(y)
    y = layers.Conv1D(filters, 3, padding="same",
                      kernel_regularizer=L2_REG)(y)
    y = layers.BatchNormalization()(y)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same",
                                 kernel_regularizer=L2_REG)(shortcut)
    return layers.ReLU()(layers.Add()([shortcut, y]))


def build_resnet(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(64, 7, padding="same", activation="relu",
                        kernel_regularizer=L2_REG)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = _residual_block(x, 64,  dilation_rate=1)
    x   = _residual_block(x, 128, dilation_rate=2)
    x   = layers.MaxPooling1D(2)(x)
    x   = _residual_block(x, 128, dilation_rate=4)
    gap = layers.GlobalAveragePooling1D()(x)
    gmp = layers.GlobalMaxPooling1D()(x)
    x   = layers.concatenate([gap, gmp])
    x   = layers.Dense(128, activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="ResNet")


def build_multiscale(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    b1  = layers.Conv1D(32, 3, padding="same", activation="relu",
                        kernel_regularizer=L2_REG)(inp)
    b2  = layers.Conv1D(32, 5, padding="same", dilation_rate=2,
                        activation="relu", kernel_regularizer=L2_REG)(inp)
    b3  = layers.Conv1D(32, 7, padding="same", dilation_rate=4,
                        activation="relu", kernel_regularizer=L2_REG)(inp)
    x   = layers.concatenate([b1, b2, b3])
    x   = layers.BatchNormalization()(x)
    x   = layers.SpatialDropout1D(SDROPOUT_RATE)(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 5, padding="same", activation="relu",
                        kernel_regularizer=L2_REG)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="MultiScale")


def _channel_attention(x, reduction=8):
    c     = x.shape[-1]
    r_dim = max(c // reduction, 1)
    gap   = layers.GlobalAveragePooling1D()(x)
    exc   = layers.Dense(r_dim, activation="relu",
                         kernel_regularizer=L2_REG)(gap)
    exc   = layers.Dense(c, activation="sigmoid")(exc)
    exc   = layers.Reshape((1, c))(exc)
    return layers.Multiply()([x, exc])


def build_attention_cnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(64, 5, padding="same", activation="relu",
                        kernel_regularizer=L2_REG)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.SpatialDropout1D(SDROPOUT_RATE)(x)
    x   = _channel_attention(x, reduction=8)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 3, padding="same", dilation_rate=2,
                        activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.BatchNormalization()(x)
    x   = _channel_attention(x, reduction=8)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation="relu", kernel_regularizer=L2_REG)(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="AttentionCNN")


MODEL_BUILDERS = {
    "CNN":        build_cnn,
    "ResNet":     build_resnet,
    "MultiScale": build_multiscale,
    "Attention":  build_attention_cnn,
}

# =============================================================================
# METRICS UTILITIES
# =============================================================================
def compute_all_metrics(y_true, y_pred, y_prob):
    oh = tf.keras.utils.to_categorical(y_true, num_classes)
    per_cls = []
    for i in range(num_classes):
        try:
            per_cls.append(float(roc_auc_score(oh[:, i], y_prob[:, i])))
        except Exception:
            per_cls.append(0.0)
    try:
        micro_auc = float(roc_auc_score(oh, y_prob, average="micro"))
    except Exception:
        micro_auc = 0.0
    return dict(
        accuracy  = float(accuracy_score(y_true, y_pred)),
        precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall    = float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1        = float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        macro_auc = float(np.mean(per_cls)),
        micro_auc = micro_auc,
        per_cls_auc = per_cls,
        report = classification_report(y_true, y_pred,
                                       target_names=class_names, digits=4),
        cm     = confusion_matrix(y_true, y_pred),
    )

def overfitting_analysis(history):
    ta  = history["accuracy"][-1]
    va  = max(history["val_accuracy"])
    vl  = min(history["val_loss"])
    gap = ta - va
    cat = (
        "Underfitting"        if va < 0.60 else
        "Good Generalization" if gap < 0.05 else
        "Mild Overfitting"    if gap < 0.15 else
        "Severe Overfitting"
    )
    return dict(
        best_val_loss   = float(vl),
        best_val_acc    = float(va),
        final_train_acc = float(ta),
        acc_gap         = float(gap),
        category        = cat,
    )

# =============================================================================
# VISUALISASI
# =============================================================================
def plot_confusion_matrix(cm, names, title, save_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linewidths=0, annot_kws={"size": 13, "weight": "bold"})
    plt.title(title, fontsize=13, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_prob, title, save_path):
    oh = tf.keras.utils.to_categorical(y_true, num_classes)
    plt.figure(figsize=(9, 7))
    per_auc = []
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(oh[:, i], y_prob[:, i])
        a = auc(fpr, tpr); per_auc.append(a)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={a:.3f})")
    fpr_m, tpr_m, _ = roc_curve(oh.ravel(), y_prob.ravel())
    plt.plot(fpr_m, tpr_m, "k--", label=f"Micro AUC={auc(fpr_m, tpr_m):.3f}")
    plt.plot([0, 1], [0, 1], "gray", linestyle=":")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"{title}\nMacro AUC={np.mean(per_auc):.3f}", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_learning_curves(history, title, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(history["loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title(f"{title} — Loss"); ax1.legend()
    ax1.set_xlabel("Epoch"); ax1.grid(alpha=0.3)
    ax2.plot(history["accuracy"], label="Train")
    ax2.plot(history["val_accuracy"], label="Val")
    ax2.set_title(f"{title} — Accuracy"); ax2.legend()
    ax2.set_xlabel("Epoch"); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# =============================================================================
# MAIN TRAINING
# =============================================================================
def run_training(model_name):
    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name}  |  Input: {INPUT_SHAPE}")
    print(f"{'='*60}")

    model = MODEL_BUILDERS[model_name](INPUT_SHAPE, num_classes)
    model.summary(line_length=90)

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes).astype(np.float32)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes).astype(np.float32)
    sample_w   = np.array([class_weights[y] for y in y_train], dtype=np.float32)

    train_ds = make_train_dataset(X_train, y_train_oh, sample_w)
    val_ds   = make_eval_dataset(X_val, y_val_oh)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

    best_path = os.path.join(OUT_DIR, f"{model_name}_best.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PAT,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=LR_REDUCE_PAT,
                          factor=LR_REDUCE_FACT, min_lr=LR_MIN, verbose=1),
        ModelCheckpoint(filepath=best_path, monitor="val_loss",
                        save_best_only=True, verbose=1),
    ]

    t0   = time.time()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=MAX_EPOCHS, callbacks=callbacks, verbose=1)
    train_time = round(time.time() - t0, 2)

    model = tf.keras.models.load_model(best_path)
    print(f"\n[INFO] Best model loaded: {best_path}")

    # ── Evaluasi ───────────────────────────────────────────────
    print("[EVAL] Inference on test set...")
    y_prob = model.predict(X_test.astype(np.float32),
                           batch_size=BATCH_SIZE, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    m = compute_all_metrics(y_test, y_pred, y_prob)
    ov = overfitting_analysis(hist.history)

    print(f"\n  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Macro AUC : {m['macro_auc']:.4f}")
    print(f"  Micro AUC : {m['micro_auc']:.4f}")
    print(f"  F1 Macro  : {m['f1']:.4f}")
    print(f"  Overfitting: {ov['category']}  (gap={ov['acc_gap']:+.4f})")

    # ── Visualisasi ────────────────────────────────────────────
    pfx = model_name
    plot_learning_curves(hist.history, pfx,
        save_path=os.path.join(OUT_DIR, f"{pfx}_learning_curves.png"))
    plot_confusion_matrix(m["cm"], class_names,
        title=f"Confusion Matrix — {pfx}  Acc={m['accuracy']:.4f}",
        save_path=os.path.join(OUT_DIR, f"{pfx}_confusion_matrix.png"))
    plot_roc_curve(y_test, y_prob,
        title=f"ROC Curve — {pfx}",
        save_path=os.path.join(OUT_DIR, f"{pfx}_roc_curve.png"))

    # ── Simpan hasil ───────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== {pfx} | Input={INPUT_SHAPE} ===\n\n")
        f.write(m["report"])
        f.write(f"\nMacro AUC : {m['macro_auc']:.4f}")
        f.write(f"\nMicro AUC : {m['micro_auc']:.4f}\n")
        f.write(f"\nPer-class AUC:\n")
        for cn, ca in zip(class_names, m["per_cls_auc"]):
            f.write(f"  {cn}: {ca:.4f}\n")
        f.write(f"\n--- Overfitting Analysis ---\n")
        for k, v in ov.items():
            f.write(f"  {k}: {v}\n")

    summary = {
        "model":       model_name,
        "input_shape": list(INPUT_SHAPE),
        "metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in m.items()
            if k not in ("report", "cm")
        },
        "overfitting":     ov,
        "training_time_s": train_time,
        "best_epoch":      len(hist.history["loss"]),
    }
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SAVED] {OUT_DIR}")
    return summary


# =============================================================================
# RUN
# =============================================================================
summary = run_training(MODEL_NAME)

print(f"\n[DONE] {MODEL_NAME}")
print(f"  Accuracy  = {summary['metrics']['accuracy']:.4f}")
print(f"  Macro AUC = {summary['metrics']['macro_auc']:.4f}")
print(f"  Output    : {OUT_DIR}")
