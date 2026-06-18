# =============================================================================
#  JETSON ORIN NANO — ECG PTB-XL INFERENCE (TensorRT FP16)
#  4 Arsitektur CNN | Input: (1000, 12) | Full PTB-XL single-label
#
#  WSM Weights:
#    Accuracy : 0.40
#    Latency  : 0.30
#    CPU      : 0.15
#    GPU      : 0.10
#    RAM      : 0.05
#
#  Usage:
#    python jetson_inference.py
#
#  Output: results_jetson/
# =============================================================================

import os, ast, gc, re, json, time, warnings, threading, subprocess
import numpy as np
import pandas as pd
import wfdb, psutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
)
warnings.filterwarnings("ignore")

# =============================================================================
# KONFIGURASI PATH
# =============================================================================
PTBXL_DB  = "/home/najwa/projects/ecg_inference/ecg-fastapi-main/dataset/ptbxl_database.csv"
SCP_PATH  = "/home/najwa/projects/ecg_inference/ecg-fastapi-main/dataset/scp_statements.csv"
BASE_DIR  = "/home/najwa/projects/ecg_inference/ecg-fastapi-main/dataset/"
MODEL_DIR = "/home/najwa/projects/ecg_inference/ecg-fastapi-main/model"
MODEL_PATHS = {
    "CNN":        os.path.join(MODEL_DIR, "CNN_best.keras"),
    "ResNet":     os.path.join(MODEL_DIR, "ResNet_best.keras"),
    "MultiScale": os.path.join(MODEL_DIR, "MultiScale_best.keras"),
    "Attention":  os.path.join(MODEL_DIR, "Attention_best.keras"),
}
PREP_CACHE_DIR = (
    "/home/najwa/projects/ecg_inference/ecg-fastapi-main/"
    "dataset/preprocessed_record"
)
RESULT_DIR = "./results_jetson"

# =============================================================================
# KONSTANTA
# =============================================================================
FS            = 100
VALID_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
RECORD_LEN    = 1000
N_LEADS       = 12
INPUT_SHAPE   = (RECORD_LEN, N_LEADS)
MODEL_NAMES   = ["CNN", "ResNet", "MultiScale", "Attention"]
BATCH_SIZE    = 32
WARM_UP_RUNS  = 10

WSM_WEIGHTS = {
    "accuracy": 0.40,
    "time_ms":  0.30,
    "cpu_pct":  0.15,
    "gpu_pct":  0.10,
    "ram_mb":   0.05,
}

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================
DIRS = {
    "root":    RESULT_DIR,
    "reports": os.path.join(RESULT_DIR, "classification_reports"),
    "cm":      os.path.join(RESULT_DIR, "confusion_matrices"),
    "plots":   os.path.join(RESULT_DIR, "plots"),
    "tables":  os.path.join(RESULT_DIR, "summary_tables"),
    "trt":     os.path.join(RESULT_DIR, "trt_models"),
    "gt":      os.path.join(RESULT_DIR, "ground_truth"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LABEL ENCODER
# =============================================================================
le = LabelEncoder()
le.fit(sorted(VALID_CLASSES))
CLASS_NAMES = list(le.classes_)
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# GPU / TEGRASTATS MONITORING
# =============================================================================
HAS_GPUUTIL = HAS_PYNVML = False
try:
    import GPUtil
    if GPUtil.getGPUs():
        HAS_GPUUTIL = True
except Exception:
    pass
if not HAS_GPUUTIL:
    try:
        import pynvml
        pynvml.nvmlInit()
        HAS_PYNVML = True
    except Exception:
        pass

def get_gpu_stats():
    if HAS_GPUUTIL:
        try:
            g = GPUtil.getGPUs()[0]
            return round(g.load * 100, 2), round(g.memoryUsed, 2)
        except Exception:
            pass
    if HAS_PYNVML:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            return float(u.gpu), round(m.used / (1024 ** 2), 2)
        except Exception:
            pass
    return 0.0, 0.0


class TegrastatsMonitor:
    def __init__(self, interval_ms=200):
        self.interval_ms  = interval_ms
        self.gpu_readings = []
        self.mem_readings = []
        self._stop        = threading.Event()
        self._proc        = None
        self._thread      = None

    def start(self):
        self._stop.clear()
        self.gpu_readings.clear()
        self.mem_readings.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                m = re.search(r"GR3D_FREQ\s+(\d+)%", line)
                if m:
                    self.gpu_readings.append(float(m.group(1)))
                m2 = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
                if m2:
                    self.mem_readings.append(float(m2.group(1)))
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        if self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def avg_gpu_pct(self):
        return float(np.mean(self.gpu_readings)) if self.gpu_readings else 0.0

    @property
    def avg_ram_mb(self):
        return float(np.mean(self.mem_readings)) if self.mem_readings else 0.0


tegra = TegrastatsMonitor(interval_ms=200)

# =============================================================================
# TF SETUP
# =============================================================================
import tensorflow as tf
from tensorflow.keras import layers, models

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"[GPU] Warning: {e}")

# =============================================================================
# SIGNAL PROCESSING
# =============================================================================
def bandpass_filter(sig):
    nyq  = FS / 2
    b, a = butter(4, [0.5 / nyq, 24 / nyq], btype="band")
    return lfilter(b, a, sig, axis=0)

# =============================================================================
# MODEL ARCHITECTURES — identik dengan training
# =============================================================================
L2_STRENGTH   = 1e-3
DROPOUT_RATE  = 0.50
SDROPOUT_RATE = 0.15
L2_REG        = tf.keras.regularizers.l2(L2_STRENGTH)

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
    y = layers.Conv1D(filters, 3, padding="same", kernel_regularizer=L2_REG)(y)
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
    exc   = layers.Dense(r_dim, activation="relu", kernel_regularizer=L2_REG)(gap)
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

MODEL_LABEL = {
    "CNN":        "1D-CNN",
    "ResNet":     "1D-ResNet",
    "MultiScale": "Multi-Scale CNN",
    "Attention":  "Attention-Based CNN",
}

MODEL_COLORS = {
    "CNN":        "#2196F3",
    "ResNet":     "#4CAF50",
    "MultiScale": "#FF9800",
    "Attention":  "#E91E63",
}

# =============================================================================
# TENSORRT CONVERSION
# =============================================================================
def convert_to_trt_fp16(keras_model, trt_save_path):
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    tmp_dir = trt_save_path + "_savedmodel_tmp"
    keras_model.save(tmp_dir, save_format="tf")

    params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=1 << 28,
        minimum_segment_size=3,
        allow_build_at_runtime=True,
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=tmp_dir,
        conversion_params=params,
    )
    converter.convert()

    def input_fn():
        for bs in [1, BATCH_SIZE]:
            dummy = np.zeros((bs, *INPUT_SHAPE), dtype=np.float32)
            yield (tf.constant(dummy),)

    print("    [TRT] Building engine...")
    converter.build(input_fn=input_fn)
    converter.save(trt_save_path)
    print(f"    [TRT] Engine saved: {trt_save_path}")


class TRTModel:
    def __init__(self, trt_path):
        self._saved  = tf.saved_model.load(trt_path)
        self._infer  = self._saved.signatures["serving_default"]
        self._outkey = self._find_output_key()

    def _find_output_key(self):
        for k, v in self._infer.structured_outputs.items():
            if hasattr(v, "shape") and len(v.shape) >= 2:
                if v.shape[-1] == NUM_CLASSES:
                    return k
        return list(self._infer.structured_outputs.keys())[0]

    def predict_batch(self, X_batch):
        t   = tf.constant(X_batch, dtype=tf.float32)
        out = self._infer(t)
        return out[self._outkey].numpy()

# =============================================================================
# LOAD METADATA — full PTB-XL single-label
# =============================================================================
print("\n" + "="*60)
print("  STEP 1: Load metadata — full PTB-XL (single-label)")
print("="*60)

df_all   = pd.read_csv(PTBXL_DB)
scp_df   = pd.read_csv(SCP_PATH, index_col=0)
scp_diag = scp_df[scp_df["diagnostic"] == 1]

def scp_to_superclass(s):
    s   = ast.literal_eval(s)
    sup = [scp_diag.loc[c, "diagnostic_class"]
           for c in s if c in scp_diag.index]
    return list(set(sup))

df_all["superclass"] = df_all["scp_codes"].apply(scp_to_superclass)
df_all = df_all[df_all["superclass"].map(len) == 1]
df_all["superclass"] = df_all["superclass"].map(lambda x: x[0])
df_all = df_all[df_all["superclass"].isin(VALID_CLASSES)].reset_index(drop=True)
eval_df = df_all.reset_index(drop=True)

print(f"  Total rekaman single-label: {len(eval_df)}")
print(f"  Distribusi kelas:\n{eval_df['superclass'].value_counts().to_string()}")

# =============================================================================
# LOAD NORMALISASI
# =============================================================================
mean_path = os.path.join(PREP_CACHE_DIR, "norm_mean.npy")
std_path  = os.path.join(PREP_CACHE_DIR, "norm_std.npy")

if not (os.path.exists(mean_path) and os.path.exists(std_path)):
    raise FileNotFoundError(
        f"Normalisasi cache tidak ditemukan: {PREP_CACHE_DIR}\n"
        f"Pastikan ecg_train_record.py sudah dijalankan terlebih dahulu."
    )

norm_mean = np.load(mean_path)
norm_std  = np.load(std_path)
print(f"\n[NORM] mean={norm_mean.shape}, std={norm_std.shape}")

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess_record(row):
    fpath  = os.path.join(BASE_DIR, row["filename_lr"])
    ecg_id = int(row["ecg_id"])
    label  = int(le.transform([row["superclass"]])[0])

    rec = wfdb.rdrecord(fpath).p_signal.astype(np.float32)
    if len(rec) < RECORD_LEN:
        pad = np.zeros((RECORD_LEN - len(rec), N_LEADS), dtype=np.float32)
        rec = np.concatenate([rec, pad], axis=0)
    rec  = rec[:RECORD_LEN]
    filt = bandpass_filter(rec)
    X    = (filt[np.newaxis] - norm_mean) / norm_std
    return X.astype(np.float32), ecg_id, label

# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y_true, y_pred, y_prob):
    oh = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)
    per_cls = []
    for i in range(NUM_CLASSES):
        try:
            per_cls.append(float(roc_auc_score(oh[:, i], y_prob[:, i])))
        except Exception:
            per_cls.append(0.0)
    try:
        micro_auc = float(roc_auc_score(oh, y_prob, average="micro"))
    except Exception:
        micro_auc = 0.0
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred,
                             average="macro", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred,
                             average="macro", zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred,
                             average="macro", zero_division=0)),
        "macro_auc":   float(np.mean(per_cls)),
        "micro_auc":   micro_auc,
        "per_cls_auc": per_cls,
        "report":      classification_report(y_true, y_pred,
                             target_names=CLASS_NAMES, digits=4),
        "cm":          confusion_matrix(y_true, y_pred),
    }

# =============================================================================
# WSM
# =============================================================================
def compute_wsm(df_sub):
    eps  = 1e-12
    norm = pd.DataFrame(index=df_sub.index)
    mn, mx = df_sub["accuracy"].min(), df_sub["accuracy"].max()
    norm["accuracy"] = (df_sub["accuracy"] - mn) / (mx - mn + eps)
    for col in ["time_avg_ms", "cpu_pct", "gpu_pct", "ram_mb"]:
        mn, mx = df_sub[col].min(), df_sub[col].max()
        norm[col] = 1.0 - (df_sub[col] - mn) / (mx - mn + eps)
    return (
        norm["accuracy"]    * WSM_WEIGHTS["accuracy"] +
        norm["time_avg_ms"] * WSM_WEIGHTS["time_ms"]  +
        norm["cpu_pct"]     * WSM_WEIGHTS["cpu_pct"]  +
        norm["gpu_pct"]     * WSM_WEIGHTS["gpu_pct"]  +
        norm["ram_mb"]      * WSM_WEIGHTS["ram_mb"]
    )

def compute_wsm_components(df_sub):
    eps = 1e-12
    out = pd.DataFrame(index=df_sub.index)
    mn, mx = df_sub["accuracy"].min(), df_sub["accuracy"].max()
    out["acc_n"] = (df_sub["accuracy"] - mn) / (mx - mn + eps)
    for col, key in zip(
        ["time_avg_ms", "cpu_pct", "gpu_pct", "ram_mb"],
        ["lat_n", "cpu_n", "gpu_n", "ram_n"]
    ):
        mn, mx = df_sub[col].min(), df_sub[col].max()
        out[key] = 1.0 - (df_sub[col] - mn) / (mx - mn + eps)
    return out

# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================
print("\n" + "="*60)
print("  STEP 2: Inference — TensorRT FP16 — 4 model")
print("="*60)

all_results    = []
gt_pred_tables = {}

for model_name in MODEL_NAMES:
    print(f"\n  {'─'*50}")
    print(f"  MODEL: {model_name}")
    print(f"  {'─'*50}")

    if not os.path.exists(MODEL_PATHS[model_name]):
        print(f"  [SKIP] Model tidak ditemukan: {MODEL_PATHS[model_name]}")
        continue

    # ── Load Keras & konversi TRT ─────────────────────────────
    keras_model = MODEL_BUILDERS[model_name](INPUT_SHAPE, NUM_CLASSES)
    keras_model.load_weights(MODEL_PATHS[model_name])

    trt_path = os.path.join(DIRS["trt"], f"{model_name}_trt_fp16")
    use_trt  = False

    if os.path.exists(trt_path):
        print(f"  [TRT] Engine sudah ada: {trt_path}")
        use_trt = True
    else:
        try:
            convert_to_trt_fp16(keras_model, trt_path)
            use_trt = True
        except Exception as e:
            print(f"  [TRT] Konversi gagal: {e} → fallback Keras")

    trt_model = None
    if use_trt:
        try:
            trt_model = TRTModel(trt_path)
            def predict_fn(X):
                return trt_model.predict_batch(X)
        except Exception as e:
            print(f"  [TRT] Load gagal: {e} → fallback Keras")
            use_trt = False

    if not use_trt:
        _km = keras_model
        def predict_fn(X):
            return _km.predict(X, verbose=0)

    if use_trt:
        del keras_model
        gc.collect()

    # ── Preprocessing ─────────────────────────────────────────
    print(f"  [PREP] {len(eval_df)} rekaman...")
    records_data = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df),
                       desc="    Preprocessing", leave=False):
        try:
            X_rec, ecg_id, label = preprocess_record(row)
            records_data.append((X_rec, ecg_id, label))
        except Exception:
            pass

    print(f"  [PREP] OK={len(records_data)}")
    if not records_data:
        continue

    X_all  = np.concatenate([r[0] for r in records_data], axis=0)
    y_all  = np.array([r[2] for r in records_data])
    id_all = np.array([r[1] for r in records_data])

    # ── Warm-up ───────────────────────────────────────────────
    for _ in range(WARM_UP_RUNS):
        _ = predict_fn(records_data[0][0])

    # ── Inferensi + monitoring ────────────────────────────────
    print(f"  [INFER] {len(records_data)} rekaman...")
    all_probs    = []
    latencies_ms = []
    cpu_readings = []

    tegra.start()
    for start in tqdm(range(0, len(X_all), BATCH_SIZE),
                      desc="    Inferensi", leave=False):
        batch = X_all[start:start + BATCH_SIZE]
        bs    = len(batch)
        cpu_readings.append(psutil.cpu_percent(interval=None))
        t0   = time.perf_counter()
        prob = predict_fn(batch)
        t1   = time.perf_counter()
        latencies_ms.extend([(t1 - t0) / bs * 1000.0] * bs)
        all_probs.append(prob)
    tegra.stop()
    time.sleep(0.3)

    # ── Resource stats ────────────────────────────────────────
    avg_cpu = float(np.mean(cpu_readings)) if cpu_readings else 0.0
    avg_gpu = tegra.avg_gpu_pct
    avg_ram = tegra.avg_ram_mb
    if avg_gpu == 0.0:
        avg_gpu, _ = get_gpu_stats()
    if avg_ram == 0.0:
        avg_ram = round(
            psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)

    lat   = np.array(latencies_ms)
    avg_t = float(np.mean(lat))
    min_t = float(np.min(lat))
    max_t = float(np.max(lat))
    std_t = float(np.std(lat))
    p95_t = float(np.percentile(lat, 95))

    print(f"  [LATENCY] avg={avg_t:.4f} min={min_t:.4f} "
          f"max={max_t:.4f} std={std_t:.4f} p95={p95_t:.4f} ms")
    print(f"  [RESOURCE] CPU={avg_cpu:.1f}% GPU={avg_gpu:.1f}% RAM={avg_ram:.1f}MB")

    # ── Metrik ───────────────────────────────────────────────
    y_probs_arr = np.concatenate(all_probs, axis=0)
    y_pred      = np.argmax(y_probs_arr, axis=1)
    m = compute_metrics(y_all, y_pred, y_probs_arr)
    print(f"  [RESULT] Acc={m['accuracy']:.4f} MacroAUC={m['macro_auc']:.4f} "
          f"MicroAUC={m['micro_auc']:.4f} F1={m['f1']:.4f}")

    # ── Ground truth vs prediksi per kelas ───────────────────
    gt_counts   = {cn: 0 for cn in CLASS_NAMES}
    pred_counts = {cn: 0 for cn in CLASS_NAMES}
    for gt_idx, pred_idx in zip(y_all, y_pred):
        gt_counts[CLASS_NAMES[gt_idx]]    += 1
        pred_counts[CLASS_NAMES[pred_idx]] += 1
    gt_pred_tables[model_name] = {
        "model":    model_name,
        "n_total":  len(y_all),
        "n_correct": int(np.sum(y_all == y_pred)),
        **{f"gt_{cn}":   gt_counts[cn]   for cn in CLASS_NAMES},
        **{f"pred_{cn}": pred_counts[cn] for cn in CLASS_NAMES},
    }

    # ── Simpan prediksi detail ────────────────────────────────
    df_pred = pd.DataFrame({
        "ecg_id":    id_all,
        "gt_label":  [CLASS_NAMES[i] for i in y_all],
        "pred_label":[CLASS_NAMES[i] for i in y_pred],
        "correct":   (y_all == y_pred).astype(int),
    })
    for i, cn in enumerate(CLASS_NAMES):
        df_pred[f"prob_{cn}"] = y_probs_arr[:, i]
    df_pred.to_csv(
        os.path.join(DIRS["gt"], f"pred_detail_{model_name}.csv"), index=False)

    # ── Confusion matrix ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {MODEL_LABEL[model_name]}\n"
                 f"TensorRT FP16 | Acc={m['accuracy']:.4f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS["cm"], f"cm_{model_name}.png"), dpi=200)
    plt.close()

    # ── Classification report ─────────────────────────────────
    with open(os.path.join(DIRS["reports"],
                           f"report_{model_name}.txt"), "w") as f:
        f.write(f"{'='*55}\n  {MODEL_LABEL[model_name]}\n"
                f"  TensorRT FP16 | Jetson Orin Nano | Full PTB-XL\n{'='*55}\n\n")
        f.write(m["report"])
        f.write(f"\nMacro AUC : {m['macro_auc']:.4f}\n")
        f.write(f"Micro AUC : {m['micro_auc']:.4f}\n")
        f.write(f"\nPer-class AUC:\n")
        for cn, ca in zip(CLASS_NAMES, m["per_cls_auc"]):
            f.write(f"  {cn}: {ca:.4f}\n")
        f.write(f"\n--- Edge Metrics ---\n")
        f.write(f"Avg Latency : {avg_t:.4f} ms\n")
        f.write(f"Min Latency : {min_t:.4f} ms\n")
        f.write(f"Max Latency : {max_t:.4f} ms\n")
        f.write(f"Std Latency : {std_t:.4f} ms\n")
        f.write(f"P95 Latency : {p95_t:.4f} ms\n")
        f.write(f"CPU         : {avg_cpu:.2f}%\n")
        f.write(f"GPU         : {avg_gpu:.2f}%\n")
        f.write(f"RAM         : {avg_ram:.2f} MB\n")
        f.write(f"Backend     : {'TensorRT FP16' if use_trt else 'Keras (fallback)'}\n")

    all_results.append({
        "model":       model_name,
        "accuracy":    round(m["accuracy"],  4),
        "precision":   round(m["precision"], 4),
        "recall":      round(m["recall"],    4),
        "f1":          round(m["f1"],        4),
        "macro_auc":   round(m["macro_auc"], 4),
        "micro_auc":   round(m["micro_auc"], 4),
        "time_avg_ms": round(avg_t, 4),
        "time_min_ms": round(min_t, 4),
        "time_max_ms": round(max_t, 4),
        "time_std_ms": round(std_t, 4),
        "time_p95_ms": round(p95_t, 4),
        "cpu_pct":     round(avg_cpu, 2),
        "gpu_pct":     round(avg_gpu, 2),
        "ram_mb":      round(avg_ram, 2),
        "n_records":   len(y_all),
        "trt_fp16":    use_trt,
    })

    if trt_model is not None:
        del trt_model
    gc.collect()

# =============================================================================
# STEP 3: WSM RANKING
# =============================================================================
print("\n" + "="*60)
print("  STEP 3: WSM Ranking")
print("="*60)

if not all_results:
    print("[ERROR] Tidak ada hasil.")
    import sys; sys.exit(1)

df = pd.DataFrame(all_results)
df["wsm_score"] = compute_wsm(df)
df["rank_wsm"]  = df["wsm_score"].rank(ascending=False).astype(int)
df_sorted       = df.sort_values("rank_wsm").reset_index(drop=True)
wsm_comp        = compute_wsm_components(df)
wsm_comp.index  = df.index

df_sorted.to_csv(os.path.join(RESULT_DIR, "metrics.csv"), index=False)
df_sorted.to_csv(os.path.join(DIRS["tables"], "ranking_wsm.csv"), index=False)

best = df_sorted.iloc[0]
print(f"\n  {'Rank':>4} | {'Model':14} | {'Acc':>7} | {'MacAUC':>7} | "
      f"{'F1':>7} | {'Lat(ms)':>8} | {'CPU%':>6} | {'GPU%':>6} | "
      f"{'RAM(MB)':>8} | {'WSM':>7}")
print("  " + "-"*85)
for _, row in df_sorted.iterrows():
    print(f"  {int(row['rank_wsm']):>4} | {MODEL_LABEL.get(row['model'],row['model']):14} | "
          f"{row['accuracy']:>7.4f} | {row['macro_auc']:>7.4f} | "
          f"{row['f1']:>7.4f} | {row['time_avg_ms']:>8.4f} | "
          f"{row['cpu_pct']:>6.2f} | {row['gpu_pct']:>6.2f} | "
          f"{row['ram_mb']:>8.2f} | {row['wsm_score']:>7.4f}")

# =============================================================================
# STEP 4: TABEL 6.12 — Efisiensi Komputasi
# =============================================================================
print("\n" + "="*60)
print("  STEP 4: Tabel 6.12 — Efisiensi Komputasi")
print("="*60)

tbl612 = []
for _, row in df_sorted.iterrows():
    tbl612.append({
        "Model":        MODEL_LABEL.get(row["model"], row["model"]),
        "Latency (ms)": f"{row['time_avg_ms']:.4f}",
        "CPU (%)":      f"{row['cpu_pct']:.2f}",
        "GPU (%)":      f"{row['gpu_pct']:.2f}",
        "RAM (MB)":     f"{row['ram_mb']:.2f}",
    })

pd.DataFrame(tbl612).to_csv(
    os.path.join(DIRS["tables"], "tabel_6_12_efisiensi.csv"), index=False)

print(f"\n  {'Model':<22} | {'Latency (ms)':>12} | {'CPU (%)':>8} | "
      f"{'GPU (%)':>8} | {'RAM (MB)':>10}")
print("  " + "-"*68)
for r in tbl612:
    print(f"  {r['Model']:<22} | {r['Latency (ms)']:>12} | {r['CPU (%)']:>8} | "
          f"{r['GPU (%)']:>8} | {r['RAM (MB)']:>10}")

# =============================================================================
# STEP 5: TABEL 6.13 — Ground Truth vs Prediksi
# =============================================================================
print("\n" + "="*60)
print("  STEP 5: Tabel 6.13 — Ground Truth vs Prediksi")
print("="*60)

tbl613 = []
for mn in MODEL_NAMES:
    if mn not in gt_pred_tables:
        continue
    row = gt_pred_tables[mn]
    tbl613.append({
        "Model":     MODEL_LABEL.get(mn, mn),
        "N Records": row["n_total"],
        "N Correct": row["n_correct"],
        "Accuracy":  f"{row['n_correct'] / row['n_total']:.4f}",
        **{f"GT_{cn}":   row[f"gt_{cn}"]   for cn in CLASS_NAMES},
        **{f"Pred_{cn}": row[f"pred_{cn}"] for cn in CLASS_NAMES},
    })

pd.DataFrame(tbl613).to_csv(
    os.path.join(DIRS["tables"], "tabel_6_13_ground_truth.csv"), index=False)

print(f"\n  {'Model':<22} | {'N':>6} | {'N Correct':>10} | {'Accuracy':>9}")
print("  " + "-"*55)
for r in tbl613:
    print(f"  {r['Model']:<22} | {r['N Records']:>6} | "
          f"{r['N Correct']:>10} | {r['Accuracy']:>9}")

# =============================================================================
# STEP 6: VISUALISASI
# =============================================================================
print("\n" + "="*60)
print("  STEP 6: Visualisasi")
print("="*60)

DPI   = 200
clrs  = [MODEL_COLORS[m] for m in df_sorted["model"]]
LBLS  = [MODEL_LABEL[m] for m in df_sorted["model"]]

def label_bar(ax, bar, value, fmt="{:.4f}", fontsize=10):
    y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_max * 0.012,
            fmt.format(value),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color="#1a1a1a")

def savefig(name):
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(DIRS["plots"], f"{name}.{ext}"),
                    dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  [PLOT] {name}")


# ── Plot 1: Accuracy ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#FAFAFA")
bars = ax.bar(LBLS, df_sorted["accuracy"], color=clrs,
              width=0.55, alpha=0.88, edgecolor="white", zorder=3)
ax.set_ylim(0, 1.10)
for bar, v in zip(bars, df_sorted["accuracy"]): label_bar(ax, bar, v)
ax.set_ylabel("Accuracy", fontsize=13)
ax.set_title("Accuracy — 4 Arsitektur CNN\n"
             "(TensorRT FP16 | Jetson Orin Nano | Full PTB-XL)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); savefig("01_accuracy")

# ── Plot 2: Macro AUC ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#FAFAFA")
bars = ax.bar(LBLS, df_sorted["macro_auc"], color=clrs,
              width=0.55, alpha=0.88, edgecolor="white", zorder=3)
ax.set_ylim(0, 1.10)
for bar, v in zip(bars, df_sorted["macro_auc"]): label_bar(ax, bar, v)
ax.set_ylabel("Macro AUC", fontsize=13)
ax.set_title("Macro AUC — 4 Arsitektur CNN\n(TensorRT FP16 | Full PTB-XL)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); savefig("02_macro_auc")

# ── Plot 3: F1-Score ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#FAFAFA")
bars = ax.bar(LBLS, df_sorted["f1"], color=clrs,
              width=0.55, alpha=0.88, edgecolor="white", zorder=3)
ax.set_ylim(0, 1.10)
for bar, v in zip(bars, df_sorted["f1"]): label_bar(ax, bar, v)
ax.set_ylabel("F1-Score (Macro)", fontsize=13)
ax.set_title("F1-Score Macro — 4 Arsitektur CNN\n(TensorRT FP16 | Full PTB-XL)",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); savefig("03_f1_score")

# ── Plot 4: Latency avg + error bar ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_facecolor("#FAFAFA")
x     = np.arange(len(df_sorted))
avgs  = df_sorted["time_avg_ms"].values
mins  = df_sorted["time_min_ms"].values
maxs  = df_sorted["time_max_ms"].values
p95s  = df_sorted["time_p95_ms"].values
bars  = ax.bar(x, avgs, color=clrs, width=0.5, alpha=0.88,
               edgecolor="white", zorder=3)
y_ceil = max(maxs) * 1.4
ax.errorbar(x, avgs, yerr=[avgs - mins, maxs - avgs],
            fmt="none", color="#333", capsize=6,
            capthick=1.8, elinewidth=1.8, zorder=5)
ax.set_ylim(0, y_ceil)
for i, (bar, avg, mn, mx) in enumerate(zip(bars, avgs, mins, maxs)):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + y_ceil*0.014,
            f"avg\n{avg:.4f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, mn - y_ceil*0.04,
            f"↓{mn:.4f}", ha="center", va="top",
            fontsize=8, color="#1565C0", fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, mx + y_ceil*0.06,
            f"↑{mx:.4f}", ha="center", va="bottom",
            fontsize=8, color="#B71C1C", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LBLS, fontsize=11)
ax.set_ylabel("Inference Time per Sample (ms)", fontsize=12)
ax.set_title("Inference Latency — Avg / Min / Max\n"
             "(TensorRT FP16 | error bar = min–max)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); savefig("04_latency")

# ── Plot 5: Resource (CPU, GPU, RAM) gabung ──────────────────
fig, ax1 = plt.subplots(figsize=(13, 7))
ax2 = ax1.twinx()
ax1.set_facecolor("#F8F9FA"); ax2.set_facecolor("#F8F9FA")
width = 0.20
x2   = np.arange(len(df_sorted))

bars_lat = ax1.bar(x2 - 1.5*width, df_sorted["time_avg_ms"], width,
                   color=clrs, alpha=0.92, edgecolor="white",
                   label="Latency (ms)", zorder=3)
bars_cpu = ax2.bar(x2 - 0.5*width, df_sorted["cpu_pct"], width,
                   color=clrs, alpha=0.65, edgecolor="white",
                   hatch="///", label="CPU (%)", zorder=3)
bars_gpu = ax2.bar(x2 + 0.5*width, df_sorted["gpu_pct"], width,
                   color=clrs, alpha=0.45, edgecolor="white",
                   hatch="xxx", label="GPU (%)", zorder=3)
bars_ram = ax2.bar(x2 + 1.5*width, df_sorted["ram_mb"]/100, width,
                   color=clrs, alpha=0.30, edgecolor="white",
                   hatch="...", label="RAM (MB÷100)", zorder=3)

for bar, v in zip(bars_lat, df_sorted["time_avg_ms"]):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, v in zip(bars_cpu, df_sorted["cpu_pct"]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, v in zip(bars_gpu, df_sorted["gpu_pct"]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
for bar, v in zip(bars_ram, df_sorted["ram_mb"]):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax1.set_xticks(x2); ax1.set_xticklabels(LBLS, fontsize=12, fontweight="bold")
ax1.set_ylabel("Avg Latency (ms)", fontsize=12, color="#333")
ax2.set_ylabel("CPU / GPU (%)  |  RAM ÷ 100", fontsize=12, color="#555")
ax1.set_ylim(0, df_sorted["time_avg_ms"].max() * 2.2)
ax2.set_ylim(0, max(df_sorted["cpu_pct"].max(),
                    df_sorted["gpu_pct"].max(),
                    (df_sorted["ram_mb"]/100).max()) * 1.7)

model_patches = [mpatches.Patch(color=MODEL_COLORS[m],
                                label=MODEL_LABEL[m], alpha=0.85)
                 for m in df_sorted["model"]]
metric_patches = [
    mpatches.Patch(facecolor="#aaa", edgecolor="#666", hatch="",
                   label="Latency (ms) — kiri"),
    mpatches.Patch(facecolor="#aaa", edgecolor="#666", hatch="///",
                   label="CPU (%)"),
    mpatches.Patch(facecolor="#aaa", edgecolor="#666", hatch="xxx",
                   label="GPU (%)"),
    mpatches.Patch(facecolor="#aaa", edgecolor="#666", hatch="...",
                   label="RAM (MB÷100)"),
]
leg1 = ax1.legend(handles=model_patches, title="Arsitektur",
                  fontsize=10, loc="upper left", framealpha=0.9)
ax1.add_artist(leg1)
ax1.legend(handles=metric_patches, title="Metrik",
           fontsize=10, loc="upper right", framealpha=0.9)
ax1.set_title("Resource Usage: Latency · CPU · GPU · RAM\n"
              "(TensorRT FP16 | Jetson Orin Nano | Full PTB-XL)",
              fontsize=13, fontweight="bold")
ax1.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
plt.tight_layout(); savefig("05_resource_metrics")

# ── Plot 6: WSM breakdown stacked bar ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("WSM Ranking — 4 Arsitektur CNN\n"
             f"Bobot: Accuracy={WSM_WEIGHTS['accuracy']}, "
             f"Latency={WSM_WEIGHTS['time_ms']}, "
             f"CPU={WSM_WEIGHTS['cpu_pct']}, "
             f"GPU={WSM_WEIGHTS['gpu_pct']}, "
             f"RAM={WSM_WEIGHTS['ram_mb']}",
             fontsize=12, fontweight="bold")

# Panel kiri: total WSM horizontal
ax = axes[0]; ax.set_facecolor("#F8F9FA")
ypos  = np.arange(len(df_sorted))
clrs_s = [MODEL_COLORS[m] for m in df_sorted["model"]]
bars_h = ax.barh(ypos, df_sorted["wsm_score"], color=clrs_s,
                 alpha=0.88, edgecolor="white", height=0.55)
x_max = df_sorted["wsm_score"].max() + 0.12
ax.set_xlim(0, x_max)
for bar, score, rank in zip(bars_h, df_sorted["wsm_score"], df_sorted["rank_wsm"]):
    ax.text(bar.get_width() + x_max*0.015,
            bar.get_y() + bar.get_height()/2,
            f"{score:.4f}", va="center", ha="left",
            fontsize=13, fontweight="bold")
    ax.text(bar.get_width()*0.05,
            bar.get_y() + bar.get_height()/2,
            f"#{int(rank)}", va="center", ha="left",
            fontsize=12, fontweight="bold", color="white")
ax.set_yticks(ypos)
ax.set_yticklabels([MODEL_LABEL[m] for m in df_sorted["model"]], fontsize=13)
ax.set_xlabel("WSM Score", fontsize=12)
ax.set_title("Total WSM Score", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--", zorder=0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel kanan: stacked breakdown
ax2 = axes[1]; ax2.set_facecolor("#F8F9FA")
comp_items = [
    ("acc_n", WSM_WEIGHTS["accuracy"], "#1976D2",
     f"Accuracy (w={WSM_WEIGHTS['accuracy']})"),
    ("lat_n", WSM_WEIGHTS["time_ms"],  "#F57C00",
     f"Latency eff. (w={WSM_WEIGHTS['time_ms']})"),
    ("cpu_n", WSM_WEIGHTS["cpu_pct"],  "#7B1FA2",
     f"CPU eff. (w={WSM_WEIGHTS['cpu_pct']})"),
    ("gpu_n", WSM_WEIGHTS["gpu_pct"],  "#C62828",
     f"GPU eff. (w={WSM_WEIGHTS['gpu_pct']})"),
    ("ram_n", WSM_WEIGHTS["ram_mb"],   "#0097A7",
     f"RAM eff. (w={WSM_WEIGHTS['ram_mb']})"),
]
x3      = np.arange(len(df_sorted))
bottoms = np.zeros(len(df_sorted))
for key, w, clr, lbl in comp_items:
    # reindex wsm_comp to match df_sorted order
    vals = np.array([
        wsm_comp.loc[df.index[df["model"] == m].tolist()[0], key] * w
        for m in df_sorted["model"]
    ])
    ax2.bar(x3, vals, bottom=bottoms, color=clr, alpha=0.85,
            edgecolor="white", linewidth=0.4, label=lbl, zorder=3, width=0.55)
    for xi, (v, bot) in enumerate(zip(vals, bottoms)):
        if v > 0.015:
            ax2.text(xi, bot + v/2, f"{v:.3f}",
                     ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")
    bottoms += vals

for xi, (mn, sc) in enumerate(zip(df_sorted["model"], df_sorted["wsm_score"])):
    rank = int(df_sorted.loc[df_sorted["model"]==mn, "rank_wsm"].values[0])
    ax2.text(xi, bottoms[xi] + 0.008,
             f"#{rank}  {sc:.4f}", ha="center", va="bottom",
             fontsize=12, fontweight="bold", color="#1a1a1a")

ax2.set_xticks(x3)
ax2.set_xticklabels([MODEL_LABEL[m] for m in df_sorted["model"]],
                    fontsize=12, fontweight="bold")
ax2.set_ylabel("Kontribusi WSM (terbobot)", fontsize=12)
ax2.set_title("Breakdown Kontribusi WSM", fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax2.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
ax2.set_ylim(0, bottoms.max() * 1.25)
plt.tight_layout(); savefig("06_wsm_breakdown")

# ── Plot 7: Confusion matrix 4 model 1 figure ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Confusion Matrix — 4 Arsitektur CNN\n"
             "(TensorRT FP16 | Jetson Orin Nano | Full PTB-XL)",
             fontsize=14, fontweight="bold")
for idx, mn in enumerate(MODEL_NAMES):
    if mn not in gt_pred_tables:
        continue
    fp = os.path.join(DIRS["gt"], f"pred_detail_{mn}.csv")
    if not os.path.exists(fp):
        continue
    d = pd.read_csv(fp)
    from sklearn.metrics import confusion_matrix as cm_fn
    cm = cm_fn(d["gt_label"], d["pred_label"], labels=CLASS_NAMES)
    acc = cm.diagonal().sum() / cm.sum()
    ax = axes[idx // 2][idx % 2]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0,
                annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True", fontsize=12, fontweight="bold")
    ax.set_title(f"{MODEL_LABEL[mn]}  |  Acc = {acc:.4f}",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
savefig("07_confusion_matrix_all_models")

# =============================================================================
# SIMPAN SUMMARY
# =============================================================================
slim_cols = ["rank_wsm", "model", "accuracy", "macro_auc", "micro_auc", "f1",
             "time_avg_ms", "time_min_ms", "time_max_ms", "time_std_ms",
             "time_p95_ms", "cpu_pct", "gpu_pct", "ram_mb", "wsm_score"]
df_sorted[slim_cols].to_csv(
    os.path.join(DIRS["tables"], "summary_thesis.csv"), index=False)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print(f"  SELESAI — Output: {RESULT_DIR}")
print("="*60)
print(f"\n  ★ BEST WSM: {MODEL_LABEL.get(best['model'], best['model'])}")
print(f"    Accuracy  = {best['accuracy']:.4f}")
print(f"    Macro AUC = {best['macro_auc']:.4f}")
print(f"    Latency   = {best['time_avg_ms']:.4f} ms")
print(f"    WSM Score = {best['wsm_score']:.4f}")
print(f"\n  Plots   : {DIRS['plots']}")
print(f"  Tables  : {DIRS['tables']}")
print(f"  Reports : {DIRS['reports']}")
print("\nDone ✅")
