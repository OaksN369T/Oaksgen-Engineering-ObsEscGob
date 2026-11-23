import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix
from xgboost import XGBClassifier

# ==============================================================================
# 1. CONFIGURACIÓN DE TRAZABILIDAD MLFLOW (OaksGen-Engineering v2.1 -> v3.0)
# ==============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("OaksGen-Engineering-v3.0-Escalabilidad-Fairness")

np.random.seed(42)

# ==============================================================================
# 2. GENERACIÓN DE DATASET INDUSTRIAL AUDITABLE (n = 1,200 registros)
#    Segmentado por Turno (Matutino, Vespertino, Nocturno) y Líneas (1 a 4)
# ==============================================================================
n_muestras = 1200
turnos = np.repeat(["Matutino", "Vespertino", "Nocturno"], n_muestras // 3)
lineas = np.tile(["Linea_1", "Linea_2", "Linea_3", "Linea_4"], n_muestras // 4)

# Temperatura ambiente (°C): caída térmica en turno nocturno provoca sesgo ambiental
temp_ambiente = np.where(
    turnos == "Nocturno",
    np.random.normal(14.0, 2.5, n_muestras),
    np.random.normal(24.5, 2.0, n_muestras)
)

# Variables físicas de prensa hidráulica
gradiente_termico = np.random.normal(48.0, 6.5, n_muestras) - (24.5 - temp_ambiente) * 0.45
presion_hidraulica = np.random.normal(210.0, 15.0, n_muestras)
vibracion_rms = np.random.normal(3.2, 0.8, n_muestras)

# Etiqueta real de falla estructural (considerando interacción térmica real)
indice_fatiga = (gradiente_termico + (24.5 - temp_ambiente) * 0.40) * 0.65 + (presion_hidraulica * 0.12) + (vibracion_rms * 2.5)
y = (indice_fatiga > np.percentile(indice_fatiga, 70)).astype(int)

df = pd.DataFrame({
    "turno": turnos,
    "linea": lineas,
    "temp_ambiente": temp_ambiente,
    "gradiente_termico": gradiente_termico,
    "presion_hidraulica": presion_hidraulica,
    "vibracion_rms": vibracion_rms,
    "falla": y
})

# ==============================================================================
# 3. AUDITORÍA DE FAIRNESS & SESGOS (ANTES VS. DESPUÉS DE MITIGACIÓN)
# ==============================================================================
def evaluar_fairness(df_eval, y_true, y_pred, etapa):
    resultados = []
    recalls_turnos = {}

    for turno in ["Matutino", "Vespertino", "Nocturno"]:
        mask = (df_eval["turno"] == turno)
        yt = y_true[mask]
        yp = y_pred[mask]
        rec = recall_score(yt, yp)
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        fpr = fp / (fp + tn)
        recalls_turnos[turno] = rec
        resultados.append({
            "Etapa": etapa,
            "Segmento": f"Turno {turno}",
            "n": int(mask.sum()),
            "Recall [MEDIDO]": round(rec, 4),
            "FPR [MEDIDO]": round(fpr * 100, 2)
        })

    ref_vespertino = recalls_turnos["Vespertino"]
    for row in resultados:
        brecha = round(row["Recall [MEDIDO]"] - ref_vespertino, 4)
        row["Brecha_Delta_Recall"] = brecha
        row["Estado_Gobernanza"] = "APROBADO (<=0.02)" if abs(brecha) <= 0.02 else "ALERTA SESGO (>0.02)"

    return pd.DataFrame(resultados), max(abs(r - ref_vespertino) for r in recalls_turnos.values())

# --- Modelo Base v2.1 (Sin considerar temp_ambiente -> Sesgo Nocturno) ---
features_base = ["gradiente_termico", "presion_hidraulica", "vibracion_rms"]
X_train_b, X_test_b, y_train, y_test, idx_train, idx_test = train_test_split(
    df[features_base], df["falla"], df.index, test_size=0.35, random_state=42, stratify=df["turno"]
)

scaler_b = StandardScaler()
modelo_v21 = XGBClassifier(n_estimators=80, max_depth=3, random_state=42, eval_metric="logloss")
modelo_v21.fit(scaler_b.fit_transform(X_train_b), y_train)
probs_v21 = modelo_v21.predict_proba(scaler_b.transform(X_test_b))[:, 1]
preds_v21 = (probs_v21 >= 0.38).astype(int)

df_fairness_antes, max_brecha_antes = evaluar_fairness(df.loc[idx_test], y_test.values, preds_v21, "v2.1 (Sin Mitigar)")

# --- Modelo Mitigado v3.0 (Incorpora temp_ambiente + calibración de umbral tau=0.32) ---
features_mitigadas = ["gradiente_termico", "presion_hidraulica", "vibracion_rms", "temp_ambiente"]
X_train_m = df.loc[idx_train, features_mitigadas]
X_test_m = df.loc[idx_test, features_mitigadas]

scaler_m = StandardScaler()
modelo_v30 = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
modelo_v30.fit(scaler_m.fit_transform(X_train_m), y_train)
probs_v30 = modelo_v30.predict_proba(scaler_m.transform(X_test_m))[:, 1]
preds_v30 = (probs_v30 >= 0.32).astype(int)

df_fairness_despues, max_brecha_despues = evaluar_fairness(df.loc[idx_test], y_test.values, preds_v30, "v3.0 (Mitigado)")

# ==============================================================================
# 4. BENCHMARK DE INFERENCIA (MEDIDO EN CPU) & PROYECCIÓN FINOPS (SIMULADA)
# ==============================================================================
# Medición real de latencia pura de inferencia XGBoost sobre lote de prueba
latencias_ms = []
X_test_scaled = scaler_m.transform(X_test_m)
for _ in range(100):
    t0 = time.perf_counter()
    _ = modelo_v30.predict_proba(X_test_scaled[:10])
    latencias_ms.append((time.perf_counter() - t0) * 1000)

latencia_modelo_p95_medida = round(float(np.percentile(latencias_ms, 95)), 2)

# Tabla comparativa explícita (separando métricas medidas vs. proyectadas/simuladas)
resumen_arquitectura = pd.DataFrame([
    {
        "Arquitectura": "Actual v2.1 (Monolítica 1x c5.2xlarge)",
        "Latencia_p95_Nominal_ms": "26.1 [MEDIDO Act.8]",
        "Latencia_p95_Pico_RAG_ms": "148.5 [SIMULADO Act.8]",
        "Throughput_req_s": "45 [PROYECTADO]",
        "Costo_Mensual_USD": 318.00,
        "Costo_10k_Pred_USD": 0.068
    },
    {
        "Arquitectura": "Propuesta v3.0 (Elástica ALB + Cache + 2-4x c6i.large)",
        "Latencia_p95_Nominal_ms": f"14.2 [PROYECTADO | Core XGB: {latencia_modelo_p95_medida}ms MEDIDO]",
        "Latencia_p95_Pico_RAG_ms": "18.4 [PROYECTADO]",
        "Throughput_req_s": "220 [PROYECTADO]",
        "Costo_Mensual_USD": 189.50,
        "Costo_10k_Pred_USD": 0.010
    }
])

# Registro en MLflow
with mlflow.start_run(run_name="Auditoria_Fairness_v2.1_PreMitigacion"):
    mlflow.log_param("version_modelo", "v2.1")
    mlflow.log_param("tipo_dato", "MEDIDO_SOBRE_TEST_SET")
    mlflow.log_metric("max_brecha_delta_recall", round(max_brecha_antes, 4))
    mlflow.log_metric("costo_mensual_usd", 318.00)

with mlflow.start_run(run_name="Auditoria_Fairness_FinOps_v3.0_PostMitigacion"):
    mlflow.log_param("version_modelo", "v3.0")
    mlflow.log_param("tipo_dato", "MEDIDO_FAIRNESS_Y_PROYECTADO_FINOPS")
    mlflow.log_metric("max_brecha_delta_recall", round(max_brecha_despues, 4))
    mlflow.log_metric("latencia_core_xgb_p95_ms_medida", latencia_modelo_p95_medida)
    mlflow.log_metric("costo_mensual_usd", 189.50)
    mlflow.log_metric("ahorro_finops_pct", 40.4)

# ==============================================================================
# 5. IMPRESIÓN DE RESULTADOS EN CONSOLA PARA EVIDENCIA
# ==============================================================================
print("=" * 85)
print("REPORTE DE AUDITORÍA FAIRNESS POR TURNO (MÉTRICAS CALCULADAS EN TEST SET)")
print("=" * 85)
print(pd.concat([df_fairness_antes, df_fairness_despues], ignore_index=True).to_string(index=False))

print("\n" + "=" * 85)
print("COMPARATIVA FINOPS Y DESEMPEÑO (MEDIDO VS. PROYECTADO)")
print("=" * 85)
print(resumen_arquitectura.to_string(index=False))

# ==============================================================================
# 6. GENERACIÓN DE DASHBOARD VISUAL (3 PANELES PARA REPORTE Y GITHUB)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("OaksGen-Engineering v3.0 — Evidencia de Escalabilidad, FinOps y Fairness", fontsize=14, fontweight="bold")

# Panel 1: Comparativa de Costos FinOps
arq_labels = ["Actual v2.1\n(Monolítica 24/7)", "Propuesta v3.0\n(Elástica + Spot)"]
costos = [318.00, 189.50]
bars1 = axes[0].bar(arq_labels, costos, color=["#d9534f", "#2ca02c"], width=0.5)
axes[0].set_title("1. Costo Mensual de Nube (USD) [-40.4%]")
axes[0].set_ylabel("USD / Mes")
axes[0].set_ylim(0, 380)
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8, f"${bar.get_height():.2f}", ha="center", fontweight="bold")

# Panel 2: Latencia p95 en Pico de Carga vs SLA
lat_labels = ["Actual v2.1\n(Pico RAG)", "Propuesta v3.0\n(Caché + ALB)"]
latencias = [148.5, 18.4]
bars2 = axes[1].bar(lat_labels, latencias, color=["#d9534f", "#1f77b4"], width=0.5)
axes[1].axhline(50.0, color="red", linestyle="--", linewidth=1.5, label="SLA Máximo (50 ms)")
axes[1].set_title("2. Latencia p95 en Pico vs. SLA (ms)")
axes[1].set_ylabel("Milisegundos (ms)")
axes[1].set_ylim(0, 175)
axes[1].legend()
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4, f"{bar.get_height():.1f} ms", ha="center", fontweight="bold")

# Panel 3: Auditoría de Fairness (Brecha Delta Recall antes vs después)
turnos_lbl = ["Matutino", "Vespertino", "Nocturno"]
x = np.arange(len(turnos_lbl))
w = 0.35
axes[2].bar(x - w/2, df_fairness_antes["Recall [MEDIDO]"], width=w, label="v2.1 (Sin Mitigar)", color="#ff7f0e")
axes[2].bar(x + w/2, df_fairness_despues["Recall [MEDIDO]"], width=w, label="v3.0 (Mitigado)", color="#2ca02c")
axes[2].axhline(0.95, color="red", linestyle="--", linewidth=1.2, label="SLA Recall (0.95)")
axes[2].set_title("3. Auditoría Fairness: Recall por Turno")
axes[2].set_xticks(x)
axes[2].set_xticklabels(turnos_lbl)
axes[2].set_ylim(0.85, 1.02)
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.savefig("dashboard_escalabilidad_fairness_oaksgen.png", dpi=300)
plt.show()
print("\n✅ Gráfica guardada como: dashboard_escalabilidad_fairness_oaksgen.png")
