import json
import os
import subprocess
import sys

def test_ejecucion_pipeline_y_quality_gates():
    """Ejecuta el pipeline completo y verifica los Quality Gates MLOps."""
    resultado = subprocess.run([sys.executable, "pipeline_fase3_oaksgen.py"], capture_output=True, text=True)
    assert resultado.returncode == 0, f"Fallo en pipeline: {resultado.stderr}"
    assert os.path.exists("metricas_oficiales_fase3.json"), "No se generó el sello de datos JSON"

def test_coherencia_sello_datos_y_slas():
    """Audita que las métricas oficiales cumplan SLA de Recall, Fairness y Latencia."""
    with open("metricas_oficiales_fase3.json", "r") as f:
        datos = json.load(f)

    assert datos["metricas_globales_test"]["recall"] >= 0.95
    assert datos["benchmark_api_fastapi"]["latencia_p95_ms"] <= 50.0
    assert datos["benchmark_api_fastapi"]["cumple_sla_latencia"] is True

    for turno, metricas in datos["auditoria_fairness_test"].items():
        assert metricas["cumple_fairness"] is True, f"Sesgo detectado en turno {turno}"
        assert abs(metricas["delta_recall"]) <= 0.02
