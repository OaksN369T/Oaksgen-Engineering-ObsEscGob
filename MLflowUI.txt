https://strategies-harris-brake-alphabetical.trycloudflare.com/

Abriendo la Interfaz Web de MLflow para tomar nuestras capturas Una vez que corra el script de arriba y nos imprima los logs de ambos incidentes y sus runbooks .

import subprocess, time, re, urllib.request

# 1. Limpiar procesos previos
!pkill -f mlflow || true
!pkill -f cloudflared || true

# 2. Arrancar MLflow UI en segundo plano guardando su log para ver cualquier detalle
log_file = open("mlflow_server.log", "w")
subprocess.Popen([
    "mlflow", "ui",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--host", "127.0.0.1",
    "--port", "5000",
    "--allowed-hosts", "*",
    "--cors-allowed-origins", "*"
], stdout=log_file, stderr=log_file)

# 3. Esperar activamente hasta que MLflow responda en localhost:5000
print("Iniciando servidor MLflow...", end="")
for _ in range(15):
    try:
        urllib.request.urlopen("http://127.0.0.1:5000", timeout=1)
        print(" ¡LISTO!")
        break
    except Exception:
        print(".", end="")
        time.sleep(1)
else:
    print("\nError al iniciar MLflow. Revisando log:")
    with open("mlflow_server.log") as f:
        print(f.read())

# 4. Abrir túnel apuntando directo a 127.0.0.1:5000
tunnel = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:5000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
)
for line in tunnel.stdout:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        print(f"\n>>> ¡AHORA SÍ! Haz clic en este enlace para abrir MLflow UI:\n👉 {match.group(0)}\n")
        break
