# Postgres
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_USER = "test-psql-user"
POSTGRES_PASSWORD = "test-psql-password"
POSTGRES_DB = "test-mindctrl"

# MQTT
MQTT_HOST = "localhost"
PROXY_MQTT_HOST = "localhost"
MQTT_PORT = 1883
MQTT_USER = "test-mqtt-user"
MQTT_PASSWORD = "test-mqtt-password"

# Multiserver
LOCAL_MULTISERVER_HOST = "127.0.0.1"
LOCAL_MULTISERVER_PORT = 5002

# Replay server
LOCAL_REPLAY_SERVER_HOST = "127.0.0.1"
LOCAL_REPLAY_SERVER_PORT = 5001

# Cluster
CLUSTER_NAME = "mindctrl"
REGISTRY_NAME = "ptmctrlreg.localhost"
REGISTRY_PORT = 12345
K8S_INGRESS_HOST = "127.0.0.1"
K8S_INGRESS_PORT = 8081
K3D_REPLAY_PV_LOCATION = "/tmp/replays"
K3D_RECORDING_PV_LOCATION = "/tmp/recordings"

# Retries, Timeouts
MAX_ATTEMPTS = 20
