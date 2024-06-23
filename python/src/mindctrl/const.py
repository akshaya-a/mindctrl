from pathlib import Path

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

SCENARIO_NAME_HEADER = "x-mctrl-scenario-name"
SCENARIO_NAME_PARAM = "scenario_name"

REPLAY_SERVER_INPUT_FILE_SUFFIX = "input.json"
REPLAY_SERVER_OUTPUT_FILE_SUFFIX = "output.json"

ROUTE_PREFIX = "/mindctrl/v1"

## Computed
BASE_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = BASE_DIR / "templates"


## Events
STOP_DEPLOYED_MODEL = "stop_deployed_model"

## Config
CONFIGURATION_STORE = "configstore"
SECRET_STORE = "secretstore"
CONFIGURATION_KEY = "mindctrl.appsettings"
CONFIGURATION_TABLE = "mindctrlconfig"
