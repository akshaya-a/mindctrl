import logging
import subprocess
from dataclasses import dataclass
from datetime import timedelta

from dapr.clients import DaprClient
from dapr.ext.workflow import DaprWorkflowContext, RetryPolicy, WorkflowActivityContext
from dapr.ext.workflow.dapr_workflow_client import DaprWorkflowClient

from mindctrl.const import STOP_DEPLOYED_MODEL

_logger = logging.getLogger(__name__)


def model_uri_to_app_id(model_uri: str) -> str:
    return model_uri.replace("/", "_").replace(":", "")


@dataclass
class ModelServeCommand:
    model_uri: str
    port: int
    pid: int
    is_healthy: bool
    app_id: str


# TODO: Kubernetes workflow needs kaniko version of this:
# https://github.com/mlflow/mlflow/blob/master/mlflow/models/python_api.py#L79
def serve_model(ctx: WorkflowActivityContext, model_serve_command: ModelServeCommand):
    # This activity serves a model from a local path
    if model_serve_command.pid >= 0:
        raise ValueError(
            f"Model {model_serve_command.model_uri} is already being served by {model_serve_command.pid}"
        )
    app_id = model_uri_to_app_id(model_serve_command.model_uri)
    _logger.info(f"Starting serving model as dapr app {app_id}")
    proc = subprocess.Popen(
        [
            "dapr",
            "run",
            "--app-id",
            app_id,
            "--app-port",
            str(model_serve_command.port),
            "--",
            "mlflow",
            "models",
            "serve",
            "-m",
            model_serve_command.model_uri,
            "--port",
            str(model_serve_command.port),
            "--no-conda",  # TODO: Add uv env build
        ]
    )
    return ModelServeCommand(
        model_uri=model_serve_command.model_uri,
        pid=proc.pid,
        port=model_serve_command.port,
        is_healthy=False,
        app_id=app_id,
    )


def stop_model(ctx: WorkflowActivityContext, model_serve_command: ModelServeCommand):
    _logger.info(f"Stopping Model serve {model_serve_command.app_id}")
    subprocess.run(["dapr", "stop", "--app-id", model_serve_command.app_id], check=True)


def stop_model_monitor(ctx: WorkflowActivityContext, child_workflow_id: str):
    _logger.info(f"Stopping monitor: {child_workflow_id}")
    wf_client = DaprWorkflowClient()
    wf_client.terminate_workflow(child_workflow_id)


def wait_for_model_serve(
    ctx: WorkflowActivityContext, model_serve_command: ModelServeCommand
) -> ModelServeCommand:
    # TODO: Check if the process is still running via Dapr API
    # TODO: Store the app id in the model serve command
    is_healthy = False
    try:
        with DaprClient() as d:
            resp = d.invoke_method(model_serve_command.app_id, "health")
            if resp.status_code != 200:
                raise Exception(f"Model serve failed to start: {resp.text}")
        is_healthy = True
    except Exception as e:
        _logger.warning(f"Error checking health: {e}")

    return ModelServeCommand(
        model_serve_command.model_uri,
        model_serve_command.port,
        model_serve_command.pid,
        is_healthy=is_healthy,
        app_id=model_serve_command.app_id,
    )


deployment_retry_policy = RetryPolicy(
    first_retry_interval=timedelta(seconds=30),
    max_number_of_attempts=3,
    backoff_coefficient=2,
    max_retry_interval=timedelta(seconds=60),
    retry_timeout=timedelta(seconds=180),
)


def check_deployment_status(
    ctx: DaprWorkflowContext, model_serve_command: ModelServeCommand
):
    model_serve_command = yield ctx.call_activity(
        wait_for_model_serve,
        input=model_serve_command,
        retry_policy=deployment_retry_policy,
    )

    check_interval = 60 if model_serve_command.is_healthy else 5
    yield ctx.create_timer(fire_at=timedelta(seconds=check_interval))

    ctx.continue_as_new(model_serve_command)


def deploy_model_workflow(
    ctx: DaprWorkflowContext, model_serve_command: ModelServeCommand
):
    if not ctx.is_replaying:
        _logger.info(
            f"Starting model deployment workflow for {model_serve_command.model_uri}"
        )
    model_serve_command = yield ctx.call_activity(
        serve_model, input=model_serve_command
    )
    monitor_id = f"{ctx.instance_id}-monitor"
    ctx.call_child_workflow(
        check_deployment_status,
        input=model_serve_command,
        instance_id=monitor_id,
    )
    # We want to perform custom termination actions, so don't rely on dapr workflow termination
    cancellation_event = yield ctx.wait_for_external_event(STOP_DEPLOYED_MODEL)
    _logger.info(f"Received stop event {cancellation_event}")
    yield ctx.call_activity(stop_model_monitor, input=monitor_id)
    _logger.info("Stopped monitor")
    yield ctx.call_activity(stop_model, input=model_serve_command)

    return {"cancellation_reason": cancellation_event}
