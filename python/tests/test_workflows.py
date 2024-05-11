import atexit
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional
import httpx
import pytest
import subprocess
import time

from durabletask.client import OrchestrationState
from dapr.conf import settings
from dapr.clients import DaprClient
from dapr.ext.workflow.workflow_state import WorkflowStatus
from dapr.ext.workflow.dapr_workflow_client import DaprWorkflowClient

from mindctrl.mlmodels import ModelInvocation
from mindctrl.workflows import WorkflowContext
from mindctrl.workflows.agent import conversation_turn_workflow
from mindctrl.workflows.deployer import ModelServeCommand, deploy_model_workflow


_logger = logging.getLogger(__name__)


def stop_dapr_app(app_id: str):
    try:
        subprocess.run(["dapr", "stop", "-a", app_id], check=True)
    except subprocess.CalledProcessError as e:
        _logger.error(f"Error stopping Dapr app {app_id}: {e}")


def wait_for_input_output(
    wf_client: DaprWorkflowClient,
    instance_id: str,
    target_input: Optional[str] = None,
    target_output: Optional[str] = None,
    target_input_val: Optional[Any] = None,
    target_output_val: Optional[Any] = None,
    timeout=120,
):
    target_match = False
    start_time = time.time()
    state = None

    if not target_input and not target_output:
        raise ValueError("Either target_input or target_output must be provided")

    if target_input and not target_input_val:
        raise ValueError(
            "target_input_val must be provided if target_input is provided"
        )
    if target_output and not target_output_val:
        raise ValueError(
            "target_output_val must be provided if target_output is provided"
        )

    while not target_match:
        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Timed out waiting for {instance_id} to reach target. State:\n"
                f"{state._WorkflowState__obj if state else None}"
            )
        state = wf_client.get_workflow_state(instance_id, fetch_payloads=True)
        assert state is not None
        orch_state: OrchestrationState = state._WorkflowState__obj
        if target_input and orch_state.serialized_input:
            target_match = (
                json.loads(orch_state.serialized_input).get(target_input)
                == target_input_val
            )
        if target_output and orch_state.serialized_output:
            target_match = target_match and (
                json.loads(orch_state.serialized_output).get(target_output)
                == target_output_val
            )
        status = state.runtime_status
        _logger.info(
            f"Workflow status: {status}, waiting...\n{state._WorkflowState__obj if state else None}"
        )
        time.sleep(5)

    state = wf_client.get_workflow_state(instance_id, fetch_payloads=True)
    assert state is not None
    return state


@pytest.fixture(scope="session")
def placement_server(deploy_mode):
    if deploy_mode.value != "local":
        # This only makes sense for local testing - dapr is initialized
        # in the container/cluster for addon/k8s
        _logger.warning(f"Unsupported deploy mode: {deploy_mode}")
        pytest.skip(f"Unsupported deploy mode: {deploy_mode}")

    placement_bin = (Path.home() / ".dapr" / "bin" / "placement").resolve()
    assert (
        placement_bin.exists()
    ), f"placement binary not found at {placement_bin}. Is Dapr installed?"
    placement_process = subprocess.Popen([str(placement_bin)])

    yield placement_process

    placement_process.terminate()


@pytest.fixture(scope="session")
def dapr_sidecar(
    tmp_path_factory: pytest.TempPathFactory,
    repo_root_dir: Path,
    request: pytest.FixtureRequest,
    deploy_mode,
    monkeypatch_session,
    placement_server,
):
    if deploy_mode.value != "local":
        # This only makes sense for local testing - dapr is initialized
        # in the container/cluster for addon/k8s
        _logger.warning(f"Unsupported deploy mode: {deploy_mode}")
        pytest.skip(f"Unsupported deploy mode: {deploy_mode}")

    state_spec = repo_root_dir / "services" / "components" / "sqlite.yaml"
    assert state_spec.exists(), f"state store spec not found at {state_spec}"
    state_store_path = tmp_path_factory.mktemp("statestore")
    target_spec = state_store_path / "sqlite.yaml"

    with monkeypatch_session.context() as m:
        m.setenv("ACTOR_STORE_CONNECTION_STRING", f"{state_store_path}/actors.db")
        with open(state_spec, "r") as f:
            content = f.read()
            content = os.path.expandvars(content)
        with open(target_spec, "w") as f:
            f.write(content)
    _logger.info(f"Generated state store spec at {target_spec}")

    dapr_process = subprocess.Popen(
        [
            "dapr",
            "run",
            "--app-id",
            request.node.name,
            "--dapr-grpc-port",
            str(settings.DAPR_GRPC_PORT),
            "--dapr-http-port",
            str(settings.DAPR_HTTP_PORT),
            # "--log-level",
            # "debug",
            "--resources-path",
            f"{state_store_path}",
        ]
    )
    yield dapr_process
    dapr_process.terminate()


@pytest.fixture(scope="session")
def workflow_client():
    with WorkflowContext():
        yield DaprWorkflowClient()


def test_smoke_workflow(dapr_sidecar, mlflow_fluent_session, workflow_client, request):
    import pandas as pd
    from mindctrl.openai_deployment import log_model
    import openai

    df = pd.DataFrame({"query": ["What's up doc?"]})
    payload = {"dataframe_split": df.to_dict(orient="split")}

    log_model(
        model="gpt-4-turbo-preview",
        task=openai.chat.completions,
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant. Answer the user's questions, even if they're incomplete.",
            },
            {
                "role": "user",
                "content": "{query}",
            },
        ],
        artifact_path="oai-chatty-cathy",
        registered_model_name="chatty_cathy",
    )
    input_invocation = ModelInvocation(
        model_uri="models:/chatty_cathy/latest",
        payload=payload,
        scenario_name=request.node.name,
        input_variables={},
    )

    # TODO: Without this, there isn't a proper actor lookup..
    time.sleep(5)
    instance_id = workflow_client.schedule_new_workflow(
        conversation_turn_workflow,
        input=input_invocation,
        instance_id=request.node.name,
    )

    state = workflow_client.wait_for_workflow_completion(instance_id)
    assert state.runtime_status == WorkflowStatus.COMPLETED


def test_deploy_workflow(dapr_sidecar, mlflow_fluent_session, workflow_client, request):
    import pandas as pd
    from mindctrl.openai_deployment import log_model
    import openai

    df = pd.DataFrame({"query": ["What's up doc?"]})
    payload = {"dataframe_split": df.to_dict(orient="split")}

    ## Add scenario name
    payload["params"] = {"scenario_name": request.node.name}

    log_model(
        model="gpt-4-turbo-preview",
        task=openai.chat.completions,
        messages=[
            {
                "role": "system",
                "content": "You're a helpful assistant. Answer the user's questions, even if they're incomplete.",
            },
            {
                "role": "user",
                "content": "{query}",
            },
        ],
        artifact_path="oai-chatty-cathy",
        registered_model_name="chatty_cathy",
    )

    model_serve_command = ModelServeCommand(
        model_uri="models:/chatty_cathy/latest",
        port=45922,
        pid=-1,
        is_healthy=False,
        app_id="",
    )
    app_id = "models_chatty_cathy_latest"
    atexit.register(lambda: stop_dapr_app(app_id))

    instance_id = workflow_client.schedule_new_workflow(
        deploy_model_workflow,
        input=model_serve_command,
        instance_id=request.node.name,
    )

    state = workflow_client.wait_for_workflow_start(instance_id)
    assert state is not None
    _logger.info(f"Model deployment running: {state._WorkflowState__obj}")

    model_monitor_instance_id = f"{instance_id}-monitor"
    state = workflow_client.wait_for_workflow_start(model_monitor_instance_id)
    assert state is not None
    _logger.info(f"Model monitor running: {state._WorkflowState__obj}")

    wait_for_input_output(
        workflow_client,
        model_monitor_instance_id,
        target_input="is_healthy",
        target_input_val=True,
    )
    resp = httpx.get(f"http://localhost:{model_serve_command.port}/health")
    assert resp.status_code == 200

    resp = httpx.post(
        f"http://localhost:{model_serve_command.port}/invocations",
        json=payload,
    )
    if resp.status_code != 200:
        print(resp.content)
    assert resp.status_code == 200
    assert "predictions" in str(resp.json())

    with DaprClient() as d:
        dapr_resp = d.invoke_method(
            app_id,
            method_name="invocations",
            data=json.dumps(payload),
            content_type="application/json",
            http_verb="POST",
        )
        if dapr_resp.status_code != 200:
            print(dapr_resp.text())
        assert dapr_resp.status_code == 200
        assert "predictions" in str(dapr_resp.json())

    # Stop the model server
    # Yes there's a const, I like to test const breakage with dupes in test
    workflow_client.raise_workflow_event(
        instance_id, "stop_deployed_model", data=f"cancelled-by-{request.node.name}"
    )

    state = workflow_client.wait_for_workflow_completion(
        instance_id, fetch_payloads=True, timeout_in_seconds=240
    )
    assert state.runtime_status == WorkflowStatus.COMPLETED
    assert (
        json.loads(state.serialized_output).get("cancellation_reason")
        == f"cancelled-by-{request.node.name}"
    )


# Some dapr stuff is easier to debug on cli
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    @dataclass
    class MockNode:
        name: str

    @dataclass
    class MockRequest:
        node: MockNode

    with WorkflowContext():
        test_smoke_workflow(
            None, None, None, request=MockRequest(MockNode("test_smoke_workflow"))
        )
