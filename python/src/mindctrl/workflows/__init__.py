# https://github.com/dapr/python-sdk/blob/main/examples/demo_workflow/app.py#LL40C1-L43C59

import json
import logging
import uuid
from typing import Optional

import mlflow.pyfunc
from dapr.ext.workflow import WorkflowRuntime
from dapr.ext.workflow.dapr_workflow_client import DaprWorkflowClient
from dapr.ext.workflow.workflow_state import WorkflowStatus

from .agent import (
    Message,
    ModelInvocation,
    append_message,
    conversation_turn_workflow,
    invoke_model,
)
from .deployer import (
    check_deployment_status,
    deploy_model_workflow,
    serve_model,
    stop_model,
    stop_model_monitor,
    wait_for_model_serve,
)

_logger = logging.getLogger(__name__)


class WorkflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.workflows = []
        self.activities = []

    def add_workflow(self, workflow):
        if len(self.workflows) > 0:
            raise ValueError("Only one workflow is supported at this time")
        self.workflows.append(workflow)

    def add_activity(self, activity):
        self.activities.append(activity)

    def predict(self, context, model_input, params=None):
        wfr = WorkflowRuntime()
        for workflow in self.workflows:
            wfr.register_workflow(workflow)
        for activity in self.activities:
            wfr.register_activity(activity)
        instance_id = uuid.uuid4().hex
        client = DaprWorkflowClient()
        instance_id = client.schedule_new_workflow(
            self.workflows[0], input=model_input, instance_id=instance_id
        )
        result = client.wait_for_workflow_completion(instance_id, fetch_payloads=True)
        assert result is not None
        return result.serialized_output


class WorkflowContext:
    def __init__(self, host: Optional[str] = None, port: Optional[str] = None):
        _logger.info(f"Initializing WorkflowContext with {host}:{port}")
        self.host = host
        self.port = port
        self._workflow_runtime = WorkflowRuntime(host=host, port=port)
        try:
            self._register_turn_workflow()
            self._register_deployer_workflow()
        except ValueError as e:
            if "already registered" not in str(e):
                raise e
            _logger.info(f"Already registered turn workflow: {e}")
        self.initialized = False

    def __enter__(self):
        self._workflow_runtime.start()
        self.initialized = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._workflow_runtime.shutdown()

    @property
    def runtime(self):
        return self._workflow_runtime

    def get_workflow_client(self):
        return DaprWorkflowClient(self.host, self.port)

    def _register_turn_workflow(self):
        _logger.info("Registering turn workflow")
        self._workflow_runtime.register_workflow(conversation_turn_workflow)
        _logger.info("Registering activities")
        self._workflow_runtime.register_activity(append_message)
        self._workflow_runtime.register_activity(invoke_model)

    def _register_deployer_workflow(self):
        _logger.info("Registering deployer workflows")
        self._workflow_runtime.register_workflow(deploy_model_workflow)
        self._workflow_runtime.register_workflow(check_deployment_status)
        _logger.info("Registering activities")
        self._workflow_runtime.register_activity(wait_for_model_serve)
        self._workflow_runtime.register_activity(serve_model)
        self._workflow_runtime.register_activity(stop_model)
        self._workflow_runtime.register_activity(stop_model_monitor)


class Conversation:
    def __init__(
        self,
        client: DaprWorkflowClient,
        model_uri: str,
        conversation_id: Optional[str] = None,
    ):
        self.model_uri = model_uri
        self.conversation_id: str = conversation_id or uuid.uuid4().hex
        self.client = client
        self.turn_ids: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def send_message(self, message: str) -> Message:
        turn_id = f"{self.conversation_id}-{len(self.turn_ids)}"
        self.turn_ids.append(turn_id)
        instance_id = self.client.schedule_new_workflow(
            conversation_turn_workflow,
            input=ModelInvocation(
                model_uri=self.model_uri,
                conversation_id=self.conversation_id,
                scenario_name=self.conversation_id,
                history=None,
                input_variables={},
            ),
            instance_id=turn_id,
        )
        self.client.raise_workflow_event(instance_id, "user_message", data=message)

        state = self.client.wait_for_workflow_completion(instance_id)
        if state is None:
            raise RuntimeError(f"Failed to get state for {turn_id}")
        if state.runtime_status != WorkflowStatus.COMPLETED:
            # print(state._WorkflowState__obj)
            # print(state)
            raise RuntimeError(f"Failed to complete {turn_id}: {state}")
        # print(state.serialized_output)
        out = json.loads(state.serialized_output)
        return Message(content=out["content"], role=out["role"])
