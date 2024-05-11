# https://github.com/dapr/python-sdk/blob/main/examples/demo_workflow/app.py#LL40C1-L43C59

import logging
from typing import Optional

from dapr.ext.workflow import (
    WorkflowRuntime,
)

from .agent import append_message, invoke_model, conversation_turn_workflow
from .deployer import (
    stop_model_monitor,
    wait_for_model_serve,
    check_deployment_status,
    deploy_model_workflow,
    stop_model,
    serve_model,
)

_logger = logging.getLogger(__name__)


class WorkflowContext:
    def __init__(self, port: Optional[str] = None):
        _logger.info("Initializing WorkflowContext")
        self.workflow_runtime = WorkflowRuntime(port=port)
        self._register_turn_workflow()
        self._register_deployer_workflow()

    def __enter__(self):
        self.workflow_runtime.start()
        return self.workflow_runtime

    def __exit__(self, exc_type, exc_value, traceback):
        self.workflow_runtime.shutdown()

    def _register_turn_workflow(self):
        _logger.info("Registering turn workflow")
        self.workflow_runtime.register_workflow(conversation_turn_workflow)
        _logger.info("Registering activities")
        self.workflow_runtime.register_activity(append_message)
        self.workflow_runtime.register_activity(invoke_model)

    def _register_deployer_workflow(self):
        _logger.info("Registering deployer workflows")
        self.workflow_runtime.register_workflow(deploy_model_workflow)
        self.workflow_runtime.register_workflow(check_deployment_status)
        _logger.info("Registering activities")
        self.workflow_runtime.register_activity(wait_for_model_serve)
        self.workflow_runtime.register_activity(serve_model)
        self.workflow_runtime.register_activity(stop_model)
        self.workflow_runtime.register_activity(stop_model_monitor)
