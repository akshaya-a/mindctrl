# https://github.com/Blueshoe/pytest-kubernetes/blob/main/pytest_kubernetes/providers/k3d.py#L5

import logging
import os
import subprocess
from typing import Dict, List, Optional
from pytest_kubernetes.providers.k3d import K3dManager
from pytest_kubernetes.options import ClusterOptions


_logger = logging.getLogger(__name__)


class LocalRegistryK3dManager(K3dManager):
    def __init__(self, registry_name: str, registry_port: int, *args, **kwargs) -> None:
        self.registry_name = f"pytest-{registry_name}"
        self.registry_port = registry_port
        self.k3d_registry_url = f"k3d-{self.registry_name}:{self.registry_port}"
        self.executions = []
        super().__init__(*args, **kwargs)

    def _exec(
        self,
        arguments: List[str],
        additional_env: Dict[str, str] = {},
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        _logger.info(f"Executing: {self.get_binary_name()} {' '.join(arguments)}")
        proc = super()._exec(arguments, additional_env, timeout)
        self.executions.append(proc)
        return proc

    def _on_create(self, cluster_options: ClusterOptions, **kwargs) -> None:
        opts = kwargs.pop("options", [])
        registry_opts = kwargs.pop("registry_options", [])
        self._exec(
            [
                "registry",
                "create",
                self.registry_name,
                "--port",
                str(self.registry_port),
            ]
            + registry_opts
        )
        new_opts = [
            "--registry-use",
            self.k3d_registry_url,
        ] + opts
        super()._on_create(cluster_options, options=new_opts, **kwargs)

    def _on_delete(self) -> None:
        super()._on_delete()
        self._exec(["registry", "delete", self.registry_name])

    def create_secret(self, name: str, from_env_var: str) -> None:
        self.kubectl(
            [
                "create",
                "secret",
                "generic",
                name,
                f"--from-literal={name}={os.environ[from_env_var]}",
            ],
            as_dict=True,
        )

    def set_kubectl_default(self):
        self._exec(
            [
                "kubeconfig",
                "merge",
                self.cluster_name,
                "--kubeconfig-merge-default",
                "--kubeconfig-switch-context",
            ]
        )

    def wait_and_get_logs(self, app: str, timeout: int = 90):
        _logger.info(f"Waiting {timeout}s for {app} to be ready")
        try:
            self.wait(
                name=f"deployments/{app}",
                waitfor="condition=Available=True",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            print(e)
            print(self.kubectl(["describe", "pod", "-l", f"app={app}"], as_dict=False))
            print(
                self.kubectl(["logs", "-l", f"app={app}", "--tail=100"], as_dict=False)
            )
            raise

    def install_dapr(self):
        _logger.info("Installing Dapr")
        # lol don't hate me
        self._exec(
            [
                "version",
                ">",
                "/dev/null",
                "&&",
                "dapr",
                "init",
                "--kubernetes",
                "--wait",
            ]
        )
        self._exec(
            ["version", ">", "/dev/null", "&&", "dapr", "status", "--kubernetes"]
        )
