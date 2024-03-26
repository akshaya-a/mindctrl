# https://github.com/Blueshoe/pytest-kubernetes/blob/main/pytest_kubernetes/providers/k3d.py#L5

import os
import subprocess
from typing import Dict, List, Optional
from pytest_kubernetes.providers.k3d import K3dManager
from pytest_kubernetes.options import ClusterOptions


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
