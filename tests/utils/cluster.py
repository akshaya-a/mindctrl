# https://github.com/Blueshoe/pytest-kubernetes/blob/main/pytest_kubernetes/providers/k3d.py#L5

import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Optional
from pytest_kubernetes.providers.k3d import K3dManager
from pytest_kubernetes.options import ClusterOptions

from .common import build_app


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

    # create configmap name-of-your-configmap --from-file=your-file.json
    def create_configmap(self, name: str, *args, **kwargs):
        command_args = [
            "create",
            "configmap",
            name,
        ]

        for arg in args:
            if isinstance(arg, Path):
                command_args.append(f"--from-file={arg}")
            else:
                raise ValueError(f"Unknown arg type: {arg}")

        for key, value in kwargs.items():
            command_args.append(f"--from-literal={key}={value}")

        self.kubectl(command_args)

    def show_configmap(self, name: str):
        # kubectl get configmaps special-config -o yaml
        return self.kubectl(["get", "configmaps", name, "-o", "yaml"], as_dict=False)

    def generate_persistent_volume(
        self, name: str, host_path: Path, mode: str = "ReadOnlyMany", size: str = "10Mi"
    ):
        return f"""
apiVersion: v1
kind: PersistentVolume
metadata:
  name: {name}
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: {size}
  accessModes:
    - {mode}
  hostPath:
    path: "{host_path}"
"""


def prepare_apps(
    source_spec_dir: Path,
    target_spec_dir: Path,
    services_dir: Path,
    registry_url: str,
    mindctrl_source: Path,
):
    _logger.info(
        f"Pulling spec templates from {source_spec_dir}, generating in {target_spec_dir}"
    )

    built_tags = []
    for app in source_spec_dir.glob("*.yaml"):
        if "dapr-local" in app.name:
            continue

        target_app = target_spec_dir / app.name

        # Don't push until the registry is created later
        if (
            "postgres" not in app.name
            and "mosquitto" not in app.name
            and "dashboard" not in app.name
        ):
            source_app = services_dir / app.stem
            assert (
                source_app / "Dockerfile"
            ).exists(), f"Missing {source_app / 'Dockerfile'}"
            built_tags.append(build_app(source_app, registry_url, mindctrl_source))

        with open(app, "r") as f:
            content = f.read()
            content = os.path.expandvars(content)
        with open(target_app, "w") as f:
            f.write(content)

    _logger.info(f"Built tags {built_tags}")
    return built_tags
