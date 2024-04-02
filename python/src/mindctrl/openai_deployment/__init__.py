"""
The ``mlflow.openai`` module provides an API for logging and loading OpenAI models, but it directly hits OAI,
defeating the purpose of the mlflow deployment server. This "custom flavor" is an openai flavor clone modified to use the deployment server
To maintain compatibility, it also registers the native openai flavor, but overrides the pyfunc
Ideally this is just a load-time parameter on model load to override the target uri? this flavor should die
"""

import itertools
import logging
import os
import warnings
from string import Formatter
from typing import Any, Dict, NamedTuple, Optional, Set

import yaml
from packaging.version import Version

import mlflow

from mlflow.deployments import get_deployments_target, get_deploy_client
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_OPENAI_SECRET_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, Schema, TensorSpec
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import (
    check_databricks_secret_scope_access,
    is_in_databricks_runtime,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.openai_utils import (
    REQUEST_URL_CHAT,
    REQUEST_URL_COMPLETIONS,
    REQUEST_URL_EMBEDDINGS,
    _OAITokenHolder,
    _OpenAIApiConfig,
    _OpenAIEnvVar,
    _validate_model_params,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "openai_deployment"
MODEL_FILENAME = "model.yaml"
_PYFUNC_SUPPORTED_TASKS = ("chat.completions", "embeddings", "completions")

model_data_artifact_paths = []

_logger = logging.getLogger(__name__)


@experimental
def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return list(map(_get_pinned_requirement, ["openai", "tiktoken", "tenacity"]))


@experimental
def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _get_obj_to_task_mapping():
    import openai

    if Version(_get_openai_package_version()).major < 1:
        from openai import api_resources as ar

        return {
            ar.Audio: ar.Audio.OBJECT_NAME,
            ar.ChatCompletion: ar.ChatCompletion.OBJECT_NAME,
            ar.Completion: ar.Completion.OBJECT_NAME,
            ar.Edit: ar.Edit.OBJECT_NAME,
            ar.Deployment: ar.Deployment.OBJECT_NAME,
            ar.Embedding: ar.Embedding.OBJECT_NAME,
            ar.Engine: ar.Engine.OBJECT_NAME,
            ar.File: ar.File.OBJECT_NAME,
            ar.Image: ar.Image.OBJECT_NAME,
            ar.FineTune: ar.FineTune.OBJECT_NAME,
            ar.Model: ar.Model.OBJECT_NAME,
            ar.Moderation: "moderations",
        }
    else:
        return {
            openai.audio: "audio",
            openai.chat.completions: "chat.completions",
            openai.completions: "completions",
            openai.images.edit: "images.edit",
            openai.embeddings: "embeddings",
            openai.files: "files",
            openai.images: "images",
            openai.fine_tuning: "fine_tuning",
            openai.moderations: "moderations",
            openai.models: "models",
        }


def _get_model_name(model):
    import openai

    if isinstance(model, str):
        return model

    if Version(_get_openai_package_version()).major < 1 and isinstance(
        model, openai.Model
    ):
        return model.id

    raise mlflow.MlflowException(
        f"Unsupported model type: {type(model)}", error_code=INVALID_PARAMETER_VALUE
    )


def _get_task_name(task):
    mapping = _get_obj_to_task_mapping()
    if isinstance(task, str):
        if task not in mapping.values():
            raise mlflow.MlflowException(
                f"Unsupported task: {task}", error_code=INVALID_PARAMETER_VALUE
            )
        return task
    else:
        task_name = mapping.get(task)
        if task_name is None:
            raise mlflow.MlflowException(
                f"Unsupported task object: {task}", error_code=INVALID_PARAMETER_VALUE
            )
        return task_name


def _get_openai_package_version():
    import openai

    try:
        return openai.__version__
    except AttributeError:
        # openai < 0.27.5 doesn't have a __version__ attribute
        return openai.version.VERSION


def _log_secrets_yaml(local_model_dir, scope):
    with open(os.path.join(local_model_dir, "openai.yaml"), "w") as f:
        yaml.safe_dump({e.value: f"{scope}:{e.secret_key}" for e in _OpenAIEnvVar}, f)


def _parse_format_fields(s) -> Set[str]:
    """Parses format fields from a given string, e.g. "Hello {name}" -> ["name"]."""
    return {fn for _, fn, _, _ in Formatter().parse(s) if fn is not None}


def _get_input_schema(task, content):
    if content:
        formatter = _ContentFormatter(task, content)
        variables = formatter.variables
        if len(variables) == 1:
            return Schema([ColSpec(type="string")])
        elif len(variables) > 1:
            return Schema([ColSpec(name=v, type="string") for v in variables])
        else:
            return Schema([ColSpec(type="string")])
    else:
        return Schema([ColSpec(type="string")])


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    task,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    example_no_conversion=False,
    **kwargs,
):
    """
    Save an OpenAI model to a path on the local file system.

    Args:
        model: The OpenAI model name.
        task: The task the model is performing, e.g., ``openai.chat.completions`` or
            ``'chat.completions'``.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
            The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
        example_no_conversion: {{ example_no_conversion }}
        kwargs: Keyword arguments specific to the OpenAI task, such as the ``messages`` (see
            :ref:`mlflow.openai.messages` for more details on this parameter)
            or ``top_p`` value to use for chat completion.

    .. code-block:: python

        import mlflow
        import openai

        # Chat
        mlflow.openai.save_model(
            model="gpt-3.5-turbo",
            task=openai.chat.completions,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            path="model",
        )

        # Completions
        mlflow.openai.save_model(
            model="text-davinci-002",
            task=openai.completions,
            prompt="{text}. The general sentiment of the text is",
            path="model",
        )

        # Embeddings
        mlflow.openai.save_model(
            model="text-embedding-ada-002",
            task=openai.embeddings,
            path="model",
        )
    """
    import numpy as np

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)
    task = _get_task_name(task)

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        if signature.params:
            _validate_model_params(
                task, kwargs, {p.name: p.default for p in signature.params.params}
            )
        mlflow_model.signature = signature
    elif task == "chat.completions":
        messages = kwargs.get("messages", [])
        if messages and not (
            all(isinstance(m, dict) for m in messages)
            and all(map(_is_valid_message, messages))
        ):
            raise mlflow.MlflowException.invalid_parameter_value(
                "If `messages` is provided, it must be a list of dictionaries with keys "
                "'role' and 'content'."
            )

        mlflow_model.signature = ModelSignature(
            inputs=_get_input_schema(task, messages),
            outputs=Schema([ColSpec(type="string", name=None)]),
        )
    elif task == "completions":
        prompt = kwargs.get("prompt")
        mlflow_model.signature = ModelSignature(
            inputs=_get_input_schema(task, prompt),
            outputs=Schema([ColSpec(type="string", name=None)]),
        )
    elif task == "embeddings":
        mlflow_model.signature = ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
        )

    if input_example is not None:
        _save_example(mlflow_model, input_example, path, example_no_conversion)
    if metadata is not None:
        mlflow_model.metadata = metadata
    model_data_path = os.path.join(path, MODEL_FILENAME)
    model_dict = {
        "model": _get_model_name(model),
        "task": task,
        **kwargs,
    }
    with open(model_data_path, "w") as f:
        yaml.safe_dump(model_dict, f)

    if task in _PYFUNC_SUPPORTED_TASKS:
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mindctrl.openai_deployment",
            data=MODEL_FILENAME,
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_dir_subpath,
        )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        openai_version=_get_openai_package_version(),
        data=MODEL_FILENAME,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    task,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    example_no_conversion=False,
    **kwargs,
):
    """
    Log an OpenAI model as an MLflow artifact for the current run.

    Args:
        model: The OpenAI model name or reference instance, e.g.,
            ``openai.Model.retrieve("gpt-3.5-turbo")``.
        task: The task the model is performing, e.g., ``openai.chat.completions`` or
            ``'chat.completions'``.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
            The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
        example_no_conversion: {{ example_no_conversion }}
        kwargs: Keyword arguments specific to the OpenAI task, such as the ``messages`` (see
            :ref:`mlflow.openai.messages` for more details on this parameter)
            or ``top_p`` value to use for chat completion.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python

        import mlflow
        import openai

        # Chat
        with mlflow.start_run():
            info = mlflow.openai.log_model(
                model="gpt-3.5-turbo",
                task=openai.chat.completions,
                messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
                artifact_path="model",
            )
            model = mlflow.pyfunc.load_model(info.model_uri)
            df = pd.DataFrame({"animal": ["cats", "dogs"]})
            print(model.predict(df))

        # Embeddings
        with mlflow.start_run():
            info = mlflow.openai.log_model(
                model="text-embedding-ada-002",
                task=openai.embeddings,
                artifact_path="embeddings",
            )
            model = mlflow.pyfunc.load_model(info.model_uri)
            print(model.predict(["hello", "world"]))
    """

    import sys

    return Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        registered_model_name=registered_model_name,
        model=model,
        task=task,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        example_no_conversion=example_no_conversion,
        **kwargs,
    )


def _load_model(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _is_valid_message(d):
    return isinstance(d, dict) and "content" in d and "role" in d


class _ContentFormatter:
    def __init__(self, task, template=None):
        if task == "completions":
            template = template or "{prompt}"
            if not isinstance(template, str):
                raise mlflow.MlflowException.invalid_parameter_value(
                    f"Template for task {task} expects type `str`, but got {type(template)}."
                )

            self.template = template
            self.format_fn = self.format_prompt
            self.variables = sorted(_parse_format_fields(self.template))
        elif task == "chat.completions":
            if not template:
                template = [{"role": "user", "content": "{content}"}]
            if not all(map(_is_valid_message, template)):
                raise mlflow.MlflowException.invalid_parameter_value(
                    f"Template for task {task} expects type `dict` with keys 'content' "
                    f"and 'role', but got {type(template)}."
                )

            self.template = template.copy()
            self.format_fn = self.format_chat
            self.variables = sorted(
                set(
                    itertools.chain.from_iterable(
                        _parse_format_fields(message.get("content"))
                        | _parse_format_fields(message.get("role"))
                        for message in self.template
                    )
                )
            )
            if not self.variables:
                self.template.append({"role": "user", "content": "{content}"})
                self.variables.append("content")
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Task type ``{task}`` is not supported for formatting."
            )

    def format(self, **params):
        if missing_params := set(self.variables) - set(params):
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Expected parameters {self.variables} to be provided, "
                f"only got {list(params)}, {list(missing_params)} are missing."
            )
        return self.format_fn(**params)

    def format_prompt(self, **params):
        return self.template.format(**{v: params[v] for v in self.variables})

    def format_chat(self, **params):
        format_args = {v: params[v] for v in self.variables}
        return [
            {
                "role": message.get("role").format(**format_args),
                "content": message.get("content").format(**format_args),
            }
            for message in self.template
        ]


def _first_string_column(pdf):
    iter_str_cols = (c for c, v in pdf.iloc[0].items() if isinstance(v, str))
    col = next(iter_str_cols, None)
    if col is None:
        raise mlflow.MlflowException.invalid_parameter_value(
            f"Could not find a string column in the input data: {pdf.dtypes.to_dict()}"
        )
    return col


# This is the meat of the changes
class _OpenAIDeploymentWrapper:
    def __init__(self, model):
        task = model.pop("task")
        if task not in _PYFUNC_SUPPORTED_TASKS:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Unsupported task: {task}. Supported tasks: {_PYFUNC_SUPPORTED_TASKS}."
            )
        self.model: dict = model
        self.task: str = task

        if self.task != "embeddings":
            self._setup_completions()

    def _setup_completions(self):
        if self.task == "chat.completions":
            self.template = self.model.get("messages", [])
        else:
            self.template = self.model.get("prompt")
        self.formater = _ContentFormatter(self.task, self.template)

    def format_completions(self, params_list):
        return [self.formater.format(**params) for params in params_list]

    def get_params_list(self, data):
        if len(self.formater.variables) == 1:
            variable = self.formater.variables[0]
            if variable in data.columns:
                return data[[variable]].to_dict(orient="records")
            else:
                first_string_column = _first_string_column(data)
                return [{variable: s} for s in data[first_string_column]]
        else:
            return data[self.formater.variables].to_dict(orient="records")

    def _predict_chat(self, data, params):
        # Don't get into this business for now
        # from mlflow.openai.api_request_parallel_processor import process_api_requests
        from mlflow.deployments import MlflowDeploymentClient

        _validate_model_params(self.task, self.model, params)
        messages_list = self.format_completions(self.get_params_list(data))
        # requests.exceptions.HTTPError: 422 Client Error: Unprocessable Entity for url: http://127.0.0.1:5001/endpoints/chat4t/invocations.
        # Response text: {"detail":"The parameter 'model' is not permitted to be passed. The route being queried already defines a model instance."}
        # We should change this behavior to only 4** if the model id is *different* from the endpoint model id
        model_kwargs = self.model.copy()
        model_kwargs.pop("model")

        requests = [
            {**model_kwargs, **params, "messages": messages}
            for messages in messages_list
        ]

        _logger.info(f"Making {len(requests)} requests to {get_deployments_target()}")
        deploy_client = get_deploy_client()
        assert deploy_client is not None
        assert isinstance(deploy_client, MlflowDeploymentClient)

        available_endpoints = deploy_client.list_endpoints()
        _logger.info(
            f"Available endpoints: {available_endpoints}, matching {self.model.get('model')} for {self.task}"
        )

        # For now, try to match model + task. Warn if multiple matches and pick the first one
        # Yeah, yeah I know this is a buggy filter, but fix it later when mainlined
        matched_tasks = [
            endpoint
            for endpoint in available_endpoints
            if "chat" in endpoint.endpoint_type
        ]
        matched_models = [
            endpoint
            for endpoint in matched_tasks
            if self.model["model"] in endpoint.model.name
            and endpoint.model.provider == "openai"
        ]
        if len(matched_models) == 0:
            raise mlflow.MlflowException(
                f"No matching endpoints found for model:\n{self.model}\ntask:\n{self.task}\navailable:\n{available_endpoints}"
            )
        if len(matched_models) > 1:
            _logger.warning(
                f"Multiple matching endpoints found for model:\n{self.model}\ntask:\n{self.task}\navailable:\n{available_endpoints}"
            )
        matched_endpoint = matched_models[0]

        responses = []
        for r in requests:
            response = deploy_client.predict(
                endpoint=matched_endpoint.name,
                inputs=r,
            )
            responses.append(response)

        return [r["choices"][0]["message"]["content"] for r in responses]

    def _predict_completions(self, data, params):
        raise NotImplementedError()

    def _predict_embeddings(self, data, params):
        raise NotImplementedError()

    def predict(self, data, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                           release without warning.

        Returns:
            Model predictions.
        """

        if self.task == "chat.completions":
            return self._predict_chat(data, params or {})
        elif self.task == "completions":
            return self._predict_completions(data, params or {})
        elif self.task == "embeddings":
            return self._predict_embeddings(data, params or {})


def _load_pyfunc(path):
    """Loads PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``openai`` flavor.
    """
    return _OpenAIDeploymentWrapper(_load_model(path))


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load an OpenAI model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A dictionary representing the OpenAI model.
    """
    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_data_path = os.path.join(
        local_model_path, flavor_conf.get("data", MODEL_FILENAME)
    )
    return _load_model(model_data_path)
