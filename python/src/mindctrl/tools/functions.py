import inspect
import json
import logging
from typing import Callable, Optional


_logger = logging.getLogger(__name__)

_specific_json_types = {
    str: "string",
    int: "number",
}


def generate_json_schema(
    name: str, description: Optional[str], **params: inspect.Parameter
) -> dict:
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    for param_name, param in params.items():
        param_schema = {
            "type": _specific_json_types.get(param.annotation, "object"),
            "description": param.default
            if param.default != inspect.Parameter.empty
            else None,
        }

        if param.default == inspect.Parameter.empty:
            schema["function"]["parameters"]["required"].append(param_name)

        schema["function"]["parameters"]["properties"][param_name] = param_schema

    return schema


async def call_function(func):
    ret = func()
    return await ret if inspect.isawaitable(ret) else ret


def generate_function_schema(func: Callable) -> dict:
    signature = inspect.signature(func)
    parameters = signature.parameters

    schema = generate_json_schema(func.__name__, func.__doc__, **parameters)
    _logger.debug(f"Generated schema: {schema}")
    return schema


if __name__ == "__main__":
    # Example usage
    def get_current_weather(location: str, unit: str = "celsius"):
        """
        Get the current weather in a given location

        Args:
            location (str): The city and state, e.g. San Francisco, CA
            unit (str, optional): The unit of temperature. Defaults to "celsius".

        Returns:
            dict: The weather information
        """
        pass

    json_schema = generate_function_schema(get_current_weather)
    print(json.dumps(json_schema, indent=4))

    class FakeHass:
        def get_weather(self, location: str, unit: str = "celsius"):
            pass

        async def get_weather_async(self, location: str, unit: str = "celsius"):
            pass

    hass = FakeHass()
    json_schema = generate_function_schema(hass.get_weather_async)
    print(json.dumps(json_schema, indent=4))
