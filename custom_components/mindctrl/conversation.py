from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.core import (
    HomeAssistant,
)
from homeassistant.config_entries import ConfigEntry
from typing import Literal
from homeassistant.util import ulid
from homeassistant.helpers import intent, template
from homeassistant.exceptions import (
    TemplateError,
)
import mlflow
import logging


_LOGGER = logging.getLogger(__name__)


class MLflowAgent(conversation.AbstractConversationAgent):
    """MLflow conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, list[dict]] = {}

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""

        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            try:
                prompt = self._async_generate_prompt("tell me a joke")
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [{"role": "system", "content": prompt}]

        messages.append({"role": "user", "content": user_input.text})

        _LOGGER.debug("Prompt for %s: %s", "mlflow", messages)

        try:
            print("invoke the model")
        except mlflow.MlflowException as err:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        _LOGGER.debug("Response %s", "lol")
        response = "i can't say anything"
        messages.append({"role": "system", "content": response})
        self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        )

    def _async_generate_prompt(self, raw_prompt: str) -> str:
        """Generate a prompt for the user."""
        # TODO: Turn this into a RAG lookup instead of an up-front prompt stuff
        # In fact, I'm not sure we should even be doing this here.
        # Maybe we should just be passing the prompt to the model and letting it
        # decide which states to look up via function calling and injecting the
        # HASS Core API / Services into the function calling params
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
            },
            parse_result=False,
        )
