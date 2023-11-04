# Evaluating Agents and LLMs
***Am I any good at this***

## Self-evaluation
We do it at work, why not at home? Currently called GPT-assisted metrics, the technique involves carefully crafting a prompt with a previous response to the "grading LLM" that basically asks how good the response was at a certain task:
* [being grounded](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2#groundedness) - using facts from truthful context in the generated content
* [being coherent](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/concept-model-monitoring-generative-ai-evaluation-metrics?view=azureml-api-2#coherence) - using facts from previous generated content in the generated content

... and more that you can read about at the links above.
