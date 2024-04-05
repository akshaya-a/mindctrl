# LLMOps for Home Assistant

Manage your home automation LLM prompts, available LLMs and evaluate changes with MLflow

[![Addon Builder](https://github.com/akshaya-a/mindctrl/actions/workflows/addon-builder.yaml/badge.svg)](https://github.com/akshaya-a/mindctrl/actions/workflows/addon-builder.yaml)
[![Addon Integration](https://github.com/akshaya-a/mindctrl/actions/workflows/integration-addon.yaml/badge.svg)](https://github.com/akshaya-a/mindctrl/actions/workflows/integration-addon.yaml)
[![Local Integration](https://github.com/akshaya-a/mindctrl/actions/workflows/integration-local.yaml/badge.svg)](https://github.com/akshaya-a/mindctrl/actions/workflows/integration-local.yaml)
[![K3D Integration](https://github.com/akshaya-a/mindctrl/actions/workflows/integration-k3d.yaml/badge.svg)](https://github.com/akshaya-a/mindctrl/actions/workflows/integration-k3d.yaml)

---

![promptlab](assets/mlflowhass-promptlab.png)

## Why?

- Enable users to manage their LLM prompts and available LLMs from Home Assistant
- Enable users to evaluate changes to their LLM prompts and LLM parameters with versioning + change management
- Centralize credential management for LLMs

### Better manage conversational prompts

Home Assistant has a convenient prompt template tool to generate prompts for LLMs. However, it's not easy to manage these prompts. I can't tell if my new prompt is better than the last one, change tracking is not easy, and live editing means switching between the developer tools and the prompt template tool. There's a better way! Enter MLflow with its new PromptLab UI and integrated evaluation tools + tracking.

### Reason about state changes

LLMs are excellent reasoning agents with unstructured input. Someone familiar with home automation is used to kludgy workarounds:

- presence detection like wasp-in-a-box, or
- using a combination of sensors to determine if someone is home via a weak heuristic, or
- playing around with the bayesian sensor because it's awesome, then slowly losing your mind tweaking the priors

LLMs can be used to reason about state changes more naturally. Can we send the state of multiple sensors and ask the LLM to decide a higher level status in the home?

- (motion, time of day, device usage) -> "is AK asleep?"

### Wait what about Langchain?

You heard about Langchain but not this whole MLflow business - what's up with that? [Read more about how all this fits together!](/docs/prompt-techniques.md)

## How?

### Getting Started

1. Install the MLflow Gateway
2. Install the MLflow Tracking Service
3. Install the MLflow Home Integration

## What?

- LLM: Large Language Model, a model that can generate text based on a prompt
- [MLflow](https://mlflow.org/): An open source platform for the machine learning lifecycle
- [Home Assistant](https://www.home-assistant.io/): An open source home automation platform
