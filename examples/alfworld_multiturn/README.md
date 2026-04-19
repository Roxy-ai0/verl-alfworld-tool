# ALFWorld TextWorld Tool-Calling GRPO

This example adds an ALFWorld TextWorld multi-turn tool-calling task on top of veRL's existing `ToolAgentLoop`.

## Architecture

- Rollout loop: official `verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop`
- Tool parsing: official Hermes `<tool_call>...</tool_call>` parser
- Tool response path: official `ToolResponse(text=...)` fed back as `role="tool"`
- Reward path: official reward manager with a custom `compute_score`
- Training algorithm: standard GRPO

The only tool exposed to the model is `alfworld_step(action: string)`.

## What Was Added

- `verl_ext/alfworld/`
- `examples/data_preprocess/alfworld_multiturn_w_tool.py`
- `examples/alfworld_multiturn/configs/alfworld_grpo.yaml`
- `examples/alfworld_multiturn/configs/tool_config/alfworld_tool_config.yaml`
- `examples/alfworld_multiturn/run_alfworld_grpo.sh`
- `examples/alfworld_multiturn/README.md`

No official veRL source files were modified.

## Environment Requirements

Install ALFWorld TextWorld dependencies first. The local environment used for this implementation did not already have them installed.

Example:

```bash
pip install alfworld textworld
```

You also need an ALFWorld dataset/config. The preprocess script expects an official ALFWorld YAML config such as `configs/base_config.yaml`, with the dataset paths resolved correctly, typically through `ALFWORLD_DATA`.

## Preprocess

```bash
python examples/data_preprocess/alfworld_multiturn_w_tool.py \
  --config_path /path/to/alfworld/configs/base_config.yaml \
  --local_save_dir ~/data/alfworld_multiturn
```

This writes:

- `~/data/alfworld_multiturn/train.parquet`
- `~/data/alfworld_multiturn/valid_seen.parquet`
- `~/data/alfworld_multiturn/valid_unseen.parquet`

Each sample contains:

- `agent_name=tool_agent`
- the required system/user prompts
- `extra_info.tools_kwargs.alfworld_step.create_kwargs.session_init`
- task metadata needed to reopen the same TextWorld episode

## Train

```bash
bash examples/alfworld_multiturn/run_alfworld_grpo.sh
```

Useful overrides:

```bash
ALFWORLD_TRAIN_FILE=~/data/alfworld_multiturn/train.parquet \
ALFWORLD_VAL_FILE=~/data/alfworld_multiturn/valid_seen.parquet \
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
bash examples/alfworld_multiturn/run_alfworld_grpo.sh
```

## Reward

The rollout-level reward is:

```text
final_score = (10.0 if success else 0.0) + sum(tool_rewards)
```

Current non-zero tool reward:

- invalid action: `-0.1`

The custom reward function returns:

- `score`
- `success`
- `invalid_action_count`
- `final_env_score`

## Session Persistence

`ToolAgentLoop` recreates/releases the tool object on each call, so episode state is not stored in the tool instance.

Instead, `verl_ext/alfworld/session_registry.py` keeps a persistent session keyed by `agent_data.request_id`. Each session stores:

- `env`
- current observation
- current admissible actions
- final env score
- `step_id`
- `done`
- `success`
- invalid action count

Done sessions are kept long enough to answer stray post-terminal tool calls correctly, then lazily cleaned by TTL. This avoids changing official veRL lifecycle code.

## Scope

- Supported: ALFWorld TextWorld / `AlfredTWEnv`
- Not supported: Thor, hybrid, visual mode
- No custom `<tool_response>` protocol is introduced
