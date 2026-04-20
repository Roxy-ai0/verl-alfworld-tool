import copy

from verl.utils.dataset.rl_dataset import RLHFDataset

from verl_ext.alfworld import agent_loop as _agent_loop  # noqa: F401
from verl_ext.alfworld.dataset import build_system_prompt


class AlfworldRLDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        if row_dict.get("data_source") != "alfworld_textworld":
            return row_dict

        row_dict["agent_name"] = "alfworld_tool_agent"

        raw_prompt = copy.deepcopy(row_dict.get("raw_prompt"))
        if isinstance(raw_prompt, list):
            if raw_prompt and raw_prompt[0].get("role") == "system":
                raw_prompt[0]["content"] = build_system_prompt()
            else:
                raw_prompt.insert(0, {"role": "system", "content": build_system_prompt()})
            row_dict["raw_prompt"] = raw_prompt

        return row_dict
