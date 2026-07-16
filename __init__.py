from .ltx_director import LTXDirector
from .cloud_output import KlingVideoOutput
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class WhatDreamsCostExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            LTXDirector,
        ]


async def comfy_entrypoint() -> WhatDreamsCostExtension:
    return WhatDreamsCostExtension()


NODE_CLASS_MAPPINGS = {
    "LTXDirector": LTXDirector,
    "KlingVideoOutput": KlingVideoOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirector": "LTX Director (Kling)",
    "KlingVideoOutput": "Kling Video Output",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
