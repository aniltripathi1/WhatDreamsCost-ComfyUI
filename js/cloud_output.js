import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Inline <video> preview for the Kling Video Output node.
app.registerExtension({
    name: "Comfy.KlingVideoOutput",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "KlingVideoOutput") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const el = document.createElement("video");
            el.controls = true;
            el.loop = true;
            el.muted = false;
            el.playsInline = true;
            el.style.width = "100%";
            el.style.background = "#000";
            el.style.borderRadius = "6px";
            el.style.display = "block";
            this._klingVideoEl = el;

            this.addDOMWidget("kling_preview", "preview", el, { serialize: false });

            const w = Math.max((this.size && this.size[0]) || 340, 340);
            const h = ((this.size && this.size[1]) || 180) + 230;
            this.setSize && this.setSize([w, h]);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;
            try {
                const info = message && message.kling_video && message.kling_video[0];
                if (info && this._klingVideoEl) {
                    const params = new URLSearchParams({
                        filename: info.filename || "",
                        subfolder: info.subfolder || "",
                        type: info.type || "output",
                    });
                    this._klingVideoEl.src = api.apiURL(`/view?${params.toString()}`);
                    const p = this._klingVideoEl.play();
                    if (p && p.catch) p.catch(() => {});
                }
            } catch (e) {
                console.error("[KlingVideoOutput] preview error", e);
            }
            return r;
        };
    },
});
