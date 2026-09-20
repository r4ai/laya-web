import { load } from "laya-web";
import type { Agent } from "laya-web";
import type { Request, Response } from "./protocol.js";
let agent: Agent | undefined;
let requestedBackend: Request["backend"];
const send = (message: Response) => postMessage(message);
onmessage = async ({ data }: MessageEvent<Request>) => {
  try {
    if (agent && requestedBackend !== data.backend) {
      await agent.dispose();
      agent = undefined;
    }
    if (!agent) {
      agent = await load({
        modelUrl: data.modelUrl,
        backend: data.backend,
        wasmPaths: data.wasmPaths,
        onProgress: (progress) => send({ type: "progress", progress }),
      });
      requestedBackend = data.backend;
    }
    send({ type: "running" });
    const start = performance.now();
    const result = await agent.predict(data.state, data.questions);
    send({
      type: "result",
      result,
      backend: agent.backend,
      elapsed: performance.now() - start,
    });
  } catch (error) {
    send({
      type: "error",
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
