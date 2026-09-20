import demo from "../../examples/minimal/vite.config.js";
import { fileURLToPath } from "node:url";
export default {
  ...demo,
  publicDir: fileURLToPath(
    new URL("../../examples/minimal/public", import.meta.url),
  ),
};
