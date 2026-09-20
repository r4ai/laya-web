import { render } from "solid-js/web";
import { App } from "./app.js";
import "./style.css";

const dispose = render(() => <App />, document.getElementById("app")!);
if (import.meta.hot) import.meta.hot.dispose(dispose);
