const worker = new Worker(new URL("./worker.ts", import.meta.url), {
  type: "module",
});
const button = document.querySelector("button")!;
const status = document.querySelector("pre")!;
button.onclick = () => {
  button.disabled = true;
  status.textContent = "Loading";
  worker.postMessage({
    backend: document.querySelector("select")!.value,
    base: location.origin,
  });
};
worker.onmessage = ({ data }) => {
  status.textContent = JSON.stringify(data, null, 2);
  if (data.complete) button.disabled = false;
};
worker.onerror = (event) => {
  status.textContent = event.message;
  button.disabled = false;
};
