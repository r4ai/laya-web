const button = document.querySelector<HTMLButtonElement>("#run")!;
const output = document.querySelector("#measurements")!;
button.onclick = async () => {
  button.disabled = true;
  const measurements: unknown[] = [];
  try {
    for (let run = 1; run <= 3; run++) {
      output.textContent = JSON.stringify({ run, measurements }, null, 2);
      measurements.push(await measure());
    }
    output.textContent = JSON.stringify(
      { complete: true, measurements },
      null,
      2,
    );
  } catch (error) {
    output.textContent = String(error);
  } finally {
    button.disabled = false;
  }
};
function measure() {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./loading-worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = ({ data }) => {
      worker.terminate();
      if (data.error) reject(new Error(data.error));
      else resolve(data);
    };
    worker.onerror = (event) => {
      worker.terminate();
      reject(new Error(event.message));
    };
    worker.postMessage(location.origin);
  });
}
