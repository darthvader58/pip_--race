const form = document.querySelector("#telemetryForm");
const statusPill = document.querySelector("#statusPill");
const riskValue = document.querySelector("#riskValue");
const degradationValue = document.querySelector("#degradationValue");
const confidenceValue = document.querySelector("#confidenceValue");
const latencyValue = document.querySelector("#latencyValue");
const rowsBody = document.querySelector("#rowsBody");
const chartWrap = document.querySelector("#chartWrap");

const throttle = form.elements.throttle;
const brake = form.elements.brake;
const lap = form.elements.lap;
const throttleOut = document.querySelector("#throttleOut");
const brakeOut = document.querySelector("#brakeOut");
const lapOut = document.querySelector("#lapOut");
let inferenceTimer;

function syncRangeOutputs() {
  lapOut.value = Number(lap.value).toFixed(0);
  throttleOut.value = Number(throttle.value).toFixed(2);
  brakeOut.value = Number(brake.value).toFixed(2);
}

function scheduleInference() {
  window.clearTimeout(inferenceTimer);
  inferenceTimer = window.setTimeout(() => runInference(), 180);
}

function payloadFromForm() {
  return Object.fromEntries(new FormData(form).entries());
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function setStatus(status) {
  statusPill.textContent = status;
  statusPill.dataset.status = status;
}

function renderRows(rows) {
  rowsBody.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.car_id}</td>
          <td>${Math.round(row.lap_distance_m)} m</td>
          <td>${formatPercent(row.pit_risk)}</td>
          <td><span class="mini-status" data-status="${row.status}">${row.status}</span></td>
          <td>${row.tire_age_laps.toFixed(1)}</td>
          <td>${Math.round(row.speed_kph)}</td>
        </tr>
      `,
    )
    .join("");
}

async function runInference(event) {
  event?.preventDefault();
  setStatus("RUNNING");

  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadFromForm()),
  });
  const data = await response.json();
  if (!response.ok) {
    setStatus("ERROR");
    chartWrap.textContent = data.error || "Request failed";
    return;
  }

  const inference = data.frame.inference;
  setStatus(data.frame.status);
  riskValue.textContent = formatPercent(inference.pit_risk);
  degradationValue.textContent = formatPercent(inference.tire_degradation);
  confidenceValue.textContent = formatPercent(inference.confidence);
  latencyValue.textContent = `${Math.round(inference.model_latency_ns / 1000)} us`;
  chartWrap.innerHTML = data.svg;
  renderRows(data.rows);
}

form.addEventListener("submit", runInference);
form.addEventListener("input", () => {
  syncRangeOutputs();
  scheduleInference();
});
form.addEventListener("change", () => {
  syncRangeOutputs();
  scheduleInference();
});
syncRangeOutputs();
runInference();
