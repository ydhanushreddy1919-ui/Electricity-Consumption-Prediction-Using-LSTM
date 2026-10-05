/**
 * WattCast — LSTM electricity consumption prediction
 * ---------------------------------------------------
 * A real recurrent neural network (LSTM) trained fully in the browser
 * with TensorFlow.js. Pipeline:
 *   1. Generate (or upload) hourly kWh meter data
 *   2. Min-max scale + build sliding-window sequences with time features
 *   3. Train LSTM → Dropout → Dense(1) with Adam + early stopping
 *   4. Evaluate on a held-out test split (MAE / RMSE / MAPE / R²)
 *   5. Recursive multi-step forecast with a growing 95% interval
 */
import * as tf from '@tensorflow/tfjs';
import Chart from 'chart.js/auto';

/* ============================ DOM helpers ============================ */
const $ = (id) => document.getElementById(id);
const els = {
  days: $('days'), generateBtn: $('generateBtn'),
  csvFile: $('csvFile'), uploadBtn: $('uploadBtn'),
  lookback: $('lookback'), units: $('units'), epochs: $('epochs'),
  batchSize: $('batchSize'), lr: $('lr'), trainBtn: $('trainBtn'),
  horizon: $('horizon'), predictBtn: $('predictBtn'), pipelineBtn: $('pipelineBtn'),
  progressFill: $('progressFill'), progressLabel: $('progressLabel'),
  statusData: $('statusData'), statusModel: $('statusModel'),
  modelSummary: $('modelSummary'), log: $('log'),
  statMae: $('statMae'), statRmse: $('statRmse'),
  statMape: $('statMape'), statR2: $('statR2'),
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const N_FEATURES = 5; // [normalized load, sin/cos hour, sin/cos weekday]

/* ============================ App state ============================ */
const state = {
  data: [],          // { t: Date, value: number } hourly kWh
  splitIndex: 0,     // train/test boundary
  scaler: null,      // { min, max } fitted on train portion only
  features: [],      // normalized feature rows per timestamp
  lookback: 48,
  model: null,
  trained: false,
  training: false,
  metrics: null,
  testFit: [],       // one-step predictions on the test region
  forecast: null,    // [{ t, value, upper, lower }]
};
const lossHistory = { train: [], val: [] };

/* ============================ RNG utils ============================ */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ============================ Synthetic data ============================ */
// Typical commercial-building load shape across 24 hours (relative).
const HOUR_SHAPE = [
  0.52, 0.48, 0.46, 0.45, 0.48, 0.56, 0.72, 0.92, 1.08, 1.16, 1.20, 1.23,
  1.26, 1.24, 1.21, 1.18, 1.16, 1.22, 1.32, 1.30, 1.16, 0.96, 0.76, 0.60,
];

function generateSyntheticData(days, seed) {
  const rng = mulberry32(seed);
  const start = new Date();
  start.setMinutes(0, 0, 0);
  start.setHours(start.getHours() - days * 24);
  const out = [];
  const n = days * 24;
  for (let i = 0; i < n; i++) {
    const t = new Date(start.getTime() + i * 3600000);
    const h = t.getHours();
    const dow = t.getDay();
    const weekend = dow === 0 || dow === 6 ? 0.74 : 1.0;
    const seasonal = 1 + 0.14 * Math.sin((2 * Math.PI * (i / 24)) / 365.25 + 1.3);
    const trend = 1 + 0.00005 * i;
    const noise = 1 + gaussian(rng) * 0.05;
    const spike = rng() < 0.004 ? 10 + rng() * 22 : 0;
    const value = Math.max(6, 88 * HOUR_SHAPE[h] * weekend * seasonal * trend * noise + spike);
    out.push({ t, value });
  }
  return out;
}

/* ============================ CSV parsing ============================ */
function parseCSV(text) {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const withTs = [];
  const plain = [];
  for (const line of lines) {
    const parts = line.split(/[,;\t]/).map((p) => p.replace(/^"|"$/g, '').trim());
    if (parts.length >= 2) {
      const t = new Date(parts[0]);
      const v = parseFloat(parts[parts.length - 1]);
      if (!isNaN(t.getTime()) && isFinite(v)) { withTs.push({ t, value: v }); continue; }
    }
    const v = parseFloat(parts[parts.length - 1]);
    if (isFinite(v)) plain.push(v);
  }
  if (withTs.length > 0 && withTs.length >= plain.length) return withTs;
  const end = new Date();
  end.setMinutes(0, 0, 0);
  return plain.map((value, i) => ({
    t: new Date(end.getTime() - (plain.length - 1 - i) * 3600000),
    value,
  }));
}

/* ============================ Feature engineering ============================ */
function timeFeatures(t) {
  const h = t.getHours();
  const d = t.getDay() + h / 24;
  return [
    Math.sin((2 * Math.PI * h) / 24),
    Math.cos((2 * Math.PI * h) / 24),
    Math.sin((2 * Math.PI * d) / 7),
    Math.cos((2 * Math.PI * d) / 7),
  ];
}

function denorm(n) {
  const { min, max } = state.scaler;
  return min + n * (max - min);
}

/**
 * Scale on the TRAIN portion only (no leakage), build the feature matrix
 * and sliding-window sequences, split by target index.
 */
function prepareDataset(lookback) {
  const N = state.data.length;
  const trainCount = Math.floor(N * 0.8);
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < trainCount; i++) {
    const v = state.data[i].value;
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  const range = (vmax - vmin) || 1;
  state.scaler = { min: vmin, max: vmax };
  state.splitIndex = trainCount;
  state.features = state.data.map(
    (d) => [(d.value - vmin) / range, ...timeFeatures(d.t)]
  );

  const nSeq = N - lookback;
  const trainRows = [], testRows = [];
  for (let i = 0; i < nSeq; i++) {
    (i + lookback < trainCount ? trainRows : testRows).push(i);
  }

  const toArrays = (rows) => {
    const xs = new Float32Array(rows.length * lookback * N_FEATURES);
    const ys = new Float32Array(rows.length);
    rows.forEach((start, r) => {
      for (let s = 0; s < lookback; s++) {
        xs.set(state.features[start + s], (r * lookback + s) * N_FEATURES);
      }
      ys[r] = state.features[start + lookback][0];
    });
    return { xs, ys };
  };

  return { train: toArrays(trainRows), test: toArrays(testRows), testRows };
}

/* ============================ Charts ============================ */
Chart.defaults.color = '#8b9bb4';
Chart.defaults.font.family = "'JetBrains Mono', ui-monospace, monospace";
Chart.defaults.font.size = 10.5;
Chart.defaults.borderColor = 'rgba(148,163,184,0.08)';

let mainChart = null;
let lossChart = null;

const fmtLabel = (t) =>
  t.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });

const baseLineOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: {
      labels: {
        usePointStyle: true, boxWidth: 7, boxHeight: 7, padding: 14,
        filter: (item) => !item.text.startsWith('_'),
      },
    },
    tooltip: {
      backgroundColor: '#0c1322',
      borderColor: 'rgba(148,163,184,0.2)',
      borderWidth: 1,
      padding: 10,
      filter: (item) => !item.dataset.label.startsWith('_'),
      callbacks: {
        label: (ctx) =>
          ctx.parsed.y == null ? null : ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} kWh`,
      },
    },
  },
};

function initCharts() {
  mainChart = new Chart($('mainChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      ...baseLineOptions,
      scales: {
        x: { ticks: { maxTicksLimit: 9, maxRotation: 0, autoSkip: true }, grid: { display: false } },
        y: { title: { display: true, text: 'kWh' }, grid: { color: 'rgba(148,163,184,0.07)' } },
      },
    },
  });
  lossChart = new Chart($('lossChart'), {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      ...baseLineOptions,
      plugins: { ...baseLineOptions.plugins, legend: { labels: { usePointStyle: true, boxWidth: 7, boxHeight: 7 } } },
      scales: {
        x: { title: { display: true, text: 'epoch' }, ticks: { maxTicksLimit: 10 }, grid: { display: false } },
        y: { type: 'logarithmic', title: { display: true, text: 'loss (log)' }, grid: { color: 'rgba(148,163,184,0.07)' } },
      },
    },
  });
}

function renderMainChart() {
  const data = state.data;
  if (!data.length) return;
  const horizonPts = state.forecast || [];
  const N = data.length;
  const split = state.splitIndex || Math.floor(N * 0.8);
  const total = N + horizonPts.length;

  const labels = [
    ...data.map((d) => fmtLabel(d.t)),
    ...horizonPts.map((p) => fmtLabel(p.t)),
  ];
  const history = data.map((d, i) => (i < split ? d.value : null));
  const test = data.map((d, i) => (i >= split - 1 ? d.value : null));
  const fit = new Array(total).fill(null);
  for (const p of state.testFit) fit[p.idx] = p.value;
  const forecast = new Array(total).fill(null);
  const upper = new Array(total).fill(null);
  const lower = new Array(total).fill(null);
  if (horizonPts.length) {
    forecast[N - 1] = data[N - 1].value;
    horizonPts.forEach((p, i) => {
      forecast[N + i] = p.value;
      upper[N + i] = p.upper;
      lower[N + i] = p.lower;
    });
  }

  mainChart.data.labels = labels;
  mainChart.data.datasets = [
    {
      label: 'History (train)', data: history, borderColor: '#64748b',
      backgroundColor: 'rgba(100,116,139,0.08)', borderWidth: 1.4,
      pointRadius: 0, fill: true, tension: 0.25,
    },
    {
      label: 'Actual (test)', data: test, borderColor: '#fbbf24',
      borderWidth: 1.6, pointRadius: 0, tension: 0.25,
    },
    {
      label: 'LSTM 1-step fit', data: fit, borderColor: '#22d3ee',
      borderWidth: 1.2, borderDash: [5, 4], pointRadius: 0, tension: 0.25,
    },
    {
      label: 'Forecast', data: forecast, borderColor: '#34d399',
      borderWidth: 2, borderDash: [2, 3], pointRadius: 0, tension: 0.25,
    },
    {
      label: '95% interval', data: upper, borderColor: 'transparent',
      pointRadius: 0, fill: '+1', backgroundColor: 'rgba(52,211,153,0.12)',
    },
    { label: '_band', data: lower, borderColor: 'transparent', pointRadius: 0 },
  ];
  mainChart.update();
}

function renderLossChart() {
  lossChart.data.labels = lossHistory.train.map((_, i) => i + 1);
  lossChart.data.datasets = [
    { label: 'train', data: lossHistory.train, borderColor: '#22d3ee', borderWidth: 1.6, pointRadius: 0, tension: 0.3 },
    { label: 'validation', data: lossHistory.val, borderColor: '#fbbf24', borderWidth: 1.6, pointRadius: 0, tension: 0.3 },
  ];
  lossChart.update();
}

/* ============================ UI helpers ============================ */
function log(msg, level = 'info') {
  const line = document.createElement('div');
  line.className = `log-line log-${level}`;
  const time = document.createElement('span');
  time.className = 'log-time';
  time.textContent = new Date().toLocaleTimeString();
  const text = document.createElement('span');
  text.textContent = msg;
  line.append(time, text);
  els.log.appendChild(line);
  els.log.scrollTop = els.log.scrollHeight;
}

function setBusy(busy) {
  for (const btn of [els.generateBtn, els.uploadBtn, els.trainBtn, els.predictBtn, els.pipelineBtn]) {
    btn.disabled = busy;
  }
}

function updateStatus() {
  const N = state.data.length;
  els.statusData.textContent = N
    ? `${N.toLocaleString()} pts · hourly`
    : 'no data';
  els.statusModel.textContent = state.training
    ? 'model · training…'
    : state.trained
      ? 'model · trained ✓'
      : 'model · idle';
  els.statusModel.classList.toggle('ok', state.trained && !state.training);
  els.statusModel.classList.toggle('busy', state.training);
}

function updateStats() {
  const m = state.metrics;
  els.statMae.textContent = m ? m.mae.toFixed(2) : '—';
  els.statRmse.textContent = m ? m.rmse.toFixed(2) : '—';
  els.statMape.textContent = m ? m.mape.toFixed(1) : '—';
  els.statR2.textContent = m ? m.r2.toFixed(3) : '—';
}

function renderModelSummary(model, lookback) {
  const rows = model.layers.map((l) => {
    let shape = '—';
    try {
      const s = l.outputShape;
      if (Array.isArray(s)) shape = s.map((x) => (x == null ? '?' : x)).join(' × ');
    } catch { /* shape unavailable for some layers */ }
    return `<tr><td>${l.name}</td><td>${shape}</td><td class="num">${l.countParams().toLocaleString()}</td></tr>`;
  }).join('');
  els.modelSummary.innerHTML = `
    <table class="summary-table">
      <thead><tr><th>Layer</th><th>Output shape</th><th>Params</th></tr></thead>
      <tbody>
        <tr><td>input</td><td>? × ${lookback} × ${N_FEATURES}</td><td class="num">—</td></tr>
        ${rows}
      </tbody>
      <tfoot><tr><td colspan="2">Total trainable parameters</td><td class="num">${model.countParams().toLocaleString()}</td></tr></tfoot>
    </table>
    <p class="summary-note">Features per timestep: normalized load, sin/cos hour-of-day, sin/cos day-of-week.</p>`;
}

/* ============================ Data actions ============================ */
function setData(points, message) {
  state.data = points;
  state.splitIndex = Math.floor(points.length * 0.8);
  state.trained = false;
  state.metrics = null;
  state.testFit = [];
  state.forecast = null;
  state.scaler = null;
  updateStats();
  updateStatus();
  renderMainChart();
  log(message, 'ok');
}

function generateData() {
  const days = clamp(parseInt(els.days.value, 10) || 60, 14, 365);
  const seed = Math.floor(Math.random() * 1e9);
  const pts = generateSyntheticData(days, seed);
  setData(pts, `Generated ${pts.length.toLocaleString()} hourly readings (${days} days, seed ${seed}).`);
}

/* ============================ Training ============================ */
async function trainModel() {
  if (state.training) return;
  if (state.data.length < 200) {
    log('Need at least ~200 data points — generate or upload data first.', 'warn');
    return;
  }

  const lookback = clamp(parseInt(els.lookback.value, 10) || 48, 6, 168);
  const units = clamp(parseInt(els.units.value, 10) || 32, 8, 128);
  const epochs = clamp(parseInt(els.epochs.value, 10) || 30, 1, 200);
  const batchSize = clamp(parseInt(els.batchSize.value, 10) || 32, 8, 256);
  const lr = clamp(parseFloat(els.lr.value) || 0.005, 1e-5, 0.1);

  if (state.data.length - lookback < 50) {
    log(`Lookback ${lookback}h is too long for ${state.data.length} points — reduce it or add more data.`, 'warn');
    return;
  }

  state.training = true;
  setBusy(true);
  updateStatus();
  log(`Preparing sequences — lookback ${lookback}h, 80/20 chronological split…`);
  await tf.nextFrame();

  const { train, test, testRows } = prepareDataset(lookback);
  state.lookback = lookback;
  log(`Sequences ready — train ${train.ys.length.toLocaleString()} · test ${test.ys.length.toLocaleString()}.`);

  const xsTrain = tf.tensor3d(train.xs, [train.ys.length, lookback, N_FEATURES]);
  const ysTrain = tf.tensor2d(train.ys, [train.ys.length, 1]);
  const xsTest = tf.tensor3d(test.xs, [test.ys.length, lookback, N_FEATURES]);

  if (state.model) { state.model.dispose(); state.model = null; }
  const model = tf.sequential();
  model.add(tf.layers.lstm({ units, inputShape: [lookback, N_FEATURES] }));
  model.add(tf.layers.dropout({ rate: 0.15 }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
  state.model = model;

  log(`Model built — LSTM(${units}) → Dropout(0.15) → Dense(1) · ${model.countParams().toLocaleString()} params. Training…`);
  renderModelSummary(model, lookback);

  lossHistory.train = [];
  lossHistory.val = [];
  els.progressFill.style.width = '0%';
  els.progressLabel.textContent = 'starting…';

  try {
    await model.fit(xsTrain, ysTrain, {
      epochs,
      batchSize,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: [
        tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 8, restoreBestWeights: true }),
        {
          onEpochEnd: async (epoch, logs) => {
            lossHistory.train.push(logs.loss);
            lossHistory.val.push(logs.val_loss);
            renderLossChart();
            els.progressFill.style.width = `${(((epoch + 1) / epochs) * 100).toFixed(0)}%`;
            els.progressLabel.textContent =
              `epoch ${epoch + 1}/${epochs} · loss ${logs.loss.toFixed(5)} · val ${logs.val_loss.toFixed(5)}`;
            await tf.nextFrame(); // keep the UI responsive
          },
        },
      ],
    });
  } catch (err) {
    log(`Training failed: ${err.message}`, 'err');
    state.training = false;
    setBusy(false);
    updateStatus();
    xsTrain.dispose(); ysTrain.dispose(); xsTest.dispose();
    return;
  }

  /* ---- evaluate one-step predictions on the held-out test split ---- */
  const predTensor = model.predict(xsTest);
  const preds = await predTensor.data();
  predTensor.dispose();

  let mae = 0, rmse = 0, mape = 0, ssRes = 0, ssTot = 0, mean = 0;
  const actuals = Array.from(test.ys, (n) => denorm(n));
  const fitted = Array.from(preds, (n) => denorm(clamp(n, 0, 1)));
  for (let i = 0; i < actuals.length; i++) mean += actuals[i];
  mean /= actuals.length || 1;
  for (let i = 0; i < actuals.length; i++) {
    const err = actuals[i] - fitted[i];
    mae += Math.abs(err);
    rmse += err * err;
    mape += Math.abs(err) / Math.max(actuals[i], 1);
    ssRes += err * err;
    ssTot += (actuals[i] - mean) ** 2;
  }
  const n = actuals.length || 1;
  const residualsStd = Math.sqrt(ssRes / n);
  state.metrics = {
    mae: mae / n,
    rmse: Math.sqrt(rmse / n),
    mape: (mape / n) * 100,
    r2: ssTot > 0 ? 1 - ssRes / ssTot : 0,
    resStd: residualsStd,
  };
  state.testFit = testRows.map((start, i) => ({ idx: start + lookback, value: fitted[i] }));
  state.trained = true;
  state.forecast = null;

  xsTrain.dispose(); ysTrain.dispose(); xsTest.dispose();

  els.progressFill.style.width = '100%';
  els.progressLabel.textContent = 'done';
  updateStats();
  updateStatus();
  renderMainChart();
  log(
    `Training complete — MAE ${state.metrics.mae.toFixed(2)} kWh · RMSE ${state.metrics.rmse.toFixed(2)} kWh · ` +
    `MAPE ${state.metrics.mape.toFixed(1)}% · R² ${state.metrics.r2.toFixed(3)}`,
    'ok'
  );

  state.training = false;
  setBusy(false);
}

/* ============================ Forecasting ============================ */
async function runForecast() {
  if (!state.model || !state.trained) {
    log('Train the model before forecasting.', 'warn');
    return;
  }
  const horizon = clamp(parseInt(els.horizon.value, 10) || 48, 1, 336);
  const lookback = state.lookback;
  const window = state.features.slice(-lookback).map((r) => r.slice());
  const lastT = state.data[state.data.length - 1].t;
  const resStd = state.metrics ? state.metrics.resStd : (state.scaler.max - state.scaler.min) * 0.03;

  const points = [];
  for (let k = 1; k <= horizon; k++) {
    const t = new Date(lastT.getTime() + k * 3600000);
    let yNorm = 0;
    tf.tidy(() => {
      const x = tf.tensor3d([window.slice(-lookback).flat()], [1, lookback, N_FEATURES]);
      yNorm = state.model.predict(x).dataSync()[0];
    });
    yNorm = clamp(yNorm, 0, 1);
    const value = denorm(yNorm);
    const spread = 1.96 * resStd * Math.sqrt(k); // uncertainty grows with horizon
    points.push({ t, value, upper: value + spread, lower: Math.max(0, value - spread) });
    window.push([yNorm, ...timeFeatures(t)]);
  }

  state.forecast = points;
  renderMainChart();
  const total = points.reduce((s, p) => s + p.value, 0);
  const peak = points.reduce((best, p) => (p.value > best.value ? p : best), points[0]);
  log(
    `Forecast generated — next ${horizon}h · est. total ${total.toFixed(0)} kWh · ` +
    `avg ${(total / horizon).toFixed(1)} kWh/h · peak ${peak.value.toFixed(1)} kWh at ${fmtLabel(peak.t)}`,
    'ok'
  );
}

/* ============================ Full pipeline ============================ */
async function runPipeline() {
  if (state.training) return;
  log('Auto-run: dataset → train → forecast.', 'info');
  if (!state.data.length) generateData();
  await trainModel();
  if (state.trained) await runForecast();
}

/* ============================ Events ============================ */
els.generateBtn.addEventListener('click', generateData);
els.trainBtn.addEventListener('click', trainModel);
els.predictBtn.addEventListener('click', runForecast);
els.pipelineBtn.addEventListener('click', runPipeline);
els.uploadBtn.addEventListener('click', () => els.csvFile.click());
els.csvFile.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const text = await file.text();
  const pts = parseCSV(text);
  if (pts.length < 100) {
    log(`CSV rejected — only ${pts.length} numeric rows parsed (need ≥ 100).`, 'err');
  } else {
    setData(pts, `Loaded ${pts.length.toLocaleString()} points from ${file.name}.`);
  }
  e.target.value = '';
});

/* ============================ Boot ============================ */
(async function boot() {
  initCharts();
  await tf.ready();
  log(`TensorFlow.js ready — backend: ${tf.getBackend()} · version ${tf.version.tfjs}.`);
  generateData();
  log('Tip: press “Auto-run full pipeline” to generate → train → forecast in one click.');
  updateStatus();
})();
