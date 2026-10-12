/**
 * PolishML dashboard — client-side rendering.
 *
 * Two data sources are expected as embedded <script type="application/json">
 * blocks (see dashboard_template.html.j2):
 *
 *   #shap-data    -> array of { feature: str, value: float, kind: "correct"|"error" }
 *                    sorted however you like; renderShapChart re-sorts by |value|.
 *
 *   #pileup-data  -> array of { base: "G"|"A"|..., strand: "+"|"-", bq: int,
 *                                mq: int, range: [lo, hi] }
 *
 * Everything lives under the PolishMLDash namespace so the template can call
 * PolishMLDash.renderShapChart(...) / PolishMLDash.renderPileupTable(...).
 */
(function (global) {
  "use strict";

  const COLORS = {
    error: "#f36c6c",     // pushes toward error  (matches the red bars in the mock)
    correct: "#5fb3ff",   // pushes toward correct
    gBase: "#57c785",
    aBase: "#f0a35c",
    grid: "rgba(255,255,255,0.08)",
    text: "#c9d3e0",
    bgPanel: "#161c2c",
  };

  const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: COLORS.text, family: "Inter, system-ui, sans-serif", size: 12 },
    margin: { l: 140, r: 30, t: 10, b: 30 },
  };

  const PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
  };

  const readList = document.getElementById("read-support-list");
  const readSummary = document.getElementById("read-support-summary");
  let readToken = 0;

  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  // ---------------------------------------------------------------------
  // SHAP contribution chart — diverging horizontal bar, one bar per feature
  // ---------------------------------------------------------------------
  function renderShapChart(containerId, features, topK = 10) {
    if (!features || !features.length) return;

    // Sort so the largest |contribution| sits at the top (Plotly draws
    // bottom-to-top for horizontal bars, so reverse after sorting).
    const sorted = [...features]
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, topK === Infinity ? features.length : topK)
      .reverse();

    const y = sorted.map((f) => f.feature);
    const x = sorted.map((f) => f.value);
    const colors = sorted.map((f) => (f.value >= 0 ? COLORS.error : COLORS.correct));
    const labels = sorted.map((f) => (f.value >= 0 ? `+${f.value.toFixed(3)}` : f.value.toFixed(3)));

    const trace = {
      type: "bar",
      orientation: "h",
      x,
      y,
      marker: { color: colors },
      text: labels,
      textposition: "outside",
      textfont: { color: COLORS.text, size: 11 },
      hovertemplate: "%{y}<br>SHAP value: %{x:.3f}<extra></extra>",
    };

    const maxAbs = Math.max(...x.map(Math.abs), 0.05);

    const layout = {
      ...PLOTLY_LAYOUT_BASE,
      height: Math.max(280, sorted.length * 42),
      xaxis: {
        range: [-maxAbs * 1.35, maxAbs * 1.35],
        zeroline: true,
        zerolinecolor: "rgba(255,255,255,0.3)",
        zerolinewidth: 1,
        gridcolor: COLORS.grid,
        showticklabels: true,
        title: { text: "SHAP contribution", font: { size: 11 } },
      },
      yaxis: {
        automargin: true,
        showgrid: false,
        tickfont: { size: 11 },
      },
      bargap: 0.28,
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
  }

  async function renderKernelShap(site, chartId, topKSelectorId) {
    const query = new URLSearchParams({
      contig: site.contig,
      position: String(site.position),
    });
    const response = await fetch(`/api/shap?${query.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`KernelSHAP request failed (${response.status})`);
    const result = await response.json();
    const selector = document.getElementById(topKSelectorId);
    const topK = selector?.value === "all" ? Infinity : Number(selector?.value || 10);
    renderShapChart(chartId, result.features, topK);
    document.getElementById("shap-sum").textContent = `${result.shap_sum >= 0 ? "+" : ""}${result.shap_sum.toFixed(3)}`;
    document.getElementById("shap-final-value").textContent = Number(result.p_error).toFixed(3);
  }

  // ---------------------------------------------------------------------
  // Pileup table — custom HTML (Plotly's table trace can't do per-cell
  // badge styling cleanly), built once from JSON.
  function renderPileupTable(containerId, reads) {
    if (!reads || !reads.length) return;
    const container = document.getElementById(containerId);
    if (!container) return;

    const rows = reads
      .map((r) => {
        const baseClass = r.base === "G" ? "base-g" : r.base === "A" ? "base-a" : "base-o";
        const strandGlyph = r.strand === "+" ? "+" : "\u2212";
        const bqPct = Math.min(100, Math.round((r.bq / 60) * 100));
        const range = Array.isArray(r.range) ? `${r.range[0]}-${r.range[1]}` : "";
        return `
            <td class="pu-bq-num">${r.bq}</td>
            <td><span class="pu-base ${baseClass}">${r.base}</span></td>
            <td class="pu-strand">${strandGlyph}</td>
            <td class="pu-bq-bar-cell">
              <div class="pu-bq-bar-track">
                <div class="pu-bq-bar-fill" style="width:${bqPct}%"></div>
              </div>
            </td>
            <td class="pu-range">${range}</td>
          </tr>`;
      })
      .join("");

    container.innerHTML = `
      <table class="pileup-table">
        <thead>
          <tr><th>BQ</th><th>Base</th><th>Str</th><th>BQ range</th><th>Pos range</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  async function renderReadSupport(site, showAll = false) {
    const token = ++readToken;
    let reads;
    try {
      const query = new URLSearchParams({
        contig: site.contig,
        position: String(site.position),
      });
      const res = await fetch(`/api/pileup?${query.toString()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      reads = await res.json();
    } catch (err) {
      if (token === readToken) { readSummary.textContent = ""; readList.textContent = `Reads unavailable: ${err.message}`; }
      return;
    }
    if (token !== readToken) return;

    document.getElementById("read-support-title").textContent =
      `Pileup at position ${Number(site.position).toLocaleString()}`;

    const counts = {};
    reads.forEach((r) => { counts[r.base] = (counts[r.base] || 0) + 1; });
    const parts = Object.entries(counts).sort((a, b) => b[1] - a[1])
      .map(([b, n]) => `${n} ${b}`).join(" vs ");
    readSummary.textContent = `${reads.length} reads aligned — ${parts}`;
    if (!reads.length) {
      document.getElementById("rationale-summary").textContent = `The model predicts ${site.decision} at ${site.contig}:${site.position} with error confidence ${Number(site.p_error).toFixed(3)}. The draft base is ${site.assembly_base} and the corrected base is ${site.corrected_base}.`;
      document.getElementById("rationale-pileup").textContent =
        `Pileup argument at ${site.contig}:${site.position}: no usable reads were found.`;
      readList.textContent = "No reads at this position.";
      return;
    }
    const support = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([base, count]) => `${base}: ${count} (${(count / reads.length * 100).toFixed(1)}%)`)
      .join(", ");
    const meanMq = reads.reduce((sum, read) => sum + Number(read.mq || 0), 0) / reads.length;
    const reverse = reads.filter((read) => read.strand === "-").length;
    const strandBias = Math.abs(reads.length - (reverse * 2)) / reads.length;
    document.getElementById("rationale-pileup").textContent =
      `Pileup argument at ${site.contig}:${site.position}: depth ${reads.length}; ${support}; mean MAPQ ${meanMq.toFixed(1)}; strand bias ${strandBias.toFixed(3)}.`;
    document.getElementById("rationale-summary").textContent = `The model predicts ${site.decision} at ${site.contig}:${site.position} with error confidence ${Number(site.p_error).toFixed(3)}. The draft base is ${site.assembly_base} and the corrected base is ${site.corrected_base}.`;

    reads.sort((a, b) => (b.base === site.corrected_base) - (a.base === site.corrected_base)
                          || a.base.localeCompare(b.base));
    const shown = showAll ? reads : reads.slice(0, 12);

    readList.replaceChildren(...shown.map((r) => {
      const row = el("div", "read-row");
      row.append(
        el("span", "read-off", r.offset),
        el("span", `read-base base-${r.base}`, r.base),
        el("span", "read-strand", r.strand),
        el("span", "read-bq", `BQ ${r.bq}`),
        (() => {
          const bar = el("span", "read-bar");
          const filled = Math.round((Math.min(r.bq, 60) / 60) * 20);
          for (let i = 0; i < 20; i++) bar.append(el("i", i < filled ? `on on-${r.base}` : ""));
          return bar;
        })(),
        el("span", "read-mq", `MQ ${r.mq}`),
      );
      return row;
    }));

    if (!showAll && reads.length > shown.length) {
      const more = el("button", "read-more", `…${reads.length - shown.length} more reads`);
      more.type = "button";
      more.addEventListener("click", () => renderReadSupport(site, true));
      readList.append(more);
    }
  }

  function baseCode(base) {
    if (base === "A") return 1;
    if (base === "C") return 2;
    if (base === "G") return 3;
    if (base === "T") return 4;
    if (base === "-") return 5;
    return 0;
  }

  function createAlignmentRow(label, className) {
    const row = document.createElement("div");
    row.className = "align-row";
    const labelElement = document.createElement("span");
    labelElement.className = "align-label";
    labelElement.textContent = label;
    const content = document.createElement("div");
    content.className = "alignment-content";
    content.dataset.row = className;
    row.append(labelElement, content);
    return { row, content };
  }

  async function initializeLazyAlignments(selector, manifestUrl) {
    const containers = [...document.querySelectorAll(selector)];
    if (!containers.length) return;
    const manifestResponse = await fetch(manifestUrl);
    if (!manifestResponse.ok) {
      throw new Error(`Could not load alignment manifest (${manifestResponse.status})`);
    }
    const manifest = await manifestResponse.json();
    const cellWidth = 24;

    for (const container of containers) {
      const contig = manifest.contigs[Number(container.dataset.contigIndex)];
      if (!contig) continue;
      const raw = createAlignmentRow("Raw", "raw");
      const connection = createAlignmentRow("Connection", "connection");
      const corrected = createAlignmentRow("Corrected", "corrected");
      container.append(raw.row, connection.row, corrected.row);
      const rows = [raw, connection, corrected];
      const alignmentPanel = container.closest(".contig-alignment");
      const previousButton = alignmentPanel.querySelector(".alignment-nav-prev");
      const nextButton = alignmentPanel.querySelector(".alignment-nav-next");
      const positionOutput = alignmentPanel.querySelector(".alignment-position-value");
      let loadedChunk = -1;
      let loadPromise = null;
      let currentChunk = 0;
      let loadedChunkData = null;
      let focusRange = null;

      const updateAlignmentFocus = () => {
        rows.forEach((row) => {
          const window = row.content.firstElementChild;
          if (!window) return;
          const cells = [...window.children];
          const visibleCells = cells.filter((cell) => {
            const position = Number(cell.dataset.position);
            if (!Number.isFinite(position)) return true;
            const visible = !focusRange || (position >= focusRange.start && position <= focusRange.end);
            cell.classList.toggle("alignment-focus-hidden", !visible);
            return visible;
          });
          const columnCount = focusRange
            ? Math.min(256, Math.max(1, visibleCells.length))
            : Math.min(256, Math.max(1, cells.length));
          window.style.gridTemplateColumns = `repeat(${columnCount}, minmax(0, 1fr))`;
        });
      };

      const updateNavigation = () => {
        previousButton.disabled = currentChunk <= 0;
        nextButton.disabled = currentChunk >= contig.chunks.length - 1;
      };

      const renderChunk = async (chunkIndex) => {
        if (chunkIndex === loadedChunk) return;
        if (loadPromise) await loadPromise;
        if (chunkIndex === loadedChunk) return;
        loadPromise = fetch(contig.chunks[chunkIndex]).then((response) => {
          if (!response.ok) {
            throw new Error(`Could not load alignment window (${response.status})`);
          }
          return response.json();
        });
        const chunk = await loadPromise;
        loadPromise = null;
        loadedChunkData = chunk;
        const rawBases = [...chunk.raw];
        const correctedBases = [...chunk.corrected];
        const metadata = Array.isArray(chunk.metadata) ? chunk.metadata : [];
        while (rawBases.length < correctedBases.length) rawBases.push("-");
        while (correctedBases.length < rawBases.length) correctedBases.push("-");
        const flags = await global.PolishMLShapWasm.alignmentFlags(
          rawBases.map(baseCode),
          correctedBases.map(baseCode),
        );
        rows.forEach((row) => {
          const window = document.createElement("div");
          window.className = "alignment-window";
          const columnCount = Math.min(256, Math.max(1, rawBases.length));
          window.style.gridTemplateColumns = `repeat(${columnCount}, minmax(0, 1fr))`;
          if (row === connection) {
            flags.forEach((flag) => {
              const cell = document.createElement("span");
              cell.className = "conn-cell";
              cell.textContent = flag ? "x" : "|";
              window.appendChild(cell);
            });
          } else {
            const bases = row === raw ? rawBases : correctedBases;
            bases.forEach((base, index) => {
              const cell = document.createElement("span");
              const baseClass = base === "A" || base === "C" || base === "G" || base === "T"
                ? `base-${base.toLowerCase()}`
                : base === "-" ? "base-gap" : "base-unknown";
              const statusClass = flags[index] ? (row === raw ? "error" : "applied") : "plain";
              cell.className = `base-cell ${baseClass} ${statusClass}`;
              cell.dataset.position = chunk.start + index + 1;
              cell.setAttribute("aria-label", `${row === raw ? "Raw" : "Corrected"} base ${base}`);
              cell.title = base;
              const site = metadata[index];
              if (site) {
                cell.classList.add("selectable-base");
                cell.tabIndex = 0;
                cell.setAttribute("role", "button");
                const selectBase = () => global.dispatchEvent(new CustomEvent("polishml:base-selected", {
                  detail: { ...site, contig: contig.contig },
                }));
                cell.addEventListener("click", selectBase);
                cell.addEventListener("keydown", (event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    selectBase();
                  }
                });
              }
              window.appendChild(cell);
            });
          }
          row.content.replaceChildren(window);
        });
        updateAlignmentFocus();
        loadedChunk = chunkIndex;
        currentChunk = chunkIndex;
        const windowLength = Math.max(rawBases.length, correctedBases.length);
        const windowEnd = Math.min(chunk.start + windowLength, contig.length);
        positionOutput.textContent = `${(chunk.start + 1).toLocaleString()}–${windowEnd.toLocaleString()}`;
        global.dispatchEvent(new CustomEvent("polishml:alignment-window-changed", {
          detail: { contig: contig.contig, start: chunk.start + 1, end: windowEnd, window: Math.floor(chunk.start / 256) + 1 },
        }));
        updateNavigation();
      };

      const moveToChunk = async (delta) => {
        const nextChunk = Math.max(0, Math.min(contig.chunks.length - 1, currentChunk + delta));
        if (nextChunk === currentChunk) return;
        previousButton.disabled = true;
        nextButton.disabled = true;
        try {
          await renderChunk(nextChunk);
        } finally {
          updateNavigation();
        }
      };

      previousButton.addEventListener("click", () => void moveToChunk(-1));
      nextButton.addEventListener("click", () => void moveToChunk(1));
      global.addEventListener("polishml:base-selected", (event) => {
        container.querySelectorAll(".selected-base").forEach((cell) => cell.classList.remove("selected-base"));
        container.querySelectorAll(`[data-position="${event.detail.position}"]`).forEach((cell) => cell.classList.add("selected-base"));
      });
      global.addEventListener("polishml:alignment-focus", (event) => {
        if (event.detail.contig !== contig.contig) return;
        focusRange = { start: Number(event.detail.start), end: Number(event.detail.end) };
        updateAlignmentFocus();
      });
      global.addEventListener("polishml:location-requested", async (event) => {
        const request = event.detail;
        if (request.contig !== contig.contig) return;
        const requestedWindow = Number(request.window);
        const requestedPosition = Number(request.position);
        const targetPosition = Number.isFinite(requestedPosition) && requestedPosition > 0
          ? requestedPosition
          : Math.max(1, (requestedWindow - 1) * 256 + 1);
        const chunkIndex = Math.min(
          contig.chunks.length - 1,
          Math.max(0, Math.floor((targetPosition - 1) / manifest.chunkBases)),
        );
        await renderChunk(Math.max(0, chunkIndex));
        const metadata = Array.isArray(loadedChunkData?.metadata) ? loadedChunkData.metadata : [];
        const metadataIndex = targetPosition - loadedChunkData.start - 1;
        const site = metadata[metadataIndex];
        if (site) {
          global.dispatchEvent(new CustomEvent("polishml:base-selected", {
            detail: { ...site, contig: contig.contig },
          }));
        }
      });
      await renderChunk(0);
    }
  }

  global.PolishMLDash = {
    renderShapChart,
    renderKernelShap,
    renderPileupTable,
    renderReadSupport,
    initializeLazyAlignments,
    COLORS,
  };
})(window);
