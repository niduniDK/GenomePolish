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

  // ---------------------------------------------------------------------
  // SHAP contribution chart — diverging horizontal bar, one bar per feature
  // ---------------------------------------------------------------------
  function renderShapChart(containerId, features) {
    if (!features || !features.length) return;

    // Sort so the largest |contribution| sits at the top (Plotly draws
    // bottom-to-top for horizontal bars, so reverse after sorting).
    const sorted = [...features].sort((a, b) => Math.abs(a.value) - Math.abs(b.value));

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
      height: Math.max(220, sorted.length * 34),
      xaxis: {
        range: [-maxAbs * 1.35, maxAbs * 1.35],
        zeroline: true,
        zerolinecolor: "rgba(255,255,255,0.3)",
        zerolinewidth: 1,
        gridcolor: COLORS.grid,
        showticklabels: false,
      },
      yaxis: {
        automargin: true,
        showgrid: false,
      },
      bargap: 0.35,
    };

    Plotly.newPlot(containerId, [trace], layout, PLOTLY_CONFIG);
  }

  async function initializeWindowShap(
    selectorId,
    positionId,
    chartId,
    rows,
    featureNames,
    baseline,
    fallbackFeatures,
  ) {
    const selector = document.getElementById(selectorId);
    const position = document.getElementById(positionId);
    if (!selector || !rows.length || !featureNames.length) {
      renderShapChart(chartId, fallbackFeatures);
      return;
    }

    selector.max = String(rows.length - 1);
    const update = async () => {
      const row = rows[Number(selector.value)];
      position.textContent = row.position;
      const values = await global.PolishMLShapWasm.contributions(
        row.features,
        baseline,
      );
      renderShapChart(
        chartId,
        featureNames.map((feature, index) => ({
          feature,
          value: values[index],
        })),
      );
    };

    selector.addEventListener("input", update);
    await update();
  }

  // ---------------------------------------------------------------------
  // Pileup table — custom HTML (Plotly's table trace can't do per-cell
  // badge styling cleanly), built once from JSON.
  // ---------------------------------------------------------------------
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
          <tr>
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

  function baseCode(base) {
    if (base === "A") return 1;
    if (base === "C") return 2;
    if (base === "G") return 3;
    if (base === "T") return 4;
    if (base === "-") return 5;
    return 0;
  }

  function createAlignmentRow(label, className, contentWidth) {
    const row = document.createElement("div");
    row.className = "align-row";
    const labelElement = document.createElement("span");
    labelElement.className = "align-label";
    labelElement.textContent = label;
    const scroll = document.createElement("div");
    scroll.className = "alignment-scroll";
    scroll.dataset.row = className;
    const content = document.createElement("div");
    content.className = "alignment-content";
    content.style.width = `${contentWidth}px`;
    scroll.appendChild(content);
    row.append(labelElement, scroll);
    return { row, scroll, content };
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
      const contentWidth = contig.length * cellWidth;
      const raw = createAlignmentRow("Raw", "raw", contentWidth);
      const connection = createAlignmentRow("Connection", "connection", contentWidth);
      const corrected = createAlignmentRow("Corrected", "corrected", contentWidth);
      container.append(raw.row, connection.row, corrected.row);
      const rows = [raw, connection, corrected];
      const positionSlider = document.querySelector(
        `.alignment-position[data-contig-index="${container.dataset.contigIndex}"]`,
      );
      const positionOutput = positionSlider
        ?.closest(".contig-alignment")
        ?.querySelector(".alignment-position-value");
      let loadedChunk = -1;
      let loadPromise = null;

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
        const rawBases = [...chunk.raw];
        const correctedBases = [...chunk.corrected];
        while (rawBases.length < correctedBases.length) rawBases.push("-");
        while (correctedBases.length < rawBases.length) correctedBases.push("-");
        const flags = await global.PolishMLShapWasm.alignmentFlags(
          rawBases.map(baseCode),
          correctedBases.map(baseCode),
        );
        rows.forEach((row) => {
          const window = document.createElement("div");
          window.className = "alignment-window";
          window.style.left = `${chunk.start * cellWidth}px`;
          if (row === raw || row === corrected) {
            const bases = row === raw ? rawBases : correctedBases;
            bases.forEach((base, index) => {
              const cell = document.createElement("span");
              cell.className = `base-cell ${flags[index] ? (row === raw ? "error" : "applied") : "plain"}`;
              cell.textContent = base;
              window.appendChild(cell);
            });
          } else {
            flags.forEach((flag) => {
              const cell = document.createElement("span");
              cell.className = "conn-cell";
              cell.textContent = flag ? "x" : "|";
              window.appendChild(cell);
            });
          }
          row.content.replaceChildren(window);
        });
        loadedChunk = chunkIndex;
      };

      const syncScroll = (source) => {
        const basePosition = Math.round(source.scrollLeft / cellWidth);
        if (positionSlider && Number(positionSlider.value) !== basePosition) {
          positionSlider.value = String(basePosition);
        }
        if (positionOutput) {
          const start = basePosition + 1;
          const end = Math.min(basePosition + manifest.windowBases, contig.length);
          positionOutput.textContent = `${start.toLocaleString()}–${end.toLocaleString()}`;
        }
        rows.forEach((row) => {
          if (row.scroll !== source) row.scroll.scrollLeft = source.scrollLeft;
        });
        const chunkIndex = Math.min(
          contig.chunks.length - 1,
          Math.floor(source.scrollLeft / (manifest.windowBases * cellWidth)),
        );
        void renderChunk(Math.max(0, chunkIndex));
      };
      const moveToPosition = () => {
        const basePosition = Number(positionSlider.value);
        rows.forEach((row) => {
          row.scroll.scrollLeft = basePosition * cellWidth;
        });
        syncScroll(rows[0].scroll);
      };
      rows.forEach((row) => row.scroll.addEventListener("scroll", () => syncScroll(row.scroll)));
      positionSlider?.addEventListener("input", moveToPosition);
      await renderChunk(0);
    }
  }

  global.PolishMLDash = {
    renderShapChart,
    renderPileupTable,
    initializeWindowShap,
    initializeLazyAlignments,
    COLORS,
  };
})(window);
