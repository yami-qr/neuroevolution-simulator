import { chartGeometry } from "./data.js";

const namespace = "http://www.w3.org/2000/svg";
function svgElement(tag, attributes = {}, text) {
  const node = document.createElementNS(namespace, tag);
  for (const [key, value] of Object.entries(attributes))
    node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

const number = (value) =>
  new Intl.NumberFormat("en-GB", {
    maximumFractionDigits: 1,
    notation: Math.abs(value) >= 10000 ? "compact" : "standard",
  }).format(value);

export class Chart {
  constructor(host, { completion = false, select }) {
    this.host = host;
    this.completion = completion;
    this.select = select;
    this.rows = [];
    this.observer = new ResizeObserver(() => this.render());
    this.observer.observe(host);
  }

  update(rows, selected) {
    this.rows = rows;
    this.selected = selected;
    this.render();
  }

  render() {
    const { rows, host, completion } = this;
    if (!rows.length) {
      const message = document.createElement("p");
      message.className = "chart-empty";
      message.textContent = "Your first saved generation starts the curve.";
      host.replaceChildren(message);
      return;
    }
    const width = Math.max(260, host.clientWidth);
    const height = host.clientHeight || 210;
    const geometry = chartGeometry(rows, width, height, completion);
    const { x, y, min, max, last, left, right, top, bottom } = geometry;
    const svg = svgElement("svg", {
      viewBox: `0 0 ${width} ${height}`,
      "aria-hidden": "true",
    });
    for (let index = 0; index <= 4; index++) {
      const value = min + ((max - min) * index) / 4;
      svg.append(
        svgElement("line", {
          x1: left,
          x2: width - right,
          y1: y(value),
          y2: y(value),
          class: "gridline",
        }),
      );
      svg.append(
        svgElement(
          "text",
          { x: left - 10, y: y(value) + 4, "text-anchor": "end" },
          completion ? `${Math.round(value * 100)}%` : number(value),
        ),
      );
    }
    const tickCount = Math.min(5, last + 1);
    const ticks = [
      ...new Set(
        Array.from({ length: tickCount }, (_, i) =>
          Math.round((i * last) / Math.max(1, tickCount - 1)),
        ),
      ),
    ];
    for (const tick of ticks)
      svg.append(
        svgElement(
          "text",
          { x: x(tick), y: height - 12, "text-anchor": "middle" },
          tick,
        ),
      );
    const series = completion
      ? [["completion_rate", "#e6c489"]]
      : [
          ["mean_fitness", "#9badcc"],
          ["best_fitness", "#a2e3ba"],
        ];
    for (const [key, color] of series) {
      const points = rows
        .map((row) => `${x(row.generation)},${y(row[key])}`)
        .join(" ");
      svg.append(
        svgElement("polyline", {
          points,
          fill: "none",
          stroke: color,
          "stroke-width": 2,
          "stroke-linejoin": "round",
        }),
      );
      for (const row of rows.length <= 60
        ? rows
        : rows.filter((row) => row.generation === this.selected)) {
        svg.append(
          svgElement("circle", {
            cx: x(row.generation),
            cy: y(row[key]),
            r: row.generation === this.selected ? 4 : 2.5,
            fill: color,
            stroke: "#181e20",
            "stroke-width": 1.5,
          }),
        );
      }
    }
    const selection = rows.find((row) => row.generation === this.selected);
    if (selection)
      svg.append(
        svgElement("line", {
          x1: x(selection.generation),
          x2: x(selection.generation),
          y1: top,
          y2: height - bottom,
          stroke: "#899995",
          "stroke-dasharray": "3 5",
          opacity: 0.5,
        }),
      );
    const cursor = svgElement("line", {
      y1: top,
      y2: height - bottom,
      stroke: "#899995",
      opacity: 0,
    });
    svg.append(cursor);
    const tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip";
    tooltip.hidden = true;
    const nearest = (event) => {
      const rectangle = svg.getBoundingClientRect();
      const generation = geometry.generationAt(
        ((event.clientX - rectangle.left) * width) / rectangle.width,
      );
      return rows.reduce((best, row) =>
        Math.abs(row.generation - generation) <
        Math.abs(best.generation - generation)
          ? row
          : best,
      );
    };
    svg.addEventListener("pointermove", (event) => {
      const row = nearest(event);
      cursor.setAttribute("x1", x(row.generation));
      cursor.setAttribute("x2", x(row.generation));
      cursor.setAttribute("opacity", "0.6");
      tooltip.textContent = completion
        ? `Generation ${row.generation} · ${(row.completion_rate * 100).toFixed(1)}% reached the goal`
        : `Generation ${row.generation} · Best ${number(row.best_fitness)} · Mean ${number(row.mean_fitness)}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(8, Math.min(x(row.generation) - 70, width - tooltip.offsetWidth - 8))}px`;
    });
    svg.addEventListener("pointerleave", () => {
      tooltip.hidden = true;
      cursor.setAttribute("opacity", "0");
    });
    svg.addEventListener("click", (event) =>
      this.select(nearest(event).generation),
    );
    host.replaceChildren(svg, tooltip);
  }
}
