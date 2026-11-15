// Shared browser data operations. Kept independent of the DOM for Node tests.
export class ApiClient {
  constructor(fetcher = (...args) => globalThis.fetch(...args)) {
    this.fetcher = fetcher;
  }

  async request(path, { signal, body, method = "GET", blob = false } = {}) {
    const timeout = AbortSignal.timeout(30000);
    const response = await this.fetcher(path, {
      method,
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) {
      let message = `Request failed (${response.status}).`;
      try {
        const { detail } = await response.json();
        if (typeof detail === "string") message = detail;
        else if (Array.isArray(detail))
          message = detail
            .map((item) => `${item.loc.slice(1).join(" · ")}: ${item.msg}`)
            .join(". ");
      } catch {
        // Gateways may return non-JSON errors.
      }
      throw new Error(message);
    }
    return blob ? response.blob() : response.json();
  }

  async pages(path, { signal, limit = 100 } = {}) {
    const result = [];
    let offset = 0;
    while (true) {
      const separator = path.includes("?") ? "&" : "?";
      const page = await this.request(
        `${path}${separator}offset=${offset}&limit=${limit}`,
        { signal },
      );
      result.push(...page.items);
      offset += page.items.length;
      if (!page.items.length || offset >= page.total) return result;
    }
  }
}

// Changing a selection invalidates both the request and its eventual response.
export class SelectionGate {
  next() {
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    return {
      signal: controller.signal,
      current: () =>
        this.controller === controller && !controller.signal.aborted,
    };
  }
}

export function frameIndex(frames, time) {
  if (!frames.length) return -1;
  let low = 0;
  let high = frames.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (frames[middle].time <= time) low = middle;
    else high = middle - 1;
  }
  return low;
}

export function advancePlayback(time, deltaSeconds, speed, duration) {
  const next = Math.min(duration, Math.max(0, time + deltaSeconds * speed));
  return { time: next, playing: next < duration };
}

export function fitnessCsv(rows) {
  const fields = [
    "generation",
    "population_size",
    "best_fitness",
    "mean_fitness",
    "completed_count",
    "completion_rate",
    "best_progress",
    "best_checkpoints",
  ];
  return (
    [
      fields.join(","),
      ...rows.map((row) => fields.map((key) => row[key]).join(",")),
    ].join("\r\n") + "\r\n"
  );
}

export function chartGeometry(rows, width, height, completion = false) {
  const left = 52,
    right = 18,
    top = 18,
    bottom = 34;
  let min = completion
    ? 0
    : Math.min(...rows.flatMap((row) => [row.best_fitness, row.mean_fitness]));
  let max = completion
    ? 1
    : Math.max(...rows.flatMap((row) => [row.best_fitness, row.mean_fitness]));
  if (!completion) {
    const padding = Math.max((max - min) * 0.12, 1);
    min -= padding;
    max += padding;
  }
  const last = Math.max(1, rows.at(-1)?.generation ?? 0);
  return {
    left,
    right,
    top,
    bottom,
    min,
    max,
    last,
    x: (generation) => left + (generation / last) * (width - left - right),
    y: (value) =>
      height - bottom - ((value - min) / (max - min)) * (height - top - bottom),
    generationAt: (pixel) =>
      Math.max(
        0,
        Math.min(last, ((pixel - left) / (width - left - right)) * last),
      ),
  };
}
