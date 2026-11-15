import test from "node:test";
import assert from "node:assert/strict";
import {
  ApiClient,
  SelectionGate,
  advancePlayback,
  chartGeometry,
  fitnessCsv,
  frameIndex,
} from "../../src/neuroevolution/server/static/data.js";

test("the default client calls browser fetch with its window receiver", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = function () {
    assert.equal(this, globalThis);
    return Promise.resolve(Response.json({ status: "ready" }));
  };
  try {
    assert.deepEqual(await new ApiClient().request("/ready"), {
      status: "ready",
    });
  } finally {
    globalThis.fetch = original;
  }
});

test("pagination preserves filters and consumes every page", async () => {
  const requests = [];
  const client = new ApiClient(async (url) => {
    requests.push(url);
    const offset = Number(
      new URL(url, "http://localhost").searchParams.get("offset"),
    );
    return Response.json({ items: offset === 0 ? [1, 2] : [3], total: 3 });
  });
  assert.deepEqual(
    await client.pages("/runs?experiment_id=abc", { limit: 2 }),
    [1, 2, 3],
  );
  assert.deepEqual(requests, [
    "/runs?experiment_id=abc&offset=0&limit=2",
    "/runs?experiment_id=abc&offset=2&limit=2",
  ]);
});

test("API errors retain validation context, including safe non-JSON fallback", async () => {
  const client = new ApiClient(async () =>
    Response.json(
      {
        detail: [
          { loc: ["body", "settings", "evolution"], msg: "Invalid population" },
        ],
      },
      { status: 422 },
    ),
  );
  await assert.rejects(
    client.request("/experiments"),
    /settings · evolution: Invalid population/,
  );
  const unavailable = new ApiClient(
    async () => new Response("Unavailable", { status: 503 }),
  );
  await assert.rejects(unavailable.request("/ready"), /503/);
});

test("changing selection aborts and invalidates stale results", () => {
  const gate = new SelectionGate();
  const old = gate.next();
  assert.equal(old.current(), true);
  const current = gate.next();
  assert.equal(old.signal.aborted, true);
  assert.equal(old.current(), false);
  assert.equal(current.current(), true);
});

test("recorded frames use simulation time and preserve terminal states", () => {
  const frames = [{ time: 0 }, { time: 0.1 }, { time: 1.5 }];
  assert.equal(frameIndex([], 0), -1);
  assert.equal(frameIndex(frames, -1), 0);
  assert.equal(frameIndex(frames, 0.8), 1);
  assert.equal(frameIndex(frames, 1.5), 2);
  assert.equal(frameIndex(frames, 99), 2);
  assert.deepEqual(advancePlayback(1, 0.2, 5, 1.5), {
    time: 1.5,
    playing: false,
  });
  assert.deepEqual(advancePlayback(0, 0.1, 1, 1.5), {
    time: 0.1,
    playing: true,
  });
});

test("chart axes support a single generation, negative and constant scores", () => {
  const rows = [{ generation: 0, best_fitness: -4, mean_fitness: -4 }];
  const geometry = chartGeometry(rows, 400, 210);
  assert.ok(Number.isFinite(geometry.y(-4)));
  assert.ok(geometry.min < -4 && geometry.max > -4);
  assert.equal(geometry.generationAt(-100), 0);
  assert.equal(geometry.generationAt(1000), 1);
  const completion = chartGeometry(rows, 400, 210, true);
  assert.ok(completion.y(1) < completion.y(0));
});

test("CSV exports actual metrics with stable headers and CRLF rows", () => {
  const csv = fitnessCsv([
    {
      generation: 0,
      population_size: 3,
      best_fitness: -2,
      mean_fitness: -5,
      completed_count: 0,
      completion_rate: 0,
      best_progress: 0.02,
      best_checkpoints: 0,
    },
  ]);
  assert.ok(csv.startsWith("generation,population_size,best_fitness"));
  assert.ok(csv.endsWith("0,3,-2,-5,0,0,0.02,0\r\n"));
});
