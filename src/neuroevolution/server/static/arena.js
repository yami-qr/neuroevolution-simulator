const colors = {
  mint: "#a2e3ba",
  gold: "#e6c489",
  muted: "#657570",
  red: "#e5a397",
};

export class Arena {
  constructor(canvas, { select, hover }) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.data = {};
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas);
    const nearest = (event) => {
      if (!this.project) return null;
      const bounds = canvas.getBoundingClientRect();
      const point = [event.clientX - bounds.left, event.clientY - bounds.top];
      let selected = null,
        distance = 12;
      for (const agent of this.data.population ?? []) {
        const position = this.project(agent.final_state.position);
        const gap = Math.hypot(position[0] - point[0], position[1] - point[1]);
        if (gap < distance) {
          distance = gap;
          selected = agent;
        }
      }
      return selected;
    };
    canvas.addEventListener("click", (event) => {
      const agent = nearest(event);
      if (agent) select(agent.index);
    });
    canvas.addEventListener("pointermove", (event) => {
      const agent = nearest(event);
      canvas.style.cursor = agent ? "pointer" : "default";
      hover(agent, event);
    });
    canvas.addEventListener("pointerleave", () => hover(null));
  }

  update(data) {
    this.data = data;
    this.draw();
  }

  draw() {
    const { canvas, context: ctx, data } = this;
    const width = canvas.clientWidth,
      height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(devicePixelRatio || 1, 3);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const {
      course,
      replay,
      frame,
      frameIndex = 0,
      population = [],
      sensors,
      trail,
    } = data;
    if (!course) return;
    const padding = width < 400 ? 22 : 30;
    const scale = Math.min(
      (width - 2 * padding) / course.width,
      (height - 2 * padding) / course.height,
    );
    const offsetX = (width - course.width * scale) / 2;
    const offsetY = (height - course.height * scale) / 2;
    const project = ([x, y]) => [
      offsetX + x * scale,
      height - offsetY - y * scale,
    ];
    this.project = project;
    const circle = (point, radius, fill, stroke) => {
      const [x, y] = project(point);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (fill) {
        ctx.fillStyle = fill;
        ctx.fill();
      }
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.stroke();
      }
    };
    const line = (a, b, color) => {
      ctx.beginPath();
      ctx.moveTo(...project(a));
      ctx.lineTo(...project(b));
      ctx.strokeStyle = color;
      ctx.stroke();
    };
    const label = (text, point, color, xOffset = 0, yOffset = 0) => {
      const [x, y] = project(point);
      ctx.fillStyle = color;
      ctx.fillText(text, x + xOffset, y + yOffset);
    };
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#1e292b";
    for (let x = 0; x <= course.width; x += Math.max(50, course.width / 30))
      line([x, 0], [x, course.height], "#1e292b");
    for (let y = 0; y <= course.height; y += Math.max(50, course.height / 20))
      line([0, y], [course.width, y], "#1e292b");
    ctx.strokeStyle = "#40504d";
    ctx.strokeRect(
      offsetX,
      offsetY,
      course.width * scale,
      course.height * scale,
    );
    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "center";
    for (const [index, checkpoint] of course.checkpoints.entries()) {
      ctx.setLineDash([3, 5]);
      circle(
        checkpoint.center,
        checkpoint.radius * scale,
        "#a2e3ba04",
        "#3c5045",
      );
      ctx.setLineDash([]);
      label(String(index + 1).padStart(2, "0"), checkpoint.center, "#6a8174");
    }
    for (const obstacle of course.obstacles) {
      const [x, y] = project([obstacle.x, obstacle.y + obstacle.height]);
      ctx.fillStyle = "#2b3638";
      ctx.fillRect(x, y, obstacle.width * scale, obstacle.height * scale);
      ctx.strokeStyle = "#546060";
      ctx.strokeRect(x, y, obstacle.width * scale, obstacle.height * scale);
    }
    const goal = course.goal;
    circle(
      goal.center,
      Math.max(8, goal.radius * scale),
      "#e6c48912",
      colors.gold,
    );
    circle(goal.center, 2.5, colors.gold);
    label(
      "GOAL",
      goal.center,
      colors.gold,
      0,
      -Math.max(8, goal.radius * scale) - 10,
    );
    circle(course.spawn, 4, "#a2e3ba20", "#6a8174");
    label("START", course.spawn, "#81978a", 0, 19);
    for (const agent of population) {
      circle(
        agent.final_state.position,
        agent.index === replay?.agent.index ? 3 : 2.5,
        agent.final_state.status === "reached" ? colors.gold : "#82938d80",
      );
    }
    if (!frame) return;
    const state = frame.state;
    if (trail && replay) {
      ctx.beginPath();
      for (let index = 0; index <= frameIndex; index++) {
        const point = project(replay.frames[index].state.position);
        if (index === 0) ctx.moveTo(...point);
        else ctx.lineTo(...point);
      }
      ctx.strokeStyle = "#a2e3ba88";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    if (sensors && replay) {
      const { ray_angles, max_distance } = replay.sensor_config;
      ray_angles.forEach((angle, index) => {
        const direction = state.heading + angle;
        const distance = frame.inputs[index] * max_distance;
        const end = [
          state.position[0] + Math.cos(direction) * distance,
          state.position[1] + Math.sin(direction) * distance,
        ];
        ctx.lineWidth = 1;
        line(state.position, end, "#a2e3ba42");
        circle(end, 1.6, "#a2e3ba99");
      });
    }
    const radius = Math.max(5, replay.agent_config.radius * scale);
    const [x, y] = project(state.position);
    circle(state.position, radius + 5, "#a2e3ba10");
    ctx.lineWidth = 1.5;
    circle(state.position, radius, "#1c3828", colors.mint);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(
      x + Math.cos(state.heading) * (radius + 4),
      y - Math.sin(state.heading) * (radius + 4),
    );
    ctx.strokeStyle = colors.mint;
    ctx.stroke();
  }
}
