const colors = {
  signal: "#ff8a3d",
  goal: "#7fb6ff",
  muted: "#6f685c",
  red: "#ff6b5e",
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
    ctx.strokeStyle = "#1a1916";
    for (let x = 0; x <= course.width; x += Math.max(50, course.width / 30))
      line([x, 0], [x, course.height], "#1a1916");
    for (let y = 0; y <= course.height; y += Math.max(50, course.height / 20))
      line([0, y], [course.width, y], "#1a1916");
    ctx.strokeStyle = "#3a372f";
    ctx.strokeRect(
      offsetX,
      offsetY,
      course.width * scale,
      course.height * scale,
    );
    ctx.font = '10px "IBM Plex Mono", Consolas, monospace';
    ctx.textAlign = "center";
    for (const [index, checkpoint] of course.checkpoints.entries()) {
      ctx.setLineDash([3, 5]);
      circle(
        checkpoint.center,
        checkpoint.radius * scale,
        "#ece6da05",
        "#4a4538",
      );
      ctx.setLineDash([]);
      label(String(index + 1).padStart(2, "0"), checkpoint.center, "#7d7566");
    }
    for (const obstacle of course.obstacles) {
      const [x, y] = project([obstacle.x, obstacle.y + obstacle.height]);
      ctx.fillStyle = "#24221d";
      ctx.fillRect(x, y, obstacle.width * scale, obstacle.height * scale);
      ctx.strokeStyle = "#4d483d";
      ctx.strokeRect(x, y, obstacle.width * scale, obstacle.height * scale);
    }
    const goal = course.goal;
    circle(
      goal.center,
      Math.max(8, goal.radius * scale),
      "#7fb6ff14",
      colors.goal,
    );
    circle(goal.center, 2.5, colors.goal);
    label(
      "GOAL",
      goal.center,
      colors.goal,
      0,
      -Math.max(8, goal.radius * scale) - 10,
    );
    circle(course.spawn, 4, "#ff8a3d20", "#7d7566");
    label("START", course.spawn, "#8a8274", 0, 19);
    for (const agent of population) {
      circle(
        agent.final_state.position,
        agent.index === replay?.agent.index ? 3 : 2.5,
        agent.final_state.status === "reached" ? colors.goal : "#8a827488",
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
      ctx.strokeStyle = "#ff8a3d88";
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
        line(state.position, end, "#ff8a3d42");
        circle(end, 1.6, "#ff8a3d99");
      });
    }
    const radius = Math.max(5, replay.agent_config.radius * scale);
    const [x, y] = project(state.position);
    circle(state.position, radius + 5, "#ff8a3d18");
    ctx.lineWidth = 1.5;
    circle(state.position, radius, "#2a170a", colors.signal);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(
      x + Math.cos(state.heading) * (radius + 4),
      y - Math.sin(state.heading) * (radius + 4),
    );
    ctx.strokeStyle = colors.signal;
    ctx.stroke();
  }
}
