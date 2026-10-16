// Auswahl-Modell (Formen in Bildkoordinaten) und Hilfen zum Zeichnen.
//
// Eine Auswahl ist eine Folge von Formen, die nacheinander hinzugefügt oder abgezogen
// werden – genau so rechnet auch der Server die Maske aus.

export class Selection {
  constructor(shapes = [], feather = 0) {
    this.shapes = shapes;
    this.feather = feather;
  }

  get empty() {
    return !this.shapes.some((s) => s.mode === 'add');
  }

  toJSON() {
    return this.empty ? null : { shapes: this.shapes, feather: this.feather, invert: false };
  }

  /** Begrenzung der hinzugefügten Formen (ganzzahlig, im Bild geclippt) oder null. */
  bounds(width, height) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const s of this.shapes) {
      if (s.mode !== 'add') continue;
      const b = shapeBounds(s);
      x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]);
      x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
    }
    if (x0 === Infinity) return null;
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(width, Math.ceil(x1)); y1 = Math.min(height, Math.ceil(y1));
    return x1 > x0 && y1 > y0 ? [x0, y0, x1, y1] : null;
  }
}

export function fullRect(width, height, mode = 'add') {
  return { type: 'rect', x: 0, y: 0, w: width, h: height, mode };
}

function isFullRect(s, width, height) {
  return s.type === 'rect' && s.mode === 'add' && s.x <= 0 && s.y <= 0 && s.x + s.w >= width && s.y + s.h >= height;
}

/** Kehrt die Auswahl exakt um: Komplement = ganzes Bild, danach alle Formen mit umgekehrtem Modus. */
export function invertShapes(shapes, width, height) {
  const flip = (list) => list.map((s) => ({ ...s, mode: s.mode === 'add' ? 'subtract' : 'add' }));
  if (shapes.length && isFullRect(shapes[0], width, height)) return flip(shapes.slice(1));
  return [fullRect(width, height), ...flip(shapes)];
}

export function shapeBounds(s) {
  if (s.type === 'rect' || s.type === 'ellipse') {
    return [Math.min(s.x, s.x + s.w), Math.min(s.y, s.y + s.h), Math.max(s.x, s.x + s.w), Math.max(s.y, s.y + s.h)];
  }
  const r = s.type === 'brush' ? s.size / 2 : 0;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of s.points) {
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  return [x0 - r, y0 - r, x1 + r, y1 + r];
}

/**
 * Legt den Pfad einer Form an. `map` rechnet Bild- in Zeichenkoordinaten um,
 * `scale` ist der Maßstab für Längen (Radien, Pinselbreite).
 */
export function tracePath(ctx, s, map, scale) {
  ctx.beginPath();
  if (s.type === 'rect') {
    const [x0, y0, x1, y1] = shapeBounds(s);
    const [a, b] = map(x0, y0);
    ctx.rect(a, b, (x1 - x0) * scale, (y1 - y0) * scale);
  } else if (s.type === 'ellipse') {
    const [x0, y0, x1, y1] = shapeBounds(s);
    const [cx, cy] = map((x0 + x1) / 2, (y0 + y1) / 2);
    ctx.ellipse(cx, cy, Math.max(0.01, ((x1 - x0) / 2) * scale), Math.max(0.01, ((y1 - y0) / 2) * scale), 0, 0, Math.PI * 2);
  } else {
    s.points.forEach(([x, y], i) => {
      const [a, b] = map(x, y);
      if (i === 0) ctx.moveTo(a, b);
      else ctx.lineTo(a, b);
    });
    if (s.type === 'polygon') ctx.closePath();
  }
}

/** Füllt (oder bei Pinsel: zieht) eine Form in den aktuellen Kontext. */
export function paintShape(ctx, s, map, scale) {
  if (s.type === 'brush') {
    const width = Math.max(1, s.size * scale);
    if (s.points.length === 1) {
      const [a, b] = map(...s.points[0]);
      ctx.beginPath();
      ctx.arc(a, b, width / 2, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    tracePath(ctx, s, map, scale);
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    return;
  }
  tracePath(ctx, s, map, scale);
  ctx.fill('evenodd');
}
