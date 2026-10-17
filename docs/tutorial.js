// The first-visit walkthrough: walk to a tree, chop two, place a table, then go for a diamond. Each step moves on from what
// the frames show (hud.js: the wood count, the tiles around the player) and from the reward head, or after enough moves
// anyway, so nobody gets stuck. Two looks: 'a' a bubble next to the target, 'b' one caption over the top of the frame;
// both outline the target.
import { tiles, route, STEP } from './hud.js';

const ARROW = { 1: '←', 2: '→', 3: '↑', 4: '↓' }, KEY = { 1: 'A', 2: 'D', 3: 'W', 4: 'S' };
const BUDGET = [26, 26, 0, 16];   // moves on a step before the next one shows anyway (step 2 moves on by itself)
const CHOP_HINT = 14;             // moves without reaching a tree before the chop hint shows anyway
const SURE = 600;                 // a wood count is taken from one frame when the second best digit is this much worse

export function tutorial({ root, canvas, touch, look = 'a', top, face, onEnd }) {
  const layer = document.createElement('div');
  layer.className = `coach look-${look}`;
  layer.innerHTML = '<div class="ring"></div><div class="bub"></div><button class="skip" data-a="skip">skip</button>';
  root.append(layer);
  const ring = layer.querySelector('.ring'), bub = layer.querySelector('.bub');
  let step = 0, since = 0, got = 0, wood = 0, last = -1, shown = null, timer = 0, gone = false;

  function box(t) {  // a target as a box in the root's pixels: an element, or [x, y, w, h] in canvas pixels
    const R = root.getBoundingClientRect();
    if (t instanceof Element) {
      const r = t.getBoundingClientRect();
      return { x: r.left - R.left, y: r.top - R.top, w: r.width, h: r.height };
    }
    const C = canvas.getBoundingClientRect(), s = C.width / 63;
    return { x: C.left - R.left + t[0] * s, y: C.top - R.top + t[1] * s, w: t[2] * s, h: t[3] * s };
  }

  function draw() {  // the hint on show, placed for the current layout
    if (!shown || gone) {
      layer.hidden = true;
      for (const el of root.querySelectorAll('.cue')) el.classList.remove('cue');
      return;
    }
    layer.hidden = false;
    const { text, target, glyph = '', flash = false, cue = null, sides = ['top', 'bottom', 'left', 'right'] } = shown;
    for (const el of root.querySelectorAll('.cue')) if (el !== cue) el.classList.remove('cue');
    cue?.classList.add('cue');   // a key that does the same, pulsing too
    const R = root.getBoundingClientRect(), C = box(canvas);
    if (target) {
      const t = box(target), pad = 3;
      Object.assign(ring.style, { left: t.x - pad + 'px', top: t.y - pad + 'px', width: t.w + 2 * pad + 'px', height: t.h + 2 * pad + 'px',
        fontSize: Math.round(t.h * 0.8) + 'px' });
      ring.textContent = glyph;
      ring.className = 'ring' + (flash ? ' flash' : '');
      ring.hidden = false;
    } else ring.hidden = true;
    bub.innerHTML = text;
    bub.className = 'bub';
    const sk = layer.querySelector('.skip');   // the way out, on the empty corner of the inventory strip
    Object.assign(sk.style, { left: C.x + C.w - sk.offsetWidth - 4 + 'px', top: C.y + C.h - sk.offsetHeight - 4 + 'px' });
    if (look === 'b' || !target) {   // one caption over the top of the frame (the score strip and the top of the map)
      if (look === 'b') {
        Object.assign(bub.style, { left: C.x + 'px', top: C.y + 'px', width: C.w + 'px', minHeight: Math.max(44, top * C.w / 63) + 'px' });
        bub.classList.add('cap');
      } else {   // no target: the middle of the map
        const w = bub.offsetWidth, h = bub.offsetHeight;
        Object.assign(bub.style, { left: C.x + (C.w - w) / 2 + 'px', top: C.y + C.w * (top + 14) / 63 - h / 2 + 'px', width: '', minHeight: '' });
      }
      return;
    }
    // a bubble on the map stays on the frame, one on a key on the screen; on one line if it fits on a side, else on two
    const t = box(target), gap = 14, cx = t.x + t.w / 2, cy = t.y + t.h / 2, map = Array.isArray(target);
    const lo = map ? C.x : 6 - R.left, hi = map ? C.x + C.w : innerWidth - R.left - 6;
    let w, h, at;
    const measure = wrap => {
      Object.assign(bub.style, { width: '', minHeight: '', left: '0px', top: '0px', whiteSpace: wrap ? 'normal' : '',
        maxWidth: wrap ? Math.max(t.x - gap - lo, hi - t.x - t.w - gap, 120) + 'px' : '' });
      [w, h] = [bub.offsetWidth, bub.offsetHeight];
      at = { top: [cx - w / 2, t.y - h - gap], bottom: [cx - w / 2, t.y + t.h + gap], left: [t.x - w - gap, cy - h / 2],
        right: [t.x + t.w + gap, cy - h / 2] };
    };
    const fits = s => s === 'top' ? R.top + at.top[1] > 4 && (!map || at.top[1] >= C.y)
      : s === 'bottom' ? !map || at.bottom[1] + h <= C.y + C.h : at[s][0] >= lo && at[s][0] + w <= hi;
    const side = sides.find(s => (measure(false), fits(s)) || (s === 'left' || s === 'right') && (measure(true), fits(s)))
      ?? (measure(false), sides[0]);
    let [x, y] = at[side];
    if (side === 'top' || side === 'bottom') x = Math.min(Math.max(x, lo), hi - w);
    Object.assign(bub.style, { left: x + 'px', top: y + 'px' });
    bub.classList.add(side);
    bub.style.setProperty('--ax', Math.min(Math.max(cx - x, 14), w - 14) + 'px');
    bub.style.setProperty('--ay', Math.min(Math.max(cy - y, 14), h - 14) + 'px');
  }

  function show(s) {  // a hint: { text, target, glyph, flash, sides }; the target is scrolled into view if it is off screen
    shown = s;
    draw();
    const el = s?.target instanceof Element ? s.target : null, r = el?.getBoundingClientRect();
    if (r && (r.bottom > innerHeight || r.top < 0)) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  const tile = (r, c) => [7 * c, top + 7 * r, 7, 7];   // a map tile in canvas pixels
  const key = a => root.querySelector(`.key[data-a="${a}"]`);
  const move = d => ({ text: touch ? `Tap to move ${ARROW[d]}` : `Press ${KEY[d]} to move ${ARROW[d]}`,
    target: tile(3 + STEP[d][0], 4 + STEP[d][1]), glyph: ARROW[d], cue: key(d), sides: d === 3 || d === 4 ? ['right', 'left', 'top'] : ['top', 'bottom'] });
  const chop = again => ({ text: again ? 'Chop one more' : touch ? 'Tap ACTION to chop' : 'Press SPACE to chop', target: key(5),
    sides: ['top', 'left', 'right', 'bottom'] });

  function guide(rgb, again) {  // to the nearest tree and chop it
    const way = route(tiles(rgb), face);
    if (way?.facing || step === 0 && since >= CHOP_HINT) return show(chop(again));   // next to it, or the hint anyway
    show(way ? move(way.next) : { text: 'Find a tree', target: null });
  }

  function next(to) {  // to a step: 0 first wood, 1 second wood, 2 craft, 3 table, 4 diamond, 5 done
    step = to;
    since = 0;
    clearTimeout(timer);
    if (step === 2) {
      show({ text: 'Now craft!', target: root.querySelector('.acts'), flash: true, sides: ['top', 'left', 'bottom'] });
      timer = setTimeout(() => step === 2 && next(3), 2200);   // three flashes, then the table
    } else if (step === 3) {
      show({ text: touch ? 'Tap to place table' : 'Press T for table', target: key(8), sides: ['top', 'left', 'bottom', 'right'] });
    } else if (step === 4) {
      // the diamond's inventory slot (column 1, row 2 of the strip, as Craftax renders it), where it shows once you hold one
      show({ text: 'Now find a diamond! <img src="icons/diamond.png" alt="">', target: tile(8, 0), sides: ['top'] });
      timer = setTimeout(() => step === 4 && end(), 12000);
    } else if (step >= 5) end();
  }

  function count(read, a, r) {  // the wood held: a read sure on one frame, or the same on two frames in a row
    if (read.margin > SURE || read.n === last) wood = read.n;
    last = read.n;
    if (a === 5 && r > 0.5 && step === 0) wood = Math.max(wood, 1);   // the reward head: the first wood is an achievement
    return wood;
  }

  function frame({ rgb, a, r, inv, over }) {  // after every step; over: the run has ended
    if (gone) return;
    if (a >= 1 && a <= 4) face = a;
    since++;
    if (over) { shown = null; return draw(); }   // the run is over: only a new world goes on, and it ends the tutorial
    const before = wood, w = count(inv.wood, a, r);
    if (step <= 1) {
      got = Math.max(got, w);
      if (got >= 2) return next(2);
      if (step === 0 && got >= 1) {
        next(1);
        return show({ text: 'Wood! Chop one more', target: [35, top + 49, 7, 7], sides: ['top'], wood: true });
      }
      if (since >= BUDGET[step]) return next(step + 1);
      if (step === 1 && since <= 2 && shown?.wood) return draw();   // the wood stays pointed out for two moves
      return guide(rgb, step === 1);
    }
    if (step === 2) {
      if (a >= 6) next(3);
      else return draw();
    }
    if (step === 3) {
      const facing = route(tiles(rgb), face, 'table')?.facing;
      if (a === 8 && (r > 0.5 || w <= before - 2 || facing)) return next(4);
      if (since >= BUDGET[3]) return next(4);
      return draw();
    }
    if (step === 4 && since >= 8) return end();
    draw();
  }

  function begin(rgb, inv) {  // the first frame of the tutorial world
    const read = inv.wood;
    wood = got = read.n;
    last = read.n;
    if (got >= 2) return next(2);
    next(got);
    if (step === 0) guide(rgb, false);
    else show({ text: 'Chop one more', target: [35, top + 49, 7, 7], sides: ['top'] });
  }

  function end(skipped = false) {
    if (gone) return;
    gone = true;
    clearTimeout(timer);
    draw();
    layer.remove();
    onEnd?.(skipped);
  }

  return { begin, frame, end, layout: draw, get on() { return !gone; }, get step() { return step; } };
}
