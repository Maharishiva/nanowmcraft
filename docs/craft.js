// What each place and make action needs in Craftax-Classic (craftax 1.5.0, game_logic.py), and a bubble in the tutorial's
// look over its key the first time it is pressed on this browser. "Next to" is one of the 8 tiles around the player.
const I = n => `<img src="icons/${n}.png" alt="${n}">`;
const NEAR_T = 'next to a table ' + I('place_table'), NEAR_TF = `next to a table ${I('place_table')} and a furnace ${I('place_furnace')}`;
const IRON = `1 ${I('wood')} 1 ${I('stone')} 1 ${I('coal')} 1 ${I('iron')}`;
export const RECIPES = {
  7: ['Place stone', `1 ${I('stone')}`, 'on grass, sand, path, water or lava'],
  8: ['Table', `2 ${I('wood')}`, 'chop trees for wood'],
  9: ['Furnace', `1 ${I('stone')}`, 'mine stone with a wood pickaxe'],
  10: ['Plant', `1 ${I('sapling')}`, 'on grass; hit grass for saplings'],
  11: ['Wood pickaxe', `1 ${I('wood')}`, `${NEAR_T}; mines stone and coal`],
  12: ['Stone pickaxe', `1 ${I('wood')} 1 ${I('stone')}`, `${NEAR_T}; mines iron`],
  13: ['Iron pickaxe', IRON, `${NEAR_TF}; mines diamonds`],
  14: ['Wood sword', `1 ${I('wood')}`, NEAR_T],
  15: ['Stone sword', `1 ${I('wood')} 1 ${I('stone')}`, NEAR_T],
  16: ['Iron sword', IRON, NEAR_TF],
};
const KEY = 'tinywmcraft.recipes', SHOW_MS = 5000;

export function recipes(root) {
  const layer = document.createElement('div');
  layer.className = 'coach look-a recipe';
  layer.innerHTML = '<div class="ring"></div><div class="bub top"></div>';
  layer.hidden = true;
  root.append(layer);
  const ring = layer.querySelector('.ring'), bub = layer.querySelector('.bub');
  const seen = new Set((localStorage.getItem(KEY) ?? '').split(',').filter(Boolean).map(Number));
  let timer = 0, on = null;

  function place() {  // over the key, the pointer on its middle, kept inside the player
    if (!on) return;
    const R = root.getBoundingClientRect(), r = on.getBoundingClientRect(), pad = 3;
    const t = { x: r.left - R.left, y: r.top - R.top, w: r.width, h: r.height };
    Object.assign(ring.style, { left: t.x - pad + 'px', top: t.y - pad + 'px', width: t.w + 2 * pad + 'px', height: t.h + 2 * pad + 'px' });
    bub.style.maxWidth = R.width - 8 + 'px';
    const w = bub.offsetWidth, h = bub.offsetHeight, cx = t.x + t.w / 2;
    const x = Math.min(Math.max(cx - w / 2, 4), R.width - w - 4);
    Object.assign(bub.style, { left: x + 'px', top: t.y - pad - 12 - h + 'px' });
    bub.style.setProperty('--ax', Math.min(Math.max(cx - x, 14), w - 14) + 'px');
  }

  function hide() {
    clearTimeout(timer);
    layer.hidden = true;
    on = null;
  }

  function show(a) {  // the recipe of action a over its key, once per action on this browser
    if (!(a in RECIPES) || seen.has(a)) return;
    hide();
    seen.add(a);
    localStorage.setItem(KEY, [...seen].join(','));
    const [name, needs, where] = RECIPES[a];
    on = root.querySelector(`.acts .key[data-a="${a}"]`);
    if (!on) return;
    bub.innerHTML = `${name}: needs ${needs}<small>${where}</small>`;
    layer.hidden = false;
    place();
    timer = setTimeout(hide, SHOW_MS);
  }

  addEventListener('resize', place);
  return { show, hide };
}
