/*
 * Procedural tower with LED fixtures, drawn in code (illustration, not model output).
 * FacadeScene.mount(canvas, { labels: bool, cx: 0..1 })
 * Loop (16 s): element outlines draw in over a dim rig -> rig lights up with a slow
 * rising drift that resolves into a calm crown glow. Reduced motion: one still frame.
 */
(function () {
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const sstep = (a, b, x) => { x = clamp((x - a) / (b - a)); return x * x * (3 - 2 * x); };
  const LEVELS = 48;
  // champagne ramp: lightness climbs, chroma peaks mid-ramp so the top end reads near-white
  const ramp = Array.from({ length: LEVELS }, (_, i) => {
    const k = i / (LEVELS - 1);
    const L = 0.27 + 0.69 * Math.pow(k, 0.9);
    const C = 0.018 + 0.34 * k * (1 - k) * 0.9;
    return `oklch(${L.toFixed(3)} ${C.toFixed(3)} ${(88 - 10 * k).toFixed(0)})`;
  });
  const MASK = 'oklch(0.84 0.10 82 / ';
  const CYCLE = 16;
  const BLOOM = 6;

  function mount(canvas, opts) {
    opts = opts || {};
    const ctx = canvas.getContext('2d');
    const bloom = document.createElement('canvas');
    const bctx = bloom.getContext('2d');
    const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let w = 0, h = 0, dpr = 1, geo = null, raf = 0, visible = true;
    const t0 = performance.now();

    function layout() {
      const r = canvas.getBoundingClientRect();
      w = Math.max(1, r.width); h = Math.max(1, r.height);
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      bloom.width = Math.ceil(w / BLOOM); bloom.height = Math.ceil(h / BLOOM);
      const H = Math.min(h * 0.86, w * 1.15);
      const gy = h * 0.95;
      const cx = w * (opts.cx || 0.5);
      const sw = H * 0.30;
      const lowTop = gy - 0.68 * H, upTop = gy - 0.90 * H, podTop = gy - 0.11 * H;
      const pw = sw * 1.85;
      geo = {
        H, gy, cx, sw, pw, lowTop, upTop, podTop, crownTop: gy - H,
        secs: [
          { id: 'pod', y0: podTop, y1: gy, n: Math.floor(pw / 13), wAt: () => pw },
          { id: 'low', y0: lowTop, y1: podTop, n: Math.floor(sw / 12), wAt: () => sw },
          { id: 'up', y0: upTop, y1: lowTop, n: Math.floor(sw * 0.76 / 12), wAt: () => sw * 0.76 },
          { id: 'crown', y0: gy - H, y1: upTop, n: 7, wAt: (y) => sw * 0.76 * (0.22 + 0.78 * sstep(gy - H, upTop, y)) },
        ],
        sky: Array.from({ length: 9 }, (_, i) => ({
          x: w * (0.02 + i * 0.115) + ((i * 37) % 17), w: w * (0.06 + ((i * 53) % 7) * 0.008),
          top: gy - H * (0.18 + ((i * 29) % 11) * 0.03),
        })).filter((b) => Math.abs(b.x + b.w / 2 - cx) > pw / 2 + 10),
      };
    }

    // brightness 0..1 for fixture column i of n at height y
    function bright(g, i, n, y, t, lit) {
      const u = clamp((g.gy - y) / g.H);
      const drift = 0.5 + 0.5 * Math.sin(Math.PI * 2 * (u * 1.25 - t / 9) + 0.5 * Math.sin(i * 0.45 + t * 0.2));
      const wave = Math.pow(drift, 3.2) * 0.85;
      const crown = sstep(0.78, 1, u) * (0.45 + 0.35 * Math.sin(t / 2.6)) ;
      return clamp(Math.max(wave, crown) * lit + 0.04);
    }

    function frame(now, forced) {
      if (!geo) layout();
      const g = geo;
      const t = forced != null ? forced : (now - t0) / 1000;
      const ph = t % CYCLE;
      const ov = forced != null ? 0.85 : sstep(0.4, 1.6, ph) * (1 - sstep(7.2, 8.8, ph));
      const lit = forced != null ? 1 : 0.22 + 0.78 * sstep(2.5, 6.5, ph);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      bctx.setTransform(1, 0, 0, 1, 0, 0);
      bctx.clearRect(0, 0, bloom.width, bloom.height);

      // distant skyline, unlit
      ctx.fillStyle = 'oklch(0.185 0.009 70)';
      g.sky.forEach((b) => ctx.fillRect(b.x, b.top, b.w, g.gy - b.top));

      // tower bodies
      ctx.fillStyle = 'oklch(0.215 0.011 70)';
      g.secs.forEach((s) => {
        if (s.id === 'crown') {
          ctx.beginPath();
          ctx.moveTo(g.cx - s.wAt(s.y0) / 2, s.y0); ctx.lineTo(g.cx + s.wAt(s.y0) / 2, s.y0);
          ctx.lineTo(g.cx + s.wAt(s.y1) / 2, s.y1); ctx.lineTo(g.cx - s.wAt(s.y1) / 2, s.y1);
          ctx.closePath(); ctx.fill();
        } else ctx.fillRect(g.cx - s.wAt() / 2, s.y0, s.wAt(), s.y1 - s.y0);
      });
      // spire
      ctx.fillStyle = 'oklch(0.3 0.012 72)';
      ctx.fillRect(g.cx - 1, g.crownTop - g.H * 0.05, 2, g.H * 0.05);

      // floor lines (glass bands)
      ctx.strokeStyle = 'oklch(0.285 0.011 70)'; ctx.lineWidth = 1;
      ctx.beginPath();
      g.secs.forEach((s) => {
        if (s.id === 'crown') return;
        const x0 = g.cx - s.wAt() / 2, x1 = g.cx + s.wAt() / 2;
        for (let y = s.y1 - 22; y > s.y0; y -= 22) { ctx.moveTo(x0, Math.round(y) + 0.5); ctx.lineTo(x1, Math.round(y) + 0.5); }
      });
      ctx.stroke();

      // fins on the lower shaft
      const low = g.secs[1];
      ctx.fillStyle = 'oklch(0.31 0.013 72)';
      for (let i = 4; i < low.n; i += 5) ctx.fillRect(g.cx - g.sw / 2 + (i + 0.5) * g.sw / low.n - 1, low.y0, 2, low.y1 - low.y0);

      // fixtures: LED bar segments on each mullion column
      const seg = 7, gap = 5;
      g.secs.forEach((s) => {
        for (let y = s.y1 - seg - 3; y > s.y0; y -= seg + gap) {
          const wy = s.wAt(y), x0 = g.cx - wy / 2;
          for (let i = 0; i < s.n; i++) {
            const x = x0 + (i + 0.5) * wy / s.n;
            const idx = Math.round(bright(g, i, s.n, y, t, lit) * (LEVELS - 1));
            ctx.fillStyle = ramp[idx];
            ctx.fillRect(x - 1.5, y, 3, seg);
            if (idx > 14) { bctx.fillStyle = ramp[idx]; bctx.fillRect(x / BLOOM, y / BLOOM, 1, 1); }
          }
        }
      });

      // bloom from the low-res fixture map
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 0.5;
      ctx.filter = 'blur(14px)';
      ctx.drawImage(bloom, 0, 0, w, h);
      ctx.restore();

      // ground line
      ctx.fillStyle = 'oklch(0.3 0.012 72)';
      ctx.fillRect(0, g.gy, w, 1);

      if (opts.labels && ov > 0.01) overlay(g, ov);
    }

    function overlay(g, a) {
      const low = g.secs[1], up = g.secs[2], crown = g.secs[3];
      const right = g.cx + g.sw / 2 + 40;
      const finI = low.n - 1 - ((low.n - 1 - 4) % 5);
      const finX = g.cx - g.sw / 2 + (finI + 0.5) * g.sw / low.n;
      const bandY = low.y0 + (low.y1 - low.y0) * 0.38;
      const items = [
        { n: '01', name: 'Crown', y: (crown.y0 + crown.y1) / 2, edge: g.cx + crown.wAt((crown.y0 + crown.y1) / 2) / 2 },
        { n: '02', name: 'Setback', y: low.y0, edge: g.cx + g.sw / 2 },
        { n: '03', name: 'Glass band', y: bandY + 11, edge: g.cx + g.sw / 2 },
        { n: '04', name: 'Fins', y: (low.y0 + low.y1) * 0.5 + 60, edge: finX + 7 },
      ];
      ctx.save();
      ctx.globalAlpha = a;
      ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]); ctx.strokeStyle = 'oklch(0.84 0.10 82)';
      ctx.fillStyle = MASK + '0.14)';
      // crown mask
      ctx.beginPath();
      ctx.moveTo(g.cx - crown.wAt(crown.y0) / 2, crown.y0); ctx.lineTo(g.cx + crown.wAt(crown.y0) / 2, crown.y0);
      ctx.lineTo(g.cx + crown.wAt(crown.y1) / 2, crown.y1); ctx.lineTo(g.cx - crown.wAt(crown.y1) / 2, crown.y1);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      // setback: step edge on the lower roofline
      ctx.beginPath(); ctx.moveTo(g.cx - g.sw / 2 - 10, low.y0); ctx.lineTo(g.cx + g.sw / 2 + 10, low.y0); ctx.stroke();
      // glass band
      ctx.fillRect(g.cx - g.sw / 2, bandY, g.sw, 22); ctx.strokeRect(g.cx - g.sw / 2, bandY, g.sw, 22);
      // fin
      ctx.fillRect(finX - 7, low.y0, 14, low.y1 - low.y0); ctx.strokeRect(finX - 7, low.y0, 14, low.y1 - low.y0);
      ctx.setLineDash([]); ctx.lineWidth = 1;
      ctx.font = '600 14px Archivo, "Helvetica Neue", Arial, sans-serif';
      ctx.textBaseline = 'middle';
      items.forEach((it) => {
        ctx.beginPath(); ctx.moveTo(it.edge + 4, it.y); ctx.lineTo(right - 8, it.y); ctx.stroke();
        ctx.fillStyle = 'oklch(0.84 0.10 82)';
        ctx.fillText(it.n, right, it.y);
        ctx.fillStyle = 'oklch(0.94 0.008 80)';
        ctx.fillText(it.name.toUpperCase(), right + 30, it.y);
      });
      ctx.restore();
    }

    function loop(now) { if (visible) frame(now); raf = requestAnimationFrame(loop); }

    layout();
    if (still) {
      frame(0, 10.5);
      new ResizeObserver(() => { layout(); frame(0, 10.5); }).observe(canvas);
    } else {
      new ResizeObserver(layout).observe(canvas);
      new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(canvas);
      raf = requestAnimationFrame(loop);
    }
  }

  window.FacadeScene = { mount };
})();
