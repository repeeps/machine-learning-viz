/* ============================================================
   머신러닝 엔진 (VZ.ML — 지도학습)
   "머신러닝, 눈으로 보기" 전용 렌더러. 순수 함수(상태→SVG 문자열),
   전부 방어적(빈 배열·0 division·음수·NaN 가드). 베이스 lib.js
   (VZ.AL 스텝 플레이어 + heatColor + VZ.LA.tween + linePlot) 재사용.
   - svg / chip / arrow : 기본 · plane(좌표 매핑 [0..1]→픽셀, y 위가 큼)
   - scatter(산점도+적합선+잔차) / boundary(결정 경계 영역 채색) / curve(적합 곡선)
   - errPlot(train/test 오차 곡선) / dartboard(편향-분산 과녁)
   - tree(결정트리 분기) / confusion(혼동행렬) / bars(정밀도·재현율) / roc / kfold
   색: class0=청록(--q) class1=앰버(--hot) class2=초록(--good)
       적합선=보라(--v) 잔차=코랄(--dead) train=청록 test=앰버 정답=초록 오답=코랄
   ============================================================ */
(function (global) {
  'use strict';
  const VZ = global.VZ;
  const LA = VZ.LA, AL = VZ.AL, clamp = VZ.clamp;
  const CLS = ['var(--q)', 'var(--hot)', 'var(--good)'];

  const C = {
    cls: CLS, fit: 'var(--v)', resid: 'var(--dead)', train: 'var(--q)', test: 'var(--hot)',
    good: 'var(--good)', bad: 'var(--dead)', dead: 'var(--dead)', ink: 'var(--ink)', muted: 'var(--muted)',
    faint: 'var(--faint)', line: 'var(--line)', q: 'var(--q)', v: 'var(--v)', hot: 'var(--hot)',
    pink: 'var(--pink)', blue: 'var(--blue)', slate: 'var(--slate)',
  };
  const num = (v, d = 0) => (isFinite(v) ? v : d);
  const esc = s => String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const clsCol = i => CLS[((i % CLS.length) + CLS.length) % CLS.length] || CLS[0];

  function svg(W, H, inner, aria) {
    return `<svg viewBox="0 0 ${num(W, 100)} ${num(H, 100)}" width="100%" role="img" aria-label="${esc(aria) || '머신러닝 그림'}" style="max-width:100%;display:block;background:var(--panel-2);border:1px solid var(--line);border-radius:12px">${inner || ''}</svg>`;
  }
  function rng(seed) { let s = (seed >>> 0) || 1; return function () { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  function chip(cx, cy, text, opts = {}) {
    const col = opts.color || C.q, w = Math.max(opts.minW || 30, String(text).length * 7 + 14), h = opts.h || 20;
    return `<rect x="${(cx - w / 2).toFixed(1)}" y="${(cy - h / 2).toFixed(1)}" width="${w.toFixed(1)}" height="${h}" rx="6" fill="${opts.fill || 'var(--panel)'}" stroke="${col}" stroke-width="1.2"${opts.dim ? ' opacity="0.45"' : ''}/>` +
      `<text x="${cx}" y="${cy + 3.5}" text-anchor="middle" font-size="${opts.fs || 10.5}" font-family="JetBrains Mono" font-weight="700" fill="${col}">${esc(text)}</text>`;
  }
  function arrow(x1, y1, x2, y2, opts = {}) {
    const col = opts.color || C.line;
    if (opts.dash) return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="${opts.lw || 1.6}" stroke-dasharray="${opts.dash}"${opts.dim ? ' opacity="0.4"' : ''}/>`;
    return LA.arrowPx(x1, y1, x2, y2, col, { lw: opts.lw || 1.8 });
  }

  // 좌표 매핑: 데이터 [0..1]×[0..1] → 박스 픽셀 (y는 위가 큼)
  function plane(box) {
    const { x, y, w, h } = box, pad = 6;
    return {
      px: nx => x + pad + clamp(nx, 0, 1) * (w - 2 * pad),
      py: ny => y + h - pad - clamp(ny, 0, 1) * (h - 2 * pad),
      frame: () => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="var(--bg)" stroke="var(--line)" stroke-width="1"/>`,
    };
  }

  // ---- 산점도 (+ 선택 적합선 + 잔차) ----
  // scatter(x,y,w,h,opts) opts:{pts:[{x,y,c}], line:{m,b}, residuals(bool), r, axes:[xlab,ylab], title}
  function scatter(x, y, w, h, opts = {}) {
    const P = plane({ x, y, w, h }), pts = opts.pts || [], r = opts.r || 4;
    let g = P.frame();
    if (opts.title != null) g += `<text x="${x + w / 2}" y="${y - 6}" text-anchor="middle" font-size="10" font-family="JetBrains Mono" fill="${C.muted}">${esc(opts.title)}</text>`;
    if (opts.axes) { g += `<text x="${x + w / 2}" y="${y + h + 13}" text-anchor="middle" font-size="8.5" font-family="JetBrains Mono" fill="${C.faint}">${esc(opts.axes[0])}</text>`;
      g += `<text x="${x - 4}" y="${y + h / 2}" text-anchor="middle" font-size="8.5" font-family="JetBrains Mono" fill="${C.faint}" transform="rotate(-90 ${x - 4} ${y + h / 2})">${esc(opts.axes[1])}</text>`; }
    const ln = opts.line;
    if (opts.residuals && ln) pts.forEach(p => { const ly = ln.m * p.x + ln.b; g += `<line x1="${P.px(p.x).toFixed(1)}" y1="${P.py(p.y).toFixed(1)}" x2="${P.px(p.x).toFixed(1)}" y2="${P.py(ln.m * p.x + ln.b).toFixed(1)}" stroke="${C.resid}" stroke-width="1.4"/>`; });
    if (ln) { const y0 = ln.b, y1 = ln.m + ln.b; g += `<line x1="${P.px(0)}" y1="${P.py(clamp(y0, -0.5, 1.5))}" x2="${P.px(1)}" y2="${P.py(clamp(y1, -0.5, 1.5))}" stroke="${opts.lineColor || C.fit}" stroke-width="2.4"/>`; }
    pts.forEach(p => { g += `<circle cx="${P.px(p.x).toFixed(1)}" cy="${P.py(p.y).toFixed(1)}" r="${r}" fill="${p.c != null ? clsCol(p.c) : C.q}" opacity="0.92" stroke="var(--bg)" stroke-width="1"/>`; });
    return g;
  }

  // ---- 결정 경계 (영역 채색) ----
  // boundary(x,y,w,h,opts) opts:{regionFn(nx,ny)->classIdx, pts, cols(grid), title}
  function boundary(x, y, w, h, opts = {}) {
    const P = plane({ x, y, w, h }), fn = opts.regionFn || (() => 0), cols = opts.cols || 14, rows = opts.rows || 10;
    let g = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="var(--bg)" stroke="var(--line)" stroke-width="1"/>`;
    const cw = (w - 12) / cols, ch = (h - 12) / rows;
    for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
      const nx = (i + 0.5) / cols, ny = (j + 0.5) / rows, ci = fn(nx, ny);
      const cx = x + 6 + i * cw, cyy = y + h - 6 - (j + 1) * ch;
      g += `<rect x="${cx.toFixed(1)}" y="${cyy.toFixed(1)}" width="${(cw + 0.6).toFixed(1)}" height="${(ch + 0.6).toFixed(1)}" fill="${clsCol(ci)}" opacity="0.16"/>`;
    }
    (opts.pts || []).forEach(p => { g += `<circle cx="${P.px(p.x).toFixed(1)}" cy="${P.py(p.y).toFixed(1)}" r="${opts.r || 4.5}" fill="${clsCol(p.c)}" opacity="0.95" stroke="var(--bg)" stroke-width="1.2"/>`; });
    if (opts.title != null) g += `<text x="${x + w / 2}" y="${y - 6}" text-anchor="middle" font-size="10" font-family="JetBrains Mono" fill="${C.muted}">${esc(opts.title)}</text>`;
    return g;
  }

  // ---- 적합 곡선 (train 점 + 함수 곡선) ----
  // curve(x,y,w,h,opts) opts:{pts, fn(nx)->ny, samples, color, title, testPts}
  function curve(x, y, w, h, opts = {}) {
    const P = plane({ x, y, w, h }), fn = opts.fn, N = opts.samples || 48;
    let g = P.frame();
    if (opts.title != null) g += `<text x="${x + w / 2}" y="${y - 6}" text-anchor="middle" font-size="10" font-family="JetBrains Mono" fill="${C.muted}">${esc(opts.title)}</text>`;
    if (fn) { let d = ''; for (let i = 0; i <= N; i++) { const nx = i / N, ny = clamp(fn(nx), -0.2, 1.2); d += (i === 0 ? 'M' : 'L') + P.px(nx).toFixed(1) + ',' + P.py(ny).toFixed(1) + ' '; } g += `<path d="${d}" fill="none" stroke="${opts.color || C.fit}" stroke-width="2.4"/>`; }
    (opts.pts || []).forEach(p => { g += `<circle cx="${P.px(p.x).toFixed(1)}" cy="${P.py(p.y).toFixed(1)}" r="4" fill="${C.train}" opacity="0.9" stroke="var(--bg)" stroke-width="1"/>`; });
    (opts.testPts || []).forEach(p => { g += `<circle cx="${P.px(p.x).toFixed(1)}" cy="${P.py(p.y).toFixed(1)}" r="4" fill="${C.test}" opacity="0.9" stroke="var(--bg)" stroke-width="1"/>`; });
    return g;
  }

  // ---- train/test 오차 곡선 (복잡도 축) ----
  // errPlot(x,y,w,h,opts) opts:{train:[..], test:[..], mark(index), title}
  function errPlot(x, y, w, h, opts = {}) {
    const P = plane({ x, y, w, h }), tr = opts.train || [], te = opts.test || [], n = Math.max(tr.length, te.length, 2);
    let g = P.frame();
    if (opts.title != null) g += `<text x="${x + w / 2}" y="${y - 6}" text-anchor="middle" font-size="10" font-family="JetBrains Mono" fill="${C.muted}">${esc(opts.title)}</text>`;
    const path = (arr, col) => { let d = ''; arr.forEach((v, i) => { d += (i === 0 ? 'M' : 'L') + P.px(i / (n - 1)).toFixed(1) + ',' + P.py(clamp(v, 0, 1)).toFixed(1) + ' '; }); return `<path d="${d}" fill="none" stroke="${col}" stroke-width="2.2"/>`; };
    g += path(tr, C.train); g += path(te, C.test);
    if (opts.mark != null) { const mx = P.px(opts.mark / (n - 1)); g += `<line x1="${mx}" y1="${y + 4}" x2="${mx}" y2="${y + h - 4}" stroke="var(--hot)" stroke-width="1.2" stroke-dasharray="3 3"/>`; }
    g += `<text x="${x + 10}" y="${y + 14}" font-size="8.5" font-family="JetBrains Mono" fill="${C.train}">▬ 훈련 오차</text>`;
    g += `<text x="${x + 10}" y="${y + 26}" font-size="8.5" font-family="JetBrains Mono" fill="${C.test}">▬ 시험 오차</text>`;
    const xlab = opts.xlabel != null ? opts.xlabel : '모델 복잡도 →';
    if (xlab) g += `<text x="${x + w / 2}" y="${y + h + 13}" text-anchor="middle" font-size="8.5" font-family="JetBrains Mono" fill="${C.faint}">${esc(xlab)}</text>`;
    return g;
  }

  // ---- 편향-분산 과녁 ----
  // dartboard(cx,cy,opts) opts:{shots:[{x,y}] (중심 기준 -1..1), r, label}
  function dartboard(cx, cy, opts = {}) {
    const r = opts.r || 56, shots = opts.shots || [];
    let g = '';
    [1, 0.66, 0.33].forEach((f, i) => g += `<circle cx="${cx}" cy="${cy}" r="${(r * f).toFixed(1)}" fill="${i === 2 ? 'rgba(52,211,153,.18)' : 'none'}" stroke="var(--line)" stroke-width="1.2"/>`);
    g += `<circle cx="${cx}" cy="${cy}" r="3" fill="var(--good)"/>`;
    shots.forEach(s => { g += `<circle cx="${(cx + s.x * r).toFixed(1)}" cy="${(cy + s.y * r).toFixed(1)}" r="3.4" fill="${C.hot}" opacity="0.9"/>`; });
    if (opts.label != null) g += `<text x="${cx}" y="${cy + r + 16}" text-anchor="middle" font-size="10" font-family="JetBrains Mono" font-weight="700" fill="${C.muted}">${esc(opts.label)}</text>`;
    return g;
  }

  // ---- 결정트리 (노드·분기) ----
  // tree(opts) opts:{nodes:[{id,x,y,label,leaf,cls}], edges:[[a,b,lab]], W,H}  좌표는 픽셀
  function tree(opts = {}) {
    const nodes = opts.nodes || [], edges = opts.edges || [];
    const find = id => nodes.find(n => n.id === id) || { x: 0, y: 0 };
    let g = '';
    edges.forEach(e => { const a = find(e[0]), b = find(e[1]); g += `<line x1="${a.x}" y1="${a.y + 14}" x2="${b.x}" y2="${b.y - 14}" stroke="var(--line)" stroke-width="1.4"/>`;
      if (e[2] != null) g += `<text x="${((a.x + b.x) / 2).toFixed(1)}" y="${((a.y + b.y) / 2).toFixed(1)}" text-anchor="middle" font-size="8" font-family="JetBrains Mono" fill="${C.faint}">${esc(e[2])}</text>`; });
    nodes.forEach(n => { const leaf = n.leaf, col = leaf ? clsCol(n.cls || 0) : C.v, wn = Math.max(54, String(n.label).length * 7 + 14);
      g += `<rect x="${(n.x - wn / 2).toFixed(1)}" y="${n.y - 14}" width="${wn}" height="28" rx="${leaf ? 14 : 7}" fill="${leaf ? col : 'var(--panel)'}" opacity="${leaf ? 0.85 : 1}" stroke="${col}" stroke-width="1.5"/>`;
      g += `<text x="${n.x}" y="${n.y + 4}" text-anchor="middle" font-size="10" font-family="JetBrains Mono" font-weight="700" fill="${leaf ? '#0b0e14' : col}">${esc(n.label)}</text>`; });
    return g;
  }

  // ---- 혼동행렬 (2x2) ----
  // confusion(x,y,opts) opts:{tp,fp,fn,tn, cell, title}
  function confusion(x, y, opts = {}) {
    const cell = opts.cell || 64, v = { tp: opts.tp || 0, fp: opts.fp || 0, fn: opts.fn || 0, tn: opts.tn || 0 };
    const data = [['TP', v.tp, C.good, '맞게 양성'], ['FN', v.fn, C.bad, '놓친 양성'], ['FP', v.fp, C.bad, '잘못 양성'], ['TN', v.tn, C.good, '맞게 음성']];
    let g = '';
    g += `<text x="${x + cell}" y="${y - 6}" text-anchor="middle" font-size="9" font-family="JetBrains Mono" fill="${C.muted}">예측 →</text>`;
    g += `<text x="${x - 10}" y="${y + cell}" text-anchor="middle" font-size="9" font-family="JetBrains Mono" fill="${C.muted}" transform="rotate(-90 ${x - 10} ${y + cell})">실제 →</text>`;
    [['양성', x + cell * 0.5], ['음성', x + cell * 1.5]].forEach(c => g += `<text x="${c[1]}" y="${y - 18}" text-anchor="middle" font-size="8.5" font-family="JetBrains Mono" fill="${C.faint}">예측 ${c[0]}</text>`);
    [['양성', y + cell * 0.5], ['음성', y + cell * 1.5]].forEach(c => g += `<text x="${x - 22}" y="${c[1] + 3}" text-anchor="end" font-size="8.5" font-family="JetBrains Mono" fill="${C.faint}">실제 ${c[0]}</text>`);
    data.forEach((d, i) => { const cx = x + (i % 2) * cell, cyy = y + Math.floor(i / 2) * cell;
      g += `<rect x="${cx}" y="${cyy}" width="${cell - 2}" height="${cell - 2}" rx="6" fill="${d[2]}" opacity="0.16" stroke="${d[2]}" stroke-width="1.3"/>`;
      g += `<text x="${cx + cell / 2}" y="${cyy + cell / 2 - 4}" text-anchor="middle" font-size="16" font-family="JetBrains Mono" font-weight="700" fill="${d[2]}">${d[1]}</text>`;
      g += `<text x="${cx + cell / 2}" y="${cyy + cell / 2 + 12}" text-anchor="middle" font-size="8" font-family="JetBrains Mono" fill="${C.faint}">${d[0]} · ${d[3]}</text>`; });
    return g;
  }

  // ---- 가로 막대 (정밀도·재현율 등) ----
  // bars(x,y,opts) opts:{items:[{label,val(0..1),color}], w, title}
  function bars(x, y, opts = {}) {
    const items = opts.items || [], w = opts.w || 200, bh = 22, gap = 12;
    let g = '';
    items.forEach((it, i) => { const yy = y + i * (bh + gap); g += `<text x="${x}" y="${yy + 14}" font-size="10" font-family="JetBrains Mono" font-weight="700" fill="${it.color || C.q}">${esc(it.label)}</text>`;
      g += `<rect x="${x + 84}" y="${yy}" width="${w}" height="${bh}" rx="5" fill="var(--panel)" stroke="var(--line)"/>`;
      g += `<rect x="${x + 84}" y="${yy}" width="${(w * clamp(it.val, 0, 1)).toFixed(1)}" height="${bh}" rx="5" fill="${it.color || C.q}" opacity="0.85"/>`;
      g += `<text x="${x + 84 + w - 8}" y="${yy + 15}" text-anchor="end" font-size="9.5" font-family="JetBrains Mono" font-weight="700" fill="#0b0e14">${Math.round(clamp(it.val, 0, 1) * 100)}%</text>`; });
    return g;
  }

  // ---- ROC 곡선 ----
  // roc(x,y,w,h,opts) opts:{pts:[{x,y}] (0..1), mark, title}
  function roc(x, y, w, h, opts = {}) {
    const P = plane({ x, y, w, h }), pts = opts.pts || [];
    let g = P.frame();
    g += `<line x1="${P.px(0)}" y1="${P.py(0)}" x2="${P.px(1)}" y2="${P.py(1)}" stroke="var(--faint)" stroke-width="1" stroke-dasharray="3 3"/>`;
    let d = ''; pts.forEach((p, i) => { d += (i === 0 ? 'M' : 'L') + P.px(p.x).toFixed(1) + ',' + P.py(p.y).toFixed(1) + ' '; });
    g += `<path d="${d}" fill="none" stroke="${C.v}" stroke-width="2.4"/>`;
    if (opts.mark) g += `<circle cx="${P.px(opts.mark.x)}" cy="${P.py(opts.mark.y)}" r="4.5" fill="${C.hot}" stroke="var(--bg)" stroke-width="1.4"/>`;
    g += `<text x="${x + w / 2}" y="${y + h + 13}" text-anchor="middle" font-size="8" font-family="JetBrains Mono" fill="${C.faint}">거짓 양성률 →</text>`;
    g += `<text x="${x - 4}" y="${y + h / 2}" text-anchor="middle" font-size="8" font-family="JetBrains Mono" fill="${C.faint}" transform="rotate(-90 ${x - 4} ${y + h / 2})">재현율 →</text>`;
    return g;
  }

  // ---- k-fold 띠 ----
  // kfold(x,y,opts) opts:{k, testFold, w, title}
  function kfold(x, y, opts = {}) {
    const k = opts.k || 5, test = opts.testFold == null ? 0 : opts.testFold, w = opts.w || 360, fw = (w - (k - 1) * 4) / k, fh = opts.fh || 30;
    let g = opts.title != null ? `<text x="${x}" y="${y - 7}" font-size="9.5" font-family="JetBrains Mono" font-weight="700" fill="${C.muted}">${esc(opts.title)}</text>` : '';
    for (let i = 0; i < k; i++) { const fx = x + i * (fw + 4), isT = i === test;
      g += `<rect x="${fx.toFixed(1)}" y="${y}" width="${fw.toFixed(1)}" height="${fh}" rx="5" fill="${isT ? C.hot : C.train}" opacity="${isT ? 0.85 : 0.5}" stroke="${isT ? C.hot : C.train}" stroke-width="1.3"/>`;
      g += `<text x="${(fx + fw / 2).toFixed(1)}" y="${y + fh / 2 + 4}" text-anchor="middle" font-size="9" font-family="JetBrains Mono" font-weight="700" fill="#0b0e14">${isT ? '시험' : '훈련'}</text>`; }
    return g;
  }

  // ---- 비교 카드 ----
  function card(x, y, w, title, rows, opts = {}) {
    const r = rows || [], h = opts.h || (28 + r.length * 18), col = opts.color || C.q;
    let g = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="10" fill="var(--panel)" stroke="${col}" stroke-width="1.6"/>`;
    g += `<text x="${x + 12}" y="${y + 19}" font-size="12" font-family="'Pretendard'" font-weight="700" fill="${col}">${esc(title)}</text>`;
    r.forEach((row, i) => { const ry = y + 36 + i * 18; g += `<text x="${x + 12}" y="${ry}" font-size="9.5" font-family="JetBrains Mono" fill="${C.muted}">${esc(row[0])}</text><text x="${x + w - 12}" y="${ry}" text-anchor="end" font-size="9.5" font-family="JetBrains Mono" fill="${C.ink}">${esc(row[1])}</text>`; });
    return g;
  }

  VZ.ML = { C, svg, rng, num, esc, clsCol, chip, arrow, plane, scatter, boundary, curve, errPlot, dartboard, tree, confusion, bars, roc, kfold, card, heat: AL.heatColor };
})(window);
