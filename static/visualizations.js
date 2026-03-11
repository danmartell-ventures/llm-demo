// ─── Visualization Renderers ────────────────────────────
const VIZ = (() => {
  const COLORS = ['#818cf8','#86efac','#fdba74','#f9a8d4','#67e8f9','#c4b5fd','#fde047','#fca5a5'];
  const CAT_COLORS = {
    animals:'#22c55e',colors:'#ef4444',emotions:'#ec4899',food:'#f97316',
    nature:'#06b6d4',body:'#eab308',tech:'#6366f1',actions:'#a855f7',
    size:'#14b8a6',time:'#f43f5e'
  };
  const dpr = window.devicePixelRatio || 1;

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, w: rect.width, h: rect.height };
  }

  // ─── Tokenizer ─────────────────────────────────────
  function animateTokens(tokens, container) {
    container.innerHTML = '';
    tokens.forEach((t, i) => {
      const chip = document.createElement('div');
      chip.className = `token-chip tc${i % 8}`;
      chip.innerHTML = `<span>${escHtml(t.text)}</span><span class="token-id">${t.id}</span>`;
      container.appendChild(chip);
      setTimeout(() => chip.classList.add('visible'), 100 + i * 120);
    });
  }

  // ─── Scatter Plot (Embeddings) ─────────────────────
  function drawScatter(canvas, embeddings, highlightTokens, legendEl) {
    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    // Find bounds
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    embeddings.forEach(e => {
      if (e.x < minX) minX = e.x; if (e.x > maxX) maxX = e.x;
      if (e.y < minY) minY = e.y; if (e.y > maxY) maxY = e.y;
    });
    const pad = 40;
    const scaleX = (v) => pad + ((v - minX) / (maxX - minX)) * (w - 2 * pad);
    const scaleY = (v) => pad + ((v - minY) / (maxY - minY)) * (h - 2 * pad);

    // Draw grid
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const x = pad + (i / 4) * (w - 2 * pad);
      const y = pad + (i / 4) * (h - 2 * pad);
      ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, h - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
    }

    // Draw all points
    embeddings.forEach(e => {
      const px = scaleX(e.x), py = scaleY(e.y);
      const col = CAT_COLORS[e.category] || '#94a3b8';
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
    });

    // Highlight tokens
    ctx.globalAlpha = 1;
    if (highlightTokens && highlightTokens.length) {
      highlightTokens.forEach((ht, i) => {
        const match = embeddings.find(e => e.word === ht.word);
        if (!match) return;
        const px = scaleX(match.x), py = scaleY(match.y);
        const col = COLORS[i % COLORS.length];

        // Glow
        ctx.shadowColor = col;
        ctx.shadowBlur = 12;
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(ht.word, px, py - 12);
      });
    }

    // Legend
    if (legendEl) {
      legendEl.innerHTML = '';
      const cats = new Set(embeddings.map(e => e.category));
      cats.forEach(cat => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-dot" style="background:${CAT_COLORS[cat]||'#94a3b8'}"></span>${cat}`;
        legendEl.appendChild(item);
      });
    }
  }

  // ─── Attention Heatmap ─────────────────────────────
  function drawAttention(canvas, tokens, matrix) {
    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    const n = tokens.length;
    if (n === 0) return;

    const labelSpace = 80;
    const cellW = Math.min(40, (w - labelSpace - 20) / n);
    const cellH = Math.min(40, (h - labelSpace - 20) / n);
    const ox = labelSpace;
    const oy = labelSpace;

    // Column labels (top)
    ctx.save();
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${Math.min(11, cellW - 2)}px Inter, sans-serif`;
    ctx.textAlign = 'right';
    tokens.forEach((t, j) => {
      ctx.save();
      ctx.translate(ox + j * cellW + cellW / 2, oy - 6);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(t.length > 8 ? t.slice(0, 7) + '…' : t, 0, 0);
      ctx.restore();
    });

    // Row labels (left)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    tokens.forEach((t, i) => {
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(t.length > 8 ? t.slice(0, 7) + '…' : t, ox - 6, oy + i * cellH + cellH / 2);
    });

    // Cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const val = matrix[i][j];
        const intensity = Math.min(1, val * 3);
        ctx.fillStyle = `rgba(99,102,241,${intensity})`;
        ctx.fillRect(ox + j * cellW, oy + i * cellH, cellW - 1, cellH - 1);

        if (cellW > 20) {
          ctx.fillStyle = intensity > 0.5 ? 'white' : '#64748b';
          ctx.font = '9px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(val.toFixed(2), ox + j * cellW + cellW / 2, oy + i * cellH + cellH / 2);
        }
      }
    }
    ctx.restore();
  }

  // ─── Prediction Bar Chart ─────────────────────────
  function drawPredictions(canvas, predictions) {
    const { ctx, w, h } = setupCanvas(canvas);
    ctx.clearRect(0, 0, w, h);

    if (!predictions || !predictions.length) return;

    const top = predictions.slice(0, 15);
    const pad = { top: 20, right: 20, bottom: 40, left: 70 };
    const chartW = w - pad.left - pad.right;
    const chartH = h - pad.top - pad.bottom;
    const barH = Math.min(28, chartH / top.length - 4);
    const maxProb = Math.max(...top.map(p => p.prob));

    top.forEach((p, i) => {
      const y = pad.top + i * (barH + 4);
      const barW = (p.prob / maxProb) * chartW;

      // Bar
      const gradient = ctx.createLinearGradient(pad.left, 0, pad.left + barW, 0);
      gradient.addColorStop(0, '#6366f1');
      gradient.addColorStop(1, '#06b6d4');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.roundRect(pad.left, y, barW, barH, 4);
      ctx.fill();

      // Label
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(p.token, pad.left - 8, y + barH / 2);

      // Probability
      ctx.fillStyle = '#94a3b8';
      ctx.font = '11px monospace';
      ctx.textAlign = 'left';
      ctx.fillText((p.prob * 100).toFixed(1) + '%', pad.left + barW + 6, y + barH / 2);
    });
  }

  // ─── Forward Pass Diagram ─────────────────────────
  function buildForwardPass(container, onComplete) {
    container.innerHTML = '';

    const blocks = [
      { label: '📝 Input Tokens', cls: 'embed-block', id: 'fp-input' },
      { label: '🔢 Token Embeddings', cls: 'embed-block', id: 'fp-embed' },
      { label: '📍 + Positional Encoding', cls: 'pos-block', id: 'fp-pos' },
      { type: 'layer-start' },
      { label: '🔎 Multi-Head Attention', cls: 'attn-block', id: 'fp-attn' },
      { type: 'residual', label: 'Add & Norm' },
      { label: '⚡ Feed-Forward Network', cls: 'ffn-block', id: 'fp-ffn' },
      { type: 'residual', label: 'Add & Norm' },
      { type: 'layer-end' },
      { label: '📊 Final LayerNorm', cls: 'norm-block', id: 'fp-norm' },
      { label: '🎯 Output Probabilities', cls: 'out-block', id: 'fp-out' },
    ];

    let layerDiv = null;
    blocks.forEach(b => {
      if (b.type === 'layer-start') {
        layerDiv = document.createElement('div');
        layerDiv.className = 'fp-layer-bracket';
        layerDiv.innerHTML = '<span class="fp-layer-label">× N layers</span>';
        container.appendChild(layerDiv);
        return;
      }
      if (b.type === 'layer-end') {
        layerDiv = null;
        return;
      }
      if (b.type === 'residual') {
        const res = document.createElement('div');
        res.className = 'fp-residual';
        res.textContent = b.label;
        const arrow = document.createElement('div');
        arrow.className = 'fp-arrow';
        (layerDiv || container).appendChild(arrow);
        (layerDiv || container).appendChild(res);
        const arrow2 = document.createElement('div');
        arrow2.className = 'fp-arrow';
        (layerDiv || container).appendChild(arrow2);
        return;
      }

      const arrow = document.createElement('div');
      arrow.className = 'fp-arrow';
      if ((layerDiv || container).children.length > 0) {
        (layerDiv || container).appendChild(arrow);
      }
      const el = document.createElement('div');
      el.className = `fp-block ${b.cls}`;
      el.id = b.id;
      el.innerHTML = `${b.label}<div class="fp-data-flow"></div>`;
      (layerDiv || container).appendChild(el);
    });

    // Animate sequentially
    const allBlocks = container.querySelectorAll('.fp-block');
    let i = 0;
    function lightNext() {
      if (i >= allBlocks.length) {
        if (onComplete) onComplete();
        return;
      }
      allBlocks[i].classList.add('lit');
      i++;
      setTimeout(lightNext, 400);
    }
    setTimeout(lightNext, 300);
  }

  // ─── Helpers ───────────────────────────────────────
  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  return { animateTokens, drawScatter, drawAttention, drawPredictions, buildForwardPass, COLORS };
})();
