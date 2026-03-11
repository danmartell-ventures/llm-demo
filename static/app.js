// ─── LLM Journey App ────────────────────────────────────
(function() {
  'use strict';

  // State
  let currentStep = 0; // 0 = landing, 1-5 = steps
  let phrase = '';
  let sampledTokens = [];
  let technicalMode = false;
  let apiData = {}; // cached API responses

  // DOM refs
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const landing = $('#landing');
  const phraseInput = $('#phrase-input');
  const beginBtn = $('#begin-btn');
  const progressBar = $('#progress-bar');
  const modeToggle = $('#mode-toggle');
  const navButtons = $('#nav-buttons');
  const prevBtn = $('#prev-btn');
  const nextBtn = $('#next-btn');
  const startOverBtn = $('#start-over-btn');
  const toggleTech = $('#toggle-technical');
  const progressFill = $('#progress-fill');
  const progressStepsEl = $('#progress-steps');

  const stepNames = ['Tokenize', 'Embed', 'Attention', 'Forward Pass', 'Predict'];

  // ─── Init ─────────────────────────────────────────
  function init() {
    // Build progress steps
    stepNames.forEach((name, i) => {
      const el = document.createElement('div');
      el.className = 'progress-step';
      el.textContent = `${i + 1}. ${name}`;
      el.addEventListener('click', () => goToStep(i + 1));
      progressStepsEl.appendChild(el);
    });

    // Events
    beginBtn.addEventListener('click', handleBegin);
    phraseInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') handleBegin();
    });
    $$('.suggestion').forEach(btn => {
      btn.addEventListener('click', () => {
        phraseInput.value = btn.dataset.text;
        handleBegin();
      });
    });
    prevBtn.addEventListener('click', () => goToStep(currentStep - 1));
    nextBtn.addEventListener('click', () => goToStep(currentStep + 1));
    startOverBtn.addEventListener('click', startOver);
    toggleTech.addEventListener('click', toggleTechnical);

    // Sliders
    $('#temp-slider').addEventListener('input', handleSliderChange);
    $('#topp-slider').addEventListener('input', handleSliderChange);
    $('#sample-btn').addEventListener('click', sampleToken);
  }

  // ─── Begin Journey ────────────────────────────────
  function handleBegin() {
    phrase = phraseInput.value.trim();
    if (!phrase) {
      phrase = 'The cat sat on the mat';
      phraseInput.value = phrase;
    }
    sampledTokens = [];
    apiData = {};
    goToStep(1);
  }

  // ─── Navigation ───────────────────────────────────
  function goToStep(step) {
    if (step < 0 || step > 5) return;

    // Hide all sections
    $$('.step').forEach(s => {
      s.classList.remove('active');
      s.style.display = 'none';
    });

    currentStep = step;

    if (step === 0) {
      landing.style.display = 'flex';
      landing.classList.add('active');
      progressBar.classList.add('hidden');
      modeToggle.classList.add('hidden');
      navButtons.classList.add('hidden');
      return;
    }

    // Show target step
    const section = $(`#step-${step}`);
    section.style.display = 'flex';
    // Force reflow for transition
    void section.offsetWidth;
    section.classList.add('active');

    // UI chrome
    progressBar.classList.remove('hidden');
    modeToggle.classList.remove('hidden');
    navButtons.classList.remove('hidden');

    // Progress
    progressFill.style.width = `${(step / 5) * 100}%`;
    $$('.progress-step').forEach((el, i) => {
      el.classList.toggle('active', i === step - 1);
      el.classList.toggle('completed', i < step - 1);
    });

    // Nav buttons
    prevBtn.disabled = step <= 1;
    nextBtn.style.display = step >= 5 ? 'none' : '';

    // Load step data
    loadStep(step);
  }

  // ─── Load Step Data ───────────────────────────────
  async function loadStep(step) {
    // Delay to ensure DOM layout is complete after display change
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    switch (step) {
      case 1: await loadTokenize(); break;
      case 2: await loadEmbed(); break;
      case 3: await loadAttention(); break;
      case 4: loadForwardPass(); break;
      case 5: await loadPredict(); break;
    }
  }

  // ─── Step 1: Tokenize ─────────────────────────────
  async function loadTokenize() {
    $('#original-phrase').textContent = `"${phrase}"`;

    if (!apiData.tokens) {
      const res = await fetch(`/api/tokenize?text=${enc(phrase)}`);
      apiData.tokens = await res.json();
    }

    const { tokens, count } = apiData.tokens;
    VIZ.animateTokens(tokens, $('#token-display'));
    $('#token-stats').textContent = `${count} token${count !== 1 ? 's' : ''} • Vocabulary maps text → integer IDs`;
  }

  // ─── Step 2: Embed ────────────────────────────────
  async function loadEmbed() {
    if (!apiData.tokens) {
      const res = await fetch(`/api/tokenize?text=${enc(phrase)}`);
      apiData.tokens = await res.json();
    }
    if (!apiData.embeddings) {
      const res = await fetch('/api/embeddings');
      apiData.embeddings = await res.json();
    }

    const tokens = apiData.tokens.tokens;
    const embeddings = apiData.embeddings.embeddings;

    // Token buttons
    const embedTokensEl = $('#embed-tokens');
    embedTokensEl.innerHTML = '';
    const highlightTokens = [];

    tokens.forEach((t, i) => {
      const cleanWord = t.text.replace(/^Ġ/, '').toLowerCase();
      const match = embeddings.find(e => e.word === cleanWord);
      const btn = document.createElement('button');
      btn.className = `embed-token tc${i % 8}`;
      btn.textContent = t.text;
      btn.dataset.word = cleanWord;
      if (match) {
        highlightTokens.push({ word: cleanWord, idx: i });
      }
      btn.addEventListener('click', () => selectEmbedToken(cleanWord, i));
      embedTokensEl.appendChild(btn);
    });

    // Draw scatter
    const canvas = $('#scatter-canvas');
    VIZ.drawScatter(canvas, embeddings, highlightTokens, $('#scatter-legend'));

    // Show first token's vector
    if (highlightTokens.length) {
      selectEmbedToken(highlightTokens[0].word, 0);
    } else {
      $('#vector-display').textContent = 'Tokens not in embedding vocabulary preview';
    }
  }

  async function selectEmbedToken(word, colorIdx) {
    // Highlight button
    $$('.embed-token').forEach(btn => {
      btn.classList.toggle('selected', btn.dataset.word === word);
    });

    const match = apiData.embeddings.embeddings.find(e => e.word === word);
    if (match) {
      // Fetch vector for this specific word if not cached
      if (!match.vector) {
        const res = await fetch(`/api/embeddings?word1=${enc(word)}`);
        const data = await res.json();
        const found = data.embeddings.find(e => e.word === word);
        if (found && found.vector) match.vector = found.vector;
      }
      if (match.vector) {
        const vecStr = match.vector.map(v => v.toFixed(3)).join(', ');
        $('#vector-display').innerHTML =
          `<strong style="color:${VIZ.COLORS[colorIdx % 8]}">"${word}"</strong> → [${vecStr}]` +
          `<br><span style="color:#64748b">Category: ${match.category} • ${match.vector.length} dimensions (real models: 768–12288)</span>`;
      } else {
        $('#vector-display').innerHTML =
          `<strong style="color:${VIZ.COLORS[colorIdx % 8]}">"${word}"</strong> — category: ${match.category}`;
      }
    }
  }

  // ─── Step 3: Attention ────────────────────────────
  async function loadAttention() {
    if (!apiData.attention) {
      const res = await fetch(`/api/attention?text=${enc(phrase)}`);
      apiData.attention = await res.json();
    }

    const { tokens, heads, averaged } = apiData.attention;
    const headNames = ['Averaged', ...Object.keys(heads)];
    const matrices = { Averaged: averaged };
    Object.assign(matrices, heads);

    // Build tabs
    const tabsEl = $('#attention-tabs');
    tabsEl.innerHTML = '';
    headNames.forEach((name, i) => {
      const tab = document.createElement('button');
      tab.className = `head-tab${i === 0 ? ' active' : ''}`;
      const labels = { local: '🔍 Local', global: '🌐 Global', positional: '📍 Positional', content: '💡 Content', Averaged: '📊 Averaged' };
      tab.textContent = labels[name] || name;
      tab.addEventListener('click', () => {
        $$('.head-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        VIZ.drawAttention($('#attention-canvas'), tokens, matrices[name]);
        showAttentionInfo(name);
      });
      tabsEl.appendChild(tab);
    });

    VIZ.drawAttention($('#attention-canvas'), tokens, averaged);
    showAttentionInfo('Averaged');
  }

  function showAttentionInfo(headName) {
    const descriptions = {
      Averaged: 'Average across all attention heads — shows the overall attention pattern.',
      local: 'Local attention — each token primarily attends to its neighbors. Common in early layers.',
      global: 'Global attention — certain "anchor" tokens (articles, punctuation) receive attention from everywhere.',
      positional: 'Positional attention — tokens attend strongly to the beginning of the sequence.',
      content: 'Content-based attention — semantically related tokens (pronouns → nouns) attend to each other.'
    };
    $('#attention-arcs').textContent = descriptions[headName] || '';
  }

  // ─── Step 4: Forward Pass ─────────────────────────
  function loadForwardPass() {
    VIZ.buildForwardPass($('#forward-pass-diagram'));
  }

  // ─── Step 5: Predict ──────────────────────────────
  async function loadPredict() {
    updateCurrentPhrase();
    await fetchPredictions();
  }

  async function fetchPredictions() {
    const fullPhrase = getFullPhrase();
    const temp = $('#temp-slider').value;
    const topp = $('#topp-slider').value;

    const res = await fetch(`/api/predict?text=${enc(fullPhrase)}&temperature=${temp}&top_p=${topp}`);
    apiData.predictions = await res.json();

    VIZ.drawPredictions($('#predict-canvas'), apiData.predictions.predictions);
  }

  function handleSliderChange() {
    $('#temp-val').textContent = parseFloat($('#temp-slider').value).toFixed(1);
    $('#topp-val').textContent = parseFloat($('#topp-slider').value).toFixed(2);
    if (currentStep === 5) fetchPredictions();
  }

  function sampleToken() {
    if (!apiData.predictions || !apiData.predictions.predictions.length) return;

    // Weighted random sampling
    const preds = apiData.predictions.predictions;
    const r = Math.random();
    let cum = 0;
    let chosen = preds[0].token;
    for (const p of preds) {
      cum += p.prob;
      if (r <= cum) { chosen = p.token; break; }
    }

    sampledTokens.push(chosen);
    updateCurrentPhrase();
    fetchPredictions();
  }

  function getFullPhrase() {
    return phrase + (sampledTokens.length ? ' ' + sampledTokens.join(' ') : '');
  }

  function updateCurrentPhrase() {
    const el = $('#current-phrase');
    if (sampledTokens.length === 0) {
      el.innerHTML = getFullPhrase();
    } else {
      el.innerHTML = escHtml(phrase) + ' <span class="sampled">' + escHtml(sampledTokens.join(' ')) + '</span>';
    }
  }

  // ─── Technical Mode ───────────────────────────────
  function toggleTechnical() {
    technicalMode = !technicalMode;
    toggleTech.classList.toggle('active', technicalMode);
    toggleTech.querySelector('.toggle-label').textContent = technicalMode ? 'Technical' : 'Simple';
    $$('.technical-detail').forEach(el => {
      el.classList.toggle('visible', technicalMode);
    });
  }

  // ─── Start Over ───────────────────────────────────
  function startOver() {
    phrase = '';
    sampledTokens = [];
    apiData = {};
    phraseInput.value = '';
    goToStep(0);
    phraseInput.focus();
  }

  // ─── Helpers ──────────────────────────────────────
  function enc(s) { return encodeURIComponent(s); }
  function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // Boot
  init();
})();
