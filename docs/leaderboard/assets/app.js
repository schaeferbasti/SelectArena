// SelectArena leaderboard: a static page over the generated files in data/.
//
// data/meta.json lists every published subset; each subset folder
// data/subsets/tasks_<t>/models_<m>/budgets_<b>/ holds elo_explorer.html and leaderboard_table.html,
// which are embedded as iframes, and pareto.svg / pareto.csv. The folder names mirror
// selectarena.website.subsets.Subset.rel_path. The lasagna plots are in data/lasagna/ (meta.lasagna):
// stability/tasks_<t>/<variant>.svg and validity/noise_<level>/tasks_<t>/<variant>.svg.
// Every embedded page posts {saFrameHeight: <px>} to this window; the listener below resizes its iframe.
(function () {
  'use strict';

  var ALL = 'all';
  var TASK_LABELS = {
    all: 'All tasks',
    classification: 'Classification',
    binary: 'Binary',
    multiclass: 'Multiclass',
    regression: 'Regression'
  };
  var DEFAULT_VARIANT_LABELS = {
    plain: 'Heat map',
    overlay_top_3: 'Top-3 outlined',
    binary_thr_003: 'Within 0.03 of the best'
  };
  var TABS = ['leaderboard', 'tradeoffs', 'stability', 'validity'];
  // Old tab names (links to earlier versions of the site).
  var TAB_ALIASES = {
    summary: 'leaderboard', about: 'leaderboard', performance: 'tradeoffs', data: 'tradeoffs', datasets: 'tradeoffs',
    robustness: 'stability'
  };
  var DEFAULTS = {
    tab: 'leaderboard', tasks: ALL, model: ALL, budget: ALL,
    stasks: ALL, svariant: 'plain', vtasks: ALL, vvariant: 'plain', noise: '', group: 'features'
  };
  var state = Object.assign({}, DEFAULTS);
  var meta = null;
  var published = {};

  function $(id) { return document.getElementById(id); }
  function key(s) { return s.tasks + '|' + s.model + '|' + s.budget; }
  function subsetDir(s) {
    return 'data/subsets/tasks_' + s.tasks + '/models_' + s.model + '/budgets_' + s.budget + '/';
  }
  function withAxis(axis, value) {
    var s = { tasks: state.tasks, model: state.model, budget: state.budget };
    s[axis] = value;
    return s;
  }

  // 0.5 -> "0.5", 0.75 -> "0.75", 1 -> "1.0"
  function noiseLabel(value) { return Number(value).toFixed(2).replace(/0$/, ''); }
  // Folder name of a noise level, as Python's f"{noise:g}": 0.5 -> "0.5", 1 -> "1"
  function noiseKey(value) { return String(Number(value)); }

  // Short chip labels: the control's label already says "Tasks", "Model" or "Budget".
  function chipLabel(axis, value) {
    if (value === ALL) return 'All';
    if (axis === 'budget') return budgetLabel(Number(value), meta.total_budget).replace('Budget ', '');
    return null;
  }
  function budgetLabel(index, total) {
    var label = 'Budget ' + (index + 1);
    if (total && index === 0) return label + ' (smallest)';
    if (total && index === total - 1) return label + ' (largest)';
    return label;
  }

  // ---- URL hash: #tab=...&tasks=...&model=... (shareable selections) ----
  function readHash() {
    var params = new URLSearchParams(window.location.hash.slice(1));
    Object.keys(DEFAULTS).forEach(function (k) {
      var v = params.get(k);
      if (v) state[k] = v;
    });
    state.tab = TAB_ALIASES[state.tab] || state.tab;
    if (TABS.indexOf(state.tab) < 0) state.tab = DEFAULTS.tab;
  }
  function writeHash() {
    var params = new URLSearchParams();
    Object.keys(DEFAULTS).forEach(function (k) {
      if (state[k] && state[k] !== DEFAULTS[k]) params.set(k, state[k]);
    });
    var hash = params.toString();
    try {
      history.replaceState(null, '', hash ? '#' + hash : window.location.pathname + window.location.search);
    } catch (e) { /* sandboxed frame: the selection still works, it just is not in the URL */ }
  }

  // ---- iframes and images ----
  function setFrame(frame, src) {
    // Anonymized build: Anonymous GitHub refuses to show its pages inside frames, so each embedded
    // page is loaded as a script (<page>.html.js, which sets SA_PAGES[<page>]) and shown via srcdoc.
    if (frame.getAttribute('data-src') === src) return;
    frame.setAttribute('data-src', src);
    loadPage(src).then(function (html) {
      if (frame.getAttribute('data-src') === src) frame.srcdoc = html;
    });
  }
  var pageLoads = {};
  function loadPage(src) {
    if (!pageLoads[src]) {
      pageLoads[src] = new Promise(function (resolve) {
        var script = document.createElement('script');
        script.src = src + '.js';
        script.onload = function () { resolve((window.SA_PAGES || {})[src] || ''); };
        script.onerror = function () { delete pageLoads[src]; resolve(''); };
        document.head.appendChild(script);
      });
    }
    return pageLoads[src];
  }
  window.addEventListener('message', function (event) {
    var data = event.data;
    if (!data || typeof data.saFrameHeight !== 'number') return;
    document.querySelectorAll('iframe.sa-frame').forEach(function (frame) {
      if (frame.contentWindow === event.source) frame.style.height = data.saFrameHeight + 'px';
    });
  });

  // A row of chips for one choice. `options` are {value, label, disabled?}; `onPick` gets the value.
  function renderChips(containerId, options, selected, onPick) {
    var container = $(containerId);
    container.textContent = '';
    options.forEach(function (option) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'chip';
      button.textContent = option.label;
      button.setAttribute('aria-pressed', String(option.value === selected));
      if (option.disabled && option.value !== selected) {
        button.disabled = true;
        button.title = option.disabledTitle || 'Not available for this selection';
      }
      button.addEventListener('click', function () { onPick(option.value); writeHash(); });
      container.appendChild(button);
    });
    updateConfigState(container);
  }

  // The controls of each tab sit in a collapsed <details> ("Customize"); its button shows the
  // current choice of every control, e.g. "Tasks: All · Model: All · Budget: All".
  function updateConfigState(container) {
    var details = container.closest('details.config');
    if (!details) return;
    var parts = [];
    details.querySelectorAll('.control').forEach(function (control) {
      if (control.hidden) return;
      var label = control.querySelector('.control-label');
      var chip = control.querySelector('.chip[aria-pressed="true"]');
      if (label && chip) parts.push(label.textContent + ': ' + chip.textContent);
    });
    details.querySelector('.config-state').textContent = parts.join(' · ');
  }

  // ---- tabs ----
  function showTab(tab) {
    state.tab = tab;
    TABS.forEach(function (t) {
      $('tab-' + t).setAttribute('aria-selected', String(t === tab));
      $('tab-' + t).tabIndex = t === tab ? 0 : -1;
      $('panel-' + t).hidden = t !== tab;
    });
    if (tab === 'stability') renderStability();
    if (tab === 'validity') renderValidity();
    writeHash();
  }
  // A tab switched while scrolled down opens at its top (right below the tabs that stay on screen).
  function scrollToTabs() {
    // The tabs' own position is the stuck one while they stick: measure from the intro above them.
    var tabs = document.querySelector('.tabs');
    var header = document.querySelector('.header');
    var intro = document.querySelector('.intro');
    var top = intro.getBoundingClientRect().bottom + window.scrollY + parseFloat(getComputedStyle(tabs).marginTop)
      - header.offsetHeight;
    if (getComputedStyle(tabs).position === 'sticky' && window.scrollY > top) window.scrollTo(0, top);
  }
  function initTabs() {
    document.querySelectorAll('[role="tab"]').forEach(function (button) {
      button.addEventListener('click', function () { showTab(button.dataset.tab); scrollToTabs(); });
      button.addEventListener('keydown', function (event) {
        var i = TABS.indexOf(button.dataset.tab);
        if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
          var next = TABS[(i + (event.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
          showTab(next);
          $('tab-' + next).focus();
        }
      });
    });
  }

  // ---- performance ----
  function axisOptions() {
    var taskTypes = meta.task_types || [ALL];
    var models = meta.models || {};
    return {
      tasks: Object.keys(TASK_LABELS)
        .filter(function (k) { return taskTypes.indexOf(k) >= 0; })
        .map(function (k) { return { value: k, label: chipLabel('tasks', k) || TASK_LABELS[k] }; }),
      model: [{ value: ALL, label: 'All' }].concat(
        Object.keys(models).map(function (k) { return { value: k, label: models[k] }; })),
      budget: [{ value: ALL, label: 'All' }].concat(
        (meta.budgets || []).map(function (b) { return { value: String(b), label: chipLabel('budget', String(b)) }; }))
    };
  }

  // The task type, model and budget chips of the Performance and Leaderboard tabs share the
  // selection; `prefix` is '' (Performance Trade-offs) or 'lb-' (Leaderboard).
  function renderSelectionChips(prefix) {
    var options = axisOptions();
    ['tasks', 'model', 'budget'].forEach(function (axis) {
      renderChips(prefix + 'chips-' + axis, options[axis].map(function (o) {
        return {
          value: o.value, label: o.label,
          disabled: !published[key(withAxis(axis, o.value))],
          disabledTitle: 'No leaderboard for this combination'
        };
      }), state[axis], function (value) { state[axis] = value; renderSelection(); });
    });
  }
  function scopeText(s) {
    return [
      TASK_LABELS[s.tasks] || s.tasks,
      s.model === ALL ? 'all downstream models' : (meta.models[s.model] || s.model),
      s.budget === ALL ? 'all budgets' : budgetLabel(Number(s.budget), meta.total_budget)
    ].join(' · ');
  }
  // One short line under the chips (which already show the selection).
  function writeSummary(element, s, info) {
    element.title = scopeText(s);
    element.textContent = info
      ? info.n_methods + ' methods · ' + info.n_datasets + ' datasets · ' +
        info.n_comparisons.toLocaleString('en-US') + ' results compared'
      : '';
  }
  function renderSelection() {
    renderLeaderboard();
    renderTradeoffs();
  }

  // ---- trade-offs (Pareto front of performance vs. selection time) ----
  function renderTradeoffs() {
    renderSelectionChips('');
    var info = published[key(state)];
    writeSummary($('summary'), state, info);
    $('subset-missing').hidden = !!info;
    $('subset-content').hidden = !info;
    if (!info) return;
    var dir = subsetDir(state);
    $('pareto-missing').hidden = !!info.pareto;
    $('pareto-figure').hidden = !info.pareto;
    if (info.pareto) {
      setFrame($('plot-pareto'), dir + 'pareto.html');
    }
  }

  // ---- leaderboard (Elo figure above the table) ----
  function renderLeaderboard() {
    renderSelectionChips('lb-');
    var info = published[key(state)];
    writeSummary($('lb-summary'), state, info);
    $('lb-missing').hidden = !!info;
    $('lb-content').hidden = !info;
    if (!info) return;
    var dir = subsetDir(state);
    setFrame($('frame-figure'), dir + 'elo_explorer.html');
    var table = $('frame-table');
    var src = dir + 'leaderboard_table.html';
    if (table.getAttribute('data-src') !== src) table.style.height = (info.n_methods * 37 + 90) + 'px';
    setFrame(table, src);
  }

  // ---- lasagna plots (stability and validity) ----
  function variantOptions() {
    var lasagna = (meta && meta.lasagna) || {};
    var labels = lasagna.variant_labels || DEFAULT_VARIANT_LABELS;
    return (lasagna.variants || Object.keys(DEFAULT_VARIANT_LABELS)).map(function (v) {
      return { value: v, label: labels[v] || v };
    });
  }
  function taskOptions(available) {
    return Object.keys(TASK_LABELS)
      .filter(function (k) { return available.indexOf(k) >= 0; })
      .map(function (k) { return { value: k, label: chipLabel('tasks', k) || TASK_LABELS[k] }; });
  }
  function showLasagna(prefix, folder, tasks, variant) {
    var dir = 'data/' + folder + '/tasks_' + tasks + '/';
    setFrame($('plot-' + prefix), dir + variant + '.html');
  }

  function renderStability() {
    var info = meta && meta.lasagna && meta.lasagna.stability;
    var hasTable = !!(meta && meta.has_stability);
    $('stability-missing').hidden = !!(info || hasTable);
    $('stability-content').hidden = !(info || hasTable);
    if (hasTable) setFrame($('frame-stability'), 'data/stability_table.html');
    var figure = $('plot-stability').closest('figure');
    figure.hidden = !info;
    $('chips-stab-tasks').closest('.controls').hidden = !info;
    if (!info) return;
    if (info.task_types.indexOf(state.stasks) < 0) state.stasks = info.task_types[0];
    renderChips('chips-stab-tasks', taskOptions(info.task_types), state.stasks,
      function (value) { state.stasks = value; renderStability(); });
    var variants = variantOptions();
    if (!variants.some(function (v) { return v.value === state.svariant; })) state.svariant = variants[0].value;
    renderChips('chips-stab-variant', variants, state.svariant,
      function (value) { state.svariant = value; renderStability(); });
    showLasagna('stability', info.path, state.stasks, state.svariant);
  }

  function renderValidity() {
    var levels = (meta && meta.lasagna && meta.lasagna.validity) || [];
    var tables = (meta && meta.validity_tables) || [];
    $('validity-missing').hidden = levels.length > 0 || tables.length > 0;
    $('validity-content').hidden = !(levels.length || tables.length);

    // Noise levels from the plots, or else from the tables.
    var noises = (levels.length ? levels : tables).map(function (l) { return noiseKey(l.noise); });
    if (!noises.length) return;
    if (noises.indexOf(state.noise) < 0) {
      var preferred = meta.validity_noise != null ? noiseKey(meta.validity_noise) : noises[0];
      state.noise = noises.indexOf(preferred) >= 0 ? preferred : noises[0];
    }
    renderChips('chips-val-noise', noises.map(function (n) {
      return { value: n, label: noiseLabel(n) + ' per feature' };
    }), state.noise, function (value) { state.noise = value; renderValidity(); });

    var table = tables.filter(function (t) { return noiseKey(t.noise) === state.noise; })[0];
    $('frame-validity').hidden = !table;
    if (table) setFrame($('frame-validity'), 'data/' + table.path);

    var level = levels.filter(function (l) { return noiseKey(l.noise) === state.noise; })[0];
    $('plot-validity').closest('figure').hidden = !level;
    $('chips-val-tasks').closest('.control').hidden = !level;
    $('chips-val-variant').closest('.control').hidden = !level;
    if (!level) return;
    if (level.task_types.indexOf(state.vtasks) < 0) state.vtasks = level.task_types[0];
    renderChips('chips-val-tasks', taskOptions(level.task_types), state.vtasks,
      function (value) { state.vtasks = value; renderValidity(); });
    var variants = variantOptions();
    if (!variants.some(function (v) { return v.value === state.vvariant; })) state.vvariant = variants[0].value;
    renderChips('chips-val-variant', variants, state.vvariant,
      function (value) { state.vvariant = value; renderValidity(); });
    showLasagna('validity', level.path, state.vtasks, state.vvariant);
  }

  // ---- per dataset characteristic, model, task type and budget, and the ablations (Performance tab;
  // all results, not filtered by the chips above) ----
  // Groupings with a section of their own are not offered under "Group by".
  var OWN_SECTION = ['model', 'task_type', 'budget'];
  // meta.analysis: {path, groupings: [{key, label}], improvement_by_model, improvement_by_task_type, rank_by_size, elo_without_timeouts}.
  function renderAnalysis() {
    var info = meta && meta.analysis;
    var all = (info && info.groupings) || [];
    var groupings = all.filter(function (g) { return OWN_SECTION.indexOf(g.key) < 0; });
    if (!groupings.length) groupings = all;
    $('analysis-content').hidden = !groupings.length;
    if (!groupings.length) return;
    if (!groupings.some(function (g) { return g.key === state.group; })) state.group = groupings[0].key;
    renderChips('chips-group', groupings.map(function (g) { return { value: g.key, label: g.label }; }),
      state.group, function (value) { state.group = value; renderAnalysis(); });
    var dir = 'data/' + info.path + '/';
    setFrame($('plot-group'), dir + 'heatmap_' + state.group + '.html');
    $('model-section').hidden = !info.improvement_by_model;
    if (info.improvement_by_model) setFrame($('plot-model'), dir + 'improvement_by_model.html');
    $('task-section').hidden = !info.improvement_by_task_type;
    if (info.improvement_by_task_type) setFrame($('plot-task'), dir + 'improvement_by_task_type.html');
    // The heat map by budget, in a section of its own.
    var byBudget = all.some(function (g) { return g.key === 'budget'; });
    $('budget-section').hidden = !byBudget;
    if (byBudget) setFrame($('plot-budget'), dir + 'heatmap_budget.html');
    $('timeout-section').hidden = !info.elo_without_timeouts;
    if (info.elo_without_timeouts) setFrame($('plot-timeout'), dir + 'elo_no_timeouts.html');
    $('size-figure').hidden = !info.rank_by_size;
    if (info.rank_by_size) setFrame($('plot-size'), dir + 'rank_by_size.html');
  }

  // ---- complete results (Performance tab; not filtered by the chips) ----
  function loadResults() {
    var hasExplorer = !!meta.has_results_explorer;
    $('frame-results').hidden = !hasExplorer;
    if (hasExplorer) setFrame($('frame-results'), 'data/results_explorer.html');
  }

  // ---- theme: light / dark button; without a choice the system setting decides ----
  var darkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var chosenTheme = null;  // 'light', 'dark' or null (follow the system); saved in localStorage when possible
  try {
    var stored = localStorage.getItem('selectarena-theme');
    if (stored === 'light' || stored === 'dark') chosenTheme = stored;
  } catch (e) { /* private window or blocked storage: the choice lasts for this visit */ }
  function effectiveTheme() {
    return chosenTheme || (darkQuery && darkQuery.matches ? 'dark' : 'light');
  }
  function setThemeAttribute(root) {
    if (chosenTheme) root.setAttribute('data-theme', chosenTheme); else root.removeAttribute('data-theme');
  }
  // Pass the choice on to the embedded pages (same origin: set it directly; else by message).
  function themeFrame(frame) {
    try {
      var root = frame.contentDocument && frame.contentDocument.documentElement;
      if (root) { setThemeAttribute(root); return; }
    } catch (e) { /* cross-origin frame */ }
    if (frame.contentWindow) frame.contentWindow.postMessage({ saTheme: chosenTheme }, '*');
  }
  function renderTheme() {
    setThemeAttribute(document.documentElement);
    var dark = effectiveTheme() === 'dark';
    var button = $('theme-button');
    button.classList.toggle('is-dark', dark);
    var label = dark ? 'Switch to light mode' : 'Switch to dark mode';
    button.setAttribute('aria-label', label);
    button.title = label;
    document.querySelectorAll('iframe.sa-frame').forEach(themeFrame);
  }
  function initTheme() {
    $('theme-button').addEventListener('click', function () {
      chosenTheme = effectiveTheme() === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem('selectarena-theme', chosenTheme); } catch (e) { /* not saved */ }
      renderTheme();
    });
    document.querySelectorAll('iframe.sa-frame').forEach(function (frame) {
      frame.addEventListener('load', function () { themeFrame(frame); });
    });
    if (darkQuery && darkQuery.addEventListener) darkQuery.addEventListener('change', renderTheme);
    renderTheme();
  }

  // ---- Header and tabs stay at the top on larger screens (style.css): the tabs sit right below the
  // header, whose height changes with the window width, so it is measured into --header-h. ----
  function initStickyHeader() {
    var header = document.querySelector('.header');
    if (!header) return;
    function update() {
      document.documentElement.style.setProperty('--header-h', header.getBoundingClientRect().height + 'px');
    }
    update();
    if (window.ResizeObserver) new ResizeObserver(update).observe(header);
    else window.addEventListener('resize', update);
  }

  // ---- "Cite" button: scroll to the BibTeX entry without replacing the selection in the URL hash ----
  function initCite() {
    $('cite-button').addEventListener('click', function (event) {
      event.preventDefault();
      $('citation').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }

  // ---- "Copy" button of the BibTeX entry ----
  // The Space runs inside an iframe on huggingface.co, where navigator.clipboard can be blocked:
  // fall back to selecting the text and document.execCommand('copy').
  function copyText(text) {
    function fallback() {
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      area.remove();
      return ok ? Promise.resolve() : Promise.reject(new Error('copy failed'));
    }
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).catch(fallback);
    }
    return fallback();
  }
  function initCopyCitation() {
    var button = $('copy-citation');
    button.addEventListener('click', function () {
      copyText($('citation-bibtex').textContent).then(function () {
        button.textContent = 'Copied!';
      }, function () {
        button.textContent = 'Press Ctrl+C';  // nothing could copy: leave the text selected instead
        var range = document.createRange();
        range.selectNodeContents($('citation-bibtex'));
        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }).then(function () {
        button.classList.add('is-copied');
        setTimeout(function () { button.textContent = 'Copy'; button.classList.remove('is-copied'); }, 2000);
      });
    });
  }

  // ---- start ----
  function start() {
    readHash();
    initTabs();
    initStickyHeader();
    initTheme();
    (window.SA_META ? Promise.resolve({ ok: true, json: function () { return window.SA_META; } }) : fetch('data/meta.json', { cache: 'no-cache' }))
      .then(function (response) {
        if (!response.ok) throw new Error('HTTP ' + response.status);
        return response.json();
      })
      .then(function (data) {
        meta = data;
        (meta.subsets || []).forEach(function (s) { published[key(s)] = s; });
        if (!published[key(state)]) { state.tasks = ALL; state.model = ALL; state.budget = ALL; }
        $('leaderboard-content').hidden = false;
        $('lb-root').hidden = false;
        renderSelection();
        renderAnalysis();
        loadResults();
        showTab(state.tab);
      })
      .catch(function () {
        meta = null;
        $('no-results').hidden = false;
        $('lb-no-results').hidden = false;
        showTab(state.tab);
      });
  }

  start();
})();
