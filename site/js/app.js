'use strict';

var CONFIG = { dataRoot: 'data', animDuration: 800, staggerMs: 30 };

var INDEX_ACCENTS = {
  run_risk: 'crimson',
  alm_mismatch: 'amber',
  deposit_competition: 'emerald',
  funding_fragility: 'indigo'
};

/* ── Helpers ─────────────────────────────────── */
function fetchJSON(path) {
  return fetch(CONFIG.dataRoot + '/' + path)
    .then(function (res) {
      if (!res.ok) throw new Error('Fetch failed: ' + path + ' (' + res.status + ')');
      return res.json();
    });
}

function fmt(n) { return Number(n).toLocaleString(); }
function pct(n) { return (n * 100).toFixed(1) + '%'; }
function fmtRate(value) {
  if (value == null || Number.isNaN(Number(value))) return '\u2014';
  return Number(value).toFixed(2) + '%';
}
function fmtBp(value) {
  if (value == null || Number.isNaN(Number(value))) return '\u2014';
  return Number(value).toFixed(0) + ' bp';
}
function metricText(value, digits, suffix) {
  suffix = suffix || '';
  if (value == null || Number.isNaN(Number(value))) return '\u2014';
  return Number(value).toFixed(digits) + suffix;
}
function treasuryRegimeLabel(regime) {
  if (!regime || regime.slope_10y_3m_bp == null || Number.isNaN(Number(regime.slope_10y_3m_bp))) return '\u2014';
  return Number(regime.slope_10y_3m_bp) < 0 ? 'Inverted' : 'Upward';
}
function fmtPeerGroup(id) {
  var map = { community: 'Community', regional: 'Regional', large_regional: 'Large Regional', very_large: 'Very Large' };
  return map[id] || id;
}

function riskClass(score) {
  if (score >= 80) return 'risk-critical';
  if (score >= 60) return 'risk-high';
  if (score >= 40) return 'risk-elevated';
  if (score >= 20) return 'risk-moderate';
  return 'risk-low';
}

function el(tag, cls, html) {
  var e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

function formatBuildDate(iso) {
  if (!iso) return null;
  var date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.getUTCFullYear() + '-' +
    String(date.getUTCMonth() + 1).padStart(2, '0') + '-' +
    String(date.getUTCDate()).padStart(2, '0');
}

/* ── Counter Animation ───────────────────────── */
function animateValue(element, target, duration, suffix) {
  suffix = suffix || '';
  var startTime = performance.now();
  function tick(now) {
    var elapsed = now - startTime;
    var progress = Math.min(elapsed / duration, 1);
    var eased = 1 - Math.pow(1 - progress, 3);
    var current = Math.round(target * eased);
    element.textContent = fmt(current) + suffix;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ── Scroll Observer ─────────────────────────── */
function initScrollAnimations() {
  var staggerIndex = 0;
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      var delay = (staggerIndex++) * CONFIG.staggerMs;
      setTimeout(function () {
        entry.target.classList.add('is-visible');
        var counter = entry.target.querySelector('[data-count-to]');
        if (counter && !counter.dataset.counted) {
          var target = parseInt(counter.dataset.countTo, 10);
          var suffix = counter.dataset.countSuffix || '';
          animateValue(counter, target, CONFIG.animDuration, suffix);
          counter.dataset.counted = 'true';
        }
      }, delay);
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.05 });

  document.querySelectorAll('.animate-on-scroll').forEach(function (el) {
    observer.observe(el);
  });
}

/* ── Render: Hero Metrics ────────────────────── */
function renderHeroMetrics(manifest) {
  var p = manifest.pipeline;
  var runRisk = (manifest.indices || []).find(function (idx) { return idx.id === 'run_risk'; }) || {};
  function setMetric(id, value, suffix) {
    var e = document.getElementById(id);
    if (e) {
      if (value == null || Number.isNaN(Number(value))) {
        e.textContent = '\u2014';
        return;
      }
      e.dataset.countTo = value;
      e.dataset.countSuffix = suffix || '';
      e.textContent = '0' + (suffix || '');
    }
  }
  setMetric('metric-banks', p.unique_banks, '');
  setMetric('metric-quarters', p.quarters, '');
  setMetric('metric-auc', runRisk.failure_auc != null ? Math.round(runRisk.failure_auc * 100) : null, '%');
  setMetric('metric-failures', p.failures_tested, '');

  var buildNote = document.getElementById('metric-build-note');
  if (buildNote) {
    var d = formatBuildDate(manifest.generated_at);
    buildNote.textContent = d ? 'Data build: ' + d + ' \u2022 ' + fmt(p.bank_quarters) + ' bank-quarters' : '';
  }
}

function renderTreasuryBackdrop(manifest) {
  var regime = manifest.treasury_regime || {};
  var label = document.getElementById('treasury-regime-label');
  var y10 = document.getElementById('treasury-10y-level');
  var slope = document.getElementById('treasury-slope');
  var note = document.getElementById('treasury-backdrop-note');
  if (label) label.textContent = treasuryRegimeLabel(regime);
  if (y10) y10.textContent = fmtRate(regime.y10);
  if (slope) slope.textContent = fmtBp(regime.slope_10y_3m_bp);
  if (note) {
    if (regime.yield_date) {
      note.textContent = 'Latest observed rate backdrop for the enriched panel: ' + regime.yield_date +
        ' • 2Y ' + fmtRate(regime.y2) + ' • 10Y ' + fmtRate(regime.y10);
    } else {
      note.textContent = 'Latest observed rate backdrop for the enriched panel: \u2014';
    }
  }
}

/* ── Render: Index Cards ─────────────────────── */
function renderIndexCards(manifest) {
  var grid = document.getElementById('panels-grid');
  if (!grid) return;

  manifest.indices.forEach(function (idx) {
    var accent = INDEX_ACCENTS[idx.id] || 'indigo';
    var card = el('div', 'panel-card animate-on-scroll');
    card.setAttribute('data-panel-accent', accent);

    var topBanksHtml = '';
    if (idx.top_banks && idx.top_banks.length > 0) {
      topBanksHtml = '<div class="panel-card-top-banks">' +
        '<div class="panel-top-banks-label">Highest risk banks</div>' +
        idx.top_banks.slice(0, 3).map(function (b) {
          return '<div class="panel-bank-row">' +
            '<span class="panel-bank-name">' + b.name + '</span>' +
            '<span class="panel-bank-score">' + b.score.toFixed(1) + '</span>' +
          '</div>';
        }).join('') +
      '</div>';
    }

    card.innerHTML =
      '<div class="panel-card-accent"></div>' +
      '<div class="panel-card-body">' +
        '<h3 class="panel-card-title">' + idx.title + '</h3>' +
        '<p class="panel-card-desc">' + idx.description + '</p>' +
        '<div class="panel-card-stats">' +
          '<div class="panel-stat"><span class="panel-stat-value">' + metricText(idx.failure_auc, 2, '') + '</span><span class="panel-stat-label">Failure AUC</span></div>' +
          '<div class="panel-stat"><span class="panel-stat-value">' + metricText(idx.failure_recall_20 != null ? idx.failure_recall_20 * 100 : null, 0, '%') + '</span><span class="panel-stat-label">Recall @20%</span></div>' +
          '<div class="panel-stat"><span class="panel-stat-value">' + fmt(idx.bank_count) + '</span><span class="panel-stat-label">Banks</span></div>' +
        '</div>' +
        topBanksHtml +
        '<a href="league.html?index=' + idx.id + '" class="panel-card-cta">View league table &rarr;</a>' +
      '</div>';

    grid.appendChild(card);
  });
}

function renderMethodology(manifest) {
  var grid = document.getElementById('methodology-grid');
  if (!grid) return;

  var methods = manifest.index_methodology || {};
  var order = ['run_risk', 'alm_mismatch', 'deposit_competition', 'funding_fragility'];

  grid.innerHTML = order.map(function (id) {
    var method = methods[id];
    if (!method) return '';
    var components = Array.isArray(method.components) ? method.components : [];
    return '<article class="method-card animate-on-scroll">' +
      '<h3>' + method.title + '</h3>' +
      '<p>' + method.summary + '</p>' +
      '<div class="method-formula">' + method.formula + '</div>' +
      '<ul>' +
        components.map(function (component) {
          var weight = component.weight != null ? Math.round(Number(component.weight) * 100) + '%' : '\u2014';
          return '<li><strong>' + component.label + '</strong> <span class="component-meta">weight ' + weight + '</span></li>';
        }).join('') +
      '</ul>' +
      '<p>' + method.scale_note + '</p>' +
    '</article>';
  }).join('');
}

/* ── Render: Index Comparison ────────────────── */
function renderComparison(manifest) {
  var grid = document.getElementById('comparison-grid');
  if (!grid) return;

  var aucValues = manifest.indices.map(function (i) { return i.failure_auc; }).filter(function (v) { return v != null && !Number.isNaN(Number(v)); });
  var maxAuc = aucValues.length > 0 ? Math.max.apply(null, aucValues) : null;

  manifest.indices.forEach(function (idx) {
    var accent = INDEX_ACCENTS[idx.id] || 'indigo';
    var aucW = maxAuc && idx.failure_auc != null ? (idx.failure_auc / maxAuc) * 100 : 0;
    var recallW = idx.failure_recall_20 != null ? idx.failure_recall_20 * 100 : 0;

    var item = el('div', 'comparison-item');
    item.setAttribute('data-panel-accent', accent);
    item.innerHTML =
      '<div class="comparison-header">' +
        '<span class="comparison-title">' + idx.title + '</span>' +
        (idx.id === 'run_risk' ? '<span class="comparison-badge">Best</span>' : '') +
      '</div>' +
      '<div class="comparison-bars">' +
        compRow('AUC', aucW, metricText(idx.failure_auc, 4, '')) +
        compRow('Recall @20%', recallW, metricText(idx.failure_recall_20 != null ? idx.failure_recall_20 * 100 : null, 1, '%')) +
      '</div>';
    grid.appendChild(item);
  });
}

function compRow(label, width, value) {
  return '<div class="comparison-row">' +
    '<span class="comparison-label">' + label + '</span>' +
    '<div class="comparison-bar-track"><div class="comparison-bar-fill" style="width:' + width + '%"></div></div>' +
    '<span class="comparison-value">' + value + '</span>' +
  '</div>';
}

/* ── Render: Stress Episodes ─────────────────── */
function renderEpisodes(manifest) {
  var section = document.getElementById('stress-episodes');
  var grid = document.getElementById('episodes-grid');
  if (!grid) return;

  if (!manifest.stress_episodes || manifest.stress_episodes.length === 0) {
    if (section) section.style.display = 'none';
    return;
  }

  manifest.stress_episodes.forEach(function (ep) {
    var card = el('div', 'episode-card animate-on-scroll');
    card.innerHTML =
      '<div class="episode-year">' + ep.year + '</div>' +
      '<div class="episode-label">' + ep.label + '</div>' +
      '<p class="episode-finding">' + ep.finding + '</p>';
    grid.appendChild(card);
  });
}

/* ── Smooth Scroll ───────────────────────────── */
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(function (link) {
    link.addEventListener('click', function (e) {
      var href = link.getAttribute('href');
      if (href === '#') return;
      var target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        var h = document.querySelector('.header');
        var offset = h ? h.offsetHeight + 20 : 20;
        var top = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: top, behavior: 'smooth' });
        var nav = document.getElementById('nav');
        if (nav) nav.classList.remove('open');
      }
    });
  });
}

/* ── Search ───────────────────────────────────── */
var searchIndex = [];

function loadSearchIndex() {
  return fetchJSON('league.json').then(function (banks) {
    searchIndex = banks;
    initSearch();
  }).catch(function () {});
}

function initSearch() {
  var input = document.getElementById('search-input');
  var results = document.getElementById('search-results');
  if (!input || !results) return;

  input.addEventListener('input', function () {
    var q = input.value.toLowerCase().trim();
    if (q.length < 2) { results.innerHTML = ''; results.style.display = 'none'; return; }

    var matches = searchIndex.filter(function (b) {
      return (b.name || '').toLowerCase().includes(q) ||
        String(b.cert).includes(q) ||
        (b.state || '').toLowerCase().includes(q);
    }).slice(0, 12);

    if (matches.length === 0) {
      results.innerHTML = '<div class="search-empty">No banks found</div>';
      results.style.display = 'block'; return;
    }

    results.innerHTML = matches.map(function (b) {
      return '<a href="bank.html?cert=' + b.cert + '" class="search-result-item">' +
        '<div class="search-result-name">' + b.name + '</div>' +
        '<div class="search-result-meta">' +
          '<span class="search-result-panel">' + fmtPeerGroup(b.peer_group) + '</span>' +
          '<span class="search-result-domain">' + (b.state || '') + '</span>' +
          '<span class="search-result-score">' + (b.funding_fragility || 0).toFixed(0) + '</span>' +
        '</div></a>';
    }).join('');
    results.style.display = 'block';
  });

  document.addEventListener('click', function (e) {
    if (!e.target.closest('.search-container')) results.style.display = 'none';
  });
}

/* ── Main ────────────────────────────────────── */
function init() {
  fetchJSON('manifest.json').then(function (manifest) {
    renderHeroMetrics(manifest);
    renderTreasuryBackdrop(manifest);
    renderIndexCards(manifest);
    renderMethodology(manifest);
    renderComparison(manifest);
    renderEpisodes(manifest);
    initScrollAnimations();
    initSmoothScroll();
    loadSearchIndex();
  }).catch(function (err) {
    console.error('bankalm init failed:', err);
    var sub = document.querySelector('.hero-subtitle');
    if (sub) {
      sub.innerHTML += '<br><small style="color:#ef4444">Data loading error: ' + err.message + '</small>';
    }
  });
}

document.addEventListener('DOMContentLoaded', init);
