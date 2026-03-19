'use strict';

var CONFIG = { dataRoot: 'data' };

function fetchJSON(p) {
  return fetch(CONFIG.dataRoot + '/' + p).then(function (r) {
    if (!r.ok) throw new Error('Fetch failed: ' + p); return r.json();
  });
}
function fmt(n) { return Number(n).toLocaleString(); }
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
  if (!regime || regime.slope_10y_3m_bp == null || Number.isNaN(Number(regime.slope_10y_3m_bp))) return 'Unavailable';
  return Number(regime.slope_10y_3m_bp) < 0 ? 'inverted curve' : 'upward-sloping curve';
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
function getParam(k) { return new URLSearchParams(location.search).get(k); }

var S = {
  indexId: null, indexMeta: null,
  banks: [], filtered: [],
  sortKey: 'score', sortDir: 'desc',
  searchTerm: '', peerGroups: new Set(),
  page: 1, perPage: 25
};

/* ── Hero ──────────────────────────────────── */
function renderHero(manifest) {
  var idx = S.indexMeta;
  document.title = idx.title + ' League Table \u2014 bankALM';
  document.getElementById('league-title').textContent = idx.title + ' League Table';
  document.getElementById('league-desc').textContent = idx.description;
  document.getElementById('league-breadcrumb').textContent = idx.title;
  document.getElementById('lh-banks').textContent = fmt(idx.bank_count);
  document.getElementById('lh-auc').textContent = metricText(idx.failure_auc, 4, '');
  document.getElementById('lh-recall').textContent = metricText(idx.failure_recall_20 != null ? idx.failure_recall_20 * 100 : null, 1, '%');
  document.getElementById('lh-failures').textContent = fmt(manifest.pipeline.failures_tested);

  var regime = manifest.treasury_regime || {};
  var regimeText = document.getElementById('league-regime-text');
  if (regimeText) {
    if (regime.yield_date) {
      regimeText.textContent = 'Latest enriched-panel backdrop (' + regime.yield_date + '): ' +
        treasuryRegimeLabel(regime) + ', 10Y Treasury ' + fmtRate(regime.y10) +
        ', 10Y-3M slope ' + fmtBp(regime.slope_10y_3m_bp) + '. This is macro context, not part of the ranking formula.';
    } else {
      regimeText.textContent = 'Treasury backdrop unavailable for this dataset slice.';
    }
  }
}

/* ── Peer Group Filters ──────────────────── */
function renderFilters(manifest) {
  var el = document.getElementById('peer-filters');
  if (!el) return;

  el.innerHTML = '<button class="filter-chip active" data-peer="all">All</button>' +
    manifest.peer_groups.map(function (pg) {
      return '<button class="filter-chip" data-peer="' + pg.id + '">' +
        pg.label + ' <span class="chip-count">' + fmt(pg.count) + '</span></button>';
    }).join('');

  el.addEventListener('click', function (e) {
    var chip = e.target.closest('.filter-chip');
    if (!chip) return;
    var peer = chip.dataset.peer;
    if (peer === 'all') {
      S.peerGroups.clear();
      el.querySelectorAll('.filter-chip').forEach(function (c) { c.classList.remove('active'); });
      chip.classList.add('active');
    } else {
      el.querySelector('[data-peer="all"]').classList.remove('active');
      chip.classList.toggle('active');
      if (S.peerGroups.has(peer)) S.peerGroups.delete(peer); else S.peerGroups.add(peer);
      if (S.peerGroups.size === 0) el.querySelector('[data-peer="all"]').classList.add('active');
    }
    applyFilters();
  });
}

/* ── Filters + Sort ────────────────────────── */
function applyFilters() {
  var q = S.searchTerm.toLowerCase();
  S.filtered = S.banks.filter(function (b) {
    if (q && !(
      (b.name || '').toLowerCase().includes(q) ||
      String(b.cert).includes(q) ||
      (b.state || '').toLowerCase().includes(q)
    )) return false;
    if (S.peerGroups.size > 0 && !S.peerGroups.has(b.peer_group)) return false;
    return true;
  });
  S.page = 1;
  sortBanks();
  renderTable();
  renderPagination();
  var rc = document.getElementById('result-count');
  if (rc) rc.textContent = fmt(S.filtered.length) + ' of ' + fmt(S.banks.length) + ' banks';
}

function sortBanks() {
  var k = S.sortKey, d = S.sortDir === 'asc' ? 1 : -1;
  S.filtered.sort(function (a, b) {
    var av = a[k], bv = b[k];
    if (typeof av === 'string' || typeof bv === 'string') return d * String(av || '').localeCompare(String(bv || ''));
    return d * ((av || 0) - (bv || 0));
  });
}

/* ── Table ─────────────────────────────────── */
function renderTable() {
  var tbody = document.getElementById('league-tbody');
  if (!tbody) return;
  if (S.filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="table-empty">No banks match the current filters.</td></tr>';
    return;
  }
  var start = (S.page - 1) * S.perPage;
  var end = Math.min(start + S.perPage, S.filtered.length);
  var page = S.filtered.slice(start, end);
  tbody.innerHTML = page.map(function (b, i) {
    var rank = start + i + 1;
    var score = b.score || 0;
    var sw = Math.min(score, 100);
    var rc = riskClass(score);
    return '<tr class="bank-row" onclick="location.href=\'bank.html?cert=' + b.cert + '\'">' +
      '<td class="num-cell">' + rank + '</td>' +
      '<td><span class="bank-table-name">' + b.name + '</span></td>' +
      '<td class="score-cell"><div class="mini-score-bar"><div class="mini-score-fill ' + rc + '" style="width:' + sw + '%"></div></div><span class="score-val">' + score.toFixed(1) + '</span></td>' +
      '<td><span class="peer-badge">' + fmtPeerGroup(b.peer_group) + '</span></td>' +
      '<td class="num-cell">' + (b.assets ? '$' + (b.assets / 1e6).toFixed(1) + 'B' : '\u2014') + '</td>' +
      '<td class="num-cell">' + (b.uninsured_pct != null ? (b.uninsured_pct * 100).toFixed(1) + '%' : '\u2014') + '</td>' +
      '<td>' + (b.state || '\u2014') + '</td></tr>';
  }).join('');
}

/* ── Pagination ───────────────────────────── */
function totalPages() { return Math.max(1, Math.ceil(S.filtered.length / S.perPage)); }

function renderPagination() {
  var tp = totalPages();
  var info = document.getElementById('page-info');
  var prev = document.getElementById('page-prev');
  var next = document.getElementById('page-next');
  if (info) info.textContent = 'Page ' + S.page + ' of ' + tp;
  if (prev) prev.disabled = S.page <= 1;
  if (next) next.disabled = S.page >= tp;
  var wrap = document.getElementById('pagination');
  if (wrap) wrap.style.display = S.filtered.length <= S.perPage ? 'none' : '';
}

function initPagination() {
  var prev = document.getElementById('page-prev');
  var next = document.getElementById('page-next');
  if (prev) prev.addEventListener('click', function () {
    if (S.page > 1) { S.page--; renderTable(); renderPagination(); scrollToTable(); }
  });
  if (next) next.addEventListener('click', function () {
    if (S.page < totalPages()) { S.page++; renderTable(); renderPagination(); scrollToTable(); }
  });
}

function scrollToTable() {
  var el = document.getElementById('league-table');
  if (el) {
    var h = document.querySelector('.header');
    var offset = h ? h.offsetHeight + 12 : 12;
    var top = el.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({ top: top, behavior: 'smooth' });
  }
}

/* ── Sort Headers ──────────────────────────── */
function initSort() {
  document.querySelectorAll('.data-table th[data-sort]').forEach(function (th) {
    th.addEventListener('click', function () {
      var k = this.dataset.sort;
      if (S.sortKey === k) S.sortDir = S.sortDir === 'asc' ? 'desc' : 'asc';
      else { S.sortKey = k; S.sortDir = (k === 'name' || k === 'state' || k === 'peer_group') ? 'asc' : 'desc'; }
      document.querySelectorAll('.data-table th[data-sort]').forEach(function (h) { h.classList.remove('sort-asc', 'sort-desc'); });
      this.classList.add('sort-' + S.sortDir);
      applyFilters();
    });
  });
}

/* ── Init ──────────────────────────────────── */
function init() {
  S.indexId = getParam('index') || 'run_risk';

  Promise.all([fetchJSON('manifest.json'), fetchJSON('league.json')]).then(function (results) {
    var manifest = results[0];
    var banks = results[1];

    S.indexMeta = manifest.indices.find(function (i) { return i.id === S.indexId; });
    if (!S.indexMeta) S.indexMeta = manifest.indices[0];

    // Map the correct score field
    S.banks = banks.map(function (b) {
      var copy = Object.assign({}, b);
      copy.score = b[S.indexId] || b.funding_fragility || 0;
      return copy;
    }).sort(function (a, b) { return b.score - a.score; });

    S.filtered = S.banks.slice();
    renderHero(manifest);
    renderFilters(manifest);
    sortBanks();
    renderTable();
    renderPagination();
    initSort();
    initPagination();

    var rc = document.getElementById('result-count');
    if (rc) rc.textContent = fmt(S.filtered.length) + ' of ' + fmt(S.banks.length) + ' banks';

    var si = document.getElementById('table-search');
    if (si) si.addEventListener('input', function () { S.searchTerm = this.value; applyFilters(); });
  }).catch(function (e) {
    console.error(e);
    document.getElementById('league-title').textContent = 'Error: ' + e.message;
  });
}

document.addEventListener('DOMContentLoaded', init);
