'use strict';

var CONFIG = { dataRoot: 'data' };

function fetchJSON(p) {
  return fetch(CONFIG.dataRoot + '/' + p).then(function (r) {
    if (!r.ok) throw new Error('Fetch failed: ' + p); return r.json();
  });
}
function fetchWithFallback(primaryPath, fallbackPath) {
  return fetchJSON(primaryPath).catch(function () {
    return fallbackPath ? fetchJSON(fallbackPath) : Promise.reject(new Error('Fetch failed: ' + primaryPath));
  });
}
function fmt(n) { return Number(n).toLocaleString(); }
function fmtRate(n) {
  if (n == null || Number.isNaN(Number(n))) return '\u2014';
  return Number(n).toFixed(2) + '%';
}
function fmtBp(n) {
  if (n == null || Number.isNaN(Number(n))) return '\u2014';
  return Number(n).toFixed(0) + ' bp';
}
function fmtScore(n) {
  if (n == null || Number.isNaN(Number(n))) return '\u2014';
  return Number(n).toFixed(1);
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
function fmtDollars(thousands) {
  if (!thousands && thousands !== 0) return '\u2014';
  if (thousands >= 1e6) return '$' + (thousands / 1e6).toFixed(1) + 'B';
  if (thousands >= 1e3) return '$' + (thousands / 1e3).toFixed(0) + 'M';
  return '$' + fmt(thousands) + 'K';
}
function getParam(k) { return new URLSearchParams(location.search).get(k); }

function freshnessFromManifest(manifest) {
  var freshness = manifest && manifest.freshness ? manifest.freshness : {};
  var snapshot = freshness.site_snapshot_as_of || (manifest && manifest.pipeline && manifest.pipeline.date_range ? String(manifest.pipeline.date_range).split(' \u2013 ').pop() : null);
  return {
    site_snapshot_as_of: snapshot,
    generated_at: freshness.generated_at || manifest.generated_at || null,
    source_max_dates: freshness.source_max_dates || {},
    coverage_warnings: freshness.coverage_warnings || [],
    stale: freshness.stale
  };
}
function freshnessSummary(manifest) {
  var freshness = freshnessFromManifest(manifest);
  var parts = [];
  if (freshness.site_snapshot_as_of) parts.push('Data as of ' + freshness.site_snapshot_as_of);
  if (freshness.generated_at) parts.push('generated ' + freshness.generated_at.slice(0, 10));
  if (freshness.stale) parts.push('stale');
  if (freshness.coverage_warnings && freshness.coverage_warnings.length > 0) {
    parts.push(freshness.coverage_warnings[0]);
  }
  return parts.join(' \u2022 ');
}
function indexValidationBadge(idx) {
  if (!idx) return '';
  return idx.validation_status === 'not_backtested_yet'
    ? '<span class="comparison-badge comparison-badge-muted">Experimental</span>'
    : '';
}
function indexValidationNote(idx) {
  if (!idx || !idx.status_note) return '';
  return '<p class="index-status-note">' + escapeHtml(idx.status_note) + '</p>';
}

function competitionScore(bank) { return bank.deposit_competition != null ? bank.deposit_competition : bank.deposit_competition_pressure; }
function escapeHtml(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
function closeNav() {
  var nav = document.getElementById('nav');
  var hamburger = document.getElementById('hamburger');
  if (nav) nav.classList.remove('open');
  if (hamburger) hamburger.setAttribute('aria-expanded', 'false');
}
function toggleNav() {
  var nav = document.getElementById('nav');
  var hamburger = document.getElementById('hamburger');
  if (!nav || !hamburger) return;
  var isOpen = nav.classList.toggle('open');
  hamburger.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
}
function initMobileNav() {
  var nav = document.getElementById('nav');
  var hamburger = document.getElementById('hamburger');
  if (!nav || !hamburger) return;
  hamburger.setAttribute('aria-expanded', nav.classList.contains('open') ? 'true' : 'false');
  hamburger.addEventListener('click', function (e) {
    e.preventDefault();
    toggleNav();
  });
  document.addEventListener('click', function (e) {
    if (!nav.classList.contains('open')) return;
    if (e.target.closest('#nav') || e.target.closest('#hamburger')) return;
    closeNav();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeNav();
  });
}
function renderFreshnessBanner(manifest) {
  var freshness = manifest.freshness || {};
  var asOf = freshness.site_snapshot_as_of || '\u2014';
  var warnings = Array.isArray(freshness.coverage_warnings) ? freshness.coverage_warnings : [];
  var header = document.getElementById('header');
  if (!header || !header.parentNode) return;
  var existing = document.getElementById('freshness-banner');
  if (existing) existing.remove();
  var banner = document.createElement('div');
  banner.id = 'freshness-banner';
  banner.className = 'freshness-banner' + (freshness.stale ? ' is-stale' : '');
  banner.innerHTML =
    '<div class="container freshness-banner-inner">' +
      '<strong>Data as of ' + escapeHtml(asOf) + '</strong>' +
      (freshness.generated_at ? '<span class="freshness-build">Built ' + escapeHtml(freshness.generated_at.slice(0, 10)) + '</span>' : '') +
      (warnings.length > 0 ? '<span class="freshness-warning">' + escapeHtml(warnings[0]) + '</span>' : '') +
    '</div>';
  header.parentNode.insertBefore(banner, header.nextSibling);
}
function getIndexMeta(manifest, indexId) {
  return (manifest.indices || []).find(function (idx) { return idx.id === indexId; }) || null;
}

function renderComponentRows(components, mode) {
  if (!Array.isArray(components) || components.length === 0) {
    return '<p class="breakdown-card-note">Component detail unavailable for this bank-quarter.</p>';
  }
  return '<div class="component-list">' + components.map(function (component) {
    var right = '';
    if (mode === 'run_risk') {
      right = fmtScore(component.contribution) + ' pts' +
        '<span class="component-meta">' + fmtScore(component.percentile) + ' peer pct</span>';
    } else if (mode === 'composite') {
      right = fmtScore(component.contribution) + ' pts' +
        '<span class="component-meta">weight ' + Math.round(Number(component.weight) * 100) + '% • score ' + fmtScore(component.score) + '</span>';
    } else {
      right = fmtScore(component.contribution) + ' pts' +
        '<span class="component-meta">weight ' + Math.round(Number(component.weight) * 100) + '%</span>';
    }
    return '<div class="component-row">' +
      '<div class="component-name">' + component.label + '</div>' +
      '<div class="component-metric">' + right + '</div>' +
    '</div>';
  }).join('') + '</div>';
}

function renderScoreConstruction(bank, manifest) {
  var summary = document.getElementById('bank-score-summary');
  var grid = document.getElementById('score-breakdown-grid');
  var drivers = document.getElementById('bank-top-drivers');
  var methods = manifest.index_methodology || {};

  if (summary) {
    summary.innerHTML =
      '<p>This bank is scored against the <strong>' + fmtPeerGroup(bank.peer_group) + '</strong> peer group for <strong>' + (bank.repdte || '\u2014') + '</strong>. ' +
      'The peer group currently contains <strong>' + fmt(bank.peer_group_bank_count || 0) + '</strong> banks in the published latest-quarter snapshot.</p>' +
      '<p>Each score is a 0&ndash;100 percentile-style measure. Higher values mean more fragility for run risk, ALM mismatch, deposit competition, and the composite index. Treasury buffer is the opposite: higher means more protection.</p>';
  }

  if (grid) {
    var cards = [
      {
        id: 'run_risk',
        title: 'Run Risk Index',
        note: (methods.run_risk && methods.run_risk.formula) || 'Percentile rank of the transparent run-risk score within the peer group.',
        components: bank.run_risk_components,
        mode: 'run_risk'
      },
      {
        id: 'alm_mismatch',
        title: 'ALM Mismatch Index',
        note: (methods.alm_mismatch && methods.alm_mismatch.formula) || 'Weighted percentile composite of structural ALM proxies.',
        components: bank.alm_components,
        mode: 'contrib'
      },
      {
        id: 'deposit_competition',
        title: 'Deposit Competition Index',
        note: ((methods.deposit_competition && methods.deposit_competition.formula) || 'Percentile rank of transparent outside-option pressure within the peer group.') +
          ((getIndexMeta(manifest, 'deposit_competition') && getIndexMeta(manifest, 'deposit_competition').validation_status === 'not_backtested_yet')
            ? ' Experimental: not backtested yet.'
            : ''),
        components: bank.deposit_competition_components,
        mode: 'run_risk'
      },
      {
        id: 'treasury_buffer',
        title: 'Treasury Buffer Index',
        note: (methods.treasury_buffer && methods.treasury_buffer.formula) || 'Weighted percentile composite of Treasury/HQLA coverage ratios.',
        components: bank.treasury_buffer_components,
        mode: 'contrib'
      },
      {
        id: 'funding_fragility',
        title: 'Composite Fragility Mix',
        note: (methods.funding_fragility && methods.funding_fragility.formula) || 'Weighted mix of run risk, ALM mismatch, and inverse Treasury buffer.',
        components: bank.composite_components,
        mode: 'composite'
      }
    ];
    grid.innerHTML = cards.map(function (card) {
      var meta = getIndexMeta(manifest, card.id);
      return '<div class="breakdown-card">' +
        '<h4>' + card.title + ' ' + indexValidationBadge(meta || {}) + '</h4>' +
        '<div class="breakdown-card-note">' + escapeHtml(card.note) + '</div>' +
        renderComponentRows(card.components, card.mode) +
      '</div>';
    }).join('');
  }

  if (drivers) {
    var top = (Array.isArray(bank.composite_components) ? bank.composite_components.slice() : [])
      .sort(function (a, b) { return (b.contribution || 0) - (a.contribution || 0); })
      .slice(0, 3);
    drivers.innerHTML =
      '<h4>Top composite drivers</h4>' +
      (top.length === 0
        ? '<p class="breakdown-card-note">Composite driver detail unavailable for this bank-quarter.</p>'
        : '<div class="driver-list">' + top.map(function (component) {
          return '<div class="driver-pill"><strong>' + component.label + '</strong><span>' + fmtScore(component.contribution) + ' pts</span></div>';
        }).join('') + '</div>');
  }
}

/* ── Render ────────────────────────────────── */
function renderBank(bank, manifest) {
  var b = bank;
  var sourceIndex = getParam('index');
  var sourceMeta = getIndexMeta(manifest, sourceIndex);
  document.title = b.name + ' \u2014 bankALM';
  document.getElementById('bank-breadcrumb-name').textContent = b.name;
  var breadcrumbLeague = document.getElementById('bank-breadcrumb-league');
  if (breadcrumbLeague) {
    if (sourceMeta) {
      breadcrumbLeague.href = 'league.html?index=' + encodeURIComponent(sourceMeta.id);
      breadcrumbLeague.textContent = sourceMeta.title + ' League Table';
    } else {
      breadcrumbLeague.href = 'league.html';
      breadcrumbLeague.textContent = 'League Table';
    }
  }
  document.getElementById('bank-title').textContent = b.name;
  document.getElementById('bank-peer').textContent = fmtPeerGroup(b.peer_group);

  var composite = b.funding_fragility != null ? b.funding_fragility : 0;
  var rc = riskClass(composite);
  document.getElementById('bank-score-value').textContent = composite.toFixed(1);
  document.getElementById('bank-score-value').style.color = 'var(--' + rc + ')';
  var fill = document.getElementById('bank-score-fill');
  fill.style.width = Math.min(composite, 100) + '%';
  fill.className = 'bank-detail-score-fill ' + rc;

  document.getElementById('bs-assets').textContent = fmtDollars(b.assets);
  document.getElementById('bs-deposits').textContent = fmtDollars(b.deposits);
  document.getElementById('bs-uninsured').textContent = b.uninsured_pct != null ? (b.uninsured_pct * 100).toFixed(1) + '%' : '\u2014';
  document.getElementById('bs-state').textContent = b.state || '\u2014';

  // Index breakdown
  var indices = [
    { id: 'run_risk', title: 'Run Risk Index', accent: 'crimson' },
    { id: 'alm_mismatch', title: 'ALM Mismatch Index', accent: 'amber' },
    { id: 'deposit_competition', title: 'Deposit Competition', accent: 'emerald' },
    { id: 'funding_fragility', title: 'Composite Fragility', accent: 'indigo' }
  ];
  var grid = document.getElementById('indices-grid');
  if (grid) {
    grid.innerHTML = indices.map(function (idx) {
      var score = idx.id === 'deposit_competition' ? competitionScore(b) : (b[idx.id] != null ? b[idx.id] : 0);
      var rc2 = riskClass(score);
      var meta = getIndexMeta(manifest, idx.id);
      return '<div class="index-card" data-panel-accent="' + idx.accent + '">' +
        '<div class="index-card-header">' +
          '<span class="index-card-title">' + idx.title +
            ' ' + indexValidationBadge(meta || {}) +
          '</span>' +
          '<span class="index-card-score" style="color:var(--' + rc2 + ')">' + score.toFixed(1) + '</span>' +
        '</div>' +
        '<div class="risk-score-bar">' +
          '<div class="risk-bar-track"><div class="risk-bar-fill ' + rc2 + '" style="width:' + Math.min(score, 100) + '%"></div></div>' +
        '</div>' +
        (meta && meta.status_note ? '<p class="index-status-note">' + escapeHtml(meta.status_note) + '</p>' : '') +
      '</div>';
    }).join('');
  }

  // Metadata
  var meta = document.getElementById('metadata-grid');
  if (meta) {
    var items = [
      ['CERT', b.cert],
      ['Charter Type', b.charter || '\u2014'],
      ['State', b.state || '\u2014'],
      ['Peer Group', fmtPeerGroup(b.peer_group)],
      ['Total Assets', fmtDollars(b.assets)],
      ['Total Deposits', fmtDollars(b.deposits)],
      ['Uninsured %', b.uninsured_pct != null ? (b.uninsured_pct * 100).toFixed(1) + '%' : '\u2014'],
      ['Report Date', b.repdte || '\u2014'],
      ['Run Risk', (b.run_risk || 0).toFixed(1)],
      ['ALM Mismatch', (b.alm_mismatch || 0).toFixed(1)],
      ['Deposit Competition', competitionScore(b) != null ? competitionScore(b).toFixed(1) : '\u2014'],
      ['Composite Fragility', (b.funding_fragility != null ? b.funding_fragility : 0).toFixed(1)],
      ['Failed', b.failed ? 'Yes (' + b.fail_date + ')' : 'No']
    ];
    meta.innerHTML = items.map(function (r) {
      return '<div class="metadata-item"><span class="metadata-label">' + r[0] +
        '</span><span class="metadata-value">' + r[1] + '</span></div>';
    }).join('');
  }

  var regime = document.getElementById('treasury-regime-grid');
  if (regime) {
    var items2 = [
      ['Yield Observation', b.treasury_yield_date || '\u2014'],
      ['2Y Treasury', fmtRate(b.yc_2yr)],
      ['10Y Treasury', fmtRate(b.yc_10yr)],
      ['10Y-3M Slope', fmtBp(b.yc_10y_3m_slope_bp)],
      ['10Y-2Y Slope', fmtBp(b.yc_10y_2y_slope_bp)],
      ['10Y QoQ Change', fmtBp(b.yc_10yr_qoq_change_bp)],
      ['Treasury History', b.has_treasury_yield_history ? 'Available' : 'Not available'],
    ];
    regime.innerHTML = items2.map(function (r) {
      return '<div class="metadata-item"><span class="metadata-label">' + r[0] +
        '</span><span class="metadata-value">' + r[1] + '</span></div>';
    }).join('');
  }

  // Notes
  var notes = document.getElementById('bank-notes-content');
  if (notes) {
    var depositCompetitionMeta = getIndexMeta(manifest, 'deposit_competition');
    var freshness = freshnessSummary(manifest);
    notes.innerHTML =
      (freshness ? '<p>' + escapeHtml(freshness) + '</p>' : '') +
      '<p>Scores are percentile-ranked within the <strong>' + fmtPeerGroup(b.peer_group) + '</strong> peer group. ' +
      'A score of 80+ places this bank in the top 20% of its peer group for that risk dimension.</p>' +
      '<p>The score construction section above shows the actual weighted component contributions used in the latest published snapshot.</p>' +
      '<p>All data is sourced from FDIC BankFind quarterly filings, FDIC Summary of Deposits (annual, June 30), ' +
      'and FFIEC CDR Call Report bulk data. No proprietary data or supervisory inputs are used.</p>' +
      '<p>Deposit-life and ALM fields are public-data scenario proxies, not direct measurements of a bank\'s internal deposit behavior or full risk system.</p>' +
      (depositCompetitionMeta && depositCompetitionMeta.status_note ? '<p><strong>Deposit Competition:</strong> ' + escapeHtml(depositCompetitionMeta.status_note) + '</p>' : '') +
      '<p>The Treasury section shows the public rate backdrop observed on or before the report date. It provides macro context and is not part of the headline ranking methodology.</p>' +
      '<p>Scores reflect the most recent available quarter and should be read as exploratory screening output, not supervisory ratings or investment advice.</p>';
  }

  renderScoreConstruction(b, manifest || {});
}

/* ── Init ──────────────────────────────────── */
function init() {
  var cert = getParam('cert');
  if (!cert) {
    document.getElementById('bank-title').textContent = 'Bank not specified';
    return;
  }

  Promise.all([fetchJSON('manifest.json'), fetchWithFallback('banks/' + encodeURIComponent(cert) + '.json', 'league.json')]).then(function (results) {
    var manifest = results[0];
    var bank = results[1];
    if (Array.isArray(bank)) {
      bank = bank.find(function (b) { return String(b.cert) === cert; });
    }
    if (!bank || String(bank.cert) !== cert) throw new Error('Bank not found: CERT ' + cert);
    renderFreshnessBanner(manifest);
    initMobileNav();
    renderBank(bank, manifest);
  }).catch(function (e) {
    console.error(e);
    document.getElementById('bank-title').textContent = 'Error: ' + e.message;
  });
}

document.addEventListener('DOMContentLoaded', init);
