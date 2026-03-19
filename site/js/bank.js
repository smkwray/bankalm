'use strict';

var CONFIG = { dataRoot: 'data' };

function fetchJSON(p) {
  return fetch(CONFIG.dataRoot + '/' + p).then(function (r) {
    if (!r.ok) throw new Error('Fetch failed: ' + p); return r.json();
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

/* ── Render ────────────────────────────────── */
function renderBank(bank) {
  var b = bank;
  document.title = b.name + ' \u2014 bankALM';
  document.getElementById('bank-breadcrumb-name').textContent = b.name;
  document.getElementById('bank-title').textContent = b.name;
  document.getElementById('bank-peer').textContent = fmtPeerGroup(b.peer_group);

  var composite = b.funding_fragility || 0;
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
    { id: 'funding_fragility', title: 'Composite Fragility', accent: 'indigo' }
  ];
  var grid = document.getElementById('indices-grid');
  if (grid) {
    grid.innerHTML = indices.map(function (idx) {
      var score = b[idx.id] || 0;
      var rc2 = riskClass(score);
      return '<div class="index-card" data-panel-accent="' + idx.accent + '">' +
        '<div class="index-card-header">' +
          '<span class="index-card-title">' + idx.title + '</span>' +
          '<span class="index-card-score" style="color:var(--' + rc2 + ')">' + score.toFixed(1) + '</span>' +
        '</div>' +
        '<div class="risk-score-bar">' +
          '<div class="risk-bar-track"><div class="risk-bar-fill ' + rc2 + '" style="width:' + Math.min(score, 100) + '%"></div></div>' +
        '</div>' +
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
      ['Composite Fragility', (b.funding_fragility || 0).toFixed(1)],
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
    notes.innerHTML =
      '<p>Scores are percentile-ranked within the <strong>' + fmtPeerGroup(b.peer_group) + '</strong> peer group. ' +
      'A score of 80+ places this bank in the top 20% of its peer group for that risk dimension.</p>' +
      '<p>All data is sourced from FDIC BankFind quarterly filings, FDIC Summary of Deposits (annual, June 30), ' +
      'and FFIEC CDR Call Report bulk data. No proprietary data or supervisory inputs are used.</p>' +
      '<p>Deposit-life and ALM fields are public-data scenario proxies, not direct measurements of a bank\'s internal deposit behavior or full risk system.</p>' +
      '<p>The Treasury section shows the public rate backdrop observed on or before the report date. It provides macro context and is not part of the headline ranking methodology.</p>' +
      '<p>Scores reflect the most recent available quarter and should be read as exploratory screening output, not supervisory ratings or investment advice.</p>';
  }
}

/* ── Init ──────────────────────────────────── */
function init() {
  var cert = getParam('cert');
  if (!cert) {
    document.getElementById('bank-title').textContent = 'Bank not specified';
    return;
  }

  fetchJSON('league.json').then(function (banks) {
    var bank = banks.find(function (b) { return String(b.cert) === cert; });
    if (!bank) throw new Error('Bank not found: CERT ' + cert);
    renderBank(bank);
  }).catch(function (e) {
    console.error(e);
    document.getElementById('bank-title').textContent = 'Error: ' + e.message;
  });
}

document.addEventListener('DOMContentLoaded', init);
