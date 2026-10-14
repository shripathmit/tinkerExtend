/* tinkerExtend unified app: Q&A (public) + console tabs (access code).
 *
 * Plain JS, no framework. Every dynamic string goes through esc(); clicks
 * on rendered rows are handled by delegation on data-* attributes, so no
 * document name or question is ever interpolated into executable code.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const esc = (t) => {
  const d = document.createElement('div');
  d.textContent = t == null ? '' : String(t);
  return d.innerHTML;
};
const store = {
  get(k, dflt) { try { const v = localStorage.getItem(k); return v === null ? dflt : JSON.parse(v); } catch (e) { return dflt; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
};
const pct = (x, digits = 1) => (x == null || isNaN(x)) ? '—' : `${(x * 100).toFixed(digits)}%`;
const money = (x) => `$${(x || 0).toFixed(2)}`;
const basename = (p) => (p || '').split('/').pop();
const plain = (html) => {
  const d = document.createElement('div');
  d.innerHTML = (html || '').replace(/<\/(p|h[1-6]|li|td|th|tr|div|blockquote)>|<br\s*\/?>/gi, '$& ');
  return (d.textContent || '').replace(/\s+/g, ' ').trim();
};
const firstPage = (pagesJson) => { try { const p = JSON.parse(pagesJson || '[]'); return p.length ? p[0] : null; } catch (e) { return null; } };
const PERF_RATE = 0.025;

const TABS = ['qa', 'overview', 'ingestion', 'fidelity', 'golden'];
const ISSUES = ['Wrong chunk', 'Parse leak', 'Incomplete'];

const state = {
  view: 'qa',
  admin: false,
  stats: null,
  synth: store.get('tx.synth', true),
  searchType: store.get('tx.searchType', 'ask'),
  result: null,           // last /api/ask or /api/search response (normalized)
  active: 0,              // index into result.passages
  viewer: { docId: null, page: null, numPages: 0, focusChunk: null },
  surface: store.get('tx.surface', 'image'),   // image | reader | paper
  allChunks: false,
  pageCache: new Map(),
  issue: ISSUES[0],
  overview: null,
  telemetry: null,
  golden: null,
  diff: { docId: null, job: null },
  reparseAllArmed: false,
};

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
class LockedError extends Error {}

async function api(path, opts = {}) {
  const init = { ...opts, headers: { ...(opts.headers || {}) } };
  if (init.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(init.json);
    delete init.json;
  }
  const r = await fetch(path, init);
  if (r.status === 401 && path.startsWith('/api/admin/') && path !== '/api/admin/login') {
    state.admin = false;
    renderAdminChrome();
    throw new LockedError('locked');
  }
  let body = null;
  try { body = await r.json(); } catch (e) { /* non-JSON */ }
  if (!r.ok) throw new Error((body && (body.error || body.detail)) || `HTTP ${r.status}`);
  return body;
}

// ---------------------------------------------------------------------------
// Routing + header
// ---------------------------------------------------------------------------
function currentTab() {
  const h = (location.hash || '#qa').slice(1);
  return TABS.includes(h) ? h : 'qa';
}

function switchView(view) {
  state.view = view;
  document.querySelectorAll('.view').forEach((el) => el.classList.toggle('hidden', el.dataset.view !== view));
  document.querySelectorAll('#tabs .tab').forEach((el) => {
    const on = el.dataset.tab === view;
    el.classList.toggle('text-ink', on);
    el.classList.toggle('border-ink', on);
    el.classList.toggle('text-inkFaint', !on);
    el.classList.toggle('border-transparent', !on);
    el.classList.toggle('hover:text-inkMuted', !on);
    if (on) { el.setAttribute('aria-current', 'page'); el.scrollIntoView({ block: 'nearest', inline: 'nearest' }); } else el.removeAttribute('aria-current');
  });
  loadView(view);
}

async function loadView(view) {
  if (view === 'qa') return;
  const main = $(`view-${view}`);
  if (!state.admin) { showLock(main); return; }
  hideLock(main);
  try {
    if (view === 'overview') await loadOverview();
    else if (view === 'ingestion') await loadIngestion();
    else if (view === 'fidelity') await loadFidelity();
    else if (view === 'golden') await loadGolden();
  } catch (e) {
    if (e instanceof LockedError) showLock(main);
    else console.warn(e);
  }
}

function showLock(main) {
  const slot = main.querySelector('.lock-slot');
  main.querySelector('.view-body').classList.add('hidden');
  if (!slot.firstElementChild) {
    slot.appendChild($('lock-template').content.cloneNode(true));
    const form = slot.querySelector('.lock-form');
    form.addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const err = slot.querySelector('.lock-error');
      err.classList.add('hidden');
      try {
        await api('/api/admin/login', { json: { code: form.code.value } });
        form.code.value = '';
        state.admin = true;
        renderAdminChrome();
        loadBadges();
        switchView(state.view);
      } catch (e) {
        err.textContent = e.message === 'Wrong code.' ? 'Wrong code.' : `Could not unlock: ${e.message}`;
        err.classList.remove('hidden');
      }
    });
  }
  slot.classList.remove('hidden');
  const title = { overview: 'Overview', ingestion: 'Ingestion', fidelity: 'Parse Fidelity', golden: 'Golden Sets' }[main.dataset.view];
  slot.querySelector('.lock-title').textContent = `Enter the access code to open ${title}`;
  setTimeout(() => { const i = slot.querySelector('input'); if (i) i.focus(); }, 0);
}

function hideLock(main) {
  main.querySelector('.lock-slot').classList.add('hidden');
  main.querySelector('.view-body').classList.remove('hidden');
}

function renderAdminChrome() {
  $('lock-btn').classList.toggle('hidden', !state.admin);
  if (!state.admin) {
    $('nav-ingest-count').textContent = '';
    $('nav-golden-count').textContent = '';
    $('nav-warn-dot').classList.add('hidden');
  }
}

async function loadBadges() {
  if (!state.admin) return;
  try {
    const [ov, gold] = await Promise.all([api('/api/admin/overview'), api('/api/admin/golden')]);
    state.overview = ov; state.golden = gold;
    renderBadges();
  } catch (e) { /* badges are best-effort */ }
}

function renderBadges() {
  if (!state.admin) return;
  if (state.stats) $('nav-ingest-count').textContent = state.stats.documents;
  if (state.golden) $('nav-golden-count').textContent = state.golden.total;
  const n = state.overview ? state.overview.red_flags.length : 0;
  const dot = $('nav-warn-dot');
  dot.classList.toggle('hidden', n === 0);
  dot.title = `${n} document${n === 1 ? '' : 's'} with fidelity warnings`;
}

async function loadStats() {
  try {
    const s = await api('/api/stats');
    state.stats = s;
    const tok = s.token_estimate >= 1000 ? `${(s.token_estimate / 1000).toFixed(1)}k` : s.token_estimate;
    $('nav-corpus-meta').textContent = `${s.documents} docs · ${s.chunks} chunks · ${tok} tokens`;
    renderModeToggle();
    renderBadges();
  } catch (e) { console.warn(e); }
}

function synthEnabled() { return !!(state.stats && state.stats.llm_configured && state.synth); }

function renderModeToggle() {
  const label = $('mode-toggle-label');
  const btn = $('mode-toggle-btn');
  const configured = state.stats && state.stats.llm_configured;
  if (synthEnabled()) {
    label.textContent = 'Generative';
    label.className = 'font-mono text-emeraldOk';
    btn.title = 'Answers are synthesized by the LLM from retrieved passages. Click for extractive answers.';
  } else {
    label.textContent = 'Extractive';
    label.className = 'font-mono text-amberWarn';
    btn.title = configured
      ? 'Answers quote the best passage. Click to enable LLM synthesis.'
      : 'No LLM configured. Set TINKER_LLM_API_KEY on the server to enable synthesis.';
  }
  btn.classList.toggle('opacity-60', !configured);
  renderSubtitle();
}

function toggleSynthesisMode() {
  if (!(state.stats && state.stats.llm_configured)) {
    $('answer-subtitle').textContent = 'Synthesis needs TINKER_LLM_API_KEY on the server · extractive answers shown';
    return;
  }
  state.synth = !state.synth;
  store.set('tx.synth', state.synth);
  renderModeToggle();
  if (state.result && state.result.kind === 'ask') runQuery();
}

// ---------------------------------------------------------------------------
// Q&A workspace
// ---------------------------------------------------------------------------
function renderSearchType() {
  const on = 'px-2.5 py-1 bg-ink text-canvas font-medium transition-colors';
  const off = 'px-2.5 py-1 text-inkFaint hover:text-inkMuted transition-colors';
  $('switch-ask').className = state.searchType === 'ask' ? on : off;
  $('switch-search').className = state.searchType === 'search' ? on : off;
  $('switch-ask').setAttribute('aria-pressed', state.searchType === 'ask');
  $('switch-search').setAttribute('aria-pressed', state.searchType === 'search');
}

async function loadExamples() {
  try {
    const { examples } = await api('/api/examples');
    const el = $('examples');
    if (!examples.length) { el.innerHTML = ''; return; }
    el.innerHTML = '<span>Try:</span>' + examples.map((q) =>
      `<button type="button" data-example="${esc(q)}" class="text-inkMuted hover:text-ink transition-colors">${esc(q.length > 48 ? q.slice(0, 46) + '…' : q)}</button>`
    ).join('');
  } catch (e) { /* optional */ }
}

async function runQuery() {
  const q = $('query-input').value.trim();
  if (!q) { $('query-input').focus(); return; }
  resetFeedbackUI();
  $('answer-subtitle').textContent = state.searchType === 'ask' ? 'Retrieving and answering…' : 'Searching…';
  $('subtitle-action-link').classList.add('hidden');
  try {
    let res;
    if (state.searchType === 'ask') {
      const r = await api('/api/ask', { json: { question: q, limit: 5, synthesize: synthEnabled() } });
      res = { kind: 'ask', question: q, mode: r.mode, answerHtml: r.answer_html, answerText: r.answer_text,
              search: r.search, passages: r.passages || [], chunkIds: r.chunk_ids || [] };
    } else {
      const r = await api(`/api/search?q=${encodeURIComponent(q)}&limit=10`);
      res = { kind: 'search', question: q, mode: 'search', passages: r.hits || [],
              search: state.stats && state.stats.embed_model ? 'hybrid' : 'keyword' };
    }
    state.result = res;
    state.active = 0;
    renderWorkspace();
    if (res.passages.length) selectSource(0, { scroll: false });
    else clearViewer();
  } catch (e) {
    $('answer-subtitle').textContent = `Error: ${e.message}`;
  }
}

function renderSubtitle() {
  const sub = $('answer-subtitle');
  const link = $('subtitle-action-link');
  const r = state.result;
  link.classList.add('hidden');
  if (!r) {
    const n = state.stats ? state.stats.documents : 0;
    sub.textContent = n ? `Ask about ${n} document${n === 1 ? '' : 's'} · answers cite the exact page` : 'Knowledge base is empty · ingest documents to begin';
    return;
  }
  const how = `${r.search} search`;
  if (r.kind === 'search') {
    sub.textContent = `${r.passages.length} match${r.passages.length === 1 ? '' : 'es'} · ${how}`;
  } else if (r.mode === 'generated') {
    sub.textContent = `Synthesized answer · grounded in ${r.passages.length} retrieved chunk${r.passages.length === 1 ? '' : 's'}`;
    link.textContent = 'Switch to extractive';
    link.classList.remove('hidden');
  } else if (r.mode === 'extractive') {
    sub.textContent = `Extractive match · ${how}`;
    if (state.stats && state.stats.llm_configured) {
      link.textContent = 'Enable synthesis';
      link.classList.remove('hidden');
    }
  } else {
    sub.textContent = `No confident match · declined to answer · ${how}`;
  }
}

function matchLabel(p) {
  if (!p.match || p.match === 'keyword') return 'keyword match';
  const cos = p.vector_score != null ? ` · cos ${p.vector_score.toFixed(2)}` : '';
  return (p.match === 'both' ? 'keyword + vector' : 'vector match') + cos;
}

function renderWorkspace() {
  const r = state.result;
  renderSubtitle();
  const ans = $('answer-content');
  if (!r) { ans.innerHTML = ''; return; }
  if (r.kind === 'search') {
    ans.innerHTML = '';
  } else if (r.mode === 'extractive') {
    ans.innerHTML = `<blockquote id="answer-quote" class="answer-quote">${r.answerHtml}</blockquote>
      <button type="button" id="answer-expand" class="hidden text-xs font-mono text-inkMuted hover:text-ink underline underline-offset-4">Show full passage</button>`;
    clampAnswer();
  } else if (r.mode === 'generated') {
    ans.innerHTML = r.answerHtml;
  } else {
    ans.innerHTML = `<p class="text-inkMuted">${esc(r.answerText)}</p>`;
  }
  $('feedback').classList.toggle('hidden', !(r.kind === 'ask' && r.passages.length));
  renderSources();
}

function clampAnswer() {
  const q = $('answer-quote');
  const btn = $('answer-expand');
  if (!q || !btn) return;
  q.classList.add('is-clamped');
  // Only offer expansion when the passage is actually cut off.
  requestAnimationFrame(() => {
    const cut = q.scrollHeight > q.clientHeight + 4;
    if (!cut) { q.classList.remove('is-clamped'); return; }
    btn.classList.remove('hidden');
    btn.onclick = () => {
      const open = q.classList.toggle('is-clamped') === false;
      btn.textContent = open ? 'Show less' : 'Show full passage';
    };
  });
}

function renderSources() {
  const r = state.result;
  const block = $('sources-block');
  if (!r || !r.passages.length) { block.classList.add('hidden'); return; }
  block.classList.remove('hidden');
  $('sources-count').textContent = `${r.passages.length} retrieved`;
  $('sources-list').innerHTML = r.passages.map((p, i) => {
    const on = i === state.active;
    const page = firstPage(p.pages);
    const excerpt = plain(p.content_html);
    return `
      <div data-source="${i}" class="py-5 cursor-pointer transition-colors group" role="button" tabindex="0" aria-expanded="${on}">
        <div class="flex items-center justify-between gap-3">
          <div class="flex items-center space-x-3 min-w-0">
            <span class="font-mono text-xs ${on ? 'text-ink font-medium' : 'text-inkFaint group-hover:text-inkMuted'}">[${i + 1}]</span>
            <span class="font-mono text-xs truncate ${on ? 'text-ink font-medium' : 'text-inkMuted group-hover:text-ink'}">${esc(p.doc_name || basename(p.doc_source))}</span>
            <span class="font-mono text-xs text-inkFaint shrink-0">${page != null ? `· p.${page}` : ''}</span>
          </div>
          <span class="font-mono text-[11px] shrink-0 ${on ? 'text-amberWarn' : 'text-inkFaint opacity-0 group-hover:opacity-100'} transition-opacity">${on ? 'Active' : 'Inspect'}</span>
        </div>
        ${on ? `
          <div class="mt-3 pl-7 space-y-1.5 border-l border-amberWarn/50">
            <div class="text-[11px] font-mono text-inkFaint">${esc(matchLabel(p))}</div>
            <p class="text-xs text-inkMuted leading-relaxed">${esc(excerpt.length > 360 ? excerpt.slice(0, 358) + '…' : excerpt)}</p>
          </div>` : ''}
      </div>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Document viewer
// ---------------------------------------------------------------------------
function activePassage() { return state.result ? state.result.passages[state.active] : null; }

function selectSource(i, { scroll = true } = {}) {
  const r = state.result;
  if (!r || !r.passages[i]) return;
  state.active = i;
  const p = r.passages[i];
  const page = (p.sources && p.sources.length ? p.sources[0].page : null) || firstPage(p.pages) || 1;
  state.viewer.focusChunk = p.chunk_id;
  renderSources();
  showPage(p.doc_id, page, { scrollToActive: scroll });
}

function clearViewer() {
  state.viewer = { docId: null, page: null, numPages: 0, focusChunk: null };
  $('viewer-empty').classList.remove('hidden');
  $('document-sheet').classList.add('hidden');
  $('viewer-doc-title').textContent = 'No source selected';
  $('viewer-page-indicator').textContent = '—';
  $('viewer-active-caption').textContent = state.result ? 'No passages retrieved' : 'No citation selected';
  $('page-prev').disabled = true; $('page-next').disabled = true;
}

async function fetchPage(docId, page) {
  const key = `${docId}:${page}`;
  if (!state.pageCache.has(key)) state.pageCache.set(key, api(`/api/page/${encodeURIComponent(docId)}/${page}`));
  try { return await state.pageCache.get(key); } catch (e) { state.pageCache.delete(key); throw e; }
}

async function showPage(docId, page, { scrollToActive = true } = {}) {
  if (!docId) { clearViewer(); return; }
  let data;
  try { data = await fetchPage(docId, page); } catch (e) {
    $('viewer-active-caption').textContent = `Could not load page: ${e.message}`;
    return;
  }
  state.viewer.docId = docId;
  state.viewer.page = page;
  state.viewer.numPages = data.num_pages || page;
  $('viewer-empty').classList.add('hidden');
  $('viewer-doc-title').textContent = data.doc_name;
  $('viewer-doc-title').title = data.doc_name;
  $('viewer-page-indicator').textContent = `Page ${page} of ${state.viewer.numPages}`;
  $('page-prev').disabled = page <= 1;
  $('page-next').disabled = page >= state.viewer.numPages;
  renderSheet(data);
  const p = activePassage();
  const onPage = data.chunks.some((c) => c.chunk_id === state.viewer.focusChunk);
  $('viewer-active-caption').textContent = p && onPage && state.viewer.focusChunk === p.chunk_id
    ? `Showing citation [${state.active + 1}] · ${matchLabel(p)}`
    : onPage ? 'Showing focused passage' : `Page ${page} · ${data.chunks.length} chunk${data.chunks.length === 1 ? '' : 's'}`;
  if (scrollToActive) {
    const el = $('document-sheet').querySelector('[data-active="1"]');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function sourceIndexForChunk(chunkId) {
  return state.result ? state.result.passages.findIndex((p) => p.chunk_id === chunkId) : -1;
}

function renderSheet(data) {
  const sheet = $('document-sheet');
  sheet.classList.remove('hidden');
  const mode = state.surface === 'image' && !data.image_url ? 'reader' : state.surface;
  const paper = mode === 'paper';
  sheet.className = `w-full max-w-[560px] border transition-colors ${
    mode === 'image' ? 'bg-paperDark border-hairlineFocus p-3 sm:p-4'
    : paper ? 'bg-paperWarm border-stone-300 text-stone-900 p-6 sm:p-8 lg:p-10 space-y-6'
    : 'bg-paperDark border-hairlineFocus p-6 sm:p-8 lg:p-10 space-y-6'}`;
  $('toggle-surface-btn').textContent = `Surface: ${{ image: 'Image', reader: 'Reader', paper: 'Paper' }[state.surface]}`;
  const focus = state.viewer.focusChunk;
  const runHead = `
    <div class="pb-4 border-b ${paper ? 'border-stone-300 text-stone-500' : 'border-hairline text-inkFaint'} flex items-center justify-between gap-4 font-mono text-[11px]">
      <span class="truncate">${esc(data.doc_name.toUpperCase())}</span><span class="shrink-0">P. ${data.page}</span>
    </div>`;

  if (mode === 'image') {
    const W = data.width || 1, H = data.height || 1;
    const boxes = [];
    data.chunks.forEach((c) => {
      const isActive = c.chunk_id === focus;
      if (!isActive && !state.allChunks) return;
      const src = sourceIndexForChunk(c.chunk_id);
      c.boxes.forEach((b) => {
        const style = `left:${(b.left / W) * 100}%;top:${(b.top / H) * 100}%;width:${((b.right - b.left) / W) * 100}%;height:${((b.bottom - b.top) / H) * 100}%`;
        boxes.push(`<div class="hl ${isActive ? 'hl-active' : 'hl-other'}" style="${style}" data-chunk="${esc(c.chunk_id)}" ${isActive ? 'data-active="1"' : ''} title="${src >= 0 ? `Citation [${src + 1}]` : 'Chunk ' + c.ordinal}"></div>`);
      });
    });
    sheet.innerHTML = `
      <div class="relative bg-white"><img src="${esc(data.image_url)}" alt="Page ${data.page} of ${esc(data.doc_name)}" class="block w-full h-auto">${boxes.join('')}</div>`;
    return;
  }

  const passages = data.chunks.map((c) => {
    const isActive = c.chunk_id === focus;
    const cls = isActive
      ? (paper ? 'border-amber-600 bg-amber-500/15 text-stone-900' : 'border-amberWarn bg-amberWarn/[0.07] text-ink')
      : state.allChunks
        ? (paper ? 'border-stone-300 text-stone-700' : 'border-hairlineFocus bg-white/[0.01] text-inkMuted')
        : (paper ? 'border-transparent text-stone-600 opacity-60' : 'border-transparent text-inkMuted opacity-[0.45]');
    return `<div data-chunk="${esc(c.chunk_id)}" ${isActive ? 'data-active="1"' : ''} class="chunk-passage doc-html cursor-pointer p-4 -mx-4 border-l-2 ${cls} text-[13.5px] leading-[1.7] space-y-3">${c.content_html}</div>`;
  }).join('') || `<p class="font-mono text-xs ${paper ? 'text-stone-500' : 'text-inkFaint'}">No parsed text on this page.</p>`;
  const note = state.surface === 'image' && !data.image_url
    ? `<p class="font-mono text-[11px] ${paper ? 'text-stone-500' : 'text-inkFaint'}">Page image unavailable for this document · showing parsed text.</p>` : '';
  sheet.innerHTML = runHead + note + passages + `
    <div class="pt-4 border-t ${paper ? 'border-stone-300 text-stone-500' : 'border-hairline text-inkFaint'} flex items-center justify-between font-mono text-[11px]">
      <span>PARSED BY EXTEND</span><span>${data.chunks.length} CHUNK${data.chunks.length === 1 ? '' : 'S'}</span>
    </div>`;
}

function focusChunkInViewer(chunkId) {
  const src = sourceIndexForChunk(chunkId);
  if (src >= 0) { selectSource(src); return; }
  state.viewer.focusChunk = chunkId;
  showPage(state.viewer.docId, state.viewer.page, { scrollToActive: false });
}

// ---------------------------------------------------------------------------
// Feedback -> golden assertions
// ---------------------------------------------------------------------------
function resetFeedbackUI() {
  $('fb-drawer').classList.add('hidden');
  $('fb-ack').classList.add('hidden');
  $('fb-view-golden').classList.add('hidden');
  $('fb-yes').className = 'text-inkMuted hover:text-ink transition-colors';
  $('fb-no').className = 'text-inkMuted hover:text-ink transition-colors';
  $('fb-comment').value = '';
  pickIssue(ISSUES[0]);
}

function pickIssue(tag) {
  state.issue = tag;
  document.querySelectorAll('.issue-tag').forEach((el) => {
    el.className = el.dataset.issue === tag
      ? 'issue-tag text-ink font-medium underline underline-offset-4'
      : 'issue-tag text-inkFaint hover:text-inkMuted';
    el.setAttribute('aria-pressed', el.dataset.issue === tag);
  });
}

async function sendFeedback(rating, extra = {}) {
  const r = state.result;
  const p = activePassage();
  await api('/api/feedback', { json: {
    question: r.question, mode: r.mode, answer: r.answerText || '',
    chunk_ids: r.chunkIds || r.passages.map((x) => x.chunk_id),
    active_chunk_id: p ? p.chunk_id : '', rating, ...extra,
  } });
  state.golden = null;
}

async function onFeedbackYes() {
  try {
    await sendFeedback(1);
    $('fb-drawer').classList.add('hidden');
    $('fb-yes').className = 'text-ink font-medium underline underline-offset-4';
    $('fb-no').className = 'text-inkFaint';
    ack('Added PASS assertion.');
  } catch (e) { ack(`Not saved: ${e.message}`, false); }
}

function onFeedbackNo() {
  $('fb-ack').classList.add('hidden');
  $('fb-view-golden').classList.add('hidden');
  $('fb-drawer').classList.remove('hidden');
  $('fb-no').className = 'text-ink font-medium underline underline-offset-4';
  $('fb-yes').className = 'text-inkFaint';
  $('fb-comment').focus();
}

async function submitNegativeFeedback(ev) {
  ev.preventDefault();
  try {
    await sendFeedback(-1, { issue: state.issue, comment: $('fb-comment').value.trim() });
    $('fb-comment').value = '';
    $('fb-drawer').classList.add('hidden');
    ack(`Logged FAIL (${state.issue}).`);
  } catch (e) { ack(`Not saved: ${e.message}`, false); }
}

function ack(text, ok = true) {
  const a = $('fb-ack');
  a.textContent = text;
  a.className = `font-mono ${ok ? 'text-inkMuted' : 'text-amberWarn'}`;
  $('fb-view-golden').classList.toggle('hidden', !ok);
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------
const statusText = (flagged) => flagged
  ? '<span class="text-amberWarn">Warning</span>' : '<span class="text-emeraldOk">OK</span>';

function engineLabel(e) {
  return ({ parse_performance: 'PERFORMANCE', parse_light: 'LIGHT', default: 'DEFAULT' }[e] || (e || '—').toUpperCase());
}

async function loadOverview() {
  const [ov, gold] = await Promise.all([api('/api/admin/overview'), state.golden ? state.golden : api('/api/admin/golden')]);
  state.overview = ov; state.golden = gold;
  renderBadges();
  const o = ov.overview;
  const chunks = state.stats ? state.stats.chunks : 0;
  $('metric-docs').textContent = o.docs;
  $('metric-docs-sub').textContent = `${chunks} chunks · ${o.dead_letters} dead letter${o.dead_letters === 1 ? '' : 's'}`;
  $('metric-fidelity').textContent = pct(o.success_rate);
  const warn = ov.red_flags.length;
  $('metric-warnings-sub').textContent = warn ? `${warn} document${warn > 1 ? 's' : ''} raising red flags` : 'All documents verified clean';
  $('metric-warnings-sub').className = `font-mono text-[11px] mt-2 tabular-nums ${warn ? 'text-amberWarn' : 'text-emeraldOk'}`;
  $('metric-eval-pass').textContent = `${gold.passed} / ${gold.total}`;
  $('metric-eval-sub').textContent = `${Math.round(gold.pass_rate * 100)}% assertion pass rate`;
  $('metric-spend').textContent = money(o.cost_usd);
  const badge = $('overview-warnings-badge');
  badge.textContent = `${warn} active`;
  badge.className = `text-xs font-mono tabular-nums ${warn ? 'text-amberWarn' : 'text-emeraldOk'}`;
  $('overview-warnings-list').innerHTML = warn ? ov.red_flags.map((f) => `
    <div class="py-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div class="space-y-1 min-w-0">
        <div class="flex flex-wrap items-center gap-x-3"><span class="font-mono text-sm text-ink break-all">${esc(f.doc)}</span><span class="font-mono text-[11px] text-inkFaint">${esc(f.doc_type || 'other')}</span></div>
        <div class="text-xs text-inkFaint">${esc(f.flags.join(' · '))}</div>
      </div>
      <button type="button" data-inspect="${esc(f.doc_id)}" class="font-mono text-xs text-amberWarn hover:text-ink transition-colors shrink-0 self-start sm:self-center">Inspect &amp; Fix →</button>
    </div>`).join('')
    : `<div class="py-6 text-xs font-mono text-emeraldOk">No parse fidelity red flags across ${o.docs} document${o.docs === 1 ? '' : 's'}.</div>`;
  const flagged = new Set(ov.red_flags.map((f) => f.doc_id));
  $('overview-latest-wave').textContent = ov.waves.length ? `Wave ${ov.waves[0].name}` : '';
  $('overview-docs-tbody').innerHTML = ov.recent.length ? ov.recent.map((r) => `
    <tr class="hover:bg-surface/60 transition-colors">
      <td class="py-4 pr-6 font-mono text-xs text-ink"><button type="button" data-open-doc="${esc(r.doc_id)}" class="hover:underline underline-offset-4 text-left break-words">${esc(basename(r.source))}</button></td>
      <td class="py-4 px-4 text-xs text-inkMuted">${esc(r.doc_type || '—')}</td>
      <td class="py-4 px-4 font-mono text-xs text-inkFaint">${esc(engineLabel(r.engine))}</td>
      <td class="py-4 px-4 font-mono text-xs text-inkMuted text-right">${r.num_pages}</td>
      <td class="py-4 px-4 font-mono text-xs text-inkMuted text-right">${r.num_chunks}</td>
      <td class="py-4 pl-6 font-mono text-xs text-right">${flagged.has(r.doc_id)
        ? `<button type="button" data-inspect="${esc(r.doc_id)}" class="text-amberWarn hover:underline underline-offset-4">Warning</button>` : statusText(false)}</td>
    </tr>`).join('')
    : '<tr><td colspan="6" class="py-6 text-xs font-mono text-inkFaint">Nothing ingested yet.</td></tr>';
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------
function tierFor(target) {
  if (target <= 100) return '100-doc tier';
  if (target <= 1000) return '1k tier';
  return '10k tier';
}

async function loadIngestion() {
  const [ov, tel, dl] = await Promise.all([api('/api/admin/overview'), api('/api/admin/telemetry?limit=1000'), api('/api/admin/dead_letters')]);
  state.overview = ov; state.telemetry = tel.rows;
  renderBadges();
  const jobs = ov.jobs || {};
  $('ingestion-waves-tbody').innerHTML = ov.waves.length ? ov.waves.map((w) => {
    const job = jobs[w.name];
    const running = w.status === 'running' || (job && job.status === 'running');
    const done = job && job.progress && running
      ? (job.progress.done || 0) + (job.progress.failed || 0) + (job.progress.skipped || 0) : w.ingested;
    const total = w.target || done || 0;
    const p = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    const status = running ? 'RUNNING' : (w.status || '').toUpperCase();
    return `
      <tr class="hover:bg-surface/60 transition-colors">
        <td class="py-4 pr-6 text-ink whitespace-nowrap">${esc(w.name)}</td>
        <td class="py-4 px-4 text-inkFaint whitespace-nowrap">${tierFor(w.target)}</td>
        <td class="py-4 px-4"><div class="flex items-center space-x-3"><div class="w-40 h-1 bg-hairlineFocus overflow-hidden"><div class="h-full bg-accent" style="width:${p}%"></div></div><span class="text-inkMuted">${done}/${total}</span></div></td>
        <td class="py-4 px-4 text-right text-inkMuted">${w.ok}</td>
        <td class="py-4 px-4 text-right ${w.failed ? 'text-amberWarn' : 'text-inkFaint'}">${w.failed}</td>
        <td class="py-4 px-4 text-right text-inkMuted">${money(w.cost_usd)}</td>
        <td class="py-4 pl-6 text-right ${running ? 'text-accent' : status === 'DONE' ? 'text-emeraldOk' : 'text-amberWarn'}">${esc(status)}</td>
      </tr>`;
  }).join('') : '<tr><td colspan="7" class="py-6 text-inkFaint">No waves yet. Use + Ingest Document to start one.</td></tr>';

  const dls = dl.rows || [];
  $('ingestion-dl-count').textContent = `${dls.length} Dead Letter${dls.length === 1 ? '' : 's'}`;
  $('ingestion-dl-count').className = `text-xs font-mono tabular-nums ${dls.length ? 'text-amberWarn' : 'text-inkFaint'}`;
  $('dead-letter-section').classList.toggle('hidden', !dls.length);
  $('dead-letter-list').innerHTML = dls.map((d) => `
    <div class="py-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
      <div class="space-y-1 min-w-0">
        <div class="font-mono text-xs text-ink break-all">${esc(basename(d.source))} <span class="text-inkFaint">· ${esc(d.wave)} · ${d.attempts} attempt${d.attempts === 1 ? '' : 's'}</span></div>
        <div class="font-mono text-[11px] text-amberWarn break-all">${esc(d.last_error)}</div>
      </div>
      <div class="flex items-center space-x-5 shrink-0 font-mono text-xs">
        <button type="button" data-dl-retry="${d.id}" class="text-inkMuted hover:text-ink">Retry</button>
        <button type="button" data-dl-resolve="${d.id}" class="text-inkFaint hover:text-ink">Resolve</button>
      </div>
    </div>`).join('');

  const rows = tel.rows;
  const chunks = rows.reduce((a, r) => a + r.num_chunks, 0);
  $('ingestion-inventory-meta').textContent = `${rows.length} document${rows.length === 1 ? '' : 's'} · ${chunks} chunks`;
  $('ingestion-full-tbody').innerHTML = rows.length ? rows.map((r) => `
    <tr class="hover:bg-surface/60 transition-colors">
      <td class="py-4 pr-6 text-ink"><button type="button" data-open-doc="${esc(r.doc_id)}" class="hover:underline underline-offset-4 text-left break-words">${esc(r.name)}</button></td>
      <td class="py-4 px-3 text-inkMuted whitespace-nowrap">${esc(r.doc_type || '—')}</td>
      <td class="py-4 px-3 text-inkFaint">${esc(engineLabel(r.engine))}</td>
      <td class="py-4 px-3 text-right text-inkMuted">${r.num_pages}</td>
      <td class="py-4 px-3 text-right text-inkMuted">${r.num_chunks}</td>
      <td class="py-4 px-3 text-right text-inkMuted">${r.num_tables}</td>
      <td class="py-4 px-3 text-right text-inkFaint">${pct(r.num_chunks ? r.empty_chunks / r.num_chunks : 0)}</td>
      <td class="py-4 px-3 text-right text-inkMuted">${money(r.cost_usd)}</td>
      <td class="py-4 pl-6 text-right">${r.flags.length
        ? `<button type="button" data-inspect="${esc(r.doc_id)}" class="text-amberWarn hover:underline underline-offset-4" title="${esc(r.flags.join(' · '))}">WARN</button>`
        : '<span class="text-emeraldOk">OK</span>'}</td>
    </tr>`).join('')
    : '<tr><td colspan="9" class="py-6 text-inkFaint">No documents ingested yet.</td></tr>';
}

function ingestError(msg) {
  const el = $('ingest-error');
  el.textContent = msg || '';
  el.classList.toggle('hidden', !msg);
}

async function onIngestFiles() {
  const input = $('ingest-files');
  if (!input.files.length) return;
  const btn = $('ingest-btn');
  ingestError('');
  btn.disabled = true;
  btn.textContent = `Uploading ${input.files.length}…`;
  try {
    const fd = new FormData();
    for (const f of input.files) fd.append('files', f);
    const up = await api('/api/admin/upload', { method: 'POST', body: fd });
    if (!up.saved.length) throw new Error(up.skipped.map((s) => `${s.file}: ${s.reason}`).join('; ') || 'nothing uploaded');
    const stamp = new Date().toISOString().slice(0, 16).replace('T', '-').replace(':', '');
    const wave = await api('/api/admin/waves', { json: { dir: up.dir, wave: `upload-${stamp}`, workers: 4 } });
    if (up.skipped.length) ingestError(`Skipped: ${up.skipped.map((s) => `${s.file} (${s.reason})`).join(', ')}`);
    await followJob(wave.wave, (p) => { btn.textContent = `Parsing ${p}…`; }, () => { if (state.view === 'ingestion') loadIngestion(); });
    loadStats();
  } catch (e) {
    if (!(e instanceof LockedError)) ingestError(e.message);
  } finally {
    input.value = '';
    btn.disabled = false;
    btn.textContent = '+ Ingest Document';
    if (state.view === 'ingestion') loadIngestion();
  }
}

/** Poll /api/admin/jobs until `name` is no longer running. */
async function followJob(name, onProgress, onTick) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    const { jobs } = await api('/api/admin/jobs');
    const job = jobs[name];
    if (!job) return null;
    const p = job.progress || {};
    const done = (p.done || 0) + (p.failed || 0) + (p.skipped || 0);
    if (onProgress) onProgress(`${done}/${p.total || '?'}`);
    if (onTick) onTick(job);
    if (job.status !== 'running') {
      if (String(job.status).startsWith('failed')) throw new Error(job.status);
      return job;
    }
  }
}

// ---------------------------------------------------------------------------
// Parse fidelity + inspection sheet
// ---------------------------------------------------------------------------
function tableShapeText(r) {
  if (!r.num_tables) return 'n/a';
  return `${Math.round((r.table_shape_ok / r.num_tables) * 100)}% (${r.table_shape_ok}/${r.num_tables})`;
}

async function loadFidelity() {
  const tel = await api('/api/admin/telemetry?limit=1000');
  state.telemetry = tel.rows;
  const rows = tel.rows;
  const tables = rows.reduce((a, r) => a + r.num_tables, 0);
  const tablesOk = rows.reduce((a, r) => a + r.table_shape_ok, 0);
  const leaks = rows.reduce((a, r) => a + r.repeated_ngrams, 0);
  const cov = rows.filter((r) => r.text_coverage != null);
  const meanCov = cov.length ? cov.reduce((a, r) => a + r.text_coverage, 0) / cov.length : null;
  const shape = tables ? tablesOk / tables : null;
  const setMetric = (id, text, warnIf) => {
    $(id).textContent = text;
    $(id).className = `font-mono text-3xl font-light tabular-nums ${warnIf ? 'text-amberWarn' : 'text-ink'}`;
  };
  setMetric('fid-metric-table', shape == null ? 'n/a' : pct(shape), shape != null && shape < 0.9);
  $('fid-metric-table-sub').textContent = tables ? `${tablesOk} of ${tables} tables consistent` : 'No tables detected yet';
  setMetric('fid-metric-leak', `${leaks} chunk${leaks === 1 ? '' : 's'}`, leaks > 0);
  setMetric('fid-metric-coverage', meanCov == null ? 'n/a' : pct(meanCov), meanCov != null && meanCov < 0.95);
  $('fid-metric-coverage-sub').textContent = cov.length
    ? `Threshold ≥ 95% · measured on ${cov.length} of ${rows.length} docs` : 'Needs source PDFs on the server';

  const flagged = rows.filter((r) => r.flags.length);
  const reparseBtn = $('reparse-all-btn');
  reparseBtn.disabled = !flagged.length;
  if (!state.reparseAllArmed) reparseBtn.textContent = flagged.length ? `Re-parse All Flagged (${flagged.length})` : 'No Flagged Documents';

  $('fidelity-matrix-tbody').innerHTML = rows.length ? rows.map((r) => {
    const shapeWarn = r.num_tables && r.table_shape_ok / r.num_tables < 0.9;
    const covWarn = r.text_coverage != null && r.text_coverage < 0.95;
    return `
      <tr class="hover:bg-surface/60 transition-colors">
        <td class="py-4 pr-6 text-ink"><button type="button" data-inspect="${esc(r.doc_id)}" class="hover:underline underline-offset-4 text-left break-words">${esc(r.name)}</button></td>
        <td class="py-4 px-4 text-inkFaint">${esc(engineLabel(r.engine))}</td>
        <td class="py-4 px-4 text-right whitespace-nowrap ${shapeWarn ? 'text-amberWarn' : 'text-inkMuted'}">${tableShapeText(r)}</td>
        <td class="py-4 px-4 text-right whitespace-nowrap ${r.repeated_ngrams ? 'text-amberWarn' : 'text-inkMuted'}">${r.repeated_ngrams} chunk${r.repeated_ngrams === 1 ? '' : 's'}</td>
        <td class="py-4 px-4 text-right ${r.text_coverage == null ? 'text-inkFaint' : covWarn ? 'text-amberWarn' : 'text-emeraldOk'}">${r.text_coverage == null ? 'n/a' : pct(r.text_coverage)}</td>
        <td class="py-4 pl-6 text-right whitespace-nowrap">${r.flags.length
          ? `<button type="button" data-inspect="${esc(r.doc_id)}" class="text-amberWarn hover:text-ink underline underline-offset-4">Inspect →</button>`
          : '<span class="text-emeraldOk">Verified</span>'}</td>
      </tr>`;
  }).join('') : '<tr><td colspan="6" class="py-6 text-inkFaint">No documents ingested yet.</td></tr>';
}

function disarmReparseAll() {
  state.reparseAllArmed = false;
  $('reparse-all-cancel').classList.add('hidden');
}

async function onReparseAll() {
  const flagged = (state.telemetry || []).filter((r) => r.flags.length);
  if (!flagged.length) return;
  const btn = $('reparse-all-btn');
  const err = $('reparse-all-error');
  err.classList.add('hidden');
  if (!state.reparseAllArmed) {
    const pages = flagged.reduce((a, r) => a + (r.num_pages || 0), 0);
    state.reparseAllArmed = true;
    btn.textContent = `Confirm · ${flagged.length} doc${flagged.length === 1 ? '' : 's'} · est. ${money(pages * PERF_RATE)}`;
    $('reparse-all-cancel').classList.remove('hidden');
    return;
  }
  disarmReparseAll();
  btn.disabled = true;
  try {
    const r = await api('/api/admin/reparse', { json: { doc_ids: flagged.map((x) => x.doc_id) } });
    await followJob(r.job, (p) => { btn.textContent = `Re-parsing ${p}…`; });
  } catch (e) {
    if (!(e instanceof LockedError)) { err.textContent = e.message; err.classList.remove('hidden'); }
  } finally {
    btn.disabled = false;
    state.golden = null;
    state.pageCache.clear();
    loadFidelity();
  }
}

const CHECK_MARK = {
  true: '<span class="text-emeraldOk" aria-label="pass">✓</span>',
  false: '<span class="text-amberWarn" aria-label="fail">!</span>',
  null: '<span class="text-inkFaint" aria-label="not applicable">–</span>',
};

function renderChecks(checks) {
  // Failing checks first, then passing, then not applicable.
  const order = (c) => (c.ok === false ? 0 : c.ok === true ? 1 : 2);
  return [...checks].sort((a, b) => order(a) - order(b)).map((c) => `
    <div class="py-3 grid grid-cols-[1rem_1fr_auto] gap-x-3 items-baseline">
      <span class="font-mono text-xs">${CHECK_MARK[c.ok]}</span>
      <div class="min-w-0">
        <div class="text-xs ${c.ok === false ? 'text-ink' : 'text-inkMuted'}">${esc(c.label)}</div>
        ${c.ok === false ? `<div class="text-[11px] text-inkFaint leading-relaxed mt-1">${esc(c.why)}</div>` : ''}
      </div>
      <div class="text-right font-mono text-xs whitespace-nowrap">
        <div class="${c.ok === false ? 'text-amberWarn' : 'text-inkMuted'}">${esc(c.value)}</div>
        <div class="text-[11px] text-inkFaint mt-0.5">${c.ok === null ? '' : `target ${esc(c.threshold)}`}</div>
      </div>
    </div>`).join('');
}

function renderTableExample(t) {
  const maxCells = Math.max(...t.rows.map((r) => r.cells.length), t.expected_width);
  const firstBad = t.rows.find((r) => r.bad);
  const caption = firstBad
    ? `Row ${firstBad.index} has ${firstBad.width} column${firstBad.width === 1 ? '' : 's'}; the table's other rows have ${t.expected_width}. ${t.bad_count} of ${t.total_rows} rows don't match.`
    : `${t.bad_count} of ${t.total_rows} rows don't match the table's ${t.expected_width} columns.`;
  const rows = t.rows.map((r) => {
    const cells = r.cells.map((c) => `<td class="px-2 py-1 border-b border-hairline align-top">${esc(c.length > 40 ? c.slice(0, 38) + '…' : c)}</td>`).join('');
    const pad = '<td class="px-2 py-1 border-b border-hairline"></td>'.repeat(Math.max(0, maxCells - r.cells.length));
    const cls = r.bad ? 'bg-amberWarn/10 text-ink' : r.header ? 'text-inkFaint' : 'text-inkMuted';
    const mark = r.bad ? `<span class="text-amberWarn">${r.width}≠${t.expected_width}</span>` : '';
    return `<tr class="${cls}"><td class="pr-2 py-1 border-b border-hairline text-inkFaint text-right">${r.index}</td>${cells}${pad}<td class="pl-2 py-1 border-b border-hairline text-right whitespace-nowrap">${mark}</td></tr>`;
  }).join('');
  return `
    <p class="text-xs text-inkMuted leading-relaxed">${esc(caption)}</p>
    <div class="overflow-x-auto border border-hairline bg-surface">
      <table class="w-full font-mono text-[11px] tabular-nums border-collapse">${rows}</table>
    </div>`;
}

const EXAMPLE_HELP = {
  'Page number leaked into body': 'The number at the start or end of this passage is the page number, not content.',
  'Empty chunk': 'This passage has almost no text.',
  'Garbled text': 'This passage contains broken characters.',
};

async function openDiffSheet(docId) {
  state.diff = { docId, job: null };
  $('diff-sheet-backdrop').classList.remove('hidden');
  $('diff-doc-title').textContent = 'Loading…';
  ['diff-verdict', 'diff-checks', 'diff-examples', 'diff-right', 'diff-cost'].forEach((id) => { $(id).innerHTML = ''; });
  $('diff-close').focus();
  try {
    renderDiff(await api(`/api/admin/doc/${encodeURIComponent(docId)}/fidelity`));
  } catch (e) {
    if (!(e instanceof LockedError)) $('diff-verdict').textContent = e.message;
  }
}

function renderDiff(d) {
  $('diff-doc-title').textContent = d.name;
  $('diff-verdict').textContent = d.verdict;
  $('diff-engine').textContent = `Parsed with ${engineLabel(d.engine)} engine · ${d.pages} page${d.pages === 1 ? '' : 's'}`;
  $('diff-checks').innerHTML = renderChecks(d.checks || []);

  $('diff-examples-section').classList.toggle('hidden', !d.excerpts.length);
  $('diff-examples').innerHTML = d.excerpts.map((x) => `
    <div class="space-y-2">
      <div class="text-xs font-mono text-ink">${esc(x.kind)}${x.page != null ? ` <span class="text-inkFaint">· page ${x.page}</span>` : ''}</div>
      ${x.table ? renderTableExample(x.table) : `
        ${EXAMPLE_HELP[x.kind] ? `<p class="text-xs text-inkMuted">${esc(EXAMPLE_HELP[x.kind])}</p>` : ''}
        <div class="p-3 bg-surface border-l-2 border-amberWarn font-mono text-xs text-ink leading-relaxed break-words">${esc(x.text || '(empty)')}</div>`}
    </div>`).join('');

  const last = d.history[0];
  const btn = $('diff-reparse');
  if (last) {
    const before = last.before || {};
    const after = last.after || {};
    const row = (label, b, a, better) => {
      let mark = '';
      if (b != null && a != null && b !== a && better) {
        const good = better === 'down' ? a < b : a > b;
        mark = ` <span class="${good ? 'text-emeraldOk' : 'text-amberWarn'}">${good ? 'better' : 'worse'}</span>`;
      }
      return `<div class="flex justify-between gap-4"><span>${label}</span><span class="text-ink text-right">${b ?? '—'} → ${a ?? '—'}${mark}</span></div>`;
    };
    const tables = (r) => (r.num_tables ? `${r.table_shape_ok}/${r.num_tables}` : 'none');
    const cov = (r) => (r.text_coverage == null ? 'n/a' : pct(r.text_coverage));
    $('diff-right-label').textContent = `Before → after (${engineLabel(last.engine)})`;
    $('diff-right-status').innerHTML = `<span class="text-xs font-mono text-inkFaint">${esc(new Date(last.at).toLocaleString())}</span>`;
    $('diff-right').innerHTML = [
      row('Consistent tables', tables(before), tables(after)),
      row('Page numbers in text', before.repeated_ngrams, after.repeated_ngrams, 'down'),
      row('Text coverage', cov(before), cov(after)),
      row('Empty passages', before.empty_chunks, after.empty_chunks, 'down'),
      row('Chunks', before.num_chunks, after.num_chunks),
      `<div class="flex justify-between gap-4"><span>Cost</span><span class="text-ink">${money(last.cost_usd)}</span></div>`,
    ].join('');
  } else {
    $('diff-right-label').textContent = 'Performance engine';
    $('diff-right-status').innerHTML = '<span class="text-xs font-mono text-inkFaint">Not run yet</span>';
    $('diff-right').innerHTML = `
      <p>Re-parses this document with Extend's <span class="text-ink">parse_performance</span> engine, which is stronger on tables, scans and multi-column layouts.</p>
      <p>Replaces the document's passages, vectors and checks. The current numbers are kept, so this section then shows before → after.</p>`;
  }
  $('diff-cost').textContent = `Estimated cost: ${money(d.estimated_cost_usd)} (${d.pages} page${d.pages === 1 ? '' : 's'} × $${PERF_RATE})`;
  btn.disabled = !d.source_available;
  btn.textContent = last ? 'Re-parse Again' : 'Re-parse with Performance Engine';
  btn.title = d.source_available ? '' : 'The original file is not on the server. Re-upload it from Ingestion to re-parse.';
  if (!d.source_available) $('diff-cost').textContent += ' · original file not on server';
}

function closeDiffSheet() {
  $('diff-sheet-backdrop').classList.add('hidden');
  state.diff = { docId: null, job: null };
}

async function onDiffReparse() {
  const docId = state.diff.docId;
  if (!docId) return;
  const btn = $('diff-reparse');
  btn.disabled = true;
  btn.textContent = 'Starting…';
  try {
    const r = await api('/api/admin/reparse', { json: { doc_ids: [docId] } });
    state.diff.job = r.job;
    await followJob(r.job, (p) => { btn.textContent = `Re-parsing ${p}…`; });
    state.golden = null;
    state.pageCache.clear();
    if (state.diff.docId === docId) renderDiff(await api(`/api/admin/doc/${encodeURIComponent(docId)}/fidelity`));
    if (state.view !== 'qa') loadView(state.view);
  } catch (e) {
    if (!(e instanceof LockedError)) {
      $('diff-cost').textContent = `Re-parse failed: ${e.message}`;
      btn.disabled = false;
      btn.textContent = 'Retry Re-parse';
    }
  }
}

// ---------------------------------------------------------------------------
// Golden sets
// ---------------------------------------------------------------------------
function relTime(iso) {
  if (!iso) return '—';
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return 'Just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

const SOURCE_TAG = { golden_qa: 'Q&A set', golden_doc: 'Doc probe', feedback: 'Feedback' };

async function loadGolden(run = false) {
  const g = run ? await api('/api/admin/golden/run', { method: 'POST' }) : (state.golden || await api('/api/admin/golden'));
  state.golden = g;
  renderBadges();
  const fb = g.rows.filter((r) => r.source === 'feedback').length;
  $('golden-total-count').textContent = g.total;
  $('golden-total-sub').textContent = `${fb} from Q&A feedback`;
  $('golden-pass-rate').textContent = pct(g.pass_rate);
  $('golden-pass-rate').className = `font-mono text-3xl font-light tabular-nums ${g.pass_rate >= 0.9 ? 'text-emeraldOk' : 'text-amberWarn'}`;
  $('golden-pass-sub').textContent = `${g.passed} passing · ${g.failed} failing`;
  $('golden-last-run').textContent = relTime(g.ran_at);
  $('golden-last-run').title = g.ran_at ? new Date(g.ran_at).toLocaleString() : '';
  $('golden-last-run-sub').textContent = `Asserted against ${g.chunks} chunks`;
  const c = g.compare;
  if (c && c.keyword) {
    $('golden-hit').textContent = c.hybrid ? `${c.keyword.hit_rate.toFixed(2)} → ${c.hybrid.hit_rate.toFixed(2)}` : c.keyword.hit_rate.toFixed(2);
    $('golden-hit-sub').textContent = c.hybrid ? `keyword → hybrid · ${c.model}` : 'keyword only · embeddings off';
  } else {
    $('golden-hit').textContent = 'n/a';
    $('golden-hit-sub').textContent = 'No golden Q&A file';
  }
  $('golden-assertions-tbody').innerHTML = g.rows.length ? g.rows.map((a) => {
    const color = a.status === 'PASS' ? 'text-emeraldOk' : a.status === 'RESOLVED' ? 'text-inkFaint' : 'text-amberWarn';
    const action = a.source === 'feedback' && a.status === 'FAIL'
      ? `<button type="button" data-fb-resolve="${a.id}" class="block ml-auto mt-1 text-[11px] text-inkMuted hover:text-ink underline underline-offset-4">Resolve</button>` : '';
    return `
      <tr class="hover:bg-surface/60 transition-colors align-top">
        <td class="py-4 pr-6 text-ink break-words"><div>${esc(a.query)}</div><div class="text-[11px] text-inkFaint mt-1">${SOURCE_TAG[a.source] || ''}</div></td>
        <td class="py-4 px-4 text-inkMuted break-words">${esc(a.expected)}</td>
        <td class="py-4 px-4 text-inkFaint break-words">${esc(a.retrieved)}</td>
        <td class="py-4 px-4 font-sans text-xs text-inkMuted break-words">${esc(a.note)}</td>
        <td class="py-4 pl-6 text-right whitespace-nowrap ${color}">${esc(a.status)}${action}</td>
      </tr>`;
  }).join('') : '<tr><td colspan="5" class="py-6 text-inkFaint">No assertions yet. Add a golden Q&A file or rate answers in Q&A.</td></tr>';
}

async function onRunSuite() {
  const btn = $('run-suite-btn');
  btn.disabled = true;
  btn.textContent = 'Running Assertions…';
  $('golden-last-run').textContent = 'Running…';
  try { await loadGolden(true); } catch (e) { if (!(e instanceof LockedError)) $('golden-last-run-sub').textContent = e.message; }
  btn.disabled = false;
  btn.textContent = 'Run Golden Suite';
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------
function openDocInWorkspace(docId) {
  state.result = null;
  state.viewer.focusChunk = null;
  renderWorkspace();
  location.hash = '#qa';
  showPage(docId, 1, { scrollToActive: false });
}

document.addEventListener('click', async (ev) => {
  const t = ev.target.closest('[data-source],[data-example],[data-inspect],[data-open-doc],[data-chunk],[data-issue],[data-search-type],[data-fb-resolve],[data-dl-retry],[data-dl-resolve],.cite');
  if (!t) return;
  if (t.matches('.cite')) { const n = parseInt(t.dataset.n, 10) - 1; if (state.result && state.result.passages[n]) selectSource(n); return; }
  if (t.dataset.source !== undefined) { selectSource(parseInt(t.dataset.source, 10)); return; }
  if (t.dataset.example !== undefined) { $('query-input').value = t.dataset.example; setSearchType('ask'); return; }
  if (t.dataset.inspect !== undefined) { openDiffSheet(t.dataset.inspect); return; }
  if (t.dataset.openDoc !== undefined) { openDocInWorkspace(t.dataset.openDoc); return; }
  if (t.dataset.chunk !== undefined && t.closest('#document-sheet')) { focusChunkInViewer(t.dataset.chunk); return; }
  if (t.dataset.issue !== undefined) { pickIssue(t.dataset.issue); return; }
  if (t.dataset.searchType !== undefined) { setSearchType(t.dataset.searchType); return; }
  try {
    if (t.dataset.fbResolve !== undefined) { await api(`/api/admin/feedback/${t.dataset.fbResolve}/resolve`, { method: 'POST' }); state.golden = null; loadGolden(true); return; }
    if (t.dataset.dlRetry !== undefined) { await api(`/api/admin/dead_letters/${t.dataset.dlRetry}/retry`, { method: 'POST' }); loadIngestion(); return; }
    if (t.dataset.dlResolve !== undefined) { await api(`/api/admin/dead_letters/${t.dataset.dlResolve}/resolve`, { method: 'POST' }); loadIngestion(); return; }
  } catch (e) { if (!(e instanceof LockedError)) ingestError(e.message); }
});

document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && !$('diff-sheet-backdrop').classList.contains('hidden')) { closeDiffSheet(); return; }
  if ((ev.key === 'Enter' || ev.key === ' ') && ev.target.dataset && ev.target.dataset.source !== undefined) {
    ev.preventDefault();
    selectSource(parseInt(ev.target.dataset.source, 10));
  }
});

function setSearchType(type) {
  state.searchType = type;
  store.set('tx.searchType', type);
  renderSearchType();
  if ($('query-input').value.trim()) runQuery();
}

function init() {
  $('query-form').addEventListener('submit', (ev) => { ev.preventDefault(); runQuery(); });
  $('mode-toggle-btn').addEventListener('click', toggleSynthesisMode);
  $('subtitle-action-link').addEventListener('click', toggleSynthesisMode);
  $('refresh-btn').addEventListener('click', async () => {
    const btn = $('refresh-btn');
    state.golden = null; state.pageCache.clear();
    await loadStats();
    await loadView(state.view);
    if (state.view === 'qa' && state.viewer.docId) showPage(state.viewer.docId, state.viewer.page, { scrollToActive: false });
    btn.textContent = 'Synced';
    btn.classList.add('text-emeraldOk');
    setTimeout(() => { btn.textContent = 'Refresh'; btn.classList.remove('text-emeraldOk'); }, 900);
  });
  $('lock-btn').addEventListener('click', async () => {
    try { await api('/api/admin/logout', { method: 'POST' }); } catch (e) { /* ignore */ }
    state.admin = false; state.overview = null; state.golden = null;
    renderAdminChrome();
    switchView(state.view);
  });
  $('fb-yes').addEventListener('click', onFeedbackYes);
  $('fb-no').addEventListener('click', onFeedbackNo);
  $('fb-form').addEventListener('submit', submitNegativeFeedback);
  $('page-prev').addEventListener('click', () => { if (state.viewer.page > 1) showPage(state.viewer.docId, state.viewer.page - 1, { scrollToActive: false }); });
  $('page-next').addEventListener('click', () => { if (state.viewer.page < state.viewer.numPages) showPage(state.viewer.docId, state.viewer.page + 1, { scrollToActive: false }); });
  $('toggle-all-chunks-btn').addEventListener('click', () => {
    state.allChunks = !state.allChunks;
    const b = $('toggle-all-chunks-btn');
    b.textContent = `All chunks: ${state.allChunks ? 'On' : 'Off'}`;
    b.className = state.allChunks ? 'text-ink transition-colors' : 'text-inkFaint hover:text-ink transition-colors';
    if (state.viewer.docId) showPage(state.viewer.docId, state.viewer.page, { scrollToActive: false });
  });
  $('toggle-surface-btn').addEventListener('click', () => {
    state.surface = { image: 'reader', reader: 'paper', paper: 'image' }[state.surface];
    store.set('tx.surface', state.surface);
    if (state.viewer.docId) showPage(state.viewer.docId, state.viewer.page, { scrollToActive: true });
    else $('toggle-surface-btn').textContent = `Surface: ${{ image: 'Image', reader: 'Reader', paper: 'Paper' }[state.surface]}`;
  });
  $('ingest-btn').addEventListener('click', () => $('ingest-files').click());
  $('ingest-files').addEventListener('change', onIngestFiles);
  $('reparse-all-btn').addEventListener('click', onReparseAll);
  $('reparse-all-cancel').addEventListener('click', () => { disarmReparseAll(); loadFidelity(); });
  $('run-suite-btn').addEventListener('click', onRunSuite);
  $('diff-close').addEventListener('click', closeDiffSheet);
  $('diff-cancel').addEventListener('click', closeDiffSheet);
  $('diff-reparse').addEventListener('click', onDiffReparse);
  $('diff-sheet-backdrop').addEventListener('click', (ev) => { if (ev.target === $('diff-sheet-backdrop')) closeDiffSheet(); });
  window.addEventListener('hashchange', () => switchView(currentTab()));

  $('toggle-surface-btn').textContent = `Surface: ${{ image: 'Image', reader: 'Reader', paper: 'Paper' }[state.surface]}`;
  renderSearchType();
  pickIssue(ISSUES[0]);
  renderSubtitle();
  loadExamples();
  Promise.all([loadStats(), api('/api/session').then((s) => { state.admin = !!s.admin; }).catch(() => {})])
    .then(() => {
      renderAdminChrome();
      loadBadges();
      switchView(currentTab());
    });
}

init();
