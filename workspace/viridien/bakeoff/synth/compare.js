'use strict';
const trials = __TRIALS__;
const version = '__VERSION__';
const storageKey = 'lithology-blind-' + version;
let results = {};
try { results = JSON.parse(localStorage.getItem(storageKey) || '{}'); }
catch (_) { document.querySelector('#storage').textContent = 'Browser storage unavailable; export results before closing.'; }
if (!results || typeof results !== 'object' || Array.isArray(results)) results = {};
function persist() {
  try { localStorage.setItem(storageKey, JSON.stringify(results)); }
  catch (_) { document.querySelector('#storage').textContent = 'Could not save in browser storage. Export results to keep them.'; }
}
function update() {
  let answered = 0, correct = 0;
  trials.forEach((trial, index) => {
    const response = results[trial.id];
    const valid = response && (response.choice === 0 || response.choice === 1);
    const buttons = document.querySelectorAll('[data-trial="' + index + '"]');
    buttons.forEach(button => {
      const side = Number(button.dataset.side);
      button.disabled = Boolean(valid);
      button.classList.toggle('chosen', Boolean(valid && response.choice === side));
      button.classList.toggle('real', Boolean(valid && trial.real === side));
    });
    const answer = document.querySelector('#answer-' + index);
    if (valid) {
      answered++;
      const hit = response.choice === trial.real;
      if (hit) correct++;
      answer.textContent = (hit ? 'Correct. ' : 'Incorrect. ') + String.fromCharCode(65 + trial.real) + ' is real; ' + String.fromCharCode(65 + 1 - trial.real) + ' is generated.';
    } else answer.textContent = 'Unanswered';
  });
  document.querySelector('#score').textContent = correct + ' correct / ' + answered + ' answered' + (answered ? ' (' + Math.round(correct / answered * 100) + '%)' : '');
}
document.querySelectorAll('.pick').forEach(button => button.addEventListener('click', () => {
  const trial = trials[Number(button.dataset.trial)];
  if (results[trial.id]) return;
  results[trial.id] = {pid: trial.pid, choice: Number(button.dataset.side), real: trial.real,
    correct: Number(button.dataset.side) === trial.real, timestamp: new Date().toISOString()};
  persist(); update();
}));
document.querySelector('#reset').addEventListener('click', () => { results = {}; persist(); update(); });
document.querySelector('#export').addEventListener('click', () => {
  const entries = trials.filter(t => results[t.id]).map(t => Object.assign({trial_id: t.id}, results[t.id]));
  const payload = {version, exported_at: new Date().toISOString(), answered: entries.length,
    correct: entries.filter(r => r.correct).length, results: entries};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'lithology-blind-results.json';
  document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
update();
