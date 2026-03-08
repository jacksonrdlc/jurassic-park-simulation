// Toggle drawer
const drawer = document.getElementById('stats-drawer');
const toggleBtn = document.getElementById('drawer-toggle');
let drawerOpen = false;

function toggleDrawer() {
  drawerOpen = !drawerOpen;
  drawer.classList.toggle('open', drawerOpen);
}
toggleBtn.addEventListener('click', toggleDrawer);
document.addEventListener('keydown', e => { if (e.key === 's' || e.key === 'S') toggleDrawer(); });

// Space = pause (send to server via ws)
document.addEventListener('keydown', e => {
  if (e.key === ' ' && e.target.tagName !== 'INPUT') {
    e.preventDefault();
    window.sendWS?.({ action: 'pause_toggle' });
  }
});

// Speed buttons
document.querySelectorAll('.speed-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    window.sendWS?.({ action: 'speed', value: parseInt(btn.dataset.speed) });
  });
});

// Update stats from tick data
function updateStats(data) {
  document.getElementById('stat-step').textContent = data.step.toLocaleString();
  document.getElementById('stat-status').textContent = data.paused ? '⏸ Paused' : '▶ Running';
  document.getElementById('stat-carnivores').textContent = data.population.carnivores;
  document.getElementById('stat-herbivores').textContent = data.population.herbivores;
  document.getElementById('stat-grass').textContent = data.population.grass.toLocaleString();
  document.getElementById('stat-temp').textContent = data.temperature.toFixed(1) + '°C';
  document.getElementById('stat-rain').textContent = data.rainfall.toFixed(0) + 'mm';
  
  // Time of day
  const tod = data.time_of_day;
  const hours = Math.floor(tod * 24);
  const mins = Math.floor((tod * 24 - hours) * 60);
  document.getElementById('stat-tod').textContent = `${hours.toString().padStart(2,'0')}:${mins.toString().padStart(2,'0')}`;
  document.getElementById('time-bar').style.width = (tod * 100) + '%';
  
  // Ecosystem health (ratio of herbivores to carnivores, normalized)
  const ratio = data.population.carnivores > 0 
    ? data.population.herbivores / (data.population.carnivores * 2.5)
    : 1;
  const health = Math.min(1, Math.max(0, ratio));
  const healthBar = document.getElementById('health-bar');
  healthBar.style.width = (health * 100) + '%';
  healthBar.style.background = health > 0.6 ? '#22c55e' : health > 0.3 ? '#f59e0b' : '#ef4444';
  
  // Events
  if (data.events && data.events.length > 0) {
    const log = document.getElementById('event-log');
    data.events.forEach(evt => {
      const el = document.createElement('div');
      el.className = 'event-item';
      el.textContent = evt;
      log.insertBefore(el, log.firstChild);
    });
    // Keep last 30
    while (log.children.length > 30) log.removeChild(log.lastChild);
  }
}

window.updateDrawerStats = updateStats;
