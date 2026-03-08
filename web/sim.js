// Core structure
const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
let ws, terrainCanvas, minimapTerrainCanvas;
let worldW = 0, worldH = 0;
let cellSize = 0;
let offsetX = 0, offsetY = 0;
let latestAgents = [];
let terrainColors = {};

const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');
const minimap = document.getElementById('minimap');
const mctx = minimap.getContext('2d');
const statusDot = document.getElementById('status-dot');

const AGENT_COLORS = {
    trex: '#7f1d1d',
    velociraptor: '#4c1d95',
    triceratops: '#14532d',
    gallimimus: '#713f12',
    herbivore: '#166534',
    carnivore: '#991b1b'
};

const AGENT_SIZES = {
    trex: 10,
    triceratops: 8,
    velociraptor: 7,
    gallimimus: 6,
    herbivore: 7,
    carnivore: 8
};

function connect() {
    ws = new WebSocket(WS_URL);
    ws.onopen = () => {
        statusDot.className = 'connected';
        console.log('Connected to simulation');
    };
    ws.onclose = () => {
        statusDot.className = 'error';
        setTimeout(connect, 2000);
    };
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'init') handleInit(data);
        if (data.type === 'tick') handleTick(data);
    };

    window.sendWS = (msg) => {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(msg));
        }
    };
}

function handleInit(data) {
    worldW = data.width;
    worldH = data.height;
    terrainColors = data.terrain_colors;

    // Create offscreen terrain canvas
    terrainCanvas = document.createElement('canvas');
    terrainCanvas.width = worldW;
    terrainCanvas.height = worldH;
    const tctx = terrainCanvas.getContext('2d');
    
    const imageData = tctx.createImageData(worldW, worldH);
    const data8 = imageData.data;

    for (let y = 0; y < worldH; y++) {
        for (let x = 0; x < worldW; x++) {
            const type = data.terrain[y][x];
            const hex = terrainColors[type] || '#000000';
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            const idx = (y * worldW + x) * 4;
            data8[idx] = r;
            data8[idx + 1] = g;
            data8[idx + 2] = b;
            data8[idx + 3] = 255;
        }
    }
    tctx.putImageData(imageData, 0, 0);

    // Create minimap terrain
    minimapTerrainCanvas = document.createElement('canvas');
    minimapTerrainCanvas.width = minimap.width;
    minimapTerrainCanvas.height = minimap.height;
    const mtx = minimapTerrainCanvas.getContext('2d');
    mtx.imageSmoothingEnabled = false;
    mtx.drawImage(terrainCanvas, 0, 0, minimap.width, minimap.height);

    resize();
}

function handleTick(data) {
    latestAgents = data.agents;
    if (window.updateDrawerStats) window.updateDrawerStats(data);
    requestAnimationFrame(render);
}

function render() {
    if (!terrainCanvas) return;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw terrain
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(terrainCanvas, offsetX, offsetY, worldW * cellSize, worldH * cellSize);

    drawAgents(ctx, latestAgents);
    drawMinimap(latestAgents);
}

function drawAgents(ctx, agents) {
    agents.forEach(agent => {
        const x = offsetX + (agent.x * cellSize) + (cellSize / 2);
        const y = offsetY + (agent.y * cellSize) + (cellSize / 2);
        const color = AGENT_COLORS[agent.species] || '#ffffff';
        const size = (AGENT_SIZES[agent.species] || 5) * (cellSize / 4);

        // Draw trail
        if (agent.trail && agent.trail.length > 0) {
            agent.trail.forEach((pos, i) => {
                const tx = offsetX + (pos[0] * cellSize) + (cellSize / 2);
                const ty = offsetY + (pos[1] * cellSize) + (cellSize / 2);
                const alpha = (i + 1) / (agent.trail.length + 1) * 0.3;
                ctx.beginPath();
                ctx.arc(tx, ty, size * 0.8, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = alpha;
                ctx.fill();
            });
            ctx.globalAlpha = 1.0;
        }

        // Draw body
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Energy bar
        const barW = size * 2;
        const barH = 2;
        const ratio = agent.energy / agent.max_energy;
        const barColor = ratio > 0.6 ? '#22c55e' : ratio > 0.3 ? '#f59e0b' : '#ef4444';
        
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(x - size, y - size - 5, barW, barH);
        ctx.fillStyle = barColor;
        ctx.fillRect(x - size, y - size - 5, barW * Math.min(1, ratio), barH);
    });
}

function drawMinimap(agents) {
    mctx.drawImage(minimapTerrainCanvas, 0, 0);
    const mx = minimap.width / worldW;
    const my = minimap.height / worldH;

    agents.forEach(agent => {
        mctx.fillStyle = AGENT_COLORS[agent.species] || '#ffffff';
        mctx.fillRect(agent.x * mx, agent.y * my, 2, 2);
    });
}

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    if (worldW && worldH) {
        cellSize = Math.min(canvas.width / worldW, canvas.height / worldH);
        offsetX = (canvas.width - worldW * cellSize) / 2;
        offsetY = (canvas.height - worldH * cellSize) / 2;
    }
    render();
}

window.addEventListener('resize', resize);
connect();
