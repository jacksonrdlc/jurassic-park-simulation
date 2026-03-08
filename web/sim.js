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

// Zoom and Pan State
let zoom = 1.0;
let camX = 0; // camera center X in world coords
let camY = 0; // camera center Y in world coords
let isDragging = false;
let dragStartX = 0, dragStartY = 0;
let dragCamStartX = 0, dragCamStartY = 0;
const minZoom = 1.0;
const maxZoom = 4.0;

const AGENT_COLORS = {
    trex: '#7f1d1d',
    velociraptor: '#4c1d95',
    triceratops: '#14532d',
    gallimimus: '#713f12',
    herbivore: '#166534',
    carnivore: '#991b1b'
};

const AGENT_SIZES = {
    trex: 2.8,
    triceratops: 2.2,
    velociraptor: 2.0,
    gallimimus: 1.8,
    herbivore: 1.5,
    carnivore: 1.8
};

const AGENT_TINTS = {
    trex: 'rgba(180, 60, 60, 0.3)',
    velociraptor: 'rgba(80, 40, 120, 0.3)',
    triceratops: 'rgba(40, 100, 60, 0.3)',
    gallimimus: 'rgba(160, 120, 40, 0.3)',
    herbivore: 'rgba(40, 100, 60, 0.2)',
    carnivore: 'rgba(180, 60, 60, 0.2)'
};

const SPRITE_PATHS = {
    trex: '/sprites/TyrannosaurusRex_16x16.png',
    velociraptor: '/sprites/Spinosaurus_16x16.png',
    triceratops: '/sprites/Triceratops_16x16.png',
    gallimimus: '/sprites/Parasaurolophus_16x16.png',
    herbivore: '/sprites/Styracosaurus_16x16.png',
    carnivore: '/sprites/Archeopteryx_16x16.png'
};

const sprites = {};
let spritesReady = false;

function preloadSprites() {
    const species = Object.keys(SPRITE_PATHS);
    let loadedCount = 0;
    species.forEach(s => {
        const img = new Image();
        img.src = SPRITE_PATHS[s];
        img.onload = () => {
            // Process sprite sheet into 4x4 grid if 64x64
            // But for simplicity in this first pass, we'll store the whole sheet
            // and the drawing function will handle clipping if needed.
            // According to sprite_sheet.py, it's a 4x4 grid of 16x16 or 32x32 frames.
            sprites[s] = img;
            loadedCount++;
            if (loadedCount === species.length) {
                spritesReady = true;
                console.log('All sprites loaded');
            }
        };
        img.onerror = () => {
            console.error(`Failed to load sprite for ${s}`);
            loadedCount++;
            if (loadedCount === species.length) {
                spritesReady = true;
            }
        };
    });
}

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
    
    // Initial camera position
    camX = worldW / 2;
    camY = worldH / 2;

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

function clampCamera() {
    const visW = worldW / zoom;
    const visH = worldH / zoom;
    const halfVisW = visW / 2, halfVisH = visH / 2;
    camX = Math.max(halfVisW, Math.min(worldW - halfVisW, camX));
    camY = Math.max(halfVisH, Math.min(worldH - halfVisH, camY));
}

function dirToAngle(dx, dy) {
    if (dx === 0 && dy === -1) return 0;
    if (dx === 1 && dy === -1) return 45;
    if (dx === 1 && dy === 0) return 90;
    if (dx === 1 && dy === 1) return 135;
    if (dx === 0 && dy === 1) return 180;
    if (dx === -1 && dy === 1) return 225;
    if (dx === -1 && dy === 0) return 270;
    if (dx === -1 && dy === -1) return 315;
    return 0;
}

function getSpriteRow(dx, dy) {
    // Row 0: South, 1: West, 2: East, 3: North
    if (dx === 0 && dy === 1) return 0; // South
    if (dx === -1) return 1; // West (covers SW, W, NW)
    if (dx === 1) return 2; // East (covers SE, E, NE)
    if (dx === 0 && dy === -1) return 3; // North
    return 0;
}

function render() {
    if (!terrainCanvas) return;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    clampCamera();

    const visW = worldW / zoom;
    const visH = worldH / zoom;
    const halfVisW = visW / 2, halfVisH = visH / 2;

    const srcX = (camX - halfVisW);
    const srcY = (camY - halfVisH);
    const srcW = visW;
    const srcH = visH;

    // Draw terrain
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(terrainCanvas, srcX, srcY, srcW, srcH, 0, 0, canvas.width, canvas.height);

    const CELL_SIZE = canvas.width / visW;
    const scale = CELL_SIZE;

    function worldToScreen(wx, wy) {
        const screenX = (wx - srcX) * scale;
        const screenY = (wy - srcY) * scale;
        return [screenX, screenY];
    }

    drawAgents(ctx, latestAgents, worldToScreen, scale);
    drawMinimap(latestAgents);
    drawZoomIndicator();
}

function drawAgents(ctx, agents, worldToScreen, scale) {
    const frameIdx = Math.floor(Date.now() / 150) % 4;

    agents.forEach(agent => {
        const [x, y] = worldToScreen(agent.x + 0.5, agent.y + 0.5);
        const species = agent.species;
        const color = AGENT_COLORS[species] || '#ffffff';
        const sizeMultiplier = AGENT_SIZES[species] || 1.5;
        const size = sizeMultiplier * scale;

        // Draw trail
        if (agent.trail && agent.trail.length > 0) {
            agent.trail.forEach((pos, i) => {
                const [tx, ty] = worldToScreen(pos[0] + 0.5, pos[1] + 0.5);
                const alpha = (i + 1) / (agent.trail.length + 1) * 0.3;
                ctx.beginPath();
                ctx.arc(tx, ty, scale * 0.2, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.globalAlpha = alpha;
                ctx.fill();
            });
            ctx.globalAlpha = 1.0;
        }

        if (spritesReady && sprites[species]) {
            const img = sprites[species];
            const angle = dirToAngle(agent.direction[0], agent.direction[1]);
            const tint = AGENT_TINTS[species];
            
            ctx.save();
            ctx.translate(x, y);
            
            // If it's a sprite sheet (assuming 4x4 grid)
            if (img.width >= 64 && img.height >= 64) {
                const row = getSpriteRow(agent.direction[0], agent.direction[1]);
                const frameSize = img.width / 4;
                ctx.rotate(0); // Animation handles direction via rows
                ctx.drawImage(img, frameIdx * frameSize, row * frameSize, frameSize, frameSize, -size/2, -size/2, size, size);
            } else {
                ctx.rotate(angle * Math.PI / 180);
                ctx.drawImage(img, -size/2, -size/2, size, size);
            }

            if (tint) {
                ctx.globalCompositeOperation = 'multiply';
                ctx.fillStyle = tint;
                ctx.fillRect(-size/2, -size/2, size, size);
                ctx.globalCompositeOperation = 'source-over';
            }
            ctx.restore();
        } else {
            // Fallback
            ctx.beginPath();
            ctx.arc(x, y, size/3, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        }

        // Energy bar
        const barW = size * 0.8;
        const barH = 3;
        const ratio = agent.energy / agent.max_energy;
        const barColor = ratio > 0.6 ? '#22c55e' : ratio > 0.3 ? '#f59e0b' : '#ef4444';
        
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(x - barW/2, y - size/2 - 8, barW, barH);
        ctx.fillStyle = barColor;
        ctx.fillRect(x - barW/2, y - size/2 - 8, barW * Math.min(1, ratio), barH);
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
    
    // Draw viewport on minimap
    const visW = worldW / zoom;
    const visH = worldH / zoom;
    mctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
    mctx.lineWidth = 1;
    mctx.strokeRect((camX - visW/2) * mx, (camY - visH/2) * my, visW * mx, visH * my);
}

function drawZoomIndicator() {
    if (zoom === 1.0) return;
    ctx.fillStyle = 'white';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Zoom: ${zoom.toFixed(1)}x`, 10, canvas.height - 10);
}

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    render();
}

// Event Listeners
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
    const oldZoom = zoom;
    zoom = Math.max(minZoom, Math.min(maxZoom, zoom * zoomFactor));
    
    if (oldZoom !== zoom) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = (e.clientX - rect.left) / canvas.width;
        const mouseY = (e.clientY - rect.top) / canvas.height;
        
        const viewportWorldW = worldW / oldZoom;
        const viewportWorldH = worldH / oldZoom;
        
        const worldX = camX + (mouseX - 0.5) * viewportWorldW;
        const worldY = camY + (mouseY - 0.5) * viewportWorldH;
        
        const newViewportWorldW = worldW / zoom;
        const newViewportWorldH = worldH / zoom;
        
        camX = worldX - (mouseX - 0.5) * newViewportWorldW;
        camY = worldY - (mouseY - 0.5) * newViewportWorldH;
        
        clampCamera();
        render();
    }
}, { passive: false });

canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragCamStartX = camX;
    dragCamStartY = camY;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const visW = worldW / zoom;
    const visH = worldH / zoom;
    const dx = (e.clientX - dragStartX) / (canvas.width / visW);
    const dy = (e.clientY - dragStartY) / (canvas.height / visH);
    camX = dragCamStartX - dx;
    camY = dragCamStartY - dy;
    clampCamera();
    render();
});

canvas.addEventListener('mouseup', () => { isDragging = false; });

window.addEventListener('resize', resize);
preloadSprites();
connect();

// ─── Touch / Pinch-to-Zoom ───────────────────────────────────────────────────
let lastTouchDist = null;


function getTouchDist(touches) {
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;
  return Math.sqrt(dx * dx + dy * dy);
}

canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  if (e.touches.length === 1) {
    isDragging = true;
    dragStartX = e.touches[0].clientX;
    dragStartY = e.touches[0].clientY;
    dragCamStartX = camX;
    dragCamStartY = camY;
    lastTouchDist = null;
  } else if (e.touches.length === 2) {
    isDragging = false;
    lastTouchDist = getTouchDist(e.touches);
  }
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (e.touches.length === 1 && isDragging) {
    const rect = canvas.getBoundingClientRect();
    const dx = (e.touches[0].clientX - dragStartX) / rect.width * canvas.width;
    const dy = (e.touches[0].clientY - dragStartY) / rect.height * canvas.height;
    const worldDx = dx / (canvas.width / (visW * CELL_SIZE));
    const worldDy = dy / (canvas.height / (visH * CELL_SIZE));
    camX = dragCamStartX - worldDx;
    camY = dragCamStartY - worldDy;
    clampCamera();
  } else if (e.touches.length === 2 && lastTouchDist !== null) {
    const newDist = getTouchDist(e.touches);
    const factor = newDist / lastTouchDist;
    const newZoom = Math.max(minZoom, Math.min(4.0, zoom * factor));
    const mid = { x: (e.touches[0].clientX + e.touches[1].clientX) / 2, y: (e.touches[0].clientY + e.touches[1].clientY) / 2 };
    const rect = canvas.getBoundingClientRect();
    const mx = (mid.x - rect.left) / rect.width;
    const my = (mid.y - rect.top) / rect.height;
    const worldX = camX + (mx - 0.5) * visW;
    const worldY = camY + (my - 0.5) * visH;
    zoom = newZoom;
    camX = worldX - (mx - 0.5) * (worldW / zoom);
    camY = worldY - (my - 0.5) * (worldH / zoom);
    clampCamera();
    lastTouchDist = newDist;
  }
}, { passive: false });

canvas.addEventListener('touchend', (e) => {
  if (e.touches.length < 2) lastTouchDist = null;
  if (e.touches.length === 0) isDragging = false;
}, { passive: false });
