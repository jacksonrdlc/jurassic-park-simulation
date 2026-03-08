import asyncio
import json
import numpy as np
import threading
from typing import List, Dict, Any, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn
import time

# Simulation imports
from my_first_island import IslandModel, GrassAgent, HerbivoreAgent, CarnivoreAgent, TRex, Velociraptor, Triceratops, Gallimimus
from terrain_generator import generate_island, TerrainType, TERRAIN_COLORS

app = FastAPI()

# Configuration
WIDTH = 250
HEIGHT = 150
TICK_INTERVALS = {
    1: 0.2,
    2: 0.1,
    3: 0.05,
    4: 0.025,
    5: 0.01
}

# State
class SimState:
    def __init__(self):
        self.terrain_map, _ = generate_island(WIDTH, HEIGHT, seed=42)
        self.model = IslandModel(width=WIDTH, height=HEIGHT, terrain_map=self.terrain_map)
        self.paused = False
        self.tick_speed = 3
        self.active_connections: Set[WebSocket] = set()
        self.last_event_idx = 0
        self.lock = threading.Lock()

    def get_init_data(self):
        terrain_colors_hex = {
            str(t.value): f"#{r:02x}{g:02x}{b:02x}"
            for t, (r, g, b) in TERRAIN_COLORS.items()
        }
        return {
            "type": "init",
            "terrain": self.terrain_map.tolist(),
            "width": WIDTH,
            "height": HEIGHT,
            "terrain_colors": terrain_colors_hex
        }

    def get_tick_data(self):
        with self.lock:
            agents_data = []
            for agent in self.model.agents:
                if isinstance(agent, GrassAgent):
                    continue
                
                species = "herbivore"
                if isinstance(agent, TRex): species = "trex"
                elif isinstance(agent, Velociraptor): species = "velociraptor"
                elif isinstance(agent, Triceratops): species = "triceratops"
                elif isinstance(agent, Gallimimus): species = "gallimimus"
                elif isinstance(agent, CarnivoreAgent): species = "carnivore"
                
                max_energy = 150
                if isinstance(agent, CarnivoreAgent): max_energy = 250
                if hasattr(agent, 'reproduce_threshold'): max_energy = agent.reproduce_threshold
                
                agents_data.append({
                    "id": agent.unique_id,
                    "x": agent.pos[0],
                    "y": agent.pos[1],
                    "species": species,
                    "energy": agent.energy,
                    "max_energy": max_energy,
                    "direction": getattr(agent, 'direction', [1, 0]),
                    "is_moving": getattr(agent, 'is_moving', False),
                    "trail": getattr(agent, 'movement_history', [])
                })

            new_events = self.model.event_log[self.last_event_idx:]
            self.last_event_idx = len(self.model.event_log)

            h_count = len([a for a in self.model.agents if isinstance(a, HerbivoreAgent)])
            c_count = len([a for a in self.model.agents if isinstance(a, CarnivoreAgent)])
            g_count = len([a for a in self.model.agents if isinstance(a, GrassAgent) and a.energy > 0])

            return {
                "type": "tick",
                "step": self.model.steps,
                "paused": self.paused,
                "agents": agents_data,
                "population": {
                    "herbivores": h_count,
                    "carnivores": c_count,
                    "grass": g_count
                },
                "events": new_events,
                "temperature": getattr(self.model, 'temperature', 25.0),
                "rainfall": getattr(self.model, 'rainfall', 100.0),
                "time_of_day": getattr(self.model, 'time_of_day', 0.5)
            }

    def reset(self):
        with self.lock:
            self.model = IslandModel(width=WIDTH, height=HEIGHT, terrain_map=self.terrain_map)
            self.last_event_idx = 0

sim = SimState()

async def broadcast_tick():
    while True:
        if not sim.paused:
            with sim.lock:
                sim.model.step()
            
            data = sim.get_tick_data()
            if sim.active_connections:
                message = json.dumps(data)
                # Broadcast to all
                dead_connections = set()
                for ws in sim.active_connections:
                    try:
                        await ws.send_text(message)
                    except:
                        dead_connections.add(ws)
                sim.active_connections -= dead_connections
        
        await asyncio.sleep(TICK_INTERVALS[sim.tick_speed])

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_tick())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim.active_connections.add(websocket)
    
    # Send init data
    await websocket.send_text(json.dumps(sim.get_init_data()))
    
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "pause":
                sim.paused = True
            elif action == "resume":
                sim.paused = False
            elif action == "pause_toggle":
                sim.paused = not sim.paused
            elif action == "speed":
                speed = data.get("value", 3)
                if speed in TICK_INTERVALS:
                    sim.tick_speed = speed
            elif action == "reset":
                sim.reset()
                
    except WebSocketDisconnect:
        sim.active_connections.remove(websocket)

app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
