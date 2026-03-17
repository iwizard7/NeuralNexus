from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import numpy as np
import traceback
from simulation import SwarmEnvironment, RobotAgent, Resource, Nest
from brain import NeuralDirector

# Custom JSON encoder to handle NumPy types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app = FastAPI()
director = NeuralDirector(model_name="phi3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
env = SwarmEnvironment(width=100, height=100) # Используем расширенные дефолты

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS: NEW CONNECTION")
    try:
        while True:
            # Step the environment
            try:
                env.step()
            except Exception as e:
                print(f"SIM ERROR: {e}")
                traceback.print_exc()
                break

            # Collect data for frontend safely
            agents_data = []
            for agent in list(env.schedule.agents):
                try:
                    if not hasattr(agent, "pos") or agent.pos is None:
                        continue

                    if isinstance(agent, RobotAgent):
                        data = {
                            "id": agent.unique_id,
                            "pos": [float(agent.pos[0]), float(agent.pos[1])],
                            "type": "robot",
                            "role": agent.role,
                            "shouting": agent.is_shouting,
                            "energy": float(agent.energy),
                            "carrying": float(agent.carrying_resource),
                            "capacity": float(agent.dna["capacity"]),
                            "state": agent.state,
                            "speed": float(agent.dna["speed"])
                        }
                    elif isinstance(agent, Resource):
                        data = {
                            "id": agent.unique_id,
                            "pos": [float(agent.pos[0]), float(agent.pos[1])],
                            "type": "resource"
                        }
                    elif isinstance(agent, Nest):
                        data = {
                            "id": agent.unique_id,
                            "pos": [float(agent.pos[0]), float(agent.pos[1])],
                            "type": "nest",
                            "level": int(agent.level),
                            "progress": float(agent.resources_stored / (agent.level * 100))
                        }
                    elif agent.pos is not None: # EnemyAgent
                        data = {
                            "id": agent.unique_id,
                            "pos": [float(agent.pos[0]), float(agent.pos[1])],
                            "type": "enemy"
                        }
                    else:
                        continue
                    agents_data.append(data)
                except Exception as e:
                    pass
            
            # Global Directive logic
            if env.schedule.steps % 50 == 0:
                robots_only = [a for a in env.schedule.agents if isinstance(a, RobotAgent)]
                try:
                    stats = {
                        "robot_count": int(env.robot_count),
                        "resource_count": len([a for a in env.schedule.agents if isinstance(a, Resource)]),
                        "avg_energy": float(sum([a.energy for a in robots_only]) / max(1, len(robots_only)))
                    }
                    directive = director.analyze_and_direct(stats)
                except Exception as e:
                    print(f"STATS ERROR: {e}")
                    directive = director.latest_directive
            else:
                directive = director.latest_directive

            # Send data
            try:
                # Calculate additional stats
                nested_stats = {
                    "step": int(env.schedule.steps),
                    "age": int(env.colony_age),
                    "total_collected": float(env.nest.total_collected),
                    "reproductions": int(env.total_reproductions),
                    "role_counts": env.role_counts,
                    "enemies": len([a for a in env.schedule.agents if not isinstance(a, (RobotAgent, Resource, Nest))])
                }

                msg = json.dumps({
                    "agents": agents_data,
                    "directive": directive,
                    "step": int(env.schedule.steps),
                    "stats": nested_stats
                }, cls=NpEncoder)
                await websocket.send_text(msg)
            except Exception as e:
                print(f"SEND ERROR: {e}")
                break

            await asyncio.sleep(0.1)  # 10 FPS
    except asyncio.CancelledError:
        print("WS: Connection cancelled")
    except Exception as e:
        print(f"WS CRITICAL ERROR: {e}")
    finally:
        print("WS: CONNECTION CLOSED")
        try:
            await websocket.close()
        except:
            pass # Already closed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
