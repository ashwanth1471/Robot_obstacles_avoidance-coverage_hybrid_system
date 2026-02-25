# Robot_obstacles_avoidance-coverage_hybrid_system

# 🤖 Autonomous Robot  
## AI-Driven Coverage Planning and obstacles avoidance with Deep Learning, Evolutionary Optimization & Risk-Aware Search

---

# 📌 Abstract

This project presents a **server-intensive, database-driven, AI-powered coverage planning system** for an autonomous robot operating in obstacle-rich environments.

The system integrates:

- 🧠 Convolutional Neural Network (CNN) for spatial risk modeling  
- 📐 Boustrophedon Cellular Decomposition for structured coverage  
- 🧬 NSGA-II Multi-objective Evolutionary Optimization  
- 🗺 Risk-aware A* Search for fallback path generation  
- ⚡ FastAPI backend for server-side orchestration  
- 🗄 SQLite (WAL mode) for persistent trajectory storage  
- 🎨 Real-time browser-based visualization  
- 📊 Pareto front analytics  
- 🎥 Simulation recording engine  

This system optimizes coverage trajectories with respect to:

- Path length
- Turn count
- Traversal risk
- Coverage completeness

The architecture bridges **robotics, machine learning, evolutionary computation, backend engineering, and visualization systems**.

---

# 🏗 1. System Architecture

## 1.1 High-Level Pipeline

```
User Input (UI)
       ↓
FastAPI Backend
       ↓
Grid Construction
       ↓
CNN Risk Inference
       ↓
Boustrophedon Decomposition
       ↓
NSGA-II Optimization
       ↓
Validation Layer
       ↓
Risk-Aware A* (Fallback)
       ↓
SQLite Persistence
       ↓
Visualization + Pareto Analytics
```

---

## 1.2 Architectural Layers

### 🔹 Perception Layer
- CNN learns spatial risk from obstacle geometry.

### 🔹 Decomposition Layer
- Converts free space into monotonic sweepable cells.

### 🔹 Optimization Layer
- Evolves sweep order using NSGA-II.

### 🔹 Search Layer
- Connects sweeps using risk-aware A*.

### 🔹 Persistence Layer
- SQLite WAL-mode storage.

### 🔹 Visualization Layer
- Real-time animation and Pareto analytics.

---

# 🧠 2. CNN-Based Spatial Risk Modeling

## 2.1 Problem Formulation

We model spatial traversal risk:

\[
R(x,y) = f(\text{obstacle proximity}, \text{boundary proximity})
\]

The synthetic ground-truth risk is defined as:

\[
R = 3.5 \cdot e^{-\frac{d_{obs}}{1.8}} + 0.3 \cdot e^{-\frac{d_{boundary}}{10}}
\]

Where:

- \( d_{obs} \) = Euclidean distance from obstacles  
- \( d_{boundary} \) = distance from workspace boundary  

Risk is normalized to \([0,1]\).

---

## 2.2 Input Tensor (3 Channels)

| Channel | Meaning |
|----------|----------|
| 0 | Binary obstacle grid |
| 1 | Obstacle distance transform |
| 2 | Boundary distance |

\[
X \in \mathbb{R}^{3 \times H \times W}
\]

---

## 2.3 CNN Architecture (U-Net Inspired)

Encoder:
- Conv(3→32)
- Conv(32→32)
- MaxPool
- Conv(32→64)
- Conv(64→64)
- MaxPool

Bottleneck:
- Conv(64→128)
- Conv(128→128)

Decoder:
- Transposed Conv
- Skip connections
- Final Conv(32→1)
- Sigmoid activation

Output:

\[
\hat{R}(x,y) \in [0,1]
\]

---

## 2.4 Loss Function

Huber (Smooth L1):

\[
L =
\begin{cases}
0.5(x-y)^2 & |x-y| < 1 \\
|x-y| - 0.5 & \text{otherwise}
\end{cases}
\]

Why:
- Preserves sharp obstacle gradients
- Stabilizes large deviations
- More robust than MSE  [Experimented with mse too ,most of the cases model couldnt understand obstacles if its too thin ie couldnt figure out the obstacles properly (couldnt find out sharp edges or differentiate when multiple obstacles are close to each other )]

---

## 2.5 Training Strategy

- Random rectangles (3–8 per grid)
- Random vertical/horizontal walls
- Binary dilation for inflation
- Variable grid sizes: 64, 96, 128
- Batch size: 16
- Epochs: 2000
- Adam optimizer (lr=1e-3)
- Weight decay = 1e-4

---

# 📐 3. Boustrophedon Cellular Decomposition

## 3.1 Objective

Convert free space into monotonic cells to enable structured vertical sweeps.

## 3.2 Algorithm

1. Column-by-column sweep
2. Extract free row intervals
3. Detect interval overlaps
4. Merge/split active regions
5. Generate sweep skeleton

Each gene:

```python
{
  "col": c,
  "start": r1,
  "end": r2,
  "dir": ±1,
  "vertical": True
}
```

Guarantees complete structured coverage.

---

# 🧬 4. NSGA-II Multi-Objective Optimization

## 4.1 Objective Vector

\[
f(x) =
\begin{bmatrix}
L \\
T \\
R \\
1 - C
\end{bmatrix}
\]

Where:

- \(L\) = path length + jump penalty
- \(T\) = turns + curvature penalty
- \(R\) = average CNN risk
- \(C\) = coverage ratio

---

## 4.2 Penalty Formulations

### Jump Penalty
\[
L = L + 2 \cdot (1.5 \cdot jump)
\]

### Turn Detection
If movement vector changes:
\[
T += 1
\]

### Curvature Penalty
If reversal detected:
\[
curvature += 2
\]

### Direction Continuity Penalty
If sweep direction changes:
\[
dir\_penalty += 1
\]

---

## 4.3 Evolution Strategy

- Population = 30
- Generations = 60
- Mutation:
  - Direction flip (40%)
  - Swap segments (40%)
  - Reverse subsequence (30%)

Uses:
- Fast Non-Dominated Sorting
- Crowding Distance

Produces Pareto front.

---

# 🗺 5. Risk-Aware A* Search

If NSGA-II fails validation:

Fallback to A*.

Cost:

\[
g(n) = 1 + 25 \cdot R(n)^2
\]

Heuristic:

\[
h(n) = |x_1 - x_2| + |y_1 - y_2|
\]

Also uses risk-biased next-cell selection:

\[
argmin(|dx| + |dy| + 20 \cdot R)
\]

---

# 🗄 6. Database Layer

SQLite with WAL mode:

```sql
CREATE TABLE trajectory (
    plan_id TEXT,
    row INTEGER,
    col INTEGER
);
```

Features:

- Concurrent-safe writes
- Retry loop on lock
- UUID plan isolation
- Hybrid memory store (Pareto + costmap)

---

# ⚡ 7. Backend Execution Flow

`POST /plan`:

1. Build grid
2. CNN inference
3. Decomposition
4. NSGA-II
5. Validation
6. Fallback (if needed)
7. Store trajectory
8. Store Pareto front
9. Return plan_id

---

# 🎨 8. Visualization Engine

- Canvas-based simulation [simple for now]
- Smooth interpolation
- Steering smoothing:
\[
\theta = \theta + 0.2 \cdot \Delta \theta
\]
- Risk heatmap
- Pareto color normalization
- WebM video recording

---

# 📊 9. Design Justification

Why CNN?
- Learns spatial smoothness
- Encodes obstacle influence field

Why NSGA-II?
- Coverage planning has conflicting objectives
- Produces tradeoff frontier

Why A* fallback?
- Guarantees feasibility

Why SQLite WAL?
- Safe concurrent writes

---

# 🚀 10. Future Roadmap

## Future Roadmap (Production-Grade Evolution)

This project is currently structured as a high-performance research prototype.  
The next phase will transform it into a scalable, deployable robotics intelligence platform.

---

## 🗄 1. Database Migration: SQLite → Production Database

### Current
- SQLite (WAL mode)
- Local file-based storage
- Suitable for research & single-instance execution

### Future Upgrade

Replace SQLite with a production-grade database:

- PostgreSQL (recommended)
- Distributed NoSQL (for large-scale logging)
- Cloud-managed DB (AWS RDS / GCP Cloud SQL)

### Why Replace SQLite?

SQLite is excellent for:
- Prototyping
- Local simulation
- Low-concurrency systems

But production robotics systems require:

- High concurrency
- Distributed access
- Horizontal scalability
- Advanced indexing
- Analytics support
- Real-time telemetry storage

### Planned Architecture

```
FastAPI Backend
      ↓
Async ORM (SQLAlchemy )
      ↓
PostgreSQL Cluster
      ↓
Trajectory + Pareto + Telemetry Storage
```

### Additional Tables (Planned)

- `trajectory`
- `pareto_solutions`
- `costmaps`
- `execution_metrics`
- `robot_state_logs`
- `training_data_buffer`

This enables:

- Historical trajectory replay
- Performance benchmarking
- Model retraining datasets
- Real robot telemetry storage
- Multi-robot tracking

---

## 🎨 2. Dedicated UI Application (Beyond HTML Canvas)

### Current UI
- Basic HTML + Canvas
- Embedded inside FastAPI
- Lightweight simulation interface

### Future Upgrade

Develop a dedicated frontend application:

- React / Next.js
- Vue.js
- or Electron desktop app
- Optional 3D interface (Three.js)

---

## 🔹 Planned UI Features

### 📊 Advanced Analytics Dashboard
- Real-time Pareto surface visualization
- 3D tradeoff plots
- Objective weight sliders
- Performance metrics

### 🧠 AI Insight Panel

- ## LLM-Driven Pareto Interpretation
    Introduce LLM reasoning layer:

    - Explain tradeoffs between solutions
    - Suggest best solution based on mission goal
    - Natural language interface:
        - "Minimize time"
        - "Maximize safety"
        - "Balance turns and risk"
    - LLM-generated explanation of Pareto tradeoffs
    - Risk heatmap breakdown
    - Coverage diagnostics

    LLM acts as:
        > Strategic reasoning layer above optimization.

---

### 🤖 Robot Telemetry Panel
- Velocity tracking
- Turn rate visualization
- Energy consumption estimate
- Sensor simulation overlays

### 📈 Model Training Monitor
- Live CNN training loss graphs
- Online learning statistics
- Dataset accumulation tracking

### 🗺 3D Simulation View
- URDF-based robot visualization
- Physics-based simulation
- Real-time collision rendering
- Multi-robot coordination view

---

## 🧠 3. Continual CNN Training with Real Data

Future upgrade:

- Store executed trajectories
- Store spatial cost feedback
- Create online training dataset
- Enable continual learning

### Planned Loop

```
Robot Execution
      ↓
Telemetry Logging
      ↓
Database Storage
      ↓
Dataset Buffer
      ↓
Incremental CNN Training
      ↓
Improved Risk Model
```

This introduces **real spatial reasoning** instead of synthetic-only training.

---

## 🤖 4. Reinforcement Learning Integration (URDF-Based)

Future expansion:

- Convert robot model to URDF
- Integrate with Gazebo / PyBullet
- Apply PPO / SAC / TD3
- Learn coverage behavior end-to-end

### Hybrid Planning Strategy

CNN + NSGA-II + RL

- CNN → risk modeling
- NSGA-II → structured sweep
- RL → dynamic motion refinement

Long-term goal:
Fully autonomous adaptive coverage planning.

---

## 🌐 6. Multi-Robot Distributed System with Microservices Architecture

Future system:

```
Robot Agents
      ↓
Central Planner API
      ↓
Distributed Database
      ↓
Global Optimization Layer
```

Features:

- Task allocation
- Conflict avoidance
- Cooperative coverage
- Swarm-level Pareto optimization

Long-term transformation:

Split monolith into:

- Risk Inference Service
- Optimization Service
- Planning Service
- Telemetry Service
- Training Service
- UI Frontend App

Benefits:

- Independent scaling
- GPU inference servers
- Distributed deployment
- Cloud-native robotics platform

---

# 🏁 Long-Term Vision

This system will evolve from:

> Research-grade coverage planner  

Into:

> Scalable autonomous robotic intelligence platform  
> With learning, reasoning, optimization, and distributed execution.

---

# 🎯 Target End State

- Real robot execution
- Online learning
- Multi-objective reasoning
- LLM-based mission planning
- Distributed multi-agent coordination
- Production-grade backend
- Dedicated interactive control interface


# 🧪 Complexity Analysis

CNN inference:
\[
O(HW)
\]

NSGA-II:
\[
O(G \cdot P^2)
\]

A*:
\[
O(N \log N)
\]

---

More to do ......
