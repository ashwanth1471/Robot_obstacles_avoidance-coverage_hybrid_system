# main.py
from fastapi import FastAPI
from fastapi.responses import FileResponse , HTMLResponse
from pydantic import BaseModel
import numpy as np
import sqlite3, uuid, random
from scipy.ndimage import distance_transform_edt , binary_dilation
import torch
import torch.nn as nn

from typing import List
import time



# -------data type----
class ParetoPoint(BaseModel):
    length: float
    turns: float
    risk: float
    coverage: float


class ParetoResponse(BaseModel):
    pareto: List[ParetoPoint]


# ---------------- DB ----------------


DB_PATH = "planner.db"


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL;")  #  enable WAL
    db.execute("PRAGMA synchronous=NORMAL;")
    db.execute("""
               CREATE TABLE IF NOT EXISTS trajectory
               (
                   plan_id
                   TEXT,
                   row
                   INTEGER,
                   col
                   INTEGER
               )
               """)
    db.commit()
    db.close()


def get_db():
    return sqlite3.connect(
        DB_PATH,
        timeout=30,
        check_same_thread=False
    )


# initialize DB
init_db()

PARETO_STORE = {}
COSTMAP_STORE = {}


# ---------------- MODELS ----------------
class Obstacle(BaseModel):
    x: float
    y: float
    w: float
    h: float


class PlanRequest(BaseModel):
    width: float
    height: float
    cell_size: float
    obstacles: list[Obstacle]


# ---------------- CNN ----------------
class CostCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- Encoder -----[User's wish what to use , I have used u-net style architecture]


        # ----- Bottleneck -----


        # ----- Decoder -----

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.out(d1)


def cnn_cost(grid):
    from scipy.ndimage import distance_transform_edt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = CostCNN()
    cnn_model.load_state_dict(torch.load("cnn_cost.pth"))
    cnn_model.eval().to(device)

    size = grid.shape[0]

    obs_dist = distance_transform_edt(1 - grid)

    yy, xx = np.meshgrid(
        np.arange(size),
        np.arange(size),
        indexing="ij"
    )

    boundary_dist = np.minimum.reduce([
        yy,
        xx,
        size - 1 - yy,
        size - 1 - xx
    ])

    obs_dist_norm = obs_dist / (obs_dist.max() + 1e-6)
    boundary_dist_norm = boundary_dist / (boundary_dist.max() + 1e-6)

    input_tensor = np.stack(
        [grid.astype(np.float32),
         obs_dist_norm.astype(np.float32),
         boundary_dist_norm.astype(np.float32)],
        axis=0
    )

    x = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        cost = cnn_model(x).squeeze().numpy()

    cost[grid == 1] = 1.0

    return cost


# ---------------- GRID ----------------
def build_grid(req):
    rows = int(req.height / req.cell_size)
    cols = int(req.width / req.cell_size)

    grid = np.zeros((rows, cols), dtype=np.uint8)

    for o in req.obstacles:
        r1 = max(0, int(o.y / req.cell_size))
        r2 = min(rows, int((o.y + o.h) / req.cell_size))
        c1 = max(0, int(o.x / req.cell_size))
        c2 = min(cols, int((o.x + o.w) / req.cell_size))

        grid[r1:r2, c1:c2] = 1

    # inflate obstacles by 1 cell
    grid = binary_dilation(grid, iterations=1).astype(np.uint8)

    return grid


# ---------------- BOUSTROPHEDON ----------------
def boustrophedon_cells(grid):
    rows, cols = grid.shape
    cells = []
    active_cells = {}
    next_cell_id = 0

    for c in range(cols):

        # Step 1: extract free intervals in this column
        intervals = []
        r = 0
        while r < rows:
            while r < rows and grid[r, c] == 1:
                r += 1
            if r >= rows:
                break
            start = r
            while r < rows and grid[r, c] == 0:
                r += 1
            end = r - 1
            intervals.append((start, end))

        # Step 2: match intervals with previous active cells
        new_active = {}

        for iv in intervals:
            matched_ids = []

            for cell_id, cell in active_cells.items():
                prev_iv = cell["interval"]

                # Proper overlap check
                if iv[0] <= prev_iv[1] and iv[1] >= prev_iv[0]:
                    matched_ids.append(cell_id)

            if len(matched_ids) == 1:
                cid = matched_ids[0]

                prev_iv = active_cells[cid]["interval"]
                new_interval = (
                    min(prev_iv[0], iv[0]),
                    max(prev_iv[1], iv[1])
                )

                active_cells[cid]["interval"] = new_interval
                active_cells[cid]["cols"].append(c)
                new_active[cid] = active_cells[cid]


            elif len(matched_ids) > 1:
                cid = matched_ids[0]

                # Merge columns
                for mid in matched_ids[1:]:
                    active_cells[cid]["cols"].extend(active_cells[mid]["cols"])
                    del active_cells[mid]

                # Expand interval properly
                prev_iv = active_cells[cid]["interval"]
                new_interval = (
                    min(prev_iv[0], iv[0]),
                    max(prev_iv[1], iv[1])
                )

                active_cells[cid]["interval"] = new_interval
                active_cells[cid]["cols"].append(c)
                new_active[cid] = active_cells[cid]


            else:
                # Split or new region
                new_active[next_cell_id] = {
                    "interval": iv,
                    "cols": [c]
                }
                next_cell_id += 1

        # Step 3: close cells that did not continue
        for cid in list(active_cells.keys()):
            if cid not in new_active:
                cells.append(active_cells[cid])

        active_cells = new_active

    # Close remaining active cells
    cells.extend(active_cells.values())

    return cells


def cells_to_skeleton(cells, grid):
    skel = []

    for cell in cells:
        cols = sorted(cell["cols"])
        r1, r2 = cell["interval"]

        direction = 1

        for c in cols:
            rows = []

            # extract actual free rows in this column
            r = r1
            while r <= r2:
                while r <= r2 and grid[r, c] == 1:
                    r += 1
                if r > r2:
                    break
                start = r
                while r <= r2 and grid[r, c] == 0:
                    r += 1
                end = r - 1
                rows.append((start, end))

            for (start, end) in rows:
                skel.append({
                    "col": c,
                    "start": start if direction == 1 else end,
                    "end": end if direction == 1 else start,
                    "dir": direction,
                    "vertical": True
                })

                direction *= -1

    return skel


# ---------------- NSGA-II ----------------
class Individual:
    def __init__(self, genes):
        self.genes = [g.copy() for g in genes]
        self.objectives = None  # multi-objective vector
        self.rank = None
        self.crowding = 0.0


# evaluate
def evaluate(ind, grid, cost):
    length = 0
    turns = 0
    cnn_sum = 0

    visited = set()

    prev_cell = None
    prev_move_vec = None

    jump_penalty = 0
    curvature_penalty = 0
    dir_penalty = 0
    last_dir = None

    for g in ind.genes:

        if not g.get("vertical"):
            continue

        col = g["col"]

        sweep_cells = [
            (r, col)
            for r in range(g["start"], g["end"] + g["dir"], g["dir"])
            if 0 <= r < grid.shape[0]
        ]

        if not sweep_cells:
            continue

        # --- connector jump penalty ---
        if prev_cell is not None:
            jump = abs(prev_cell[0] - sweep_cells[0][0]) + \
                   abs(prev_cell[1] - sweep_cells[0][1])
            jump_penalty += jump * 1.5

        for cell in sweep_cells:

            if grid[cell] == 1:
                ind.objectives = {
                    "length": 1e9,
                    "turns": 1e9,
                    "risk": 1e9,
                    "coverage": 1e9
                }
                return

            visited.add(cell)
            length += 1
            cnn_sum += cost[cell]

            if prev_cell is not None:
                move_vec = (
                    cell[0] - prev_cell[0],
                    cell[1] - prev_cell[1]
                )

                #  real turns calculation
                if prev_move_vec is not None:

                    # turn detection
                    if move_vec != prev_move_vec:
                        turns += 1
                        curvature_penalty += 1

                    #  penalize reversal (zig-zag)
                    dot = prev_move_vec[0] * move_vec[0] + prev_move_vec[1] * move_vec[1]
                    if dot < 0:
                        curvature_penalty += 2

                prev_move_vec = move_vec

            prev_cell = cell

        # direction continuity penalty
        if last_dir is not None and last_dir != g["dir"]:
            dir_penalty += 1

        last_dir = g["dir"]

    # --- coverage objective ---
    total_free = np.sum(grid == 0)
    coverage_ratio = len(visited) / (total_free + 1e-6)

    # --- average CNN risk ---
    avg_risk = cnn_sum / (length + 1e-6)

    ind.objectives = {
        "length": length + jump_penalty * 2,
        "turns": turns + dir_penalty + curvature_penalty,
        "risk": float(avg_risk),
        "coverage": 1 - coverage_ratio  # minimize uncovered
    }


def mutate(ind):
    # flip direction
    if random.random() < 0.4:
        g = random.choice(ind.genes)
        g["dir"] *= -1

    # swap two segments
    if random.random() < 0.4:
        i, j = random.sample(range(len(ind.genes)), 2)
        ind.genes[i], ind.genes[j] = ind.genes[j], ind.genes[i]

    # reverse sub-sequence
    if random.random() < 0.3:
        i, j = sorted(random.sample(range(len(ind.genes)), 2))
        ind.genes[i:j] = list(reversed(ind.genes[i:j]))


def dominates(a, b):
    keys = ["length", "turns", "risk", "coverage"]

    better_or_equal = all(a[k] <= b[k] for k in keys)
    strictly_better = any(a[k] < b[k] for k in keys)

    return better_or_equal and strictly_better


def fast_non_dominated_sort(pop):
    fronts = [[]]
    S = {}
    n = {}

    for p in pop:
        if p.objectives is None:
            continue
        S[p] = []
        n[p] = 0
        for q in pop:
            if q.objectives is None:
                continue
            if dominates(p.objectives, q.objectives):
                S[p].append(q)
            elif dominates(q.objectives, p.objectives):
                n[p] += 1

        if n[p] == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def crowding_distance(front):
    if not front:
        return

    for p in front:
        p.crowding = 0.0

    keys = front[0].objectives.keys()

    for k in keys:
        front.sort(key=lambda x: x.objectives[k])
        front[0].crowding = front[-1].crowding = float("inf")

        min_v = front[0].objectives[k]
        max_v = front[-1].objectives[k]
        if max_v == min_v:
            continue

        for i in range(1, len(front) - 1):
            front[i].crowding += (
                                         front[i + 1].objectives[k] -
                                         front[i - 1].objectives[k]
                                 ) / (max_v - min_v)


def select_nsga2(pop, pop_size):
    if not pop:
        return []

    fronts = fast_non_dominated_sort(pop)
    new_pop = []

    for front in fronts:
        crowding_distance(front)
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(front)
        else:
            front.sort(key=lambda x: x.crowding, reverse=True)
            new_pop.extend(front[:pop_size - len(new_pop)])
            break

    return new_pop


def nsga2(skel, grid, cost):
    pop = []


    # As per user's wish can initialise population , sort it , evaluate it and append it to the population ,

    return pop


# ------a*----------
import heapq


def astar(grid, cost, start, goal):
    rows, cols = grid.shape

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq = [(0, start)]
    came = {start: None}
    g = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            while cur:
                path.append(cur)
            return path[::-1]

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:

                risk_weight = 25
                ng = g[cur] + 1 + (cost[nr, nc] ** 2) * risk_weight


    return []


# ---------------- VALIDATOR ----------------
def validate(ind, grid):
    visited = set()

    for g in ind.genes:

        if g.get("vertical"):
            for r in range(g["start"], g["end"] + g["dir"], g["dir"]):
                if grid[r, g["col"]] == 1:
                    return False
                visited.add((r, g["col"]))

        else:
            for c in range(g["start"], g["end"] + g["dir"], g["dir"]):
                if grid[g["row"], c] == 1:
                    return False
                visited.add((g["row"], c))

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 0 and (r, c) not in visited:
                return False

    return True


# ---------------- FASTAPI ----------------
app = FastAPI()


@app.post("/plan")
def plan(req: PlanRequest):
    import time
    start = time.time()

    grid = build_grid(req)

    cells = boustrophedon_cells(grid)
    skel = cells_to_skeleton(cells, grid)

    cost = cnn_cost(grid)

    pareto = nsga2(skel, grid, cost)

    plan_id = str(uuid.uuid4())
    COSTMAP_STORE[plan_id] = cost

    fronts = fast_non_dominated_sort(pareto)

    if fronts and fronts[0]:
        PARETO_STORE[plan_id] = [
            p.objectives for p in fronts[0]
        ]
    else:
        # fallback: store best 5
        sorted_pop = sorted(
            pareto,
            key=lambda p: p.objectives["length"]
        )
        PARETO_STORE[plan_id] = [
            p.objectives for p in sorted_pop[:5]
        ]

    valid = [p for p in pareto if validate(p, grid)]

    best = None
    use_astar_only = False

    if valid:
        fronts = fast_non_dominated_sort(valid)
        best = min(
            fronts[0],
            key=lambda p: (
                    p.objectives["length"] +
                    p.objectives["turns"] * 2 +
                    p.objectives["risk"] * 50 +
                    p.objectives["coverage"] * 100
                # weights can be modified to give priority to the features ( coverage vs obstacle avoidance or balanced )
                # based on my observation this worked fine

            )
        )

    else:
        # 🚨 NSGA-II failed → fallback to A*
        use_astar_only = True

    if use_astar_only:
        free_cells = {
            (r, c)
            for r in range(grid.shape[0])
            for c in range(grid.shape[1])
            if grid[r, c] == 0
        }

        full_path = []

        if not free_cells:
            return {"plan_id": plan_id}

        cur_cell = free_cells.pop()
        full_path.append(cur_cell)

        while free_cells:

            # choose next cell by risk-aware nearest heuristic
            next_cell = min(
                free_cells,
                key=lambda x: abs(x[0] - cur_cell[0]) +
                              abs(x[1] - cur_cell[1]) +
                              cost[x] * 20
            )

            path = astar(grid, cost, cur_cell, next_cell)

            if not path:
                free_cells.remove(next_cell)
                continue

            full_path.extend(path[1:])
            cur_cell = next_cell
            free_cells.remove(next_cell)

        # store trajectory

        for _ in range(5):
            try:
                db = get_db()
                cur = db.cursor()

                cur.execute("DELETE FROM trajectory WHERE plan_id=?", (plan_id,))

                for r, c in full_path:
                    cur.execute(
                        "INSERT INTO trajectory VALUES (?,?,?)",
                        (plan_id, int(r), int(c))
                    )

                db.commit()
                cur.close()
                db.close()
                break

            except sqlite3.OperationalError:
                time.sleep(0.1)

        return {"plan_id": plan_id}

    db = get_db()
    cur = db.cursor()

    #  clear old trajectory for this plan_id
    cur.execute("DELETE FROM trajectory WHERE plan_id=?", (plan_id,))

    full_path = []
    prev = None

    for g in best.genes:

        if not g.get("vertical"):
            continue

        col = g["col"]

        sweep = [
            (r, col)
            for r in range(g["start"], g["end"] + g["dir"], g["dir"])
            if 0 <= r < grid.shape[0]
        ]

        if not sweep:
            continue

        if prev is not None:
            conn = astar(grid, cost, prev, sweep[0])
            if not conn:
                continue
            full_path.extend(conn[:-1])

        full_path.extend(sweep)
        prev = sweep[-1]

    for r, c in full_path:
        cur.execute(
            "INSERT INTO trajectory VALUES (?,?,?)",
            (plan_id, int(r), int(c))
        )

    db.commit()
    cur.close()
    db.close()

    print("Plan time:", time.time() - start)
    return {"plan_id": plan_id}


@app.get("/trajectory/{plan_id}")
def trajectory(plan_id: str):
    db = get_db()
    cur = db.cursor()

    cur.execute(
        "SELECT row, col FROM trajectory WHERE plan_id=?",
        (plan_id,)
    )
    rows = cur.fetchall()

    cur.close()
    db.close()

    return {
        "trajectory": [{"row": int(r), "col": int(c)} for r, c in rows]
    }


@app.get("/pareto/{plan_id}", response_model=ParetoResponse)
def pareto(plan_id: str):
    raw_data = PARETO_STORE.get(plan_id, [])

    clean_data = [
        ParetoPoint(
            length=float(p["length"]),
            turns=float(p["turns"]),
            risk=float(p["risk"]),
            coverage=float(p["coverage"])
        )
        for p in raw_data
    ]

    return ParetoResponse(pareto=clean_data)


@app.get("/ui")
def ui():
    return FileResponse("ui.html")


@app.get("/costmap/{plan_id}")
def costmap(plan_id: str):
    cost = COSTMAP_STORE.get(plan_id)
    if cost is None:
        return {"error": "costmap not found"}
    return cost.tolist()



