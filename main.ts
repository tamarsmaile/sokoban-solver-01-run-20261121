// Sokoban solver: push-optimal A* per room.
//
// One program answers a whole set of 5 levels in a single run (node /bot/main.ts,
// Node 22). Each level is a hallway with 1-4 rooms off it; a box pushed out of a
// room's door can never come back, so the rooms are independent problems joined
// only by the player's walk, and the level's push-optimum P* is the sum of the
// rooms' optima. This solver therefore splits each level into its rooms (the
// same cut the kit's generator uses: the hallway's dead squares) and solves
// every room push-optimally in turn, the other rooms' boxes standing as walls.
//
// Per room it runs A* over pushes: a state is the boxes' squares plus the
// player's reachable region (named by its lowest square). The heuristic is the
// minimum-cost assignment of boxes to goals (Hungarian, by brute force for
// <= 4 boxes; sum of nearest-goal distances for more), using shortest-path
// distances that ignore the other boxes, so it never overestimates and the
// first solution popped is push-optimal. Deadlock pruning: dead squares (a box
// there can never reach a goal) and frozen 2x2 blocks of walls and boxes.
// A transposition table (Map from state key to best pushes) keeps the search
// from revisiting a state. If A* exhausts its work budget for a room, a
// breadth-first search over pushes (the same shape as the kit's reference
// solver) is tried instead; if that gives up too, the instance is left
// unanswered (an empty solution) rather than risking the whole run.
//
// Determinism: no clock, no randomness, no environment reads beyond the input
// and output directories. Every budget is a count of work, so two runs on one
// set write identical bytes. Tie-breaking in the heap is by (f, h, insertion
// order), all fixed.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (counts, never the clock). They bound the whole run far inside
// the 60-second wall budget; see TEST-RESULTS.md for measured usage.
const ASTAR_EXPANSIONS_PER_ROOM = 200_000;
const BFS_STATES_PER_ROOM = 250_000;
const GLOBAL_WORK_BUDGET = 3_000_000;
let globalWork = 0;

const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** All permutations of 0..n-1, in a fixed order (for the Hungarian heuristic). */
function permutations(n) {
  const out = [];
  const a = [];
  for (let i = 0; i < n; i++) a.push(i);
  const gen = (k) => {
    if (k === n) { out.push(a.slice()); return; }
    for (let i = k; i < n; i++) {
      const t = a[k]; a[k] = a[i]; a[i] = t;
      gen(k + 1);
      const t2 = a[k]; a[k] = a[i]; a[i] = t2;
    }
  };
  gen(0);
  return out;
}
const PERMS = [null, permutations(1), permutations(2), permutations(3), permutations(4)];

/**
 * A level in XSB text: its width and height, walls (1 a wall; short rows and
 * squares outside padded as walls, the outermost frame forced to wall), goals,
 * boxes and player.
 */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** The squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(level) {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = goals.slice();
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k][1] * w + DIRS[k][0];
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** Walking distances from `from` over open floor (walls only), -1 where unreachable. */
function bfsDist(wall, w, size, from) {
  const dist = new Int16Array(size).fill(-1);
  const queue = [from];
  dist[from] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    const nd = dist[p] + 1;
    const a = p - w, b = p + w, c = p - 1, d = p + 1;
    if (!wall[a] && dist[a] < 0) { dist[a] = nd; queue.push(a); }
    if (!wall[b] && dist[b] < 0) { dist[b] = nd; queue.push(b); }
    if (!wall[c] && dist[c] < 0) { dist[c] = nd; queue.push(c); }
    if (!wall[d] && dist[d] < 0) { dist[d] = nd; queue.push(d); }
  }
  return dist;
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a
 * box there reaches no goal) out of the floor, and each part left that holds a
 * box is a room, its squares, boxes and goals. Null when some part holds more
 * or fewer boxes than goals, or a box starts in the hallway.
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DIRS[k][1] * w + DIRS[k][0];
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Scratch arrays shared by one sublevel search (single-threaded, no reentrancy). */
function makeScratch(size) {
  return {
    seen: new Int32Array(size),
    mark: new Int32Array(size),
    stack: new Int32Array(size),
    occupied: new Uint8Array(size),
    prev: new Int32Array(size),
    how: new Array(size),
    queue: new Int32Array(size),
    seenStamp: 0,
    markStamp: 0,
  };
}

/**
 * A push-optimal solver for one sublevel (a room, or a whole level as a
 * fallback). Returns {pushes, moves} or null when its work budget runs out.
 * `astar` picks A* (Hungarian heuristic) or breadth-first search over pushes.
 */
function solvePush(level, maxWork, astar) {
  const { w, h, wall, goals, boxes: startBoxes, player: startPlayer } = level;
  const size = w * h;
  const n = goals.length;
  if (n === 0) return { pushes: 0, moves: '' };
  if (startBoxes.length !== n) return null;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  for (const b of startBoxes) if (!live[b]) return null;
  // Numeric state keys are exact while size ** (n + 1) stays under 2 ** 53;
  // beyond that, string keys.
  const numericKey = Math.pow(size, n + 1) < 2 ** 53;
  const keyOf = numericKey
    ? (bs, low) => { let k = low; for (const b of bs) k = k * size + b; return k; }
    : (bs, low) => low + ':' + bs.join(',');
  const solved = (bs) => { for (const b of bs) if (!goal[b]) return false; return true; };

  const sc = makeScratch(size);
  const { seen, mark, stack, occupied, prev, how, queue } = sc;

  // The lowest square of the player's reachable region from `from`, with
  // `occupied` in the way; marks the region in `arr` with `stamp`.
  const floodLow = (from, arr, stamp) => {
    let low = from;
    let sp = 0;
    stack[sp++] = from;
    arr[from] = stamp;
    while (sp > 0) {
      const p = stack[--sp];
      if (p < low) low = p;
      const a = p - w, b = p + w, c = p - 1, d = p + 1;
      if (!wall[a] && !occupied[a] && arr[a] !== stamp) { arr[a] = stamp; stack[sp++] = a; }
      if (!wall[b] && !occupied[b] && arr[b] !== stamp) { arr[b] = stamp; stack[sp++] = b; }
      if (!wall[c] && !occupied[c] && arr[c] !== stamp) { arr[c] = stamp; stack[sp++] = c; }
      if (!wall[d] && !occupied[d] && arr[d] !== stamp) { arr[d] = stamp; stack[sp++] = d; }
    }
    return low;
  };

  /** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
  const frozenBlock = (t) => {
    for (let k = 0; k < 4; k++) {
      const a = [t, t - 1, t - w, t - w - 1][k];
      const b = a + 1, c = a + w, d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
    }
    return false;
  };

  // Heuristic: the minimum-cost assignment of boxes to goals (Hungarian, brute
  // force up to 4 boxes; the sum of nearest-goal distances beyond that), on
  // shortest-path distances that ignore the other boxes. Admissible, and
  // consistent: one push moves one box one square, so it changes by at most 1.
  let heuristic = null;
  if (astar) {
    const distG = goals.map((g) => bfsDist(wall, w, size, g));
    if (n <= 4) {
      const perms = PERMS[n];
      heuristic = (bs) => {
        let best = Infinity;
        for (const perm of perms) {
          let s = 0;
          for (let i = 0; i < n; i++) {
            const d = distG[perm[i]][bs[i]];
            if (d < 0) { s = Infinity; break; }
            s += d;
          }
          if (s < best) best = s;
        }
        return best;
      };
    } else {
      heuristic = (bs) => {
        let s = 0;
        for (const b of bs) {
          let m = Infinity;
          for (let j = 0; j < n; j++) {
            const d = distG[j][b];
            if (d >= 0 && d < m) m = d;
          }
          if (m === Infinity) return Infinity;
          s += m;
        }
        return s;
      };
    }
    if (heuristic(startBoxes) === Infinity) return null;
  }

  // The states in the order found: their boxes, the player's square after the
  // push that made them, their pushes, their parent and that push's squares.
  const stBoxes = [startBoxes.slice()];
  const stPlayer = [startPlayer];
  const stG = [0];
  const stParent = [-1];
  const stFrom = [-1];
  const stTo = [-1];
  const stKey = [];
  const index = new Map();
  const startLow = floodLow(startPlayer, mark, ++sc.markStamp);
  stKey.push(keyOf(startBoxes, startLow));
  index.set(stKey[0], 0);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // A* open list: a binary heap on (f, h, insertion order); BFS: a plain queue.
  const heap = [];
  let heapSeq = 0;
  const heapLess = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && heapLess(heap[c + 1], heap[c])) c++;
        if (!heapLess(heap[c], heap[i])) break;
        const t = heap[i]; heap[i] = heap[c]; heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  let head = 0; // BFS queue head over stBoxes
  if (astar) heapPush([heuristic(startBoxes), heuristic(startBoxes), heapSeq++, 0]);

  let work = 0;
  let found = -1;
  while (astar ? heap.length > 0 : head < stBoxes.length) {
    const s = astar ? heapPop()[3] : head++;
    if (astar && index.get(stKey[s]) !== s) continue; // a better path to this state has been found
    if (astar && solved(stBoxes[s])) { found = s; break; } // popped with the smallest f: push-optimal
    work++;
    globalWork++;
    if (work > maxWork || globalWork > GLOBAL_WORK_BUDGET) return null;
    const boxes = stBoxes[s];
    for (const b of boxes) occupied[b] = 1;
    const here = ++sc.seenStamp;
    floodLow(stPlayer[s], seen, here);
    for (let bi = 0; bi < n && found < 0; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DIRS[k][1] * w + DIRS[k][0];
        const from = b - d;
        if (seen[from] !== here) continue;
        const t = b + d;
        if (wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(t);
        let next = null;
        let key = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, floodLow(b, mark, ++sc.markStamp));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const g2 = stG[s] + 1;
        // A better path to a known state replaces it (a consistent heuristic
        // can still generate a state first on a worse path); BFS generates
        // states in nondecreasing pushes, so the first path there is the best.
        const prevIdx = index.get(key);
        if (prevIdx !== undefined && stG[prevIdx] <= g2) continue;
        if (!astar && stBoxes.length >= maxWork) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        if (astar) {
          const h2 = heuristic(next);
          if (h2 === Infinity) continue;
          index.set(key, stBoxes.length);
          stKey.push(key);
          stBoxes.push(next);
          stPlayer.push(b);
          stG.push(g2);
          stParent.push(s);
          stFrom.push(b);
          stTo.push(t);
          heapPush([g2 + h2, h2, heapSeq++, stBoxes.length - 1]);
        } else {
          index.set(key, stBoxes.length);
          stKey.push(key);
          stBoxes.push(next);
          stPlayer.push(b);
          stG.push(g2);
          stParent.push(s);
          stFrom.push(b);
          stTo.push(t);
          if (solved(next)) { found = stBoxes.length - 1; break; }
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path = [];
  for (let s = found; s > 0; s = stParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const nextBoxes = stBoxes[s];
    const from = stFrom[s];
    const to = stTo[s];
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, prev, how, queue, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (the first direction first on ties). */
function walkTo(w, wall, occupied, prev, how, queue, from, to) {
  if (from === to) return '';
  const size = wall.length;
  prev.fill(-1);
  let qh = 0, qt = 0;
  queue[qt++] = from;
  prev[from] = from;
  while (qh < qt) {
    const p = queue[qh++];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DIRS[k][1] * w + DIRS[k][0];
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = DIRS[k][2];
      queue[qt++] = q;
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** One level's push-optimal solution: room by room, or over the whole level when it is no set of rooms. */
function solveLevel(text) {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let pushes = 0;
    let moves = '';
    for (const room of rooms) {
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const sub = {
        w: level.w, h: level.h, wall,
        goals: room.goals.slice().sort((a, b) => a - b),
        boxes: room.boxes.slice().sort((a, b) => a - b),
        player,
      };
      let solved = solvePush(sub, ASTAR_EXPANSIONS_PER_ROOM, true);
      if (solved === null) solved = solvePush(sub, BFS_STATES_PER_ROOM, false);
      if (solved === null) return null;
      pushes += solved.pushes;
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
    }
    return { pushes, moves };
  }
  let solved = solvePush(level, ASTAR_EXPANSIONS_PER_ROOM, true);
  if (solved === null) solved = solvePush(level, BFS_STATES_PER_ROOM, false);
  return solved;
}

for (const id of readdirSync(INPUT).sort()) {
  let moves = '';
  try {
    const res = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
    if (res !== null) moves = res.moves;
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
