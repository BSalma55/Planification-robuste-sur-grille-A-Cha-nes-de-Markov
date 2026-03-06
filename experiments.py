# -*- coding: utf-8 -*-

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import heapq
import time
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("\n" + "="*80)
print("MINI-PROJET : PLANIFICATION ROBUSTE SUR GRILLE")
print("Experiments FINAL - E.1, E.2, E.3, E.4")
print("="*80)

class Grid:
    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles) if obstacles else set()
    
    def is_free(self, x, y):
        return (0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles)
    
    def neighbors(self, x, y):
        return [(nx, ny) for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
                if self.is_free(nx, ny)]
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def zero_heuristic(self, pos1, pos2):
        return 0.0

class AStar:
    def __init__(self, grid, heuristic='manhattan'):
        self.grid = grid
        self.h = grid.manhattan_distance if heuristic == 'manhattan' else grid.zero_heuristic
    
    def search(self, start, goal, algorithm='astar', weight=1.0):
        if algorithm == 'ucs':
            wg, wh = 1.0, 0.0
        elif algorithm == 'greedy':
            wg, wh = 0.0, 1.0
        else:
            wg, wh = 1.0, float(weight)
        
        t0 = time.perf_counter()
        counter = 0
        open_heap = [(0.0, counter, start)]
        in_open = {start}
        closed = set()
        came_from = {}
        g_score = {start: 0}
        
        nodes_expanded = 0
        max_open_size = 0
        visited = []
        
        while open_heap:
            max_open_size = max(max_open_size, len(open_heap))
            _, _, current = heapq.heappop(open_heap)
            in_open.discard(current)
            
            if current == goal:
                return {
                    'path': self._reconstruct(came_from, current),
                    'cost': g_score[current],
                    'nodes_expanded': nodes_expanded,
                    'max_open_size': max_open_size,
                    'time_sec': time.perf_counter() - t0,
                    'success': True,
                    'visited': visited
                }
            
            if current in closed:
                continue
            closed.add(current)
            visited.append(current)
            nodes_expanded += 1
            
            for nbr in self.grid.neighbors(*current):
                if nbr in closed:
                    continue
                g_new = g_score[current] + 1
                if g_new < g_score.get(nbr, float('inf')):
                    came_from[nbr] = current
                    g_score[nbr] = g_new
                    f_new = wg * g_new + wh * self.h(nbr, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, nbr))
                    in_open.add(nbr)
        
        return {
            'path': None,
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'max_open_size': max_open_size,
            'time_sec': time.perf_counter() - t0,
            'success': False,
            'visited': visited
        }
    
    def _reconstruct(self, came_from, node):
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        path.reverse()
        return path
    
    def extract_policy(self, path):
        policy = {}
        for i in range(len(path) - 1):
            policy[path[i]] = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if path:
            policy[path[-1]] = (0, 0)
        return policy

class MarkovChain:
    def __init__(self, states, absorbing_states=None):
        self.states = list(states)
        self.n = len(self.states)
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}
        self.P = np.zeros((self.n, self.n))
        self.absorbing_states = absorbing_states if absorbing_states else {}
        self.absorbing_indices = {self.state_to_idx[s] for s in self.absorbing_states if s in self.state_to_idx}
    
    def build_from_policy(self, policy, grid, epsilon):
        valid = set(self.states)
        for state in self.states:
            i = self.state_to_idx[state]
            if state in self.absorbing_states:
                self.P[i, i] = 1.0
                continue
            action = policy.get(state, (0, 0))
            dx, dy = action
            dest_main = (state[0] + dx, state[1] + dy)
            if not grid.is_free(*dest_main) or dest_main not in valid:
                dest_main = state
            j = self.state_to_idx.get(dest_main)
            if j is not None:
                self.P[i, j] += (1.0 - epsilon)
            if action != (0, 0):
                laterals = [(0,1), (0,-1)] if dx != 0 else [(1,0), (-1,0)]
                for ldx, ldy in laterals:
                    dest_lat = (state[0] + ldx, state[1] + ldy)
                    if not grid.is_free(*dest_lat) or dest_lat not in valid:
                        dest_lat = state
                    j = self.state_to_idx.get(dest_lat)
                    if j is not None:
                        self.P[i, j] += (epsilon / 2.0)
        row_sums = self.P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.P /= row_sums
    
    def get_distribution_evolution(self, initial_state, max_steps):
        i0 = self.state_to_idx[initial_state]
        pi = np.zeros(self.n)
        pi[i0] = 1.0
        evol = []
        for _ in range(max_steps):
            pi = pi @ self.P
            evol.append(pi.copy())
        return evol
    
    def absorption_analysis(self):
        if not self.absorbing_indices:
            return None
        trans_idx = [i for i in range(self.n) if i not in self.absorbing_indices]
        abs_idx = sorted(self.absorbing_indices)
        if not trans_idx:
            return None
        Q = self.P[np.ix_(trans_idx, trans_idx)]
        R = self.P[np.ix_(trans_idx, abs_idx)]
        try:
            N = np.linalg.inv(np.eye(len(trans_idx)) - Q)
            B = N @ R
            t = N @ np.ones(len(trans_idx))
            return {'B': B, 't': t, 'N': N, 'trans_indices': trans_idx, 'abs_indices': abs_idx}
        except:
            return None

class MarkovSimulation:
    def __init__(self, markov_chain):
        self.mc = markov_chain
    
    def simulate_trajectory(self, start_state, max_steps=200):
        if start_state not in self.mc.state_to_idx:
            return None, None, None
        trajectory = [start_state]
        current_idx = self.mc.state_to_idx[start_state]
        for step in range(max_steps):
            next_idx = np.random.choice(self.mc.n, p=self.mc.P[current_idx])
            next_state = self.mc.idx_to_state[next_idx]
            trajectory.append(next_state)
            if next_state in self.mc.absorbing_states:
                return trajectory, step + 1, next_state
            current_idx = next_idx
        return trajectory, max_steps, trajectory[-1]
    
    def run_simulations(self, start_state, n_simulations=2000, max_steps=200):
        stats = {'goal_count': 0, 'fail_count': 0, 'times_to_goal': [], 'times_to_fail': [], 'final_states': defaultdict(int)}
        for _ in range(n_simulations):
            _, time_abs, final = self.simulate_trajectory(start_state, max_steps)
            if final is None:
                continue
            stats['final_states'][final] += 1
            if final in self.mc.absorbing_states:
                typ = self.mc.absorbing_states[final]
                if typ == 'GOAL':
                    stats['goal_count'] += 1
                    stats['times_to_goal'].append(time_abs)
                elif typ == 'FAIL':
                    stats['fail_count'] += 1
                    stats['times_to_fail'].append(time_abs)
        return stats

def get_test_grids():
    return [
        {'name': 'Facile', 'grid': Grid(5, 5, obstacles=[(2, 2)]), 'start': (0, 0), 'goal': (4, 4), 'desc': '5x5, 1 obstacle central'},
        {'name': 'Moyenne', 'grid': Grid(8, 8, obstacles=[(2,2), (2,3), (3,2), (5,5), (5,6), (6,5)]), 'start': (0, 0), 'goal': (7, 7), 'desc': '8x8, 6 obstacles'},
        {'name': 'Difficile', 'grid': Grid(10, 10, obstacles=([(i,5) for i in range(2,8)] + [(5,j) for j in range(2,8)])), 'start': (0, 0), 'goal': (9, 9), 'desc': '10x10, croix centrale'}
    ]

# FIGURE 1 : VISUALISATION DES CHEMINS
def figure_1():
    print("\n [FIGURE 1] Visualisation des chemins planifies...")
    grids_config = get_test_grids()
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Figure 1 : Visualisation des chemins planifies", fontsize=14, fontweight='bold')
    
    plot_idx = 1
    for grid_config in grids_config:
        grid = grid_config['grid']
        start = grid_config['start']
        goal = grid_config['goal']
        searcher = AStar(grid, heuristic='manhattan')
        results = {}
        
        for algo in ['ucs', 'greedy', 'astar']:
            result = searcher.search(start, goal, algorithm=algo)
            results[algo] = result
            print(f"  {grid_config['name']:10} {algo:8} : Cout={result['cost']}, Noeuds={result['nodes_expanded']}")
        
        for algo in ['ucs', 'greedy', 'astar']:
            ax = plt.subplot(3, 3, plot_idx)
            result = results[algo]
            viz = np.ones((grid.height, grid.width, 3))
            
            for y in range(grid.height):
                for x in range(grid.width):
                    if not grid.is_free(x, y):
                        viz[y, x] = [0, 0, 0]
            
            for node in result['visited']:
                x, y = node
                viz[y, x] = [0.7, 0.85, 1.0]
            
            if result['success'] and result['path']:
                for node in result['path'][1:-1]:
                    x, y = node
                    viz[y, x] = [0.0, 0.5, 1.0]
            
            sx, sy = start
            gx, gy = goal
            viz[sy, sx] = [1, 0, 0]
            viz[gy, gx] = [0, 1, 0]
            
            ax.imshow(viz, origin='upper', aspect='equal', interpolation='none')
            ax.grid(True, alpha=0.3, linewidth=0.5, color='black')
            ax.set_xticks(range(grid.width))
            ax.set_yticks(range(grid.height))
            
            ax.text(sx, sy, 'S', ha='center', va='center', fontweight='bold', fontsize=9, color='white', bbox=dict(boxstyle='circle', facecolor='red', alpha=0.9))
            ax.text(gx, gy, 'G', ha='center', va='center', fontweight='bold', fontsize=9, color='white', bbox=dict(boxstyle='circle', facecolor='green', alpha=0.9))
            
            algo_label = {'ucs': 'UCS', 'greedy': 'GREEDY', 'astar': 'A*'}[algo]
            ax.set_title(f"{grid_config['name']} - {algo_label} (Cout: {result['cost']})", fontsize=10)
            plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('Figure1_Visualisation_chemins_planifies.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [OK] Figure1_Visualisation_chemins_planifies.png")

# FIGURE 2 : COMPARAISON UCS / GREEDY / A*
def figure_2():
    print("\n [FIGURE 2] Comparaison UCS/Greedy/A*...")
    grids = get_test_grids()
    results = {}
    
    for cfg in grids:
        searcher = AStar(cfg['grid'], heuristic='manhattan')
        results[cfg['name']] = {}
        for algo in ['ucs', 'greedy', 'astar']:
            r = searcher.search(cfg['start'], cfg['goal'], algorithm=algo)
            results[cfg['name']][algo] = r
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 2 : Comparaison UCS / Greedy / A*", fontsize=14, fontweight='bold')
    
    algos = ['ucs', 'greedy', 'astar']
    colors = ['#2196F3', '#F44336', '#4CAF50']
    x = np.arange(len(algos))
    
    for ax, cfg in zip(axes, grids):
        res = results[cfg['name']]
        costs = [res[a]['cost'] for a in algos]
        nodes = [res[a]['nodes_expanded'] for a in algos]
        
        ax.bar(x - 0.2, costs, 0.35, label='Cout', color=colors, alpha=0.85)
        ax.bar(x + 0.2, nodes, 0.35, label='Noeuds developpes', color=colors, alpha=0.4, hatch='//')
        
        ax.set_title(f"Grille {cfg['name']}\n({cfg['desc']})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(['UCS', 'Greedy', 'A*'])
        ax.set_ylabel("Valeur")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Figure2_Comparaison_UCS_Greedy_A.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [OK] Figure2_Comparaison_UCS_Greedy_A.png")

# FIGURE 3 : IMPACT EPSILON
def figure_3():
    print("\n [FIGURE 3] Impact de l'incertitude epsilon...")
    cfg = get_test_grids()[0]
    grid, start, goal = cfg['grid'], cfg['start'], cfg['goal']
    epsilons = [0.0, 0.1, 0.2, 0.3]
    n_steps = 30
    N_SIM = 2000
    
    searcher = AStar(grid, heuristic='manhattan')
    result = searcher.search(start, goal, algorithm='astar')
    policy = searcher.extract_policy(result['path'])
    
    results = []
    for eps in epsilons:
        absorbing = {goal: 'GOAL', (-1,-1): 'FAIL'}
        all_states = set(result['path']) | {(-1,-1)}
        mc = MarkovChain(all_states, absorbing)
        mc.build_from_policy(policy, grid, epsilon=eps)
        
        evol = mc.get_distribution_evolution(start, n_steps)
        goal_idx = mc.state_to_idx[goal]
        p_goal_evol = [d[goal_idx] for d in evol]
        
        ab = mc.absorption_analysis()
        p_goal_abs = 0
        t_abs = 0
        if ab and start in mc.state_to_idx:
            i0 = mc.state_to_idx[start]
            if i0 in ab['trans_indices']:
                loc = ab['trans_indices'].index(i0)
                for j, ai in enumerate(ab['abs_indices']):
                    if mc.idx_to_state[ai] == goal:
                        p_goal_abs = ab['B'][loc, j]
                        t_abs = ab['t'][loc]
                        break
        
        sim = MarkovSimulation(mc)
        stats = sim.run_simulations(start, n_simulations=N_SIM, max_steps=100)
        p_goal_emp = stats['goal_count'] / N_SIM
        
        results.append({'epsilon': eps, 'evol': p_goal_evol, 'p_goal_abs': p_goal_abs, 'p_goal_emp': p_goal_emp, 't_abs': t_abs})
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['blue', 'green', 'orange', 'red']
    for i, res in enumerate(results):
        ax1.plot(range(1, n_steps+1), res['evol'], color=colors[i], linewidth=2.5, label=f"epsilon = {res['epsilon']:.1f}")
    ax1.set_xlabel("Pas n")
    ax1.set_ylabel("P(etre dans GOAL)")
    ax1.set_title("Evolution de P(GOAL) = pi(n)[GOAL]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = fig.add_subplot(gs[1, 0])
    eps_labels = [f"epsilon={r['epsilon']:.1f}" for r in results]
    p_abs = [r['p_goal_abs'] for r in results]
    p_emp = [r['p_goal_emp'] for r in results]
    xi = np.arange(len(results))
    ax2.bar(xi - 0.2, p_abs, 0.35, label='Theorique (absorption)', color='steelblue')
    ax2.bar(xi + 0.2, p_emp, 0.35, label=f'Empirique (N={N_SIM})', color='tomato')
    ax2.set_xticks(xi)
    ax2.set_xticklabels(eps_labels)
    ax2.set_ylabel("P(atteindre le but)")
    ax2.set_title("Probabilite de succes")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])
    
    ax3 = fig.add_subplot(gs[1, 1])
    t_abs_vals = [r['t_abs'] for r in results]
    ax3.bar(xi, t_abs_vals, color='steelblue', alpha=0.8)
    ax3.set_xticks(xi)
    ax3.set_xticklabels(eps_labels)
    ax3.set_ylabel("E[T] (etapes)")
    ax3.set_title("Temps moyen avant absorption")
    ax3.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Figure 3 : Impact de l'incertitude epsilon sur P(GOAL)", fontsize=14, fontweight='bold')
    plt.savefig('Figure3_Impact_incertitude_epsilon.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [OK] Figure3_Impact_incertitude_epsilon.png")

# FIGURE 4 : HEURISTIQUES
def figure_4():
    print("\n [FIGURE 4] Comparaison heuristiques...")
    grids = get_test_grids()
    results = {}
    
    for cfg in grids:
        results[cfg['name']] = {}
        for h_name in ['zero', 'manhattan']:
            searcher = AStar(cfg['grid'], heuristic=h_name)
            r = searcher.search(cfg['start'], cfg['goal'], algorithm='astar')
            results[cfg['name']][h_name] = r
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Figure 4 : Comparaison heuristiques h=0 vs Manhattan", fontsize=14, fontweight='bold')
    
    for ax, cfg in zip(axes, grids):
        res = results[cfg['name']]
        nodes = [res['zero']['nodes_expanded'], res['manhattan']['nodes_expanded']]
        xi = np.arange(2)
        bars = ax.bar(xi, nodes, color=['#FF9800', '#9C27B0'], alpha=0.85)
        
        for bar, val in zip(bars, nodes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(f"Grille {cfg['name']}\n({cfg['desc']})")
        ax.set_xticks(xi)
        ax.set_xticklabels(['h = 0', 'Manhattan'])
        ax.set_ylabel("Noeuds developpes")
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('Figure4_Comparaison_heuristiques.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [OK] Figure4_Comparaison_heuristiques.png")

# FIGURE 5 : WEIGHTED A*
def figure_5():
    print("\n [FIGURE 5] Weighted A*...")
    cfg = get_test_grids()[2]
    grid, start, goal = cfg['grid'], cfg['start'], cfg['goal']
    weights = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    searcher = AStar(grid, heuristic='manhattan')
    r_opt = searcher.search(start, goal, algorithm='astar', weight=1.0)
    c_opt = r_opt['cost']
    
    results = []
    for w in weights:
        r = searcher.search(start, goal, algorithm='astar', weight=w)
        speedup = r_opt['nodes_expanded'] / max(r['nodes_expanded'], 1)
        results.append({'w': w, 'nodes': r['nodes_expanded'], 'speedup': speedup})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 5 : Weighted A* - Compromis vitesse/optimalite", fontsize=14, fontweight='bold')
    
    ws = [r['w'] for r in results]
    nodes = [r['nodes'] for r in results]
    speeds = [r['speedup'] for r in results]
    
    ax1.plot(ws, nodes, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.set_xlabel("Poids w")
    ax1.set_ylabel("Noeuds developpes")
    ax1.set_title("Reduction des noeuds developpes")
    ax1.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2.plot(ws, [0]*len(ws), 's-', color='tomato', linewidth=2, markersize=8, label='Sous-optimalite (%)')
    ax2_twin.plot(ws, speeds, 'D-', color='#009688', linewidth=2, markersize=8, label='Speedup (x)')
    
    ax2.set_xlabel("Poids w")
    ax2.set_ylabel("Sous-optimalite (%)", color='tomato')
    ax2_twin.set_ylabel("Speedup (x)", color='#009688')
    ax2.set_title("Sous-optimalite et Speedup")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='tomato')
    ax2_twin.tick_params(axis='y', labelcolor='#009688')
    
    plt.tight_layout()
    plt.savefig('Figure5_Weighted_A.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  [OK] Figure5_Weighted_A.png")

# MAIN
if __name__ == "__main__":
    print("\n [START] Lancement de tous les experiments FINAL...\n")
    
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    
    print("\n" + "="*80)
    print(" [SUCCESS] TOUS LES FIGURES SONT GENEREES AVEC SUCCES")
    print("="*80)
    print("\nFichiers PNG generes :")
    print("  1. Figure1_Visualisation_chemins_planifies.png")
    print("  2. Figure2_Comparaison_UCS_Greedy_A.png")
    print("  3. Figure3_Impact_incertitude_epsilon.png")
    print("  4. Figure4_Comparaison_heuristiques.png")
    print("  5. Figure5_Weighted_A.png")
    print("\n" + "="*80 + "\n")