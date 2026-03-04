import matplotlib.pyplot as plt
import numpy as np
from astar import Grid, AStar, compare_algorithms
from markov import MarkovChain
from simulation import MarkovSimulation
import time

def experiment_1_compare_algorithms():
    """E.1: Comparer UCS vs Greedy vs A* sur 3 grilles"""
    
    # Définir 3 grilles de difficulté croissante
    grids = [
        # Grille facile (peu d'obstacles)
        {
            'name': 'Facile',
            'width': 5,
            'height': 5,
            'obstacles': [(2, 2)],
            'start': (0, 0),
            'goal': (4, 4)
        },
        # Grille moyenne
        {
            'name': 'Moyenne',
            'width': 8,
            'height': 8,
            'obstacles': [(2, 2), (2, 3), (3, 2), (5, 5), (5, 6), (6, 5)],
            'start': (0, 0),
            'goal': (7, 7)
        },
        # Grille difficile (labyrinthe)
        {
            'name': 'Difficile',
            'width': 10,
            'height': 10,
            'obstacles': [(i, 5) for i in range(2, 8)] + [(5, j) for j in range(2, 8)],
            'start': (0, 0),
            'goal': (9, 9)
        }
    ]
    
    results = {}
    
    for grid_config in grids:
        print(f"\n\n{'#'*60}")
        print(f"GRILLE {grid_config['name']}")
        print(f"{'#'*60}")
        
        grid = Grid(grid_config['width'], grid_config['height'], 
                   grid_config['obstacles'])
        
        # Comparer les algorithmes
        algo_results = compare_algorithms(grid, grid_config['start'], 
                                         grid_config['goal'])
        
        results[grid_config['name']] = algo_results
    
    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (grid_name, algo_results) in enumerate(results.items()):
        ax = axes[idx]
        
        algorithms = list(algo_results.keys())
        costs = [algo_results[a]['cost'] if algo_results[a]['success'] else 0 for a in algorithms]
        nodes = [algo_results[a]['nodes_expanded'] for a in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        ax.bar(x - width/2, costs, width, label='Coût', color='blue', alpha=0.7)
        ax.bar(x + width/2, nodes, width, label='Nœuds développés', color='red', alpha=0.7)
        
        ax.set_xlabel('Algorithmes')
        ax.set_ylabel('Valeurs')
        ax.set_title(f'Grille {grid_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_1_comparaison_algorithmes.png')
    plt.show()
    
    return results

def experiment_2_vary_epsilon():
    """E.2: Varier ε et mesurer l'impact sur la probabilité d'atteindre GOAL"""
    
    # Grille de test
    grid = Grid(5, 5, obstacles=[(2, 2), (2, 3)])
    start = (0, 0)
    goal = (4, 4)
    
    # Trouver le chemin optimal avec A*
    astar = AStar(grid)
    result = astar.search(start, goal, algorithm='astar')
    
    if not result['success']:
        print("Pas de chemin trouvé!")
        return
    
    path = result['path']
    policy = astar.extract_policy(path)
    
    # États d'échec (obstacles)
    fail_states = grid.obstacles
    
    # Tester différents epsilon
    epsilons = [0.0, 0.1, 0.2, 0.3]
    results = []
    
    # Ajouter GOAL comme état absorbant
    absorbing = {goal: 'GOAL'}
    
    # Ajouter un état FAIL
    fail_state = (-1, -1)
    absorbing[fail_state] = 'FAIL'
    
    for eps in epsilons:
        print(f"\n--- ε = {eps} ---")
        
        # Construire la chaîne de Markov
        all_states = set(path) | set(fail_states) | {fail_state}
        mc = MarkovChain(all_states, absorbing)
        
        mc.build_from_policy(policy, grid, eps, fail_states)
        
        # Analyse théorique
        i0 = mc.state_to_idx[start]
        pi_n_goal = []
        
        for n in range(1, 21):  # Jusqu'à 20 étapes
            pi_n = mc.get_distribution(start, n)
            goal_idx = mc.state_to_idx[goal]
            pi_n_goal.append(pi_n[goal_idx])
        
        # Simulations
        sim = MarkovSimulation(mc)
        sim_stats = sim.run_simulations(start, n_simulations=1000, max_steps=20)
        p_goal_emp = sim_stats['goal_count'] / 1000
        
        results.append({
            'epsilon': eps,
            'path_cost': result['cost'],
            'p_goal_theoretical': pi_n_goal[-1] if pi_n_goal else 0,
            'p_goal_empirical': p_goal_emp,
            'mean_time': np.mean(sim_stats['times_to_goal']) if sim_stats['times_to_goal'] else None,
            'evolution': pi_n_goal
        })
        
        print(f"  P(goal) théorique (20 étapes): {pi_n_goal[-1]:.4f}")
        print(f"  P(goal) empirique: {p_goal_emp:.4f}")
        print(f"  Temps moyen: {results[-1]['mean_time']:.2f}")
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Probabilités
    eps = [r['epsilon'] for r in results]
    p_theo = [r['p_goal_theoretical'] for r in results]
    p_emp = [r['p_goal_empirical'] for r in results]
    
    ax1.plot(eps, p_theo, 'b-o', label='Théorique', linewidth=2)
    ax1.plot(eps, p_emp, 'r--s', label='Empirique', linewidth=2)
    ax1.set_xlabel('ε (taux d\'incertitude)')
    ax1.set_ylabel('P(atteindre GOAL)')
    ax1.set_title('Probabilité d\'atteindre le but')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Évolution temporelle
    for i, r in enumerate(results):
        ax2.plot(range(1, 21), r['evolution'], label=f'ε={r["epsilon"]}', linewidth=2)
    
    ax2.set_xlabel('Nombre d\'étapes')
    ax2.set_ylabel('P(atteindre GOAL)')
    ax2.set_title('Évolution temporelle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_2_vary_epsilon.png')
    plt.show()
    
    return results

def experiment_3_compare_heuristics():
    """E.3: Tester deux heuristiques admissibles (Manhattan vs Zero)"""
    
    # Grille de test
    grid = Grid(10, 10, obstacles=[(i, 5) for i in range(3, 8)] + [(5, j) for j in range(3, 8)])
    start = (0, 0)
    goal = (9, 9)
    
    print("\n" + "="*60)
    print("COMPARAISON DES HEURISTIQUES")
    print("="*60)
    
    heuristics = ['manhattan', 'zero']
    results = {}
    
    for heur in heuristics:
        print(f"\n--- Heuristique: {heur.upper()} ---")
        astar = AStar(grid, heuristic=heur)
        
        start_time = time.time()
        result = astar.search(start, goal, algorithm='astar')
        elapsed = time.time() - start_time
        
        results[heur] = {
            **result,
            'time': elapsed
        }
        
        print(f"  Chemin trouvé: {'Oui' if result['success'] else 'Non'}")
        if result['success']:
            print(f"  Longueur du chemin: {len(result['path'])}")
            print(f"  Coût total: {result['cost']}")
        print(f"  Nœuds développés: {result['nodes_expanded']}")
        print(f"  Taille max OPEN: {result['max_open_size']}")
        print(f"  Temps d'exécution: {elapsed:.4f} s")
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(heuristics))
    width = 0.35
    
    nodes = [results[h]['nodes_expanded'] for h in heuristics]
    times = [results[h]['time'] * 1000 for h in heuristics]  # en ms
    
    ax.bar(x - width/2, nodes, width, label='Nœuds développés', color='blue', alpha=0.7)
    ax.bar(x + width/2, times, width, label='Temps (ms)', color='green', alpha=0.7)
    
    ax.set_xlabel('Heuristique')
    ax.set_ylabel('Valeurs')
    ax.set_title('Comparaison des heuristiques')
    ax.set_xticks(x)
    ax.set_xticklabels(heuristics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_3_compare_heuristics.png')
    plt.show()
    
    return results

def experiment_4_weighted_astar():
    """E.4: Weighted A* et analyse du compromis vitesse vs optimalité"""
    
    # Grille de test
    grid = Grid(15, 15, obstacles=[(i, 7) for i in range(5, 12)] + [(7, j) for j in range(5, 12)])
    start = (0, 0)
    goal = (14, 14)
    
    print("\n" + "="*60)
    print("WEIGHTED A* - COMPROMIS VITESSE VS OPTIMALITÉ")
    print("="*60)
    
    weights = [1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    
    astar = AStar(grid)
    
    # Trouver le chemin optimal (weight=1.0) pour référence
    optimal_result = astar.search(start, goal, algorithm='astar', weight=1.0)
    optimal_cost = optimal_result['cost'] if optimal_result['success'] else float('inf')
    
    print(f"\nChemin optimal (w=1.0): coût={optimal_cost}, nœuds={optimal_result['nodes_expanded']}")
    
    for w in weights:
        print(f"\n--- Weight = {w} ---")
        
        start_time = time.time()
        result = astar.search(start, goal, algorithm='astar', weight=w)
        elapsed = time.time() - start_time
        
        if result['success']:
            suboptimality = (result['cost'] - optimal_cost) / optimal_cost * 100 if optimal_cost != float('inf') else 0
            speedup = optimal_result['nodes_expanded'] / result['nodes_expanded'] if result['nodes_expanded'] > 0 else 0
        else:
            suboptimality = float('inf')
            speedup = 0
        
        results.append({
            'weight': w,
            'cost': result['cost'] if result['success'] else None,
            'nodes': result['nodes_expanded'],
            'time': elapsed,
            'suboptimality': suboptimality,
            'speedup': speedup,
            'success': result['success']
        })
        
        print(f"  Chemin trouvé: {'Oui' if result['success'] else 'Non'}")
        if result['success']:
            print(f"  Coût: {result['cost']} (sous-optimalité: {suboptimality:.1f}%)")
        print(f"  Nœuds développés: {result['nodes_expanded']} (speedup: {speedup:.2f}x)")
        print(f"  Temps: {elapsed:.4f} s")
    
    # Visualisation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    weights_list = [r['weight'] for r in results if r['success']]
    subopt = [r['suboptimality'] for r in results if r['success']]
    speedup = [r['speedup'] for r in results if r['success']]
    
    ax1.plot(weights_list, subopt, 'b-o', linewidth=2)
    ax1.set_xlabel('Poids w')
    ax1.set_ylabel('Sous-optimalité (%)')
    ax1.set_title('Sous-optimalité vs Poids')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(weights_list, speedup, 'r-o', linewidth=2)
    ax2.set_xlabel('Poids w')
    ax2.set_ylabel('Accélération (x)')
    ax2.set_title('Accélération vs Poids')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment_4_weighted_astar.png')
    plt.show()
    
    return results