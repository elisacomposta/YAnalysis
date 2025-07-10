import json
import sqlite3
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tqdm import tqdm
from matplotlib.patches import Patch
from src.utils import COALITIONS, MISINFO_LEVELS, SAVE, SHOW, get_files

def plot_final_graph(db_files, tag=''):
    fig, axes = plt.subplots(1, len(db_files), figsize=(5 * len(db_files), 6), squeeze=False)
    axes = axes.flatten()

    palette = sns.color_palette("Set2", n_colors=len(COALITIONS))
    soft_colors = {k: palette[i % len(palette)] for i, k in enumerate(COALITIONS)}

    for idx, (file, misinfo_level) in tqdm(enumerate(zip(db_files, MISINFO_LEVELS)), desc="Graphs", total=len(db_files), leave=False):
        conn = sqlite3.connect(file)
        cursor = conn.cursor()

        follow_data = cursor.execute("SELECT user_id, action, follower_id, round FROM follow ORDER BY round ASC").fetchall()
        cursor.execute("SELECT id, is_misinfo, leaning FROM user_mgmt")
        user_info = {row[0]: {"is_misinfo": row[1], "leaning": row[2]} for row in cursor.fetchall()}

        G = nx.DiGraph()
        #G.add_nodes_from(user_info.keys()) # Show all users, including the ones with no follow

        for user_id, action, follower_id, round in follow_data:
            if action == "follow":
                G.add_edge(user_id, follower_id, round=round)
            elif action == "unfollow" and G.has_edge(user_id, follower_id):
                G.remove_edge(user_id, follower_id)

        for node in G.nodes():
            info = user_info.get(node, {})
            G.nodes[node]['is_misinfo'] = info.get("is_misinfo", 0)
            G.nodes[node]['leaning'] = info.get("leaning", "unknown")

        ax = axes[idx]
        ax.set_title(f'Misinformation: {misinfo_level*100:.0f}%', fontsize=20)
        ax.axis('off')

        pos = nx.kamada_kawai_layout(G)

        for node in G.nodes():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node],
                node_color=[soft_colors.get(G.nodes[node]['leaning'], 'gray')],
                edgecolors='black',
                linewidths=1.5 if G.nodes[node]['is_misinfo'] else 0.5,
                node_size=250 + 20 * G.in_degree(node),
                ax=ax
            )

        nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=13, edge_color='gray', width=0.5)

        conn.close()

    # Global legend
    legend_colors = [Patch(color=color, label=coal) for coal, color in soft_colors.items()]
    fig.legend(handles=legend_colors, loc='lower center', ncol=len(legend_colors), fontsize=18)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if SAVE:
        path = "output/graphs/network"
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/graphs_{tag}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if SHOW:
        plt.show()
    else:
        plt.close()


def plot_population_composition(sim_path_list, tag=None):
    for sim_path in sim_path_list:
        db_files = get_files(sim_path)

        data = []

        for file in db_files:
            ref = sqlite3.connect(file)
            cursor = ref.cursor()

            user_data = cursor.execute("SELECT leaning, COUNT(*) FROM user_mgmt GROUP BY leaning").fetchall()

            for coalition, count in user_data:
                data.append({
                    "coalition": coalition,
                    "count": count,
                })

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='coalition', y='count', palette="Set2", order=COALITIONS, showfliers=False)
    plt.xlabel(None)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Number of Users", fontsize=16)
    
    if SAVE:
        path = "output/graphs/population_composition"
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/population_composition{f'_{tag}' if tag else ''}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if SHOW:
        plt.show()
    else:
        plt.close()



