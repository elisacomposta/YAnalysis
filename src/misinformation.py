import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from collections import defaultdict
import seaborn as sns
import pandas as pd
from src.utils import COALITIONS, MISINFO_LEVELS, SAVE, SHOW, TOPICS, get_files

def plot_shift_by_misinfo_level_runs(sim_paths, coalition, tag=None):
    # topic -> misinfo_level -> list of shifts (list of shifts per run)
    data = defaultdict(lambda: defaultdict(list))

    for sim_path, misinfo_level in zip(sim_paths, MISINFO_LEVELS):
        sim_files = get_files(sim_path)

        for sim_file in sim_files:
            conn = sqlite3.connect(sim_file)
            cur = conn.cursor()

            cur.execute("SELECT id, leaning FROM user_mgmt")
            user_leaning = {uid: leaning for uid, leaning in cur.fetchall()}

            for topic in TOPICS:
                topic_id = cur.execute("SELECT iid FROM interests WHERE interest = ?", (topic,)).fetchone()[0]

                records = cur.execute("""
                    SELECT user_id, score_llm, round 
                    FROM user_opinions
                    WHERE topic_id = ?
                """, (topic_id,)).fetchall()

                user_opinions = defaultdict(list)
                for user_id, score, round in records:
                    if user_leaning[user_id] == coalition:
                        user_opinions[user_id].append((round, score))

                shifts = []
                for scores in user_opinions.values():
                    sorted_scores = sorted(scores, key=lambda x: x[0])
                    shift = sorted_scores[-1][1] - sorted_scores[0][1]
                    shifts.append(shift)

                data[topic][misinfo_level].append(shifts)

            conn.close()

    fig, axs = plt.subplots(1, 4, figsize=(16, 8), sharey=True)
    fig.suptitle(f"Opinion Shift Distribution - {coalition}", fontsize=14)

    x_range = np.linspace(-2, 2, 200)

    for i, topic in enumerate(TOPICS):
        ax = axs[i]
        ax.set_title(topic.capitalize(), fontsize=12)

        for j, misinfo_level in enumerate(MISINFO_LEVELS):
            runs = data[topic][misinfo_level]
            y_base = j

            cmap = cm.get_cmap('tab20', len(runs))
            for k, shifts in enumerate(runs):
                if not shifts or len(set(shifts)) <= 1:
                    continue
                kde = gaussian_kde(shifts, bw_method=0.4)
                y_vals = kde(x_range)
                y_vals = y_vals / y_vals.max() * 0.6
                ax.plot(x_range, y_vals + y_base, alpha=0.8, linewidth=1.5, color=cmap(k))

            ax.axhline(y=y_base, color='gray', linestyle=':', linewidth=0.5)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.2, len(MISINFO_LEVELS) - 0.2)
        ax.set_yticks(range(len(MISINFO_LEVELS)))
        ax.set_xlabel("Opinion shift", fontsize=10)
        ax.set_yticklabels([f"{int(lvl*100)}%" for lvl in MISINFO_LEVELS], fontsize=10)
        if i == 0:
            ax.set_ylabel("Misinformation level", fontsize=10)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.06, wspace=0.25)

    if SAVE:
        path = f"output/misinformation/ridgeplots/runs"
        os.makedirs(path, exist_ok=True)
        fig.savefig(f"{path}/ridge_{coalition}{f'_{tag}' if tag else ''}.png", dpi=300)

    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def plot_shift_by_misinfo_level_coalitions(sim_paths, score_llm=True, tag=None, only_misinfo=False, only_base=False):
    score_type = "score_llm" if score_llm else "score"
    suffix = "_misinfo" if only_misinfo else "_base" if only_base else ""

    # topic -> misinfo_level -> coalition -> list of shifts
    data = {
        topic: {lvl: {coal: [] for coal in COALITIONS} for lvl in MISINFO_LEVELS}
        for topic in TOPICS
    }

    for sim_path, misinfo_level in zip(sim_paths, MISINFO_LEVELS):
        if only_misinfo and misinfo_level == 0.0:
            continue

        sim_files = get_files(sim_path)

        for sim_file in sim_files:
            conn = sqlite3.connect(sim_file)
            cur = conn.cursor()

            # Recupera utenti con leaning e is_misinfo
            cur.execute("SELECT id, leaning, is_misinfo FROM user_mgmt")
            users = cur.fetchall()
            user_df = pd.DataFrame(users, columns=["user_id", "leaning", "is_misinfo"])

            # Filtro se richiesto
            if only_misinfo:
                user_df = user_df[user_df["is_misinfo"] == 1]
            elif only_base:
                user_df = user_df[user_df["is_misinfo"] == 0]

            user_leaning = dict(zip(user_df["user_id"], user_df["leaning"]))

            for topic in TOPICS:
                topic_id = cur.execute("SELECT iid FROM interests WHERE interest = ?", (topic,)).fetchone()[0]
                records = cur.execute(f"""
                    SELECT user_id, {score_type}, round 
                    FROM user_opinions
                    WHERE topic_id = ?
                """, (topic_id,)).fetchall()

                user_opinions = defaultdict(list)
                for user_id, score, round in records:
                    if user_id in user_leaning:  # escludi utenti non nel filtro
                        user_opinions[user_id].append((round, score))

                shifts_per_coalition = defaultdict(list)
                for user_id, scores in user_opinions.items():
                    sorted_scores = sorted(scores, key=lambda x: x[0])
                    shift = sorted_scores[-1][1] - sorted_scores[0][1]
                    leaning = user_leaning[user_id]
                    shifts_per_coalition[leaning].append(shift)

                for coal in COALITIONS:
                    data[topic][misinfo_level][coal].extend(shifts_per_coalition[coal])

            conn.close()

    # Plot
    fig, axs = plt.subplots(1, len(TOPICS), figsize=(16, 8), sharey=True)
    x_range = np.linspace(-2, 2, 200)
    colors = sns.color_palette("tab10", n_colors=len(MISINFO_LEVELS))

    for col, topic in enumerate(TOPICS):
        ax = axs[col]
        ax.set_title(topic.capitalize(), fontsize=16)

        for idx, coalition in enumerate(COALITIONS):
            y_base = len(COALITIONS) - idx - 1

            for level_idx, misinfo_level in enumerate(MISINFO_LEVELS):
                shifts = data[topic][misinfo_level][coalition]
                if len(shifts) <= 1 or np.std(shifts) < 1e-6: # Pochi dati o costanti
                    continue  

                kde = gaussian_kde(shifts, bw_method=0.4)
                y_vals = kde(x_range)
                y_vals = y_vals / y_vals.max() * 0.6

                ax.plot(
                    x_range,
                    y_vals + y_base,
                    color=colors[level_idx],
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"{int(misinfo_level * 100)}%" if idx == 0 else ""
                )

            ax.axhline(y=y_base, color='gray', linestyle=':', linewidth=0.5)

        ax.set_xlim(-2, 2)
        ax.set_xticks(np.arange(-2, 3, 1))
        ax.set_xticklabels(np.arange(-2, 3, 1), fontsize=18)
        ax.set_xlabel("Opinion Shift", fontsize=18)
        ax.set_yticks(range(len(COALITIONS)))
        ax.set_yticklabels(COALITIONS[::-1], fontsize=18)

    # Legenda
    legend_labels = [f"{int(level * 100)}%" for level in MISINFO_LEVELS]
    legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(MISINFO_LEVELS))]
    fig.legend(legend_handles, legend_labels, title="Misinformation", loc='center right', fontsize=18, title_fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.87, 0.95])

    if SAVE:
        path = "output/misinformation/ridgeplots/coalitions"
        os.makedirs(path, exist_ok=True)
        filename = f"{score_type}{'_' + tag if tag else ''}{suffix}.png"
        fig.savefig(os.path.join(path, filename), dpi=300)

    if SHOW:
        plt.show()
    else:
        plt.close(fig)
