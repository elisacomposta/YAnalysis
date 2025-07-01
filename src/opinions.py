import sqlite3
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import seaborn as sns
import pandas as pd
from src.utils import COALITIONS, TOPICS, SHOW, SAVE, get_files
from matplotlib.lines import Line2D

def plot_op_score_by_init_value(sim_base_path, tag=None):
    db_files = get_files(f"{sim_base_path}")
    base_name = os.path.basename(sim_base_path)

    ref_conn = sqlite3.connect(db_files[0])
    cursor = ref_conn.cursor()
    topic_ids = [(topic, cursor.execute("SELECT iid FROM interests WHERE interest = ?", (topic,)).fetchone()[0]) for topic in TOPICS]
    all_rounds = [r[0] for r in cursor.execute("SELECT DISTINCT round FROM user_opinions ORDER BY round").fetchall()]
    ref_conn.close()

    n_sim = len(db_files)
    fig, axs = plt.subplots(2, len(topic_ids), figsize=(6 * len(topic_ids), 8), sharex=True)

    for t_index, (topic, topic_id) in enumerate(topic_ids):
        scores_llm = []
        scores = []

        for db_file in db_files:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            users = cursor.execute("SELECT id FROM user_mgmt").fetchall()

            for (user_id,) in users:
                data = cursor.execute("""
                    SELECT round, score, score_llm
                    FROM user_opinions
                    WHERE user_id = ? AND topic_id = ?
                    ORDER BY round
                """, (user_id, topic_id)).fetchall()

                if not data:
                    continue

                round_score = {r: (s, sllm) for r, s, sllm in data}
                init_score = data[0][1]
                init_llm = data[0][2]
                last_score = 0.0
                last_score_llm = 0.0

                for r in all_rounds:
                    if r in round_score:
                        last_score, last_score_llm = round_score[r]
                    scores_llm.append({"Round": r, "Score LLM": last_score_llm, "Init LLM": init_llm})
                    scores.append({"Round": r, "Score": last_score, "Init Score": init_score})

            conn.close()

        # Convert to DataFrame
        df_llm = pd.DataFrame(scores_llm)
        df_score = pd.DataFrame(scores)
        colors = sns.color_palette("tab10", n_colors=len(df_llm["Init LLM"].unique()))

        # Plot
        for i, (df, type) in enumerate([(df_llm, "LLM"), (df_score, "")]):
            ax = axs[i, t_index]
            score = "Score LLM" if type == "LLM" else "Score"
            hue = "Init LLM" if type == "LLM" else "Init Score"
            sns.lineplot(data=df, x="Round", y=score, hue=hue, ax=ax, palette=colors, errorbar=('ci', 95))
            ax.axhline(y=1, color='orange', linestyle='--', linewidth=1)
            ax.axhline(y=-1, color='orange', linestyle='--', linewidth=1)
            ax.axhline(y=0, color='black', alpha=0.5, linestyle='--', linewidth=1)
            ax.set_xticks(np.arange(0, max(all_rounds) + 1, 24 * 5))
            ax.set_xticklabels([f"{r // 24}" for r in ax.get_xticks()])
            ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
            ax.legend(title=False)
            ax.grid(True)

            if i == 0:
                ax.set_title(f"{topic.capitalize()}")
            if t_index == 0:
                ax.set_ylabel(score)

    plt.tight_layout()

    if SAVE:
        path = f"output/opinions/by_init_opinion"
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/{base_name}{f'_{tag}' if tag else ''}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if SHOW:
        plt.show()
    else:
        plt.close(fig)


def plot_op_score_by_coalition(sim_base_path, tag=None):
    db_files = get_files(f"{sim_base_path}")
    base_name = os.path.basename(sim_base_path)

    # Topic id and rounds
    ref_conn = sqlite3.connect(db_files[0])
    cursor = ref_conn.cursor()
    topic_ids = [(topic, cursor.execute("SELECT iid FROM interests WHERE interest = ?", (topic,)).fetchone()[0]) for topic in TOPICS]
    all_rounds = [r[0] for r in cursor.execute("SELECT DISTINCT round FROM user_opinions ORDER BY round").fetchall()]
    ref_conn.close()

    fig, axs = plt.subplots(2, len(topic_ids), figsize=(6 * len(topic_ids), 8), sharex=True)

    for t_index, (topic, topic_id) in enumerate(topic_ids):
        scores_llm = []
        scores = []

        for db_file in db_files:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            for coalition in COALITIONS:
                user_ids = [u[0] for u in cursor.execute("SELECT id FROM user_mgmt WHERE leaning = ?", (coalition,)).fetchall()]

                for user_id in user_ids:
                    data = cursor.execute("""
                        SELECT round, score_llm, score
                        FROM user_opinions
                        WHERE user_id = ? AND topic_id = ?
                        ORDER BY round
                    """, (user_id, topic_id)).fetchall()

                    if not data:
                        continue

                    round_score = {r: (sllm, s) for r, sllm, s in data}
                    last_score, last_score_llm = 0.0, 0.0

                    for r in all_rounds:
                        if r in round_score:
                            last_score_llm, last_score = round_score[r]
                        scores_llm.append({"Round": r, "Score LLM": last_score_llm, "Coalition": coalition})
                        scores.append({"Round": r, "Score": last_score, "Coalition": coalition})

            conn.close()

        # Dataframe
        df_llm = pd.DataFrame(scores_llm)
        df_score = pd.DataFrame(scores)
        palette = sns.color_palette("tab10", n_colors=len(COALITIONS))

        # Plot
        for i, (df, label) in enumerate([(df_llm, "Score LLM"), (df_score, "Score")]):
            ax = axs[i, t_index]
            sns.lineplot(data=df, x="Round", y=label, hue="Coalition", ax=ax, palette=palette)

            ax.axhline(y=1, color='orange', linestyle='--', linewidth=1)
            ax.axhline(y=-1, color='orange', linestyle='--', linewidth=1)
            ax.axhline(y=0, color='black', alpha=0.5, linestyle='--', linewidth=1)
            ax.set_xticks(np.arange(0, max(all_rounds) + 1, 24 * 5))
            ax.set_xticklabels([f"{r // 24}" for r in ax.get_xticks()], fontsize=18)
            ax.yaxis.set_ticks(np.arange(-1, 1.1, 0.5))
            ax.tick_params(axis='y', labelsize=18)
            ax.grid(True)
            ax.legend_.remove()     

            if i == 0:
                ax.set_title(topic.capitalize(), fontsize=20)
            if t_index == 0:
                ax.set_ylabel(label, fontsize=20)
            if i == 1:
                ax.set_xlabel("Days", fontsize=20)

    legend_handles = [Line2D([0], [0], color=palette[i], lw=2) for i in range(len(COALITIONS))]
    fig.legend(legend_handles, COALITIONS, loc='lower center', ncol=len(COALITIONS), fontsize=18, frameon=False)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    if SAVE:
        path = f"output/opinions/by_coalition"
        os.makedirs(path, exist_ok=True)
        filename = f"{path}/{base_name}{f'_{tag}' if tag else ''}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if SHOW:
        plt.show()
    else:
        plt.close(fig)
