import os
import sqlite3
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import COALITIONS, SAVE, SHOW, get_files
import numpy as np
from scipy.stats import gaussian_kde

def extract_all_toxicity(simulations_root="simulations", output_root="output/toxicity"):
    from detoxify import Detoxify
    model = Detoxify("original")

    for root, dirs, files in os.walk(simulations_root):
        for file in tqdm(files, desc="Processing files"):
            if not file.endswith(".db"):
                continue

            db_path = os.path.join(root, file)

            # Output path: simulations/ -> output/toxicity/
            relative_path = os.path.relpath(db_path, simulations_root)
            out_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            if os.path.exists(out_path):
                continue

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            user_coalitions = dict(cursor.execute("SELECT id, leaning FROM user_mgmt").fetchall())
            posts = cursor.execute("SELECT id, tweet, user_id, comment_to FROM post").fetchall()

            data = []
            for post_id, text, sender_id, comment_to in tqdm(posts, desc=f"{file}", leave=False):
                if not text.strip():
                    continue

                is_comment = comment_to != -1
                sender_coal = user_coalitions.get(sender_id)
                receiver_coal = None

                if is_comment:
                    result = cursor.execute("SELECT user_id FROM post WHERE id = ?", (comment_to,)).fetchone()
                    if result:
                        receiver_id = result[0]
                        receiver_coal = user_coalitions.get(receiver_id)

                # Tossicità
                try:
                    tox = model.predict(text)["toxicity"]
                except Exception:
                    tox = None

                if tox is not None and sender_coal:
                    data.append({
                        "id": post_id,
                        "coalition_src": sender_coal,
                        "coalition_tgt": receiver_coal,
                        "is_comment": is_comment,
                        "text": text,
                        "toxicity": tox
                    })

            conn.close()

            df = pd.DataFrame(data)
            df.to_csv(out_path, index=False)


def plot_combined_toxicity_diff(sim_paths, real_world_path, tag=None):
    all_data_sim = []

    # Simulations
    for sim_path in sim_paths:
        toxicity_dir = sim_path.replace("simulations", "output/toxicity")
        files = get_files(toxicity_dir, file_extension=".csv")

        for file in files:
            df = pd.read_csv(file)
            df = df[df["coalition_tgt"].notna()]
            df["direction"] = df.apply(
                lambda row: "in" if row["coalition_src"] == row["coalition_tgt"] else "out",
                axis=1
            )
            df["toxicity_log"] = np.log1p(df["toxicity"])
            all_data_sim.append(df)

    sim_df = pd.concat(all_data_sim, ignore_index=True)
    sim_means = sim_df.groupby(["id", "direction"])["toxicity_log"].mean().unstack()
    sim_means["toxicity_diff"] = sim_means["out"] - sim_means["in"]
    sim_means = sim_means.dropna(subset=["in", "out"])

    # Real world
    real_df = pd.read_csv(real_world_path)
    real_df["direction"] = real_df.apply(
        lambda r: "in" if r["from_coalition"] == r["target_coalition"] else "out",
        axis=1
    )
    real_df["toxicity_log"] = np.log1p(real_df["toxicity"])
    real_means = real_df.groupby(["author_id", "direction"])["toxicity_log"].mean().unstack()
    real_means["toxicity_diff"] = real_means["out"] - real_means["in"]
    real_means = real_means.dropna(subset=["in", "out"])

    # Plot
    plt.figure(figsize=(10, 5))
    sns.kdeplot(sim_means["toxicity_diff"], label="Simulations", color="steelblue", linewidth=2)
    sns.kdeplot(real_means["toxicity_diff"], label="Real world", color="purple", linewidth=2)

    plt.axvline(0, linestyle="--", color="black", linewidth=1)
    plt.yticks([])
    plt.ylabel("")
    plt.xlabel("Toxicity Delta (out - in)", fontsize=16)
    plt.xticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    if SAVE:
        out_dir = "output/toxicity/plots"
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        plt.savefig(f"{out_dir}/diff_in_out_combined{suffix}.png", dpi=300)

    if SHOW:
        plt.show()
    else:
        plt.close()


def plot_toxicity_post_comment(sim_paths, tag=None, errorbar='ci'):
    all_data = []

    for sim_path in sim_paths:
        toxicity_dir = sim_path.replace("simulations", "output/toxicity")
        files = get_files(toxicity_dir, file_extension=".csv")

        for file in files:
            df = pd.read_csv(file, low_memory=False)
            df = df.copy()
            df["type"] = df["is_comment"].map({True: "comment", False: "post"})
            all_data.append(df[["toxicity", "coalition_src", "type"]])

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.dropna(subset=["toxicity", "coalition_src"])
    full_df["coalition_src"] = pd.Categorical(full_df["coalition_src"], categories=COALITIONS, ordered=True)

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=full_df,
        x="coalition_src",
        y="toxicity",
        hue="type",
        palette={"post": "steelblue", "comment": "darkorange"},
        showfliers=False,
    )
    ax.set_yscale('log')
    plt.ylabel("Toxicity", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Toxicity", fontsize=16)
    plt.xlabel("")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)

    plt.tight_layout()

    if SAVE:
        out_dir = "output/toxicity/plots"
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        plt.savefig(f"{out_dir}/box_posts_vs_comments{suffix}.png", dpi=300)

    if SHOW:
        plt.show()
    else:
        plt.close()

