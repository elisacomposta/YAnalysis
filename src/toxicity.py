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


def plot_combined_toxicity_diff(sim_paths, real_world_path=None, tag=None):
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
    if real_world_path is not None:
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

    if real_world_path is not None:
        sns.kdeplot(real_means["toxicity_diff"], label="Real world", color="darkorange", linewidth=2)

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

def plot_toxicity_in_out_ridge(sim_paths, real_world_path=None, tag=None):
    all_data_sim = []

    # Simulated data
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
            df["source"] = "Simulations"
            all_data_sim.append(df)

    sim_df = pd.concat(all_data_sim, ignore_index=True)

    if real_world_path is not None:
        real_df = pd.read_csv(real_world_path)
        real_df["direction"] = real_df.apply(
            lambda r: "in" if r["from_coalition"] == r["target_coalition"] else "out",
            axis=1
        )
        real_df["toxicity_log"] = np.log1p(real_df["toxicity"])
        real_df["source"] = "Real world"
        full_df = pd.concat([sim_df, real_df], ignore_index=True)
    else:
        full_df = sim_df

    # Parametri per KDE
    x_range = np.linspace(0, full_df["toxicity_log"].max(), 200)
    sources = ["Simulations", "Real world"] if real_world_path is not None else ["Simulations"]
    directions = ["in", "out"]
    colors = {
        "Simulations": "steelblue",
        "Real world": "darkorange"
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    spacing = 1.2

    for i, direction in enumerate(directions):
        y_base = (len(directions) - i - 1) * spacing

        for j, source in enumerate(sources):
            subset = full_df[(full_df["direction"] == direction) & (full_df["source"] == source)]
            if len(subset) < 2:
                continue
            values = subset["toxicity_log"].dropna()
            kde = gaussian_kde(values, bw_method=0.3)
            y_vals = kde(x_range)
            y_vals = y_vals / y_vals.max() * 0.8  # normalizza altezza

            ax.plot(
                x_range,
                y_vals + y_base,
                label=source if i == 0 else "",
                color=colors[source],
                linewidth=2,
                alpha=0.9
            )

        ax.axhline(y=y_base, color="gray", linestyle=":", linewidth=0.5)

    ax.set_yticks([i * spacing for i in range(len(directions))])
    ax.set_yticklabels(directions[::-1], fontsize=14)
    ax.set_xlabel("Log(toxicity+1)", fontsize=14)
    ax.set_xlim(0, x_range.max())
    ax.set_xticks(np.round(np.linspace(0, x_range.max(), 5), 2))
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=13, loc="upper right")

    plt.tight_layout()

    if SAVE:
        out_dir = "output/toxicity/plots"
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        plt.savefig(f"{out_dir}/toxicity_in_out_ridge{suffix}.png", dpi=300)

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


def plot_toxicity_post_comment_ridge(sim_paths, tag=None):
    all_data = []

    for sim_path in sim_paths:
        toxicity_dir = sim_path.replace("simulations", "output/toxicity")
        files = get_files(toxicity_dir, file_extension=".csv")

        for file in files:
            df = pd.read_csv(file, low_memory=False)
            df = df.copy()
            df["type"] = df["is_comment"].map({True: "comment", False: "post"})
            df["toxicity_log"] = np.log1p(df["toxicity"])
            all_data.append(df[["toxicity_log", "coalition_src", "type"]])

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.dropna(subset=["toxicity_log", "coalition_src"])
    full_df["coalition_src"] = pd.Categorical(full_df["coalition_src"], categories=COALITIONS, ordered=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    x_range = np.linspace(0, full_df["toxicity_log"].max(), 300)
    types = ["post", "comment"]
    colors = {"post": "steelblue", "comment": "darkorange"}
    spacing = 1.2

    for i, coalition in enumerate(COALITIONS[::-1]): 
        y_base = i * spacing

        for content_type in types:
            subset = full_df[(full_df["coalition_src"] == coalition) & (full_df["type"] == content_type)]
            values = subset["toxicity_log"].dropna()

            if len(values) <= 1 or np.std(values) < 1e-6:
                continue

            kde = gaussian_kde(values, bw_method=0.4)
            y_vals = kde(x_range)
            y_vals = y_vals / y_vals.max() * 0.8  # normalize height

            ax.plot(
                x_range,
                y_vals + y_base,
                label=content_type if i == 0 else None,
                color=colors[content_type],
                linewidth=2,
                alpha=0.9
            )

        ax.axhline(y=y_base, color='gray', linestyle=':', linewidth=0.5)

    # Ticks e labels
    ax.set_yticks([i * spacing for i in range(len(COALITIONS))])
    ax.set_yticklabels(COALITIONS[::-1], fontsize=14)
    ax.set_xlabel("Log(Toxicity + 1)", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # Legenda
    ax.legend(title="Content Type", fontsize=13, title_fontsize=14, loc="upper right")

    plt.tight_layout()

    if SAVE:
        out_dir = "output/toxicity/plots"
        os.makedirs(out_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        plt.savefig(f"{out_dir}/ridge_posts_vs_comments{suffix}.png", dpi=300)

    if SHOW:
        plt.show()
    else:
        plt.close(fig)

