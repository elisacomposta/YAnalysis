import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import seaborn as sns
from src.utils import COALITIONS, SAVE, SHOW, get_files


def plot_pos_neg_in_vs_out_boxplot(sim_path_list, tag=None):
    interaction_types_pos = ["like", "follow"]
    interaction_types_neg = ["dislike", "unfollow"]

    data_to_plot = {
        "positive": defaultdict(list),
        "negative": defaultdict(list)
    }

    for sim_path in sim_path_list:
        db_files = get_files(sim_path)

        for file in db_files:
            ref = sqlite3.connect(file)
            cursor = ref.cursor()

            cursor.execute("SELECT id, leaning FROM user_mgmt")
            user_coalition = {uid: coalition for uid, coalition in cursor.fetchall()}

            # interaction type -> src_coal -> tgt_coal -> count
            run_interactions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

            # Follow
            data = cursor.execute("SELECT user_id, action, follower_id FROM follow").fetchall()
            for user_id, action, target_id in data:
                src = user_coalition.get(user_id)
                tgt = user_coalition.get(target_id)
                if src and tgt:
                    run_interactions[action][src][tgt] += 1

            # Reactions
            data = cursor.execute("SELECT user_id, type, post_id FROM reactions").fetchall()
            for user_id, action, post_id in data:
                author_row = cursor.execute("SELECT user_id FROM post WHERE id = ?", (post_id,)).fetchone()
                if author_row:
                    post_author = author_row[0]
                    src = user_coalition.get(user_id)
                    tgt = user_coalition.get(post_author)
                    if src and tgt:
                        run_interactions[action][src][tgt] += 1

            ref.close()

            # Compute in-out ratios
            for interaction_group, types in [("positive", interaction_types_pos), ("negative", interaction_types_neg)]:
                for coal in COALITIONS:
                    in_count = 0
                    out_count = 0
                    for interaction_type in types:
                        in_count += run_interactions[interaction_type][coal].get(coal, 0)
                        out_count += sum(
                            count for tgt, count in run_interactions[interaction_type][coal].items() if tgt != coal
                        )
                    total = in_count + out_count
                    ratio = in_count / total if total > 0 else 0.0
                    data_to_plot[interaction_group][coal].append(ratio)

    # Dataframe
    rows = []
    for interaction_group in ["positive", "negative"]:
        for coal in COALITIONS:
            for value in data_to_plot[interaction_group][coal]:
                rows.append({
                    "Coalition": coal,
                    "Interaction": interaction_group,
                    "In Ratio": value
                })

    df = pd.DataFrame(rows)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Coalition", y="In Ratio", hue="Interaction", palette="Set2")
    plt.ylim(0, 1)
    plt.ylabel("Intra-coalition interaction ratio", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.xlabel(None)
    plt.tight_layout()

    if SAVE:
        path = f"output/interactions/box/"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/pos_neg_in{f'_{tag}' if tag else ''}.png"
        plt.savefig(save_path)
    
    if SHOW:
        plt.show()
    else:
        plt.close()


def plot_interactions_per_user_boxplot(sim_path_list, tag=None):
    interaction_types = ["post", "comment", "like", "dislike", "follow", "unfollow"]
    rows = []

    for sim_path in sim_path_list:
        db_files = get_files(sim_path)

        for file in db_files:
            ref = sqlite3.connect(file)
            cursor = ref.cursor()

            # user_id -> user_type
            cursor.execute("SELECT id, is_misinfo FROM user_mgmt")
            user_info = {uid: ("misinformer" if is_misinfo else "base") for uid, is_misinfo in cursor.fetchall()}

            # user_id -> type -> count
            user_interactions = defaultdict(lambda: defaultdict(int))

            # Reactions
            data = cursor.execute("SELECT user_id, type FROM reactions").fetchall()
            for user_id, action in data:
                if action in interaction_types:
                    user_interactions[user_id][action] += 1

            # Follows
            data = cursor.execute("SELECT user_id, action FROM follow").fetchall()
            for user_id, action in data:
                if action in interaction_types:
                    user_interactions[user_id][action] += 1

            # Comments
            data = cursor.execute("SELECT user_id FROM post WHERE comment_to != -1").fetchall()
            for user_id, in data:
                user_interactions[user_id]["comment"] += 1

            # Posts
            data = cursor.execute("SELECT user_id FROM post WHERE comment_to = -1").fetchall()
            for user_id, in data:
                user_interactions[user_id]["post"] += 1

            ref.close()

            # dataframe
            for user_id, counts in user_interactions.items():
                user_type = user_info[user_id]
                for interaction_type in interaction_types:
                    count = counts.get(interaction_type, 0)
                    rows.append({
                        "User Type": user_type,
                        "Interaction Type": interaction_type,
                        "Count": count
                    })

    # dataframe
    df = pd.DataFrame(rows)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Interaction Type", y="Count", hue="User Type", palette="Set2")
    plt.xlabel(None)
    plt.ylabel("Interaction Count", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if SAVE:
        path = "output/interactions/box/"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/count_per_user{f'_{tag}' if tag else ''}.png"
        plt.savefig(save_path)

    if SHOW:
        plt.show()
    else:
        plt.close()


def plot_in_group_ratio_by_recsys(sim_path_def, sim_path_rand, tag=None):
    interaction_types = ["comment", "like", "dislike", "follow"]
    rows = []

    for recsys_label, sim_paths in [("Default", sim_path_def), ("Random", sim_path_rand)]:
        for sim_path in sim_paths:
            db_files = get_files(sim_path)

            for file in db_files:
                ref = sqlite3.connect(file)
                cursor = ref.cursor()

                # user_id -> coalition
                cursor.execute("SELECT id, leaning FROM user_mgmt")
                user_coalition = {uid: coalition for uid, coalition in cursor.fetchall()}

                # interaction_type -> src -> tgt -> count
                interaction_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

                # Follows
                data = cursor.execute("SELECT user_id, action, follower_id FROM follow").fetchall()
                for user_id, action, target_id in data:
                    src = user_coalition.get(user_id)
                    tgt = user_coalition.get(target_id)
                    if src and tgt:
                        interaction_counts[action][src][tgt] += 1

                # Reactions
                data = cursor.execute("SELECT user_id, type, post_id FROM reactions").fetchall()
                for user_id, action, post_id in data:
                    author_row = cursor.execute("SELECT user_id FROM post WHERE id = ?", (post_id,)).fetchone()
                    if author_row:
                        post_author = author_row[0]
                        src = user_coalition.get(user_id)
                        tgt = user_coalition.get(post_author)
                        if src and tgt:
                            interaction_counts[action][src][tgt] += 1

                # Comments
                data = cursor.execute("SELECT user_id, comment_to FROM post WHERE comment_to != -1").fetchall()
                for user_id, comment_to in data:
                    author_row = cursor.execute("SELECT user_id FROM post WHERE id = ?", (comment_to,)).fetchone()
                    if author_row:
                        post_author = author_row[0]
                        src = user_coalition.get(user_id)
                        tgt = user_coalition.get(post_author)
                        if src and tgt:
                            interaction_counts["comment"][src][tgt] += 1

                # Posts
                data = cursor.execute("SELECT user_id FROM post WHERE comment_to = -1").fetchall()
                for (user_id,) in data:
                    src = user_coalition.get(user_id)
                    if src:
                        interaction_counts["post"][src][src] += 1  # assume post is self-directed (content origination)

                ref.close()

                # In-group ratio per ogni interaction_type
                for interaction_type in interaction_types:
                    for src in COALITIONS:
                        in_count = interaction_counts[interaction_type][src].get(src, 0)
                        out_count = sum(
                            count for tgt, count in interaction_counts[interaction_type][src].items() if tgt != src
                        )
                        total = in_count + out_count
                        ratio = in_count / total if total > 0 else 0.0
                        rows.append({
                            "Interaction Type": interaction_type,
                            "In-Group Ratio": ratio,
                            "Recsys": recsys_label
                        })

    df = pd.DataFrame(rows)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Interaction Type", y="In-Group Ratio", hue="Recsys", palette="Set2", showfliers=False)
    plt.ylim(0, 1)
    plt.ylabel("In-group interaction ratio", fontsize=16)
    plt.xlabel(None)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if SAVE:
        path = "output/interactions/box/"
        os.makedirs(path, exist_ok=True)
        save_path = f"{path}/recsys_in_group_ratio{f'_{tag}' if tag else ''}.png"
        plt.savefig(save_path)

    if SHOW:
        plt.show()
    else:
        plt.close()
