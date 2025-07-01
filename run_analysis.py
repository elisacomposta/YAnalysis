import os
from tqdm import tqdm
from src.utils import COALITIONS, get_random_file_from_dir
from src.graph import plot_final_graph, plot_population_composition
from src.opinions import plot_op_score_by_coalition
from src.misinformation import plot_shift_by_misinfo_level_runs, plot_shift_by_misinfo_level_coalitions
from src.interactions import plot_in_group_ratio_by_recsys, plot_pos_neg_in_vs_out_boxplot, plot_interactions_per_user_boxplot
from src.toxicity import extract_all_toxicity, plot_combined_toxicity_diff, plot_toxicity_post_comment

RECSYS_RANDOM_PATH = 'simulations/ContentRecSys'
RECCSYS_DEFAULT_PATH = 'simulations/ReverseChronoFollowersPopularity'

sim_names_recsys_random = ['d21a100m00x', 'd21a100m05x', 'd21a100m10x', 'd21a100m50x']   
sim_names_recsys_default = ['d21a100m00d', 'd21a100m05d', 'd21a100m10d', 'd21a100m50d']   

sim_paths_recsys_random = [os.path.join(RECSYS_RANDOM_PATH, sim) for sim in sim_names_recsys_random]
sim_paths_recsys_default = [os.path.join(RECCSYS_DEFAULT_PATH, sim) for sim in sim_names_recsys_default]

if __name__ == "__main__":
    sim_paths = [sim_paths_recsys_random, sim_paths_recsys_default]
    tags = ['RandomRecSys', 'DefaultRecSys']

    #extract_all_toxicity()
    
    for sim_path_recsys, tag in zip(sim_paths, tags):
        print(f"\n\n** Running analysis for {tag} simulations **")
        
        # Graph
        print(f"\nPlotting final graph for {tag} simulations")
        simulations_to_plot = []
        for sim_path in sim_path_recsys:
            simulations_to_plot.append(get_random_file_from_dir(sim_path))
        plot_final_graph(simulations_to_plot, tag=tag)
        
        print("\nPlotting population composition")
        plot_population_composition(sim_path_recsys, tag=tag)
        
        # Opinions
        print("\nPlotting opinion scores by coalition")
        for idx, sim_path in tqdm(enumerate(sim_path_recsys), desc="Opinions", total=len(sim_path_recsys)):
            plot_op_score_by_coalition(sim_path, tag=tag)
        
        # Coalition interactions
        print(f"\nPlotting coalition interactions - all interactions")
        plot_pos_neg_in_vs_out_boxplot(sim_path_recsys, tag=tag)
        plot_interactions_per_user_boxplot(sim_path_recsys, tag=tag)

        # Misinformation
        print(f"\nPlotting misinformation impact on opinion shift by coalition - ridge plots")
        for coalition in tqdm(COALITIONS, desc="Coalitions", total=len(COALITIONS)):
            plot_shift_by_misinfo_level_runs(sim_path_recsys, coalition=coalition, tag=tag)
        
        print("\nPlotting opinion shift by misinformation level - ridge plots by coalition")
        plot_shift_by_misinfo_level_coalitions(sim_path_recsys, score_llm=True, tag=tag)

    # Toxicity analysis
    print("\nPlotting toxicity analysis")
    plot_toxicity_post_comment(sim_paths_recsys_random + sim_paths_recsys_default)

    comment_files = "real_data/reply_edges_between_users_with_coalition.csv"
    print("\nPlotting toxicity differences in/out for real-world dataset")
    plot_combined_toxicity_diff(sim_paths_recsys_random + sim_paths_recsys_default, comment_files)

    # RecSys comparison
    print(f"\nPlotting in-group ratio by recommender system")
    plot_in_group_ratio_by_recsys(sim_paths_recsys_default, sim_paths_recsys_random)