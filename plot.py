import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Define folders and experiment types
folders = {"Connect4d": 'connect4d', "Strands": 'strands'}
types = ['-mcts-vanilla',  '-mcts-bb', '-emcts-vanilla', '-emcts-bb']
sns.set_style("darkgrid")
colors = ["blue", "blue", "red", "red"]
linestyles = ['-', ':', '-', ':']
labels = ['MCTS-Vanilla', 'MCTS-BB', 'EMCTS-Vanilla', 'EMCTS-BB']

for game, experiment in folders.items():
    plt.figure(figsize=(10, 6))
    plt.clf()  
    ax = plt.gca()

    color_patch = []

    # Loop through each experiment type
    for exp_type, color, linestyle, label in zip(types, colors, linestyles, labels):
        event_file_path = f'experiments/{experiment}/{experiment}{exp_type}'
        event_acc = EventAccumulator(event_file_path)
        event_acc.Reload()  # Load events

        # Check available scalar tags
        scalar_tags = event_acc.Tags().get('scalars', [])
        
        # Identify the correct tag for 'baseline'
        target_tag = None
        for tag in scalar_tags:
            if 'baseline' in tag:  
                target_tag = tag
                break
        

        if target_tag:
            scalar_events = event_acc.Scalars(target_tag)

            # Extract step and value for the tag
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]

            # Plot using seaborn for better aesthetics
            sns.lineplot(x=steps, y=values, label=label, color=color, marker="o", linestyle=linestyle, linewidth=5)

        else:
            print(f"Tag '{target_tag}' not found in {event_file_path}")

    # Add labels, legend, and grid to the plot
    plt.xlabel('Iteration', fontsize=18)
    plt.ylabel('Win Rate', fontsize=18)
    plt.title(f'{game}: Win Rate Against Baseline', fontsize=20)
    
    # Add a legend with the custom colors and labels
    plt.legend(title="Variation", loc="lower right", 
               prop={'size': 20}, facecolor='white', edgecolor='black', framealpha=1)
    plt.grid(True)

    # Customize ticks
    ax.set_xticks([0, 10, 20, 30, 40, 50]) 
    ax.set_xlim(0, 50) 
    ax.set_ylim(0.33, 1) 
    ax.set_xticklabels(labels = [0, 10, 20, 30, 40, 50], fontsize=16)  # Adjust based on your time scale
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=16)

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Save the plot to the corresponding directory
    save_path = f'experiments/{experiment}_learning_curves.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Close the plot to avoid overlap
    plt.close()
