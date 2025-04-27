
"""
Simple Meta-Modal Resonance Experiment

This example demonstrates how to create and run a basic experiment with
the Meta-Modal Resonance Theory framework. It creates a single agent of
each type and exposes them to a simple environmental sequence.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.agents.meta_modal_agent import MetaModalAgent
from src.environment.context_generator import EnvironmentalContext
from src.metrics.integration_metrics import IntegrationMetrics
from src.visualization.meaning_structure_visualizer import MeaningStructureVisualizer


def create_agents():
    """Create one agent of each type for comparison."""
    agents = {}

    # Create a hedonic optimizer agent
    hedonic_agent = MetaModalAgent(
        agent_id="hedonic_agent",
        grid_size=30,
        oscillation_params={
            "transition_rates": {
                "hedonic_to_eudaimonic": 0.05,
                "hedonic_to_transcendent": 0.05,
                "eudaimonic_to_hedonic": 0.8,
                "eudaimonic_to_transcendent": 0.2,
                "transcendent_to_hedonic": 0.8,
                "transcendent_to_eudaimonic": 0.2
            },
            "modal_residence_times": {
                "hedonic": 15,
                "eudaimonic": 3,
                "transcendent": 2
            }
        },
        development_stage="intermediate"
    )
    agents["hedonic_agent"] = hedonic_agent

    # Create a eudaimonic optimizer agent
    eudaimonic_agent = MetaModalAgent(
        agent_id="eudaimonic_agent",
        grid_size=30,
        oscillation_params={
            "transition_rates": {
                "hedonic_to_eudaimonic": 0.8,
                "hedonic_to_transcendent": 0.2,
                "eudaimonic_to_hedonic": 0.05,
                "eudaimonic_to_transcendent": 0.05,
                "transcendent_to_hedonic": 0.2,
                "transcendent_to_eudaimonic": 0.8
            },
            "modal_residence_times": {
                "hedonic": 3,
                "eudaimonic": 15,
                "transcendent": 2
            }
        },
        development_stage="intermediate"
    )
    agents["eudaimonic_agent"] = eudaimonic_agent

    # Create a transcendent optimizer agent
    transcendent_agent = MetaModalAgent(
        agent_id="transcendent_agent",
        grid_size=30,
        oscillation_params={
            "transition_rates": {
                "hedonic_to_eudaimonic": 0.2,
                "hedonic_to_transcendent": 0.8,
                "eudaimonic_to_hedonic": 0.2,
                "eudaimonic_to_transcendent": 0.8,
                "transcendent_to_hedonic": 0.05,
                "transcendent_to_eudaimonic": 0.05
            },
            "modal_residence_times": {
                "hedonic": 2,
                "eudaimonic": 3,
                "transcendent": 15
            }
        },
        development_stage="intermediate"
    )
    agents["transcendent_agent"] = transcendent_agent

    # Create a meta-modal integrator agent
    integrator_agent = MetaModalAgent(
        agent_id="integrator_agent",
        grid_size=30,
        oscillation_params={
            "transition_rates": {
                "hedonic_to_eudaimonic": 0.2,
                "hedonic_to_transcendent": 0.2,
                "eudaimonic_to_hedonic": 0.2,
                "eudaimonic_to_transcendent": 0.2,
                "transcendent_to_hedonic": 0.2,
                "transcendent_to_eudaimonic": 0.2
            },
            "modal_residence_times": {
                "hedonic": 5,
                "eudaimonic": 5,
                "transcendent": 5
            }
        },
        development_stage="intermediate"
    )
    agents["integrator_agent"] = integrator_agent

    return agents


def run_simple_experiment(num_steps=50, output_dir="simple_experiment_results"):
    """
    Run a simple experiment with one agent of each type.

    Args:
        num_steps: Number of time steps to run
        output_dir: Directory for output files
    """
    print("Running simple Meta-Modal Resonance experiment...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create agents
    agents = create_agents()
    print(f"Created {len(agents)} agents: {', '.join(agents.keys())}")

    # Create environment
    environment = EnvironmentalContext(
        complexity=0.7,
        volatility=0.5,
        threat_level=0.3
    )
    print("Environment created with complexity=0.7, volatility=0.5")

    # Create metrics and visualizer
    metrics = IntegrationMetrics()
    visualizer = MeaningStructureVisualizer(style='default')

    # Run experiment
    agent_metrics_history = {agent_id: [] for agent_id in agents}

    for step in range(num_steps):
        # Get environment state
        if step == 0:
            env_state = environment.get_current_state()
        else:
            env_state = environment.update()

        # Create observation from environment
        observation = {
            "type": "environment",
            "intensity": 1.0,
            "time": env_state.get("time", 0),
            "threat_level": env_state.get("threat_level", 0.0),
            "growth_opportunity": env_state.get("growth_opportunity", 0.0),
            "connection_opportunity": env_state.get("connection_opportunity", 0.0),
            "cognitive_load": env_state.get("cognitive_load", 0.0)
        }

        # Process stimulus elements
        for stimulus in env_state.get("stimulus_elements", []):
            if stimulus.get("modality") == "hedonic":
                observation["pleasure_value"] = stimulus.get("pleasure_value", 0.0)
                observation["pain_value"] = stimulus.get("pain_value", 0.0)
            elif stimulus.get("modality") == "eudaimonic":
                observation["growth_value"] = stimulus.get("growth_value", 0.0)
                observation["purpose_value"] = stimulus.get("purpose_value", 0.0)
                observation["virtue_value"] = stimulus.get("virtue_value", 0.0)
            elif stimulus.get("modality") == "transcendent":
                observation["unity_value"] = stimulus.get("unity_value", 0.0)
                observation["connection_value"] = stimulus.get("connection_value", 0.0)
                observation["dissolution_value"] = stimulus.get("dissolution_value", 0.0)

        # Ensure at least small values for each modality
        if "pleasure_value" not in observation and "pain_value" not in observation:
            observation["pleasure_value"] = 0.1
            observation["pain_value"] = 0.1

        if "growth_value" not in observation and "purpose_value" not in observation and "virtue_value" not in observation:
            observation["growth_value"] = 0.1
            observation["purpose_value"] = 0.1
            observation["virtue_value"] = 0.1

        if "unity_value" not in observation and "connection_value" not in observation and "dissolution_value" not in observation:
            observation["unity_value"] = 0.1
            observation["connection_value"] = 0.1
            observation["dissolution_value"] = 0.1

        # Process observations for all agents
        for agent_id, agent in agents.items():
            # Process observation through agent
            response = agent.process_observation(observation)

            # Update metrics
            agent_metrics = metrics.update_all_metrics(
                agent.meaning_structure,
                [],  # No previous structures
                agent.meaning_structure.perturbation_responses if hasattr(agent.meaning_structure, 'perturbation_responses') else []
            )

            # Record metrics
            agent_metrics_history[agent_id].append(agent_metrics)

        # Print progress
        if step % 10 == 0:
            print(f"Completed step {step}/{num_steps}")

    print("Experiment completed!")

    # Generate visualizations
    print("Generating visualizations...")

    # Plot metrics history
    plot_metrics_history(agent_metrics_history, num_steps, output_dir)

    # Create summary visualizations for each agent
    for agent_id, agent in agents.items():
        # Create and save summary visualization
        fig = visualizer.create_summary_visualization(agent.meaning_structure)
        fig.savefig(os.path.join(output_dir, f"{agent_id}_summary.png"))
        plt.close(fig)

        print(f"Created summary visualization for {agent_id}")

    print(f"Results saved to {output_dir}")


def plot_metrics_history(agent_metrics_history, num_steps, output_dir):
    """
    Plot the evolution of metrics over time for different agent types.

    Args:
        agent_metrics_history: Dict mapping agent_id to list of metrics dicts
        num_steps: Number of time steps in the experiment
        output_dir: Directory for output files
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for different agent types
    colors = {
        "hedonic": "#FF9500",    # Orange
        "eudaimonic": "#4CAF50", # Green
        "transcendent": "#2196F3", # Blue
        "integrator": "#9C27B0"  # Purple
    }

    # Plot meaning resilience over time for each agent
    for agent_id, metrics_history in agent_metrics_history.items():
        # Get agent type from id
        agent_type = agent_id.split('_')[0]

        # Extract resilience values
        resilience_values = [metrics.get('meaning_resilience', 0.0) for metrics in metrics_history]

        # Plot
        ax.plot(range(len(resilience_values)), resilience_values,
               label=f"{agent_id}",
               color=colors.get(agent_type, 'gray'),
               linewidth=2)

    # Add labels and legend
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Meaning Resilience")
    ax.set_title("Meaning Resilience Over Time")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Save figure
    fig.savefig(os.path.join(output_dir, "meaning_resilience_over_time.png"))
    plt.close(fig)

    # Create separate plot for each metric
    metrics_to_plot = ['narrative_coherence', 'identity_stability', 'modal_balance']

    for metric in metrics_to_plot:
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        for agent_id, metrics_history in agent_metrics_history.items():
            # Get agent type from id
            agent_type = agent_id.split('_')[0]

            # Extract metric values
            metric_values = [metrics.get(metric, 0.0) for metrics in metrics_history]

            # Plot
            ax.plot(range(len(metric_values)), metric_values,
                   label=f"{agent_id}",
                   color=colors.get(agent_type, 'gray'),
                   linewidth=2)

        # Add labels and legend
        ax.set_xlabel("Time Step")
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f"{metric.replace('_', ' ').title()} Over Time")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

        # Save figure
        fig.savefig(os.path.join(output_dir, f"{metric}_over_time.png"))
        plt.close(fig)


if __name__ == "__main__":
    run_simple_experiment()
