"""
Modal Comparison Experiment

This experiment compares meaning construction in different agent types:
1. Hedonic optimizers - Primarily focus on hedonic processing
2. Eudaimonic optimizers - Primarily focus on eudaimonic processing
3. Transcendent optimizers - Primarily focus on transcendent processing
4. Meta-modal integrators - Dynamically integrate across modalities

The experiment exposes all agent types to identical environmental sequences
and measures differences in meaning structure qualities.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import time
import argparse
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.agents.meta_modal_agent import MetaModalAgent
from src.environment.context_generator import EnvironmentalContext
from src.metrics.integration_metrics import IntegrationMetrics
from src.visualization.meaning_structure_visualizer import MeaningStructureVisualizer


class ModalComparisonExperiment:
    """
    Experiment comparing meaning construction across different agent types.
    
    This experiment tests the hypothesis that agents capable of dynamic
    integration across modalities will develop more coherent and resilient
    meaning structures than agents primarily operating in a single modality.
    """
    
    def __init__(self, 
                num_agents_per_type: int = 5,
                environment_complexity: float = 0.7,
                environment_volatility: float = 0.5,
                experiment_duration: int = 100,
                perturbation_schedule: Optional[List[Dict[str, Any]]] = None,
                output_dir: str = "results"):
        """
        Initialize the experiment.
        
        Args:
            num_agents_per_type: Number of agents to create for each agent type
            environment_complexity: Complexity level of the environment
            environment_volatility: Volatility level of the environment
            experiment_duration: Number of time steps to run the experiment
            perturbation_schedule: Optional schedule of perturbations to apply
            output_dir: Directory for output files
        """
        self.num_agents_per_type = num_agents_per_type
        self.environment_complexity = environment_complexity
        self.environment_volatility = environment_volatility
        self.experiment_duration = experiment_duration
        self.output_dir = output_dir
        
        # Set up output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up perturbation schedule
        if perturbation_schedule is None:
            # Default schedule: apply perturbations at regular intervals
            self.perturbation_schedule = []
            
            # Add various perturbation types at different times
            perturbation_types = [
                "narrative_disruption",
                "identity_disruption",
                "value_disruption",
                "meta_cognitive_disruption",
                "random_disruption"
            ]
            
            # Schedule perturbations at regular intervals
            interval = experiment_duration // (len(perturbation_types) + 1)
            for i, p_type in enumerate(perturbation_types):
                time_step = (i + 1) * interval
                self.perturbation_schedule.append({
                    "time_step": time_step,
                    "type": p_type,
                    "intensity": 0.7,
                    "target_components": None
                })
        else:
            self.perturbation_schedule = perturbation_schedule
        
        # Initialize agents, environment, and metrics
        self.initialize_experiment()
    
    def initialize_experiment(self) -> None:
        """Initialize the experiment components."""
        # Create environment
        self.environment = EnvironmentalContext(
            complexity=self.environment_complexity,
            volatility=self.environment_volatility,
            threat_level=0.3
        )
        
        # Create metrics calculator
        self.metrics = IntegrationMetrics()
        
        # Create visualizer
        self.visualizer = MeaningStructureVisualizer(style='default')
        
        # Create agents
        self.agents = self.create_agents()
        
        # Track metrics over time
        self.metrics_history = {agent_id: [] for agent_id in self.agents}
        self.environment_history = []
        
        # Initialize experiment state
        self.current_step = 0
        self.experiment_completed = False
    
    def create_agents(self) -> Dict[str, MetaModalAgent]:
        """
        Create the different types of agents for the experiment.
        
        Returns:
            Dictionary mapping agent IDs to agent objects
        """
        agents = {}
        
        # Create hedonic optimizer agents
        for i in range(self.num_agents_per_type):
            agent_id = f"hedonic_{i+1}"
            
            # Use oscillation parameters that heavily favor hedonic processing
            oscillation_params = {
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
            }
            
            # Create agent
            agent = MetaModalAgent(
                agent_id=agent_id,
                grid_size=30,
                oscillation_params=oscillation_params,
                development_stage="intermediate"
            )
            
            agents[agent_id] = agent
        
        # Create eudaimonic optimizer agents
        for i in range(self.num_agents_per_type):
            agent_id = f"eudaimonic_{i+1}"
            
            # Use oscillation parameters that heavily favor eudaimonic processing
            oscillation_params = {
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
            }
            
            # Create agent
            agent = MetaModalAgent(
                agent_id=agent_id,
                grid_size=30,
                oscillation_params=oscillation_params,
                development_stage="intermediate"
            )
            
            agents[agent_id] = agent
        
        # Create transcendent optimizer agents
        for i in range(self.num_agents_per_type):
            agent_id = f"transcendent_{i+1}"
            
            # Use oscillation parameters that heavily favor transcendent processing
            oscillation_params = {
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
            }
            
            # Create agent
            agent = MetaModalAgent(
                agent_id=agent_id,
                grid_size=30,
                oscillation_params=oscillation_params,
                development_stage="intermediate"
            )
            
            agents[agent_id] = agent
        
        # Create meta-modal integrator agents
        for i in range(self.num_agents_per_type):
            agent_id = f"integrator_{i+1}"
            
            # Use balanced oscillation parameters that enable adaptive integration
            oscillation_params = {
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
            }
            
            # Create agent
            agent = MetaModalAgent(
                agent_id=agent_id,
                grid_size=30,
                oscillation_params=oscillation_params,
                development_stage="intermediate"
            )
            
            agents[agent_id] = agent
        
        return agents
    
    def run_single_step(self) -> Dict[str, Any]:
        """
        Run a single step of the experiment.
        
        Returns:
            Dictionary with step results
        """
        # Check if experiment already completed
        if self.experiment_completed:
            return {"status": "completed", "message": "Experiment already completed"}
        
        # Generate environmental state
        if self.current_step == 0:
            # First step - use initial state
            env_state = self.environment.get_current_state()
        else:
            # Update environment
            env_state = self.environment.update()
        
        # Store environment state
        self.environment_history.append(env_state)
        
        # Process observations for all agents
        responses = {}
        
        for agent_id, agent in self.agents.items():
            # Convert environment state to observation
            observation = self.environment_to_observation(env_state)
            
            # Process observation through agent
            response = agent.process_observation(observation)
            
            # Store response
            responses[agent_id] = response
            
            # Update metrics
            metrics = self.metrics.update_all_metrics(
                agent.meaning_structure,
                [],  # No previous structures yet
                agent.meaning_structure.perturbation_responses if hasattr(agent.meaning_structure, 'perturbation_responses') else []
            )
            
            # Store metrics history
            self.metrics_history[agent_id].append(metrics.copy())
        
        # Check for scheduled perturbations
        for perturbation in self.perturbation_schedule:
            if perturbation.get("time_step") == self.current_step:
                self.apply_perturbation(perturbation)
        
        # Increment step counter
        self.current_step += 1
        
        # Check if experiment is complete
        if self.current_step >= self.experiment_duration:
            self.experiment_completed = True
            self.generate_experiment_summary()
        
        # Return step results
        return {
            "status": "success",
            "step": self.current_step,
            "environment": env_state,
            "responses": responses,
            "completed": self.experiment_completed
        }
    
    def environment_to_observation(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert environment state to agent observation.
        
        Args:
            env_state: Environmental state
            
        Returns:
            Observation for agent processing
        """
        # Extract relevant components from environment
        observation = {
            "type": "environment",
            "intensity": 1.0,
            "time": env_state.get("time", 0)
        }
        
        # Add context variables
        observation["threat_level"] = env_state.get("threat_level", 0.0)
        observation["growth_opportunity"] = env_state.get("growth_opportunity", 0.0)
        observation["connection_opportunity"] = env_state.get("connection_opportunity", 0.0)
        observation["cognitive_load"] = env_state.get("cognitive_load", 0.0)
        
        # Check for significant event
        if env_state.get("is_significant", False):
            observation["is_significant"] = True
            observation["event_type"] = env_state.get("event_type", "unknown")
            observation["event_description"] = env_state.get("event_description", "Significant event")
        
        # Process stimulus elements
        stimulus_elements = env_state.get("stimulus_elements", [])
        
        # Add modality-specific values based on stimuli
        hedonic_total = 0.0
        eudaimonic_total = 0.0
        transcendent_total = 0.0
        
        for stimulus in stimulus_elements:
            stim_type = stimulus.get("type", "neutral")
            modality = stimulus.get("modality", "neutral")
            intensity = stimulus.get("intensity", 0.5)
            
            if modality == "hedonic":
                hedonic_total += intensity
                observation["pleasure_value"] = stimulus.get("pleasure_value", 0.0)
                observation["pain_value"] = stimulus.get("pain_value", 0.0)
            elif modality == "eudaimonic":
                eudaimonic_total += intensity
                observation["growth_value"] = stimulus.get("growth_value", 0.0)
                observation["purpose_value"] = stimulus.get("purpose_value", 0.0)
                observation["virtue_value"] = stimulus.get("virtue_value", 0.0)
            elif modality == "transcendent":
                transcendent_total += intensity
                observation["unity_value"] = stimulus.get("unity_value", 0.0)
                observation["connection_value"] = stimulus.get("connection_value", 0.0)
                observation["dissolution_value"] = stimulus.get("dissolution_value", 0.0)
        
        # Ensure at least small values for each modality to avoid null processing
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
        
        return observation
    
    def apply_perturbation(self, perturbation: Dict[str, Any]) -> None:
        """
        Apply a perturbation to all agents.
        
        Args:
            perturbation: Perturbation specification
        """
        perturbation_type = perturbation.get("type", "random_disruption")
        intensity = perturbation.get("intensity", 0.5)
        target_components = perturbation.get("target_components", None)
        
        print(f"Applying {perturbation_type} perturbation with intensity {intensity} at time step {self.current_step}")
        
        # Apply to all agents
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'meaning_structure') and hasattr(agent.meaning_structure, 'apply_perturbation'):
                agent.meaning_structure.apply_perturbation(
                    perturbation_type=perturbation_type,
                    intensity=intensity,
                    target_components=target_components
                )
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run the entire experiment from start to finish.
        
        Returns:
            Dictionary with experiment results
        """
        print(f"Starting Modal Comparison Experiment with {len(self.agents)} agents")
        print(f"Environment: complexity={self.environment_complexity}, volatility={self.environment_volatility}")
        print(f"Duration: {self.experiment_duration} time steps")
        print(f"Perturbation schedule: {len(self.perturbation_schedule)} perturbations scheduled")
        
        start_time = time.time()
        
        # Run all steps
        while not self.experiment_completed:
            step_result = self.run_single_step()
            
            # Print progress at regular intervals
            if self.current_step % 10 == 0:
                print(f"  Step {self.current_step}/{self.experiment_duration} completed")
        
        end_time = time.time()
        run_time = end_time - start_time
        
        print(f"Experiment completed in {run_time:.2f} seconds")
        
        # Return final results
        return self.generate_experiment_summary()
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of the experiment results.
        
        Returns:
            Dictionary with experiment summary
        """
        # Calculate final metrics for all agents
        final_metrics = {}
        
        for agent_id, agent in self.agents.items():
            # Get final metrics
            metrics = self.metrics.update_all_metrics(
                agent.meaning_structure,
                [],  # No previous structures for now
                agent.meaning_structure.perturbation_responses if hasattr(agent.meaning_structure, 'perturbation_responses') else []
            )
            
            final_metrics[agent_id] = metrics
        
        # Group metrics by agent type
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        metrics_by_type = {agent_type: {} for agent_type in agent_types}
        
        for agent_id, metrics in final_metrics.items():
            for agent_type in agent_types:
                if agent_id.startswith(agent_type):
                    # Add to group
                    metrics_by_type[agent_type][agent_id] = metrics
        
        # Calculate average metrics for each agent type
        avg_metrics_by_type = {}
        
        for agent_type, agent_metrics in metrics_by_type.items():
            # Skip if no agents of this type
            if not agent_metrics:
                continue
                
            # Calculate averages
            avg_metrics = {}
            
            for metric in self.metrics.metrics:
                values = [m.get(metric, 0.0) for m in agent_metrics.values()]
                avg_metrics[metric] = sum(values) / len(values)
            
            avg_metrics_by_type[agent_type] = avg_metrics
        
        # Create summary
        summary = {
            "experiment_params": {
                "num_agents_per_type": self.num_agents_per_type,
                "environment_complexity": self.environment_complexity,
                "environment_volatility": self.environment_volatility,
                "experiment_duration": self.experiment_duration,
                "perturbation_schedule": self.perturbation_schedule
            },
            "final_metrics": final_metrics,
            "metrics_by_type": metrics_by_type,
            "avg_metrics_by_type": avg_metrics_by_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, "experiment_summary.json")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualizations
        self.generate_result_visualizations()
        
        return summary
    
    def generate_result_visualizations(self) -> None:
        """Generate visualizations of the experiment results."""
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Group agents by type
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        agents_by_type = {agent_type: [] for agent_type in agent_types}
        
        for agent_id, agent in self.agents.items():
            for agent_type in agent_types:
                if agent_id.startswith(agent_type):
                    # Add to group
                    agents_by_type[agent_type].append(agent)
        
        # 1. Agent type comparison visualization
        self.plot_agent_type_comparison(agents_by_type, vis_dir)
        
        # 2. Metrics over time visualization
        self.plot_metrics_over_time(vis_dir)
        
        # 3. Individual agent visualizations (one per type)
        for agent_type, agents in agents_by_type.items():
            if agents:
                # Take first agent of each type
                agent = agents[0]
                
                # Create and save visualizations
                fig = self.visualizer.create_summary_visualization(agent.meaning_structure)
                fig.savefig(os.path.join(vis_dir, f"{agent_type}_summary.png"))
                plt.close(fig)
                
                # Create narrative timeline
                fig, ax = self.visualizer.plot_narrative_timeline(agent.meaning_structure)
                fig.savefig(os.path.join(vis_dir, f"{agent_type}_narrative.png"))
                plt.close(fig)
                
                # Create modal balance
                fig, ax = self.visualizer.plot_modal_balance(agent.meaning_structure, time_series=True)
                fig.savefig(os.path.join(vis_dir, f"{agent_type}_modal_balance.png"))
                plt.close(fig)
                
                # Create perturbation response if available
                if hasattr(agent.meaning_structure, 'perturbation_responses') and agent.meaning_structure.perturbation_responses:
                    fig, ax = self.visualizer.plot_perturbation_response(agent.meaning_structure)
                    fig.savefig(os.path.join(vis_dir, f"{agent_type}_perturbation.png"))
                    plt.close(fig)
    
    def plot_agent_type_comparison(self, agents_by_type: Dict[str, List[MetaModalAgent]], 
                                  vis_dir: str) -> None:
        """
        Plot comparison of agent types.
        
        Args:
            agents_by_type: Dictionary mapping agent types to agent lists
            vis_dir: Output directory for visualizations
        """
        # Extract metrics for each agent type
        metrics_by_type = {}
        
        for agent_type, agents in agents_by_type.items():
            if not agents:
                continue
                
            # Average metrics across agents of this type
            type_metrics = {}
            
            for metric in self.metrics.metrics:
                values = []
                for agent in agents:
                    if hasattr(agent.meaning_structure, metric):
                        values.append(getattr(agent.meaning_structure, metric))
                    else:
                        # Use metrics from our calculator
                        calculated_metrics = self.metrics.update_all_metrics(agent.meaning_structure)
                        values.append(calculated_metrics.get(metric, 0.0))
                
                if values:
                    type_metrics[metric] = sum(values) / len(values)
                else:
                    type_metrics[metric] = 0.0
            
            metrics_by_type[agent_type] = type_metrics
        
        # Create bar chart comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Choose metrics to show
        display_metrics = [
            "narrative_coherence",
            "identity_stability",
            "modal_balance",
            "meaning_resilience",
            "temporal_integration"
        ]
        
        # Set bar positions
        bar_width = 0.15
        r1 = np.arange(len(display_metrics))
        
        # Plot bars for each agent type
        colors = {
            "hedonic": "#FF9500",    # Orange
            "eudaimonic": "#4CAF50", # Green
            "transcendent": "#2196F3", # Blue
            "integrator": "#9C27B0"  # Purple
        }
        
        # Plot each agent type
        for i, (agent_type, metrics) in enumerate(metrics_by_type.items()):
            values = [metrics.get(metric, 0.0) for metric in display_metrics]
            position = [x + bar_width * i for x in r1]
            
            ax.bar(position, values, width=bar_width, label=agent_type.title(), 
                  color=colors.get(agent_type, 'gray'), alpha=0.7)
        
        # Add labels and legend
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Comparison of Agent Types")
        ax.set_xticks([r + bar_width * (len(metrics_by_type) - 1) / 2 for r in r1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in display_metrics])
        ax.legend()
        
        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fig.tight_layout()
        
        # Save figure
        fig.savefig(os.path.join(vis_dir, "agent_type_comparison.png"))
        plt.close(fig)
    
    def plot_metrics_over_time(self, vis_dir: str) -> None:
        """
        Plot metrics evolution over time.
        
        Args:
            vis_dir: Output directory for visualizations
        """
        # Extract metric histories
        metric_to_plot = "meaning_resilience"
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot average for each agent type
        for agent_type in agent_types:
            # Get all agents of this type
            type_agents = [agent_id for agent_id in self.agents if agent_id.startswith(agent_type)]
            
            if not type_agents:
                continue
                
            # Extract values over time
            values_over_time = []
            
            for step in range(self.current_step):
                step_values = []
                
                for agent_id in type_agents:
                    if step < len(self.metrics_history[agent_id]):
                        step_values.append(self.metrics_history[agent_id][step].get(metric_to_plot, 0.0))
                
                if step_values:
                    values_over_time.append(sum(step_values) / len(step_values))
                else:
                    values_over_time.append(0.0)
            
            # Plot with appropriate color
            color = {
                "hedonic": "#FF9500",    # Orange
                "eudaimonic": "#4CAF50", # Green
                "transcendent": "#2196F3", # Blue
                "integrator": "#9C27B0"  # Purple
            }.get(agent_type, 'gray')
            
            ax.plot(range(len(values_over_time)), values_over_time, 
                   label=agent_type.title(), color=color, linewidth=2)
        
        # Mark perturbations
        for perturbation in self.perturbation_schedule:
            time_step = perturbation.get("time_step", 0)
            if time_step < self.current_step:
                ax.axvline(x=time_step, color='red', linestyle='--', alpha=0.5)
                ax.text(time_step, 0.1, perturbation.get("type", "").replace("_", " ").title(), 
                       rotation=90, verticalalignment='bottom', fontsize=8)
        
        # Add labels and legend
        ax.set_xlabel("Time Step")
        ax.set_ylabel(metric_to_plot.replace('_', ' ').title())
        ax.set_title(f"{metric_to_plot.replace('_', ' ').title()} Over Time")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        fig.savefig(os.path.join(vis_dir, f"{metric_to_plot}_over_time.png"))
        plt.close(fig)


def main():
    """Run the experiment from command line."""
    parser = argparse.ArgumentParser(description='Run Modal Comparison Experiment')
    
    parser.add_argument('--agents-per-type', type=int, default=3,
                       help='Number of agents per agent type')
    parser.add_argument('--duration', type=int, default=100,
                       help='Experiment duration in time steps')
    parser.add_argument('--complexity', type=float, default=0.7,
                       help='Environment complexity (0.0-1.0)')
    parser.add_argument('--volatility', type=float, default=0.5,
                       help='Environment volatility (0.0-1.0)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = ModalComparisonExperiment(
        num_agents_per_type=args.agents_per_type,
        environment_complexity=args.complexity,
        environment_volatility=args.volatility,
        experiment_duration=args.duration,
        output_dir=args.output_dir
    )
    
    # Run experiment
    experiment.run_full_experiment()
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
