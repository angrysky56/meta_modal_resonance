"""
Perturbation Resilience Testing Experiment

This experiment tests how different types of meaning structures respond to various 
perturbations. It examines the hypothesis that integrated meaning structures will
demonstrate greater stability under perturbation than structures derived primarily
from single modalities.

The experiment subjects agents to a series of controlled perturbations and measures
their recovery trajectories across multiple metrics.
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


class ResilienceTestingExperiment:
    """
    Experiment testing meaning structure resilience to perturbations.
    
    This experiment tests the hypothesis that integrated meaning structures will
    demonstrate greater stability and faster recovery under perturbation than
    structures derived primarily from single modalities.
    """
    
    def __init__(self,
                num_agents_per_type: int = 3,
                stabilization_duration: int = 50,
                perturbation_types: Optional[List[str]] = None,
                perturbation_intensities: Optional[List[float]] = None,
                recovery_duration: int = 30,
                environment_complexity: float = 0.5,
                output_dir: str = "results_resilience"):
        """
        Initialize the experiment.
        
        Args:
            num_agents_per_type: Number of agents to create for each agent type
            stabilization_duration: Number of steps for initial stabilization
            perturbation_types: Types of perturbations to test
            perturbation_intensities: Intensities of perturbations to test
            recovery_duration: Steps to allow for recovery after perturbation
            environment_complexity: Complexity level of the environment
            output_dir: Directory for output files
        """
        self.num_agents_per_type = num_agents_per_type
        self.stabilization_duration = stabilization_duration
        self.recovery_duration = recovery_duration
        self.environment_complexity = environment_complexity
        self.output_dir = output_dir
        
        # Set up perturbation types if not provided
        if perturbation_types is None:
            self.perturbation_types = [
                "narrative_disruption",
                "identity_disruption",
                "value_disruption",
                "meta_cognitive_disruption",
                "random_disruption"
            ]
        else:
            self.perturbation_types = perturbation_types
            
        # Set up perturbation intensities if not provided
        if perturbation_intensities is None:
            self.perturbation_intensities = [0.3, 0.6, 0.9]
        else:
            self.perturbation_intensities = perturbation_intensities
            
        # Calculate total experiment duration
        self.experiment_duration = (
            self.stabilization_duration + 
            len(self.perturbation_types) * len(self.perturbation_intensities) * (self.recovery_duration + 1)
        )
        
        # Set up output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize experiment components
        self.initialize_experiment()
    
    def initialize_experiment(self) -> None:
        """Initialize the experiment components."""
        # Create environment - low volatility for cleaner testing
        self.environment = EnvironmentalContext(
            complexity=self.environment_complexity,
            volatility=0.2,
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
        
        # Track perturbation responses
        self.perturbation_responses = {agent_id: [] for agent_id in self.agents}
        
        # Initialize experiment state
        self.current_step = 0
        self.experiment_completed = False
        self.current_phase = "stabilization"
        self.current_perturbation_index = 0
        self.current_recovery_step = 0
    
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
                [],  # No previous structures for now
                agent.meaning_structure.perturbation_responses if hasattr(agent.meaning_structure, 'perturbation_responses') else []
            )
            
            # Store metrics history
            self.metrics_history[agent_id].append(metrics.copy())
        
        # Handle experiment phases
        self.manage_experiment_phases()
        
        # Increment step counter
        self.current_step += 1
        
        # Return step results
        return {
            "status": "success",
            "step": self.current_step,
            "environment": env_state,
            "responses": responses,
            "completed": self.experiment_completed
        }
    
    def manage_experiment_phases(self) -> None:
        """Manage the transitions between experiment phases."""
        # Stabilization phase
        if self.current_phase == "stabilization":
            if self.current_step >= self.stabilization_duration:
                # Move to first perturbation
                self.current_phase = "perturbation"
                print(f"Step {self.current_step}: Stabilization complete. Moving to perturbation phase.")
        
        # Perturbation phase
        elif self.current_phase == "perturbation":
            # Determine which perturbation to apply
            type_idx = self.current_perturbation_index // len(self.perturbation_intensities)
            intensity_idx = self.current_perturbation_index % len(self.perturbation_intensities)
            
            # Check if we've run out of perturbations
            if type_idx >= len(self.perturbation_types):
                self.experiment_completed = True
                self.generate_experiment_summary()
                return
            
            # Get perturbation details
            p_type = self.perturbation_types[type_idx]
            p_intensity = self.perturbation_intensities[intensity_idx]
            
            # Apply perturbation to all agents
            print(f"Step {self.current_step}: Applying {p_type} perturbation with intensity {p_intensity}")
            self.apply_perturbation(p_type, p_intensity)
            
            # Move to recovery phase
            self.current_phase = "recovery"
            self.current_recovery_step = 0
        
        # Recovery phase
        elif self.current_phase == "recovery":
            self.current_recovery_step += 1
            
            if self.current_recovery_step >= self.recovery_duration:
                # Move to next perturbation
                self.current_perturbation_index += 1
                self.current_phase = "perturbation"
                print(f"Step {self.current_step}: Recovery complete. Moving to next perturbation.")
    
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
    
    def apply_perturbation(self, perturbation_type: str, intensity: float) -> None:
        """
        Apply a perturbation to all agents.
        
        Args:
            perturbation_type: Type of perturbation to apply
            intensity: Intensity of the perturbation
        """
        # Apply to all agents
        perturbation_results = {}
        
        for agent_id, agent in self.agents.items():
            # Capture pre-perturbation metrics
            pre_metrics = self.metrics.update_all_metrics(agent.meaning_structure)
            
            # Apply perturbation
            response = agent.meaning_structure.apply_perturbation(
                perturbation_type=perturbation_type,
                intensity=intensity
            )
            
            # Capture post-perturbation metrics
            post_metrics = self.metrics.update_all_metrics(agent.meaning_structure)
            
            # Record results
            result = {
                "agent_id": agent_id,
                "perturbation_type": perturbation_type,
                "intensity": intensity,
                "time_step": self.current_step,
                "pre_metrics": pre_metrics,
                "post_metrics": post_metrics,
                "impact": {}
            }
            
            # Calculate impact on each metric
            for metric, pre_value in pre_metrics.items():
                post_value = post_metrics.get(metric, 0.0)
                if pre_value > 0:
                    relative_change = (post_value - pre_value) / pre_value
                else:
                    relative_change = 0.0 if post_value == 0 else float('inf')
                    
                result["impact"][metric] = relative_change
            
            # Store result
            self.perturbation_responses[agent_id].append(result)
            perturbation_results[agent_id] = result
        
        # Record perturbation event with timestamp
        perturbation_event = {
            "time_step": self.current_step,
            "perturbation_type": perturbation_type,
            "intensity": intensity,
            "results": perturbation_results
        }
        
        # Write event to file for later analysis
        event_file = os.path.join(
            self.output_dir, 
            f"perturbation_event_{self.current_step}_{perturbation_type}_{intensity}.json"
        )
        
        with open(event_file, 'w') as f:
            # Convert to JSON-serializable format
            serializable_event = {
                "time_step": perturbation_event["time_step"],
                "perturbation_type": perturbation_event["perturbation_type"],
                "intensity": perturbation_event["intensity"],
                "results": {
                    agent_id: {
                        "agent_id": result["agent_id"],
                        "perturbation_type": result["perturbation_type"],
                        "intensity": result["intensity"],
                        "time_step": result["time_step"],
                        "pre_metrics": result["pre_metrics"],
                        "post_metrics": result["post_metrics"],
                        "impact": result["impact"]
                    }
                    for agent_id, result in perturbation_event["results"].items()
                }
            }
            
            json.dump(serializable_event, f, indent=2)
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run the entire experiment from start to finish.
        
        Returns:
            Dictionary with experiment results
        """
        print(f"Starting Resilience Testing Experiment with {len(self.agents)} agents")
        print(f"Environment: complexity={self.environment_complexity}")
        print(f"Phases: {self.stabilization_duration} steps stabilization, " +
              f"{self.recovery_duration} steps recovery per perturbation")
        print(f"Testing {len(self.perturbation_types)} perturbation types at " +
              f"{len(self.perturbation_intensities)} intensity levels")
        
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
        # Calculate resilience metrics for all agents
        resilience_metrics = {}
        
        for agent_id, perturbation_responses in self.perturbation_responses.items():
            # Group by perturbation type
            by_type = {}
            
            for response in perturbation_responses:
                p_type = response["perturbation_type"]
                
                if p_type not in by_type:
                    by_type[p_type] = []
                    
                by_type[p_type].append(response)
            
            # Calculate average impact and recovery by type
            type_metrics = {}
            
            for p_type, responses in by_type.items():
                # Calculate average initial impact
                impacts = {
                    metric: [] for metric in ["narrative_coherence", "identity_stability", 
                                           "modal_balance", "meaning_resilience"]
                }
                
                for response in responses:
                    for metric in impacts:
                        if metric in response["impact"]:
                            impacts[metric].append(response["impact"][metric])
                
                # Calculate averages
                avg_impacts = {
                    metric: (sum(values) / len(values) if values else 0.0) 
                    for metric, values in impacts.items()
                }
                
                # Store metrics
                type_metrics[p_type] = avg_impacts
            
            # Store agent metrics
            resilience_metrics[agent_id] = {
                "by_perturbation_type": type_metrics,
                "overall_resilience": sum(
                    abs(impact["meaning_resilience"]) 
                    for impacts in type_metrics.values() 
                    for impact in [impacts] 
                    if "meaning_resilience" in impact
                ) / len(type_metrics) if type_metrics else 0.0
            }
        
        # Group by agent type
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        metrics_by_type = {agent_type: {} for agent_type in agent_types}
        
        for agent_id, metrics in resilience_metrics.items():
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
                
            # Calculate average overall resilience
            resilience_values = [m.get("overall_resilience", 0.0) for m in agent_metrics.values()]
            avg_metrics_by_type[agent_type] = {
                "avg_overall_resilience": sum(resilience_values) / len(resilience_values)
            }
            
            # Calculate average impact by perturbation type
            by_type = {}
            
            for agent_id, metrics in agent_metrics.items():
                type_metrics = metrics.get("by_perturbation_type", {})
                
                for p_type, impacts in type_metrics.items():
                    if p_type not in by_type:
                        by_type[p_type] = []
                        
                    by_type[p_type].append(impacts)
            
            # Calculate averages
            avg_by_type = {}
            
            for p_type, impacts_list in by_type.items():
                avg_impacts = {}
                
                for metric in ["narrative_coherence", "identity_stability", 
                            "modal_balance", "meaning_resilience"]:
                    values = []
                    
                    for impacts in impacts_list:
                        if metric in impacts:
                            values.append(impacts[metric])
                    
                    if values:
                        avg_impacts[metric] = sum(values) / len(values)
                
                avg_by_type[p_type] = avg_impacts
            
            avg_metrics_by_type[agent_type]["avg_by_perturbation_type"] = avg_by_type
        
        # Create summary
        summary = {
            "experiment_params": {
                "num_agents_per_type": self.num_agents_per_type,
                "stabilization_duration": self.stabilization_duration,
                "perturbation_types": self.perturbation_types,
                "perturbation_intensities": self.perturbation_intensities,
                "recovery_duration": self.recovery_duration,
                "environment_complexity": self.environment_complexity
            },
            "resilience_metrics": resilience_metrics,
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
        
        # 1. Plot impact by perturbation type for each agent type
        self.plot_impact_by_perturbation_type(vis_dir)
        
        # 2. Plot recovery trajectories for a selected perturbation
        self.plot_recovery_trajectories(vis_dir)
        
        # 3. Plot overall resilience comparison
        self.plot_overall_resilience(vis_dir)
        
        # 4. Plot individual perturbation responses
        for agent_type, agents in agents_by_type.items():
            if agents:
                # Take first agent of each type
                agent = agents[0]
                
                if hasattr(agent.meaning_structure, 'perturbation_responses') and agent.meaning_structure.perturbation_responses:
                    fig, ax = self.visualizer.plot_perturbation_response(agent.meaning_structure)
                    fig.savefig(os.path.join(vis_dir, f"{agent_type}_perturbation_response.png"))
                    plt.close(fig)
    
    def plot_impact_by_perturbation_type(self, vis_dir: str) -> None:
        """
        Plot the impact of different perturbation types on each agent type.
        
        Args:
            vis_dir: Output directory for visualizations
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Extract data from perturbation responses
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        
        # Set bar positions
        bar_width = 0.2
        perturbation_types = self.perturbation_types
        index = np.arange(len(perturbation_types))
        
        # Colors for agent types
        colors = {
            "hedonic": "#FF9500",    # Orange
            "eudaimonic": "#4CAF50", # Green
            "transcendent": "#2196F3", # Blue
            "integrator": "#9C27B0"  # Purple
        }
        
        # Plot bars for each agent type
        for i, agent_type in enumerate(agent_types):
            # Calculate average impact on meaning resilience for each perturbation type
            impacts = []
            
            for p_type in perturbation_types:
                # Get all impacts for this agent type and perturbation type
                agent_impacts = []
                
                for agent_id, agent in self.agents.items():
                    if agent_id.startswith(agent_type):
                        for response in self.perturbation_responses[agent_id]:
                            if response["perturbation_type"] == p_type:
                                if "meaning_resilience" in response["impact"]:
                                    agent_impacts.append(response["impact"]["meaning_resilience"])
                
                # Calculate average impact
                avg_impact = sum(agent_impacts) / len(agent_impacts) if agent_impacts else 0.0
                impacts.append(avg_impact)
            
            # Plot bars
            bar_positions = index + (i - 1.5) * bar_width
            ax.bar(bar_positions, impacts, bar_width, label=agent_type.title(), 
                  color=colors[agent_type], alpha=0.7)
        
        # Add labels and legend
        ax.set_xlabel("Perturbation Type")
        ax.set_ylabel("Impact on Meaning Resilience")
        ax.set_title("Impact of Perturbation Types on Agent Types")
        ax.set_xticks(index)
        ax.set_xticklabels([p.replace('_', ' ').title() for p in perturbation_types])
        ax.legend()
        
        # Add zero line
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Set y limits to be symmetric around zero
        y_max = max(0.5, max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])))
        ax.set_ylim(-y_max, y_max)
        
        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fig.tight_layout()
        
        # Save figure
        fig.savefig(os.path.join(vis_dir, "impact_by_perturbation_type.png"))
        plt.close(fig)
    
    def plot_recovery_trajectories(self, vis_dir: str) -> None:
        """
        Plot the recovery trajectories for a selected perturbation.
        
        Args:
            vis_dir: Output directory for visualizations
        """
        # Select a perturbation type to visualize
        p_type = "narrative_disruption"  # Example perturbation
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get the time steps at which this perturbation was applied
        perturbation_times = []
        
        for agent_id, responses in self.perturbation_responses.items():
            for response in responses:
                if response["perturbation_type"] == p_type:
                    step = response["time_step"]
                    if step not in perturbation_times:
                        perturbation_times.append(step)
        
        # Skip if no perturbations of this type were applied
        if not perturbation_times:
            plt.close(fig)
            return
        
        # Sort times
        perturbation_times.sort()
        
        # Pick the first perturbation time
        perturbation_time = perturbation_times[0]
        
        # Extract recovery trajectories for each agent type
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        
        # Colors for agent types
        colors = {
            "hedonic": "#FF9500",    # Orange
            "eudaimonic": "#4CAF50", # Green
            "transcendent": "#2196F3", # Blue
            "integrator": "#9C27B0"  # Purple
        }
        
        # Extract pre-perturbation values
        pre_values = {agent_type: [] for agent_type in agent_types}
        
        for agent_id, agent in self.agents.items():
            # Determine agent type
            agent_type = None
            for at in agent_types:
                if agent_id.startswith(at):
                    agent_type = at
                    break
                    
            if agent_type is None:
                continue
                
            # Get pre-perturbation value (just before perturbation)
            idx = perturbation_time - 1
            if idx >= 0 and idx < len(self.metrics_history[agent_id]):
                pre_values[agent_type].append(
                    self.metrics_history[agent_id][idx].get("meaning_resilience", 0.0)
                )
        
        # Calculate average pre-perturbation values
        avg_pre_values = {
            agent_type: sum(values) / len(values) if values else 0.0
            for agent_type, values in pre_values.items()
        }
        
        # Plot recovery trajectories
        max_steps_to_plot = min(self.recovery_duration + 5, 20)  # Limit to recovery period
        steps_to_plot = range(max_steps_to_plot)
        
        for agent_type in agent_types:
            # Track recovery values for all agents of this type
            recovery_values = [[] for _ in steps_to_plot]
            
            for agent_id, agent in self.agents.items():
                if agent_id.startswith(agent_type):
                    # Get recovery trajectory
                    for i, step in enumerate(steps_to_plot):
                        idx = perturbation_time + step
                        if idx < len(self.metrics_history[agent_id]):
                            recovery_values[i].append(
                                self.metrics_history[agent_id][idx].get("meaning_resilience", 0.0)
                            )
            
            # Calculate average recovery trajectory
            avg_recovery = [
                sum(values) / len(values) if values else None
                for values in recovery_values
            ]
            
            # Normalize to pre-perturbation value
            pre_value = avg_pre_values[agent_type]
            if pre_value != 0:
                normalized_recovery = [
                    (v / pre_value if v is not None else None)
                    for v in avg_recovery
                ]
            else:
                normalized_recovery = avg_recovery
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(normalized_recovery) if v is not None]
            valid_values = [normalized_recovery[i] for i in valid_indices]
            valid_steps = [steps_to_plot[i] for i in valid_indices]
            
            # Plot trajectory
            if valid_steps and valid_values:
                ax.plot(valid_steps, valid_values, 'o-', 
                       label=agent_type.title(), 
                       color=colors[agent_type],
                       linewidth=2)
        
        # Add perturbation line at step 0
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                  label=f"{p_type.replace('_', ' ').title()} Perturbation")
        
        # Add baseline level
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, 
                  label="Pre-Perturbation Level")
        
        # Add labels and legend
        ax.set_xlabel("Steps After Perturbation")
        ax.set_ylabel("Normalized Meaning Resilience")
        ax.set_title(f"Recovery Trajectories After {p_type.replace('_', ' ').title()} Perturbation")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save figure
        fig.savefig(os.path.join(vis_dir, f"recovery_trajectory_{p_type}.png"))
        plt.close(fig)
    
    def plot_overall_resilience(self, vis_dir: str) -> None:
        """
        Plot the overall resilience comparison between agent types.
        
        Args:
            vis_dir: Output directory for visualizations
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate average absolute impact across all perturbations for each agent type
        agent_types = ["hedonic", "eudaimonic", "transcendent", "integrator"]
        
        # Colors for agent types
        colors = {
            "hedonic": "#FF9500",    # Orange
            "eudaimonic": "#4CAF50", # Green
            "transcendent": "#2196F3", # Blue
            "integrator": "#9C27B0"  # Purple
        }
        
        # Calculate average impact magnitude for each agent type
        impact_magnitudes = []
        
        for agent_type in agent_types:
            # Get all impacts for this agent type
            agent_impacts = []
            
            for agent_id, agent in self.agents.items():
                if agent_id.startswith(agent_type):
                    for response in self.perturbation_responses[agent_id]:
                        if "meaning_resilience" in response["impact"]:
                            agent_impacts.append(abs(response["impact"]["meaning_resilience"]))
            
            # Calculate average impact magnitude
            avg_magnitude = sum(agent_impacts) / len(agent_impacts) if agent_impacts else 0.0
            impact_magnitudes.append(avg_magnitude)
        
        # Plot bars
        ax.bar(agent_types, impact_magnitudes, color=[colors[at] for at in agent_types], alpha=0.7)
        
        # Add labels
        ax.set_xlabel("Agent Type")
        ax.set_ylabel("Average Absolute Impact Magnitude")
        ax.set_title("Overall Perturbation Resilience by Agent Type")
        ax.set_xticklabels([at.title() for at in agent_types])
        
        # Lower values = better resilience, so invert y-axis
        ax.invert_yaxis()
        
        # Add "more resilient" label at the top
        y_min, y_max = ax.get_ylim()
        ax.text(len(agent_types) - 0.5, y_min + 0.05 * (y_max - y_min), 
               "More Resilient →", ha='right', va='bottom', 
               fontsize=12, fontweight='bold')
        
        # Save figure
        fig.savefig(os.path.join(vis_dir, "overall_resilience.png"))
        plt.close(fig)


def main():
    """Run the experiment from command line."""
    parser = argparse.ArgumentParser(description='Run Resilience Testing Experiment')
    
    parser.add_argument('--agents-per-type', type=int, default=3,
                       help='Number of agents per agent type')
    parser.add_argument('--stabilization', type=int, default=50,
                       help='Stabilization duration in time steps')
    parser.add_argument('--recovery', type=int, default=30,
                       help='Recovery duration per perturbation')
    parser.add_argument('--complexity', type=float, default=0.5,
                       help='Environment complexity (0.0-1.0)')
    parser.add_argument('--output-dir', type=str, default='results_resilience',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = ResilienceTestingExperiment(
        num_agents_per_type=args.agents_per_type,
        stabilization_duration=args.stabilization,
        recovery_duration=args.recovery,
        environment_complexity=args.complexity,
        output_dir=args.output_dir
    )
    
    # Run experiment
    experiment.run_full_experiment()
    
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
