"""
Simulation Runner Module

This module provides the SimulationRunner class for executing simulations using the
Meta-Modal Resonance framework. It orchestrates the interaction between agents and
environments, tracks metrics, and manages experimental conditions.

The simulation runner serves as the core experimental framework for testing hypotheses
about modal integration and meaning construction.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Import framework components
from ..agents.meta_modal_agent import MetaModalAgent
from ..environment.context_generator import EnvironmentalContext
from ..metrics.integration_metrics import IntegrationMetrics
from ..structures.meaning_structure import MeaningStructure


class SimulationRunner:
    """
    Executes simulations using the Meta-Modal Resonance framework.
    
    The SimulationRunner orchestrates interactions between agents and environments,
    manages experimental conditions, tracks metrics, and handles data collection
    for hypothesis testing.
    """
    
    def __init__(self, 
                config: Dict[str, Any] = None, 
                output_dir: str = "simulation_results"):
        """
        Initialize the simulation runner.
        
        Args:
            config: Configuration dictionary for the simulation
            output_dir: Directory to store simulation results
        """
        # Set default config if none provided
        self.config = config or {
            "num_steps": 100,
            "agents": [
                {"id": "agent1", "type": "integrated", "development_stage": "intermediate"},
                {"id": "agent2", "type": "hedonic_dominant", "development_stage": "intermediate"},
                {"id": "agent3", "type": "eudaimonic_dominant", "development_stage": "intermediate"},
                {"id": "agent4", "type": "transcendent_dominant", "development_stage": "intermediate"}
            ],
            "environment": {
                "complexity": 0.7,
                "volatility": 0.5,
                "threat_level": 0.3,
                "pattern_sequence": ["random", "cyclical", "escalating"]
            },
            "metrics": {
                "record_interval": 5,
                "perturbation_schedule": [25, 50, 75],
                "visualization_interval": 10
            },
            "experiment": {
                "name": "modal_comparison",
                "description": "Comparing integrated vs. single-modality dominant agents",
                "hypothesis": "Integrated agents demonstrate higher meaning resilience than single-modality dominant agents"
            }
        }
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize environment
        self.environment = self._initialize_environment()
        
        # Initialize metrics calculators
        self.metrics_calculators = {agent_id: IntegrationMetrics() for agent_id in self.agents}
        
        # Data collection
        self.agent_histories = {agent_id: [] for agent_id in self.agents}
        self.environment_history = []
        self.metrics_history = {agent_id: {} for agent_id in self.agents}
        
        # Track current step
        self.current_step = 0
        
        # Perturbation tracking
        self.perturbation_results = {agent_id: [] for agent_id in self.agents}
        
        # Agent interaction history
        self.interaction_history = []
        
        # Timestamp for this simulation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _initialize_agents(self) -> Dict[str, MetaModalAgent]:
        """
        Initialize agents based on configuration.
        
        Returns:
            Dictionary mapping agent IDs to agent objects
        """
        agents = {}
        
        for agent_config in self.config["agents"]:
            agent_id = agent_config["id"]
            agent_type = agent_config["type"]
            development_stage = agent_config.get("development_stage", "novice")
            
            # Create agent with appropriate configuration
            if agent_type == "integrated":
                # Balanced modal weights
                oscillation_params = {
                    "pattern": "adaptive",
                    "modal_bias": {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34}
                }
            elif agent_type == "hedonic_dominant":
                # Hedonic-focused modal weights
                oscillation_params = {
                    "pattern": "biased",
                    "modal_bias": {"hedonic": 0.6, "eudaimonic": 0.2, "transcendent": 0.2}
                }
            elif agent_type == "eudaimonic_dominant":
                # Eudaimonic-focused modal weights
                oscillation_params = {
                    "pattern": "biased",
                    "modal_bias": {"hedonic": 0.2, "eudaimonic": 0.6, "transcendent": 0.2}
                }
            elif agent_type == "transcendent_dominant":
                # Transcendent-focused modal weights
                oscillation_params = {
                    "pattern": "biased",
                    "modal_bias": {"hedonic": 0.2, "eudaimonic": 0.2, "transcendent": 0.6}
                }
            else:
                # Default to integrated
                oscillation_params = {
                    "pattern": "adaptive",
                    "modal_bias": {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34}
                }
            
            # Create agent
            agents[agent_id] = MetaModalAgent(
                agent_id=agent_id,
                grid_size=50,
                oscillation_params=oscillation_params,
                development_stage=development_stage
            )
        
        return agents
    
    def _initialize_environment(self) -> EnvironmentalContext:
        """
        Initialize environment based on configuration.
        
        Returns:
            Configured environment object
        """
        env_config = self.config["environment"]
        
        environment = EnvironmentalContext(
            complexity=env_config.get("complexity", 0.7),
            volatility=env_config.get("volatility", 0.5),
            threat_level=env_config.get("threat_level", 0.3),
            modal_bias=env_config.get("modal_bias", None)
        )
        
        # Set initial pattern if specified
        if "initial_pattern" in env_config:
            environment.set_pattern(env_config["initial_pattern"])
        
        return environment
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run the complete simulation.
        
        Returns:
            Dictionary of simulation results
        """
        print(f"Starting simulation: {self.config['experiment']['name']}")
        print(f"Hypothesis: {self.config['experiment']['hypothesis']}")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Number of steps: {self.config['num_steps']}")
        
        start_time = time.time()
        
        # Run simulation steps
        for step in range(self.config["num_steps"]):
            self._run_step()
            
            # Check for scheduled perturbations
            if step in self.config["metrics"]["perturbation_schedule"]:
                self._apply_perturbation()
            
            # Check for pattern changes
            if "pattern_sequence" in self.config["environment"]:
                pattern_sequence = self.config["environment"]["pattern_sequence"]
                sequence_length = len(pattern_sequence)
                
                if sequence_length > 0:
                    # Calculate when to change patterns
                    steps_per_pattern = self.config["num_steps"] // sequence_length
                    
                    if step > 0 and step % steps_per_pattern == 0:
                        pattern_index = (step // steps_per_pattern) % sequence_length
                        new_pattern = pattern_sequence[pattern_index]
                        
                        print(f"Step {step}: Changing environment pattern to '{new_pattern}'")
                        self.environment.set_pattern(new_pattern)
            
            # Periodic visualization
            if step % self.config["metrics"]["visualization_interval"] == 0 or step == self.config["num_steps"] - 1:
                self._create_visualizations(step)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Simulation completed in {duration:.2f} seconds")
        
        # Generate final results
        results = self._compile_results()
        
        # Save results
        self._save_results(results)
        
        # Create final visualizations
        self._create_final_visualizations()
        
        return results
    
    def _run_step(self) -> None:
        """Run a single simulation step."""
        # Update environment
        env_state = self.environment.update()
        self.environment_history.append(env_state)
        
        # Process each agent
        for agent_id, agent in self.agents.items():
            # Convert environment state to observation for agent
            observation = self._env_state_to_observation(env_state)
            
            # Agent processes observation
            response = agent.process_observation(observation)
            
            # Record agent response
            self.agent_histories[agent_id].append(response)
            
            # Update metrics
            if self.current_step % self.config["metrics"]["record_interval"] == 0:
                # Get previous structures for this agent (if any)
                previous_structures = []
                
                # Update metrics
                metrics = self.metrics_calculators[agent_id].update_metrics(
                    agent.meaning_structure,
                    previous_structures
                )
                
                # Record metrics
                self.metrics_history[agent_id][self.current_step] = metrics
        
        # Agent interactions (periodic)
        if self.current_step % 10 == 5:  # Offset from environment changes
            self._facilitate_agent_interactions()
        
        # Increment step counter
        self.current_step += 1
    
    def _env_state_to_observation(self, env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert environment state to agent observation.
        
        Args:
            env_state: Current environment state
            
        Returns:
            Observation dictionary for agent processing
        """
        # Extract relevant elements from environment state
        
        # Basic observation
        observation = {
            "time": env_state["time"],
            "is_significant": env_state["is_significant"]
        }
        
        # Add intensity based on significance
        if env_state["is_significant"]:
            observation["intensity"] = 1.0
            
            # Add event type if available
            if "event_type" in env_state:
                observation["type"] = env_state["event_type"]
            else:
                observation["type"] = "significant_event"
        else:
            observation["intensity"] = 0.7  # Default intensity
            observation["type"] = "standard"
        
        # Process stimulus elements into modality-specific components
        hedonic_total = 0.0
        eudaimonic_total = 0.0
        transcendent_total = 0.0
        
        for stimulus in env_state["stimulus_elements"]:
            # Extract stimulus values
            if stimulus["modality"] == "hedonic":
                hedonic_total += stimulus["intensity"]
                observation["pleasure_value"] = stimulus.get("pleasure_value", 0.0)
                observation["pain_value"] = stimulus.get("pain_value", 0.0)
                
            elif stimulus["modality"] == "eudaimonic":
                eudaimonic_total += stimulus["intensity"]
                observation["growth_value"] = stimulus.get("growth_value", 0.0)
                observation["purpose_value"] = stimulus.get("purpose_value", 0.0)
                observation["virtue_value"] = stimulus.get("virtue_value", 0.0)
                
            elif stimulus["modality"] == "transcendent":
                transcendent_total += stimulus["intensity"]
                observation["unity_value"] = stimulus.get("unity_value", 0.0)
                observation["connection_value"] = stimulus.get("connection_value", 0.0)
                observation["dissolution_value"] = stimulus.get("dissolution_value", 0.0)
        
        # Set type based on dominant modality if not significant
        if not env_state["is_significant"]:
            if hedonic_total > eudaimonic_total and hedonic_total > transcendent_total:
                observation["type"] = "hedonic"
            elif eudaimonic_total > hedonic_total and eudaimonic_total > transcendent_total:
                observation["type"] = "eudaimonic"
            elif transcendent_total > hedonic_total and transcendent_total > eudaimonic_total:
                observation["type"] = "transcendent"
        
        # Add context variables
        observation["threat_level"] = env_state["threat_level"]
        observation["growth_opportunity"] = env_state["growth_opportunity"]
        observation["connection_opportunity"] = env_state["connection_opportunity"]
        observation["cognitive_load"] = env_state["cognitive_load"]
        
        return observation
    
    def _facilitate_agent_interactions(self) -> None:
        """Facilitate interactions between agents."""
        # Select random pair of agents
        agent_ids = list(self.agents.keys())
        if len(agent_ids) < 2:
            return
        
        agent1_id = np.random.choice(agent_ids)
        remaining_ids = [aid for aid in agent_ids if aid != agent1_id]
        agent2_id = np.random.choice(remaining_ids)
        
        # Get agent objects
        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]
        
        # Facilitate interaction
        result1, result2 = agent1.interact_with_agent(agent2)
        
        # Record interaction
        interaction_record = {
            "time": self.current_step,
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "result1": result1,
            "result2": result2
        }
        
        self.interaction_history.append(interaction_record)
    
    def _apply_perturbation(self) -> None:
        """Apply perturbations to test resilience."""
        perturbation_types = [
            "narrative_disruption",
            "identity_disruption",
            "value_disruption",
            "meta_cognitive_disruption",
            "random_disruption"
        ]
        
        # Select random perturbation type
        perturbation_type = np.random.choice(perturbation_types)
        
        # Apply to each agent
        for agent_id, agent in self.agents.items():
            # Skip 20% of agents randomly
            if np.random.random() < 0.2:
                continue
            
            # Intensity varies by agent type (higher for single-modality agents)
            if "dominant" in agent_id or "dominant" in agent.id:
                intensity = np.random.uniform(0.6, 0.9)
            else:
                intensity = np.random.uniform(0.5, 0.8)
            
            # Apply perturbation
            print(f"Step {self.current_step}: Applying {perturbation_type} perturbation (intensity {intensity:.2f}) to {agent_id}")
            
            # Record pre-perturbation state
            pre_coherence = agent.meaning_structure.coherence_score
            pre_resilience = agent.meaning_structure.resilience_score
            
            # Apply perturbation
            response = agent.meaning_structure.apply_perturbation(
                perturbation_type=perturbation_type,
                intensity=intensity
            )
            
            # Record result
            result = {
                "time": self.current_step,
                "agent_id": agent_id,
                "perturbation_type": perturbation_type,
                "intensity": intensity,
                "pre_coherence": pre_coherence,
                "post_coherence": agent.meaning_structure.coherence_score,
                "coherence_impact": (agent.meaning_structure.coherence_score - pre_coherence) / max(0.01, pre_coherence),
                "pre_resilience": pre_resilience,
                "post_resilience": agent.meaning_structure.resilience_score,
                "resilience_impact": (agent.meaning_structure.resilience_score - pre_resilience) / max(0.01, pre_resilience)
            }
            
            self.perturbation_results[agent_id].append(result)
    
    def _create_visualizations(self, step: int) -> None:
        """
        Create visualizations for the current simulation state.
        
        Args:
            step: Current simulation step
        """
        # Skip if too early
        if step < 5:
            return
        
        # Create directory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create subdirectory for this step
        step_dir = os.path.join(viz_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Create environmental state visualization
        self._visualize_environment(step, step_dir)
        
        # Create agent metrics visualization
        self._visualize_agent_metrics(step, step_dir)
        
        # Create meaning structure visualization for each agent
        for agent_id, agent in self.agents.items():
            self._visualize_agent_meaning_structure(agent_id, step, step_dir)
    
    def _visualize_environment(self, step: int, output_dir: str) -> None:
        """
        Create visualization of environmental state.
        
        Args:
            step: Current simulation step
            output_dir: Directory to save visualization
        """
        # Get environment history up to current step
        history = self.environment_history[:step+1]
        if len(history) < 2:
            return
        
        # Extract key variables
        times = list(range(len(history)))
        threat_levels = [state["threat_level"] for state in history]
        growth_opps = [state["growth_opportunity"] for state in history]
        connection_opps = [state["connection_opportunity"] for state in history]
        cognitive_loads = [state["cognitive_load"] for state in history]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot environmental variables
        plt.plot(times, threat_levels, 'r-', label='Threat Level')
        plt.plot(times, growth_opps, 'g-', label='Growth Opportunity')
        plt.plot(times, connection_opps, 'b-', label='Connection Opportunity')
        plt.plot(times, cognitive_loads, 'k--', label='Cognitive Load')
        
        # Mark significant events
        significant_times = [i for i, state in enumerate(history) if state["is_significant"]]
        if significant_times:
            plt.plot(significant_times, [1.05] * len(significant_times), 'ro', markersize=8)
            for t in significant_times:
                plt.axvline(x=t, color='r', linestyle='--', alpha=0.3)
        
        # Add labels and legend
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title(f'Environmental State Over Time (Step {step})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'environment_state.png'))
        plt.close()
    
    def _visualize_agent_metrics(self, step: int, output_dir: str) -> None:
        """
        Create visualization of agent metrics.
        
        Args:
            step: Current simulation step
            output_dir: Directory to save visualization
        """
        # Need sufficient history
        if step < 10:
            return
        
        # Create figure for each metric type
        metric_types = [
            "narrative_coherence", 
            "identity_stability", 
            "modal_balance", 
            "meaning_resilience"
        ]
        
        for metric_type in metric_types:
            plt.figure(figsize=(10, 6))
            
            # Collect metric data for each agent
            for agent_id in self.agents:
                times = []
                values = []
                
                # Extract values from history
                for t, metrics in self.metrics_history[agent_id].items():
                    if t <= step and metric_type in metrics:
                        times.append(t)
                        values.append(metrics[metric_type])
                
                if times:
                    # Plot data
                    plt.plot(times, values, '-o', label=f'{agent_id}')
            
            # Add perturbation markers
            for pert_step in self.config["metrics"]["perturbation_schedule"]:
                if pert_step <= step:
                    plt.axvline(x=pert_step, color='r', linestyle='--', alpha=0.3)
                    plt.text(pert_step, 0.05, 'P', color='r', fontsize=12)
            
            # Add labels and legend
            plt.xlabel('Time Step')
            plt.ylabel(f'{metric_type.replace("_", " ").title()}')
            plt.title(f'{metric_type.replace("_", " ").title()} Over Time (Step {step})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric_type}.png'))
            plt.close()
    
    def _visualize_agent_meaning_structure(self, agent_id: str, step: int, output_dir: str) -> None:
        """
        Create visualization of agent's meaning structure.
        
        Args:
            agent_id: ID of agent to visualize
            step: Current simulation step
            output_dir: Directory to save visualization
        """
        # Get agent
        agent = self.agents[agent_id]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Extract modal focus history
        modal_focus_history = {
            "hedonic": [],
            "eudaimonic": [],
            "transcendent": []
        }
        
        # Get data from narrative elements
        times = []
        meanings = []
        
        if hasattr(agent.meaning_structure, 'narrative_elements'):
            elements = agent.meaning_structure.narrative_elements
            
            for elem in elements:
                if "time" in elem and elem["time"] <= step:
                    # Record time and meaning
                    times.append(elem["time"])
                    
                    if "net_meaning" in elem:
                        meanings.append(elem["net_meaning"])
                    else:
                        meanings.append(0.0)
                    
                    # Record modal weights
                    if "modal_weights" in elem:
                        for mode, weight in elem["modal_weights"].items():
                            if mode in modal_focus_history:
                                modal_focus_history[mode].append(weight)
        
        # Create subplot layout
        plt.subplot(2, 1, 1)
        
        # Plot modal focus history
        time_indices = list(range(len(modal_focus_history["hedonic"])))
        
        if time_indices:
            plt.stackplot(
                time_indices,
                [modal_focus_history["hedonic"], modal_focus_history["eudaimonic"], modal_focus_history["transcendent"]],
                labels=["Hedonic", "Eudaimonic", "Transcendent"],
                colors=["#FF9999", "#99FF99", "#9999FF"]
            )
            
            plt.xlabel('Narrative Element Index')
            plt.ylabel('Modal Weight')
            plt.title(f'Modal Focus History for {agent_id}')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
        
        # Plot meaning history
        plt.subplot(2, 1, 2)
        
        if times and meanings:
            plt.plot(times, meanings, 'k-o', label='Net Meaning')
            
            # Add reference line
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Mark perturbations
            for result in self.perturbation_results.get(agent_id, []):
                if result["time"] <= step:
                    plt.axvline(x=result["time"], color='r', linestyle='--', alpha=0.3)
                    plt.text(result["time"], min(meanings) - 0.1, 'P', color='r', fontsize=12)
            
            plt.xlabel('Time Step')
            plt.ylabel('Net Meaning')
            plt.title(f'Meaning History for {agent_id}')
            plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{agent_id}_meaning_structure.png'))
        plt.close()
    
    def _create_final_visualizations(self) -> None:
        """Create summary visualizations for the complete simulation run."""
        # Create directory for final visualizations
        viz_dir = os.path.join(self.output_dir, "final_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Create perturbation response visualization
        self._visualize_perturbation_responses(viz_dir)
        
        # Create modal integration comparison
        self._visualize_modal_integration_comparison(viz_dir)
        
        # Create resilience comparison
        self._visualize_resilience_comparison(viz_dir)
    
    def _visualize_perturbation_responses(self, output_dir: str) -> None:
        """
        Create visualization of perturbation responses.
        
        Args:
            output_dir: Directory to save visualization
        """
        # Check if we have perturbation data
        has_data = False
        for results in self.perturbation_results.values():
            if results:
                has_data = True
                break
        
        if not has_data:
            return
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create bar groups for each agent type
        agent_types = {
            "integrated": [aid for aid in self.agents.keys() if "integrated" in aid or not any(x in aid for x in ["hedonic", "eudaimonic", "transcendent"])],
            "hedonic": [aid for aid in self.agents.keys() if "hedonic" in aid],
            "eudaimonic": [aid for aid in self.agents.keys() if "eudaimonic" in aid],
            "transcendent": [aid for aid in self.agents.keys() if "transcendent" in aid]
        }
        
        # Calculate average impact by agent type
        impacts = {
            agent_type: {
                "coherence": [],
                "resilience": []
            }
            for agent_type in agent_types
        }
        
        # Collect impact data
        for agent_id, results in self.perturbation_results.items():
            # Determine agent type
            agent_type = None
            for type_name, ids in agent_types.items():
                if agent_id in ids:
                    agent_type = type_name
                    break
            
            if agent_type and results:
                # Calculate average impacts
                for result in results:
                    impacts[agent_type]["coherence"].append(result["coherence_impact"])
                    impacts[agent_type]["resilience"].append(result["resilience_impact"])
        
        # Calculate averages
        avg_impacts = {
            agent_type: {
                "coherence": np.mean(data["coherence"]) if data["coherence"] else 0,
                "resilience": np.mean(data["resilience"]) if data["resilience"] else 0
            }
            for agent_type, data in impacts.items()
        }
        
        # Prepare data for plotting
        agent_type_names = list(avg_impacts.keys())
        coherence_impacts = [avg_impacts[at]["coherence"] for at in agent_type_names]
        resilience_impacts = [avg_impacts[at]["resilience"] for at in agent_type_names]
        
        # Set up bar positions
        bar_width = 0.35
        index = np.arange(len(agent_type_names))
        
        # Create bars
        bar1 = plt.bar(index, coherence_impacts, bar_width,
                     label='Coherence Impact', color='#FF9999', alpha=0.7)
        bar2 = plt.bar(index + bar_width, resilience_impacts, bar_width,
                     label='Resilience Impact', color='#9999FF', alpha=0.7)
        
        # Add labels and legend
        plt.xlabel('Agent Type')
        plt.ylabel('Average Impact (% Change)')
        plt.title('Perturbation Impact by Agent Type')
        plt.xticks(index + bar_width / 2, [at.title() for at in agent_type_names])
        plt.legend()
        
        # Add value labels
        for i, v in enumerate(coherence_impacts):
            plt.text(i - 0.1, v + 0.01, f'{v:.2%}', fontsize=9)
        
        for i, v in enumerate(resilience_impacts):
            plt.text(i + bar_width - 0.1, v + 0.01, f'{v:.2%}', fontsize=9)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'perturbation_responses.png'))
        plt.close()
    
    def _visualize_modal_integration_comparison(self, output_dir: str) -> None:
        """
        Create visualization comparing modal integration across agent types.
        
        Args:
            output_dir: Directory to save visualization
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create line plot for each agent
        for agent_id in self.agents:
            times = []
            values = []
            
            # Extract modal integration values
            for t, metrics in self.metrics_history[agent_id].items():
                if "modal_integration" in metrics:
                    times.append(t)
                    values.append(metrics["modal_integration"])
            
            if times:
                # Determine line style based on agent type
                if "integrated" in agent_id or not any(x in agent_id for x in ["hedonic", "eudaimonic", "transcendent"]):
                    linestyle = '-'
                    linewidth = 2
                else:
                    linestyle = '--'
                    linewidth = 1.5
                
                # Plot data
                plt.plot(times, values, linestyle=linestyle, linewidth=linewidth, label=agent_id)
        
        # Add labels and legend
        plt.xlabel('Time Step')
        plt.ylabel('Modal Integration Score')
        plt.title('Modal Integration Comparison Across Agent Types')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'modal_integration_comparison.png'))
        plt.close()
    
    def _visualize_resilience_comparison(self, output_dir: str) -> None:
        """
        Create visualization comparing meaning resilience across agent types.
        
        Args:
            output_dir: Directory to save visualization
        """
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create line plot for each agent
        for agent_id in self.agents:
            times = []
            values = []
            
            # Extract resilience values
            for t, metrics in self.metrics_history[agent_id].items():
                if "meaning_resilience" in metrics:
                    times.append(t)
                    values.append(metrics["meaning_resilience"])
            
            if times:
                # Determine line style based on agent type
                if "integrated" in agent_id or not any(x in agent_id for x in ["hedonic", "eudaimonic", "transcendent"]):
                    linestyle = '-'
                    linewidth = 2
                else:
                    linestyle = '--'
                    linewidth = 1.5
                
                # Plot data
                plt.plot(times, values, linestyle=linestyle, linewidth=linewidth, label=agent_id)
        
        # Add perturbation markers
        for pert_step in self.config["metrics"]["perturbation_schedule"]:
            plt.axvline(x=pert_step, color='r', linestyle='--', alpha=0.3)
            plt.text(pert_step, 0.05, 'P', color='r', fontsize=12)
        
        # Add labels and legend
        plt.xlabel('Time Step')
        plt.ylabel('Meaning Resilience Score')
        plt.title('Meaning Resilience Comparison Across Agent Types')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resilience_comparison.png'))
        plt.close()
    
    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile simulation results.
        
        Returns:
            Dictionary of simulation results
        """
        # Calculate final metrics for each agent
        final_metrics = {}
        for agent_id in self.agents:
            if self.metrics_history[agent_id]:
                # Get latest metrics
                latest_time = max(self.metrics_history[agent_id].keys())
                final_metrics[agent_id] = self.metrics_history[agent_id][latest_time]
            else:
                final_metrics[agent_id] = {}
        
        # Calculate perturbation response statistics
        perturbation_stats = {}
        for agent_id, results in self.perturbation_results.items():
            if results:
                coherence_impacts = [r["coherence_impact"] for r in results]
                resilience_impacts = [r["resilience_impact"] for r in results]
                
                perturbation_stats[agent_id] = {
                    "coherence_impact_avg": np.mean(coherence_impacts),
                    "coherence_impact_std": np.std(coherence_impacts),
                    "resilience_impact_avg": np.mean(resilience_impacts),
                    "resilience_impact_std": np.std(resilience_impacts),
                    "num_perturbations": len(results)
                }
            else:
                perturbation_stats[agent_id] = {}
        
        # Compile environment statistics
        env_stats = {
            "avg_threat_level": np.mean([state["threat_level"] for state in self.environment_history]),
            "avg_growth_opportunity": np.mean([state["growth_opportunity"] for state in self.environment_history]),
            "avg_connection_opportunity": np.mean([state["connection_opportunity"] for state in self.environment_history]),
            "avg_cognitive_load": np.mean([state["cognitive_load"] for state in self.environment_history]),
            "num_significant_events": sum(1 for state in self.environment_history if state["is_significant"])
        }
        
        # Create results dictionary
        results = {
            "experiment": self.config["experiment"],
            "simulation_timestamp": self.timestamp,
            "num_steps": self.current_step,
            "agents": {
                agent_id: {
                    "type": next((a["type"] for a in self.config["agents"] if a["id"] == agent_id), "unknown"),
                    "development_stage": next((a["development_stage"] for a in self.config["agents"] if a["id"] == agent_id), "unknown"),
                    "final_metrics": final_metrics.get(agent_id, {}),
                    "perturbation_stats": perturbation_stats.get(agent_id, {})
                }
                for agent_id in self.agents
            },
            "environment": {
                "config": self.config["environment"],
                "statistics": env_stats
            },
            "interactions": {
                "count": len(self.interaction_history)
            }
        }
        
        # Add hypothesis evaluation
        results["hypothesis_evaluation"] = self._evaluate_hypothesis(results)
        
        return results
    
    def _evaluate_hypothesis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the experiment hypothesis based on results.
        
        Args:
            results: Compiled simulation results
            
        Returns:
            Dictionary with hypothesis evaluation
        """
        # Extract agent types
        integrated_agents = [aid for aid, info in results["agents"].items() 
                           if info["type"] == "integrated"]
        
        single_modality_agents = [aid for aid, info in results["agents"].items() 
                                if info["type"] in ["hedonic_dominant", "eudaimonic_dominant", "transcendent_dominant"]]
        
        # Compare metrics between types
        if integrated_agents and single_modality_agents:
            # Calculate average metrics for each group
            integrated_metrics = {
                "narrative_coherence": np.mean([results["agents"][aid]["final_metrics"].get("narrative_coherence", 0) 
                                            for aid in integrated_agents]),
                "identity_stability": np.mean([results["agents"][aid]["final_metrics"].get("identity_stability", 0) 
                                           for aid in integrated_agents]),
                "modal_balance": np.mean([results["agents"][aid]["final_metrics"].get("modal_balance", 0) 
                                      for aid in integrated_agents]),
                "meaning_resilience": np.mean([results["agents"][aid]["final_metrics"].get("meaning_resilience", 0) 
                                           for aid in integrated_agents]),
                "modal_integration": np.mean([results["agents"][aid]["final_metrics"].get("modal_integration", 0) 
                                          for aid in integrated_agents]),
            }
            
            single_modality_metrics = {
                "narrative_coherence": np.mean([results["agents"][aid]["final_metrics"].get("narrative_coherence", 0) 
                                            for aid in single_modality_agents]),
                "identity_stability": np.mean([results["agents"][aid]["final_metrics"].get("identity_stability", 0) 
                                           for aid in single_modality_agents]),
                "modal_balance": np.mean([results["agents"][aid]["final_metrics"].get("modal_balance", 0) 
                                      for aid in single_modality_agents]),
                "meaning_resilience": np.mean([results["agents"][aid]["final_metrics"].get("meaning_resilience", 0) 
                                           for aid in single_modality_agents]),
                "modal_integration": np.mean([results["agents"][aid]["final_metrics"].get("modal_integration", 0) 
                                          for aid in single_modality_agents]),
            }
            
            # Calculate differences
            metric_diffs = {
                metric: integrated_metrics[metric] - single_modality_metrics[metric]
                for metric in integrated_metrics
            }
            
            # Calculate perturbation response differences
            integrated_pert_response = np.mean([
                results["agents"][aid]["perturbation_stats"].get("resilience_impact_avg", 0)
                for aid in integrated_agents if "perturbation_stats" in results["agents"][aid]
            ])
            
            single_pert_response = np.mean([
                results["agents"][aid]["perturbation_stats"].get("resilience_impact_avg", 0)
                for aid in single_modality_agents if "perturbation_stats" in results["agents"][aid]
            ])
            
            pert_response_diff = integrated_pert_response - single_pert_response
            
            # Evaluate hypothesis
            hypothesis_supported = (
                metric_diffs["meaning_resilience"] > 0 and
                metric_diffs["modal_integration"] > 0 and
                pert_response_diff > 0
            )
            
            return {
                "hypothesis_supported": hypothesis_supported,
                "metric_differences": metric_diffs,
                "perturbation_response_difference": pert_response_diff,
                "integrated_agents_metrics": integrated_metrics,
                "single_modality_agents_metrics": single_modality_metrics
            }
        else:
            return {
                "hypothesis_supported": None,
                "reason": "Insufficient agent types for comparison"
            }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save simulation results to disk.
        
        Args:
            results: Simulation results to save
        """
        # Create results file
        results_file = os.path.join(self.output_dir, f"results_{self.timestamp}.json")
        
        # Save results as JSON
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
