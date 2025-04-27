"""
Integration Metrics Module

This module provides metrics for evaluating meaning construction quality
in the Meta-Modal Resonance Theory. It calculates various measures of
coherence, resilience, and integration across modalities and time.

These metrics serve as the dependent variables in experiments testing
the theory's central claims about meaning construction through modal integration.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class IntegrationMetrics:
    """
    Calculates metrics to evaluate meaning construction and integration.
    
    This class provides a suite of metrics that assess the quality of
    meaning structures produced by agents, focusing on:
    1. Narrative coherence - Consistency and connectedness over time
    2. Identity stability - Consistency of self-representation
    3. Modal balance - Distribution across processing modalities
    4. Perturbation resilience - Stability under disruption
    5. Developmental progress - Growth in integration capacity
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        # Core metrics tracked
        self.metrics = {
            "narrative_coherence": 0.0,
            "identity_stability": 0.0,
            "modal_balance": 0.0,
            "perturbation_resilience": 0.0,
            "developmental_progress": 0.0,
            "meaning_resilience": 0.0,
            "temporal_integration": 0.0,
            "structural_complexity": 0.0
        }
        
        # Tracking of metric history
        self.metrics_history = {metric: [] for metric in self.metrics}
        self.time_points = []
        
        # Reference values for comparison
        self.reference_values = {
            "narrative_coherence": {
                "novice": 0.4,
                "intermediate": 0.6,
                "advanced": 0.8
            },
            "identity_stability": {
                "novice": 0.3,
                "intermediate": 0.6,
                "advanced": 0.9
            },
            "modal_balance": {
                "novice": 0.2,
                "intermediate": 0.5,
                "advanced": 0.8
            },
            "meaning_resilience": {
                "novice": 0.25,
                "intermediate": 0.55,
                "advanced": 0.85
            }
        }
    
    def calculate_narrative_coherence(self, meaning_structure: Any) -> float:
        """
        Calculate temporal consistency of narrative elements.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Coherence score between 0.0 and 1.0
        """
        if not hasattr(meaning_structure, 'narrative_elements') or len(meaning_structure.narrative_elements) < 2:
            return 0.5  # Default for insufficient data
        
        # Extract narrative elements
        narrative_elements = meaning_structure.narrative_elements
        
        # Initialize coherence metrics
        temporal_consistency = 0.0
        thematic_consistency = 0.0
        
        # Calculate temporal consistency (smoothness of transitions)
        for i in range(1, len(narrative_elements)):
            prev = narrative_elements[i-1]
            curr = narrative_elements[i]
            
            # Use net_meaning as indicator of overall meaning trajectory
            if "net_meaning" in prev and "net_meaning" in curr:
                # Smaller differences = more consistent
                diff = abs(prev["net_meaning"] - curr["net_meaning"])
                # Convert to 0-1 scale (0 = largest possible difference, 1 = no difference)
                consistency = max(0.0, 1.0 - diff)
                temporal_consistency += consistency
        
        # Normalize
        if len(narrative_elements) > 1:
            temporal_consistency /= (len(narrative_elements) - 1)
        
        # Calculate thematic consistency (shared themes across narrative)
        # This would be more sophisticated in a full implementation
        # Here we use a simple proxy based on modal weighting consistency
        modal_weights = [
            e.get("modal_weights", {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34}) 
            for e in narrative_elements
        ]
        
        # Calculate variance in modal weights
        if modal_weights:
            hedonic_variance = np.var([w.get("hedonic", 0.33) for w in modal_weights])
            eudaimonic_variance = np.var([w.get("eudaimonic", 0.33) for w in modal_weights])
            transcendent_variance = np.var([w.get("transcendent", 0.34) for w in modal_weights])
            
            # Lower variance = higher consistency
            avg_variance = (hedonic_variance + eudaimonic_variance + transcendent_variance) / 3
            thematic_consistency = max(0.0, 1.0 - 5.0 * avg_variance)  # Scale factor of 5 to amplify small differences
        
        # Combine metrics (weighted average)
        coherence = temporal_consistency * 0.7 + thematic_consistency * 0.3
        
        # Update stored metric
        self.metrics["narrative_coherence"] = coherence
        
        return coherence
    
    def calculate_identity_stability(self, meaning_structure: Any, 
                                    previous_structures: Optional[List[Any]] = None) -> float:
        """
        Calculate stability of identity components over time.
        
        Args:
            meaning_structure: Current meaning structure
            previous_structures: Optional list of previous structures for longitudinal analysis
            
        Returns:
            Identity stability score between 0.0 and 1.0
        """
        if not hasattr(meaning_structure, 'identity_components') or not meaning_structure.identity_components:
            return 0.5  # Default for insufficient data
        
        # Basic stability measure - richness of identity components
        component_count = len(meaning_structure.identity_components)
        component_richness = min(1.0, component_count / 10.0)  # Cap at 10 components
        
        # If no previous structures, just use component richness
        if not previous_structures:
            self.metrics["identity_stability"] = component_richness
            return component_richness
        
        # Calculate longitudinal stability
        stability_scores = []
        
        # Compare with previous structures (most recent first)
        for prev in previous_structures[:5]:  # Only use the 5 most recent
            if not hasattr(prev, 'identity_components') or not prev.identity_components:
                continue
                
            # Find common components
            common_keys = set(meaning_structure.identity_components.keys()) & set(prev.identity_components.keys())
            
            if not common_keys:
                stability_scores.append(0.0)
                continue
            
            # Calculate average difference in component values
            diffs = []
            for key in common_keys:
                curr_val = meaning_structure.identity_components[key]
                prev_val = prev.identity_components[key]
                diffs.append(abs(curr_val - prev_val))
            
            # Calculate stability (inverse of difference)
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                stability = max(0.0, 1.0 - avg_diff)
                stability_scores.append(stability)
        
        # Calculate longitudinal stability (if we have scores)
        longitudinal_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0.5
        
        # Combine with component richness
        stability = component_richness * 0.4 + longitudinal_stability * 0.6
        
        # Update stored metric
        self.metrics["identity_stability"] = stability
        
        return stability
    
    def calculate_modal_balance(self, meaning_structure: Any) -> float:
        """
        Calculate the balance between different processing modalities.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Modal balance score between 0.0 and 1.0
        """
        if not hasattr(meaning_structure, 'narrative_elements') or not meaning_structure.narrative_elements:
            return 0.5  # Default for insufficient data
        
        # Extract latest modal weights
        latest = meaning_structure.narrative_elements[-1] if meaning_structure.narrative_elements else None
        
        if not latest or "modal_weights" not in latest:
            return 0.5  # Default if no weights available
        
        weights = latest["modal_weights"]
        
        # Perfect balance would be equal weights (0.33, 0.33, 0.34)
        # Calculate deviation from perfect balance
        perfect_weights = {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34}
        
        # Sum of absolute deviations
        deviation = sum(abs(weights.get(mode, 0) - perfect_weights.get(mode, 0)) for mode in perfect_weights)
        
        # Convert to balance score (0 = maximum imbalance, 1 = perfect balance)
        balance = max(0.0, 1.0 - deviation)
        
        # Update stored metric
        self.metrics["modal_balance"] = balance
        
        return balance
    
    def calculate_perturbation_resilience(self, 
                                        meaning_structure: Any,
                                        perturbation_responses: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Calculate resilience of meaning structure to perturbations.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            perturbation_responses: Optional list of perturbation response records
            
        Returns:
            Resilience score between 0.0 and 1.0
        """
        # If no perturbation data available, estimate from structure properties
        if not perturbation_responses:
            # Use a combination of other metrics as proxy for resilience
            coherence = self.metrics.get("narrative_coherence", 0.5)
            stability = self.metrics.get("identity_stability", 0.5)
            balance = self.metrics.get("modal_balance", 0.5)
            
            estimated_resilience = coherence * 0.4 + stability * 0.4 + balance * 0.2
            
            # Update stored metric
            self.metrics["perturbation_resilience"] = estimated_resilience
            
            return estimated_resilience
        
        # Calculate actual resilience from perturbation responses
        impact_scores = []
        recovery_scores = []
        
        for response in perturbation_responses:
            # Calculate impact (how much did metrics change)
            if "overall_impact" in response:
                # Lower impact = higher resilience
                impact = 1.0 - min(1.0, abs(response["overall_impact"]))
                impact_scores.append(impact)
            
            # Calculate recovery (how well did it return to baseline)
            if "pre_coherence" in response and "post_coherence" in response:
                recovery = 1.0 - min(1.0, abs(response["pre_coherence"] - response["post_coherence"]))
                recovery_scores.append(recovery)
        
        # Combine scores
        impact_resilience = sum(impact_scores) / len(impact_scores) if impact_scores else 0.5
        recovery_resilience = sum(recovery_scores) / len(recovery_scores) if recovery_scores else 0.5
        
        resilience = impact_resilience * 0.6 + recovery_resilience * 0.4
        
        # Update stored metric
        self.metrics["perturbation_resilience"] = resilience
        
        return resilience
    
    def calculate_meaning_resilience(self, meaning_structure: Any) -> float:
        """
        Calculate overall meaning resilience score.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Meaning resilience score between 0.0 and 1.0
        """
        # Use meaning structure's own resilience score if available
        if hasattr(meaning_structure, 'resilience_score'):
            resilience = meaning_structure.resilience_score
        else:
            # Otherwise calculate from component metrics
            coherence = self.metrics.get("narrative_coherence", 0.5)
            stability = self.metrics.get("identity_stability", 0.5)
            balance = self.metrics.get("modal_balance", 0.5)
            perturbation_resilience = self.metrics.get("perturbation_resilience", 0.5)
            
            resilience = (
                coherence * 0.3 +
                stability * 0.3 +
                balance * 0.2 +
                perturbation_resilience * 0.2
            )
        
        # Update stored metric
        self.metrics["meaning_resilience"] = resilience
        
        return resilience
    
    def calculate_temporal_integration(self, meaning_structure: Any) -> float:
        """
        Calculate integration across time periods.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Temporal integration score between 0.0 and 1.0
        """
        # Use meaning structure's own temporal integration if available
        if hasattr(meaning_structure, 'temporal_integration'):
            integration = meaning_structure.temporal_integration
        else:
            # Default to narrative coherence as proxy
            integration = self.metrics.get("narrative_coherence", 0.5)
        
        # Update stored metric
        self.metrics["temporal_integration"] = integration
        
        return integration
    
    def calculate_structural_complexity(self, meaning_structure: Any) -> float:
        """
        Calculate the structural complexity of the meaning structure.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Structural complexity score between 0.0 and 1.0
        """
        # Initialize component counts
        narrative_count = 0
        identity_count = 0
        value_count = 0
        meta_count = 0
        
        # Count narrative elements
        if hasattr(meaning_structure, 'narrative_elements'):
            narrative_count = len(meaning_structure.narrative_elements)
        
        # Count identity components
        if hasattr(meaning_structure, 'identity_components'):
            identity_count = len(meaning_structure.identity_components)
        
        # Count value framework elements
        if hasattr(meaning_structure, 'value_framework'):
            value_count = len(meaning_structure.value_framework)
        
        # Count meta-cognitive elements
        if hasattr(meaning_structure, 'meta_cognitive'):
            meta_count = len(meaning_structure.meta_cognitive)
        
        # Calculate complexity score
        # We use a log scale to prevent unlimited growth
        complexity = min(1.0, (
            0.2 * np.log1p(narrative_count / 5) +
            0.3 * np.log1p(identity_count / 8) +
            0.3 * np.log1p(value_count / 6) +
            0.2 * np.log1p(meta_count / 4)
        ))
        
        # Update stored metric
        self.metrics["structural_complexity"] = complexity
        
        return complexity
    
    def calculate_developmental_progress(self, meaning_structure: Any) -> float:
        """
        Calculate developmental progress in integration capability.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Developmental progress score between 0.0 and 1.0
        """
        # This metric requires historical data to be meaningful
        # For demonstration, we'll use a combination of other metrics
        
        # Calculate distance from reference values
        stage_distances = {}
        
        for stage in ["novice", "intermediate", "advanced"]:
            distance_sum = 0
            count = 0
            
            for metric in ["narrative_coherence", "identity_stability", "modal_balance", "meaning_resilience"]:
                if metric in self.metrics and metric in self.reference_values and stage in self.reference_values[metric]:
                    ref_value = self.reference_values[metric][stage]
                    current_value = self.metrics[metric]
                    distance = abs(current_value - ref_value)
                    distance_sum += distance
                    count += 1
            
            if count > 0:
                stage_distances[stage] = distance_sum / count
        
        # Determine closest stage
        if not stage_distances:
            closest_stage = "novice"
        else:
            closest_stage = min(stage_distances, key=stage_distances.get)
        
        # Convert to progress score
        if closest_stage == "novice":
            progress = 0.25
        elif closest_stage == "intermediate":
            progress = 0.6
        else:  # advanced
            progress = 0.9
        
        # Update stored metric
        self.metrics["developmental_progress"] = progress
        
        return progress
    
    def update_all_metrics(self, meaning_structure: Any,
                         previous_structures: Optional[List[Any]] = None,
                         perturbation_responses: Optional[List[Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Update all metrics at once for a given meaning structure.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            previous_structures: Optional list of previous structures
            perturbation_responses: Optional list of perturbation responses
            
        Returns:
            Dictionary of all metrics
        """
        # Calculate each metric
        self.calculate_narrative_coherence(meaning_structure)
        self.calculate_identity_stability(meaning_structure, previous_structures)
        self.calculate_modal_balance(meaning_structure)
        self.calculate_perturbation_resilience(meaning_structure, perturbation_responses)
        self.calculate_meaning_resilience(meaning_structure)
        self.calculate_temporal_integration(meaning_structure)
        self.calculate_structural_complexity(meaning_structure)
        self.calculate_developmental_progress(meaning_structure)
        
        # Record metrics history
        self.time_points.append(len(self.time_points))
        for metric, value in self.metrics.items():
            self.metrics_history[metric].append(value)
        
        return self.metrics.copy()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get the current metrics."""
        return self.metrics.copy()
    
    def get_metrics_history(self) -> Dict[str, List[float]]:
        """Get the history of metrics over time."""
        return self.metrics_history.copy()
    
    def plot_metrics(self, metrics_to_plot: Optional[List[str]] = None) -> Tuple[Figure, Axes]:
        """
        Plot the evolution of metrics over time.
        
        Args:
            metrics_to_plot: Optional list of specific metrics to plot
            
        Returns:
            matplotlib Figure and Axes objects
        """
        if not self.time_points:
            # Create empty plot with message if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No metrics data available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xlabel("Time")
            ax.set_ylabel("Metric Value")
            ax.set_title("Integration Metrics Over Time")
            return fig, ax
        
        # Determine which metrics to plot
        if not metrics_to_plot:
            metrics_to_plot = list(self.metrics.keys())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each metric
        for metric in metrics_to_plot:
            if metric in self.metrics_history:
                ax.plot(self.time_points, self.metrics_history[metric], 
                       label=metric.replace("_", " ").title())
        
        # Add labels and legend
        ax.set_xlabel("Time")
        ax.set_ylabel("Metric Value")
        ax.set_title("Integration Metrics Over Time")
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    def plot_radar_chart(self, stage_comparison: bool = False) -> Figure:
        """
        Plot a radar chart of the current metrics.
        
        Args:
            stage_comparison: Whether to include reference values for different stages
            
        Returns:
            matplotlib Figure
        """
        metrics_to_plot = [
            "narrative_coherence", 
            "identity_stability",
            "modal_balance", 
            "meaning_resilience",
            "temporal_integration",
            "structural_complexity"
        ]
        
        # Set up radar chart
        num_metrics = len(metrics_to_plot)
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        
        # Close the loop
        angles += angles[:1]
        
        # Get values
        values = [self.metrics.get(metric, 0.0) for metric in metrics_to_plot]
        values += values[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot current values
        ax.plot(angles, values, 'o-', linewidth=2, label='Current')
        ax.fill(angles, values, alpha=0.25)
        
        # Plot reference values if requested
        if stage_comparison:
            for stage in ["novice", "intermediate", "advanced"]:
                stage_values = []
                for metric in metrics_to_plot:
                    if metric in self.reference_values and stage in self.reference_values[metric]:
                        stage_values.append(self.reference_values[metric][stage])
                    else:
                        # Default values if not specified
                        if stage == "novice":
                            stage_values.append(0.3)
                        elif stage == "intermediate":
                            stage_values.append(0.6)
                        else:  # advanced
                            stage_values.append(0.9)
                
                # Close the loop
                stage_values += stage_values[:1]
                
                # Plot
                ax.plot(angles, stage_values, '--', linewidth=1, label=stage.title())
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics_to_plot])
        
        # Set y limits
        ax.set_ylim(0, 1)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set title
        ax.set_title("Integration Metrics Radar Chart")
        
        return fig
    
    def compare_agents(self, 
                     agent_metrics: Dict[str, Dict[str, float]], 
                     metrics_to_plot: Optional[List[str]] = None) -> Figure:
        """
        Compare metrics across multiple agents.
        
        Args:
            agent_metrics: Dictionary mapping agent IDs to their metrics
            metrics_to_plot: Optional list of specific metrics to plot
            
        Returns:
            matplotlib Figure
        """
        if not agent_metrics:
            # Create empty plot with message if no data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No agent data available", 
                   horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Determine which metrics to plot
        if not metrics_to_plot:
            # Use intersection of all available metrics
            all_metrics = set()
            for agent_id, metrics in agent_metrics.items():
                all_metrics.update(metrics.keys())
            metrics_to_plot = list(all_metrics)
        
        # Set up grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Number of agents and metrics
        num_agents = len(agent_metrics)
        num_metrics = len(metrics_to_plot)
        
        # Set up positions
        index = np.arange(num_metrics)
        bar_width = 0.8 / num_agents
        
        # Plot each agent's metrics
        for i, (agent_id, metrics) in enumerate(agent_metrics.items()):
            values = [metrics.get(metric, 0.0) for metric in metrics_to_plot]
            position = index + i * bar_width
            ax.bar(position, values, bar_width, alpha=0.7, label=agent_id)
        
        # Add labels and legend
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title("Comparison of Agent Integration Metrics")
        ax.set_xticks(index + bar_width * (num_agents - 1) / 2)
        ax.set_xticklabels([metric.replace("_", " ").title() for metric in metrics_to_plot])
        ax.legend()
        
        # Rotate tick labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fig.tight_layout()
        
        return fig
    
    def evaluate_stage(self, meaning_structure: Any) -> str:
        """
        Evaluate the developmental stage of meaning integration.
        
        Args:
            meaning_structure: The meaning structure to evaluate
            
        Returns:
            Developmental stage classification
        """
        # Update metrics if needed
        self.update_all_metrics(meaning_structure)
        
        # Calculate distance from reference values for each stage
        stage_distances = {}
        
        for stage in ["novice", "intermediate", "advanced"]:
            distance_sum = 0
            count = 0
            
            for metric in ["narrative_coherence", "identity_stability", "modal_balance", "meaning_resilience"]:
                if metric in self.metrics and metric in self.reference_values and stage in self.reference_values[metric]:
                    ref_value = self.reference_values[metric][stage]
                    current_value = self.metrics[metric]
                    distance = abs(current_value - ref_value)
                    distance_sum += distance
                    count += 1
            
            if count > 0:
                stage_distances[stage] = distance_sum / count
        
        # Determine closest stage
        if not stage_distances:
            return "undetermined"
        
        closest_stage = min(stage_distances, key=stage_distances.get)
        return closest_stage
    
    def reset(self) -> None:
        """Reset all metrics to initial values."""
        for metric in self.metrics:
            self.metrics[metric] = 0.0
        
        self.metrics_history = {metric: [] for metric in self.metrics}
        self.time_points = []
