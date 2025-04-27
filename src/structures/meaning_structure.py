"""
Meaning Structure Module

This module implements the MeaningStructure class that represents the integrated
output of multi-modal processing. The meaning structure contains narrative elements,
identity components, and value frameworks that emerge from the integration of
hedonic, eudaimonic, and transcendent processing.

The structure implements the temporal integration component of the Meta-Modal
Resonance Theory, enabling stable meaning to emerge from oscillatory processing
across different modalities.
"""

import numpy as np
import json
import copy
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime


class MeaningStructure:
    """
    Represents the integrated output of multi-modal processing.
    
    The meaning structure contains:
    1. Narrative elements - Temporally ordered experiences
    2. Identity components - Stable self-representations
    3. Value framework - Evaluative structures
    4. Meta-cognitive elements - Awareness of meaning processes
    
    It tracks coherence, resilience, and other metrics that characterize
    the quality of meaning integration.
    """
    
    def __init__(self, agent_id: str = "default"):
        """
        Initialize an empty meaning structure.
        
        Args:
            agent_id: ID of the agent that owns this structure
        """
        self.agent_id = agent_id
        self.created_at = datetime.now().isoformat()
        
        # Core structural components
        self.narrative_elements = []  # Temporally ordered experiences
        self.identity_components = {} # Stable self-representations
        self.value_framework = {}     # Evaluative structures
        self.meta_cognitive = {}      # Awareness of meaning processes
        
        # Integration metrics
        self.coherence_score = 0.0    # Measure of internal consistency
        self.resilience_score = 0.0   # Ability to maintain integrity under perturbation
        self.temporal_integration = 0.0  # Connection across time
        self.modal_integration = 0.0     # Connection across modalities
        
        # Tracking
        self.modification_history = []  # History of significant modifications
        self.perturbation_responses = []  # Responses to perturbations
        self.current_age = 0  # Age in time steps
    
    def integrate_modal_outputs(self, 
                              hedonic_output: Dict[str, Any],
                              eudaimonic_output: Dict[str, Any],
                              transcendent_output: Dict[str, Any],
                              modal_weights: Dict[str, float],
                              previous_structure: Optional['MeaningStructure'] = None) -> None:
        """
        Integrate outputs from all modalities into coherent structure.
        
        Args:
            hedonic_output: Output from hedonic processing
            eudaimonic_output: Output from eudaimonic processing
            transcendent_output: Output from transcendent processing
            modal_weights: Current weights for each modality
            previous_structure: Previous meaning structure if any
        """
        # Create new narrative element
        narrative_element = {
            "time": self.current_age,
            "hedonic_component": hedonic_output.get("net_value", 0.0),
            "eudaimonic_component": eudaimonic_output.get("net_value", 0.0),
            "transcendent_component": transcendent_output.get("net_value", 0.0),
            "modal_weights": modal_weights.copy(),
            "net_meaning": (
                hedonic_output.get("net_value", 0.0) * modal_weights.get("hedonic", 0.33) +
                eudaimonic_output.get("net_value", 0.0) * modal_weights.get("eudaimonic", 0.33) +
                transcendent_output.get("net_value", 0.0) * modal_weights.get("transcendent", 0.34)
            )
        }
        
        # Add to narrative elements
        self.narrative_elements.append(narrative_element)
        
        # Update identity components
        self._update_identity_components(hedonic_output, eudaimonic_output, transcendent_output, modal_weights)
        
        # Update value framework
        self._update_value_framework(hedonic_output, eudaimonic_output, transcendent_output, modal_weights)
        
        # Update meta-cognitive elements
        self._update_meta_cognitive(hedonic_output, eudaimonic_output, transcendent_output, modal_weights)
        
        # If previous structure provided, ensure temporal integration
        if previous_structure:
            self._integrate_with_previous(previous_structure)
        
        # Recalculate coherence and resilience
        self._calculate_coherence()
        self._calculate_resilience()
        self._calculate_temporal_integration()
        self._calculate_modal_integration()
        
        # Record this integration as a modification
        self.modification_history.append({
            "time": self.current_age,
            "type": "integration",
            "modal_weights": modal_weights.copy(),
            "resulting_coherence": self.coherence_score,
            "resulting_resilience": self.resilience_score
        })
        
        # Increment age
        self.current_age += 1
    
    def _update_identity_components(self,
                                   hedonic_output: Dict[str, Any],
                                   eudaimonic_output: Dict[str, Any],
                                   transcendent_output: Dict[str, Any],
                                   modal_weights: Dict[str, float]) -> None:
        """
        Update identity components based on modal outputs.
        
        This function extracts identity-relevant information from each modality
        and integrates it into the identity structure.
        """
        # Extract identity-relevant information from each modality
        
        # Hedonic identity components (preferences, desires, aversions)
        if "dominant_vector" in hedonic_output:
            # Approach/avoidance orientation
            approach_orientation = 1.0 if hedonic_output["dominant_vector"] == "approach" else -1.0
            
            # Update hedonic orientation with weighted moving average
            self.identity_components["hedonic_orientation"] = (
                self.identity_components.get("hedonic_orientation", 0.0) * 0.9 +
                approach_orientation * hedonic_output.get("intensity", 0.0) * 0.1
            )
            
            # Record specific preferences if present
            if "preferences" in hedonic_output:
                for pref_name, pref_value in hedonic_output["preferences"].items():
                    pref_key = f"preference_{pref_name}"
                    # Update with weighted average
                    self.identity_components[pref_key] = (
                        self.identity_components.get(pref_key, 0.0) * 0.9 +
                        pref_value * 0.1
                    )
        
        # Eudaimonic identity components (goals, virtues, purpose)
        if "goal_alignment" in eudaimonic_output:
            # Purpose clarity
            self.identity_components["purpose_clarity"] = (
                self.identity_components.get("purpose_clarity", 0.0) * 0.9 +
                eudaimonic_output.get("goal_alignment", 0.0) * 0.1
            )
            
            # Record growth orientation
            if "growth_contribution" in eudaimonic_output:
                self.identity_components["growth_orientation"] = (
                    self.identity_components.get("growth_orientation", 0.0) * 0.9 +
                    eudaimonic_output.get("growth_contribution", 0.0) * 0.1
                )
                
            # Record virtue orientation
            if "virtue_contribution" in eudaimonic_output:
                self.identity_components["virtue_orientation"] = (
                    self.identity_components.get("virtue_orientation", 0.0) * 0.9 +
                    eudaimonic_output.get("virtue_contribution", 0.0) * 0.1
                )
            
        # Transcendent identity components (connectedness, boundaries, unity)
        if "integration_level" in transcendent_output:
            # Self-boundary strength (inverse of boundary dissolution)
            self.identity_components["self_boundary_strength"] = (
                self.identity_components.get("self_boundary_strength", 1.0) * 0.9 +
                (1.0 - transcendent_output.get("boundary_dissolution", 0.0)) * 0.1
            )
            
            # Record connectedness
            if "connection_level" in transcendent_output:
                self.identity_components["connectedness"] = (
                    self.identity_components.get("connectedness", 0.0) * 0.9 +
                    transcendent_output.get("connection_level", 0.0) * 0.1
                )
            
            # Record unity experience
            if "unity_level" in transcendent_output:
                self.identity_components["unity_experience"] = (
                    self.identity_components.get("unity_experience", 0.0) * 0.9 +
                    transcendent_output.get("unity_level", 0.0) * 0.1
                )
    
    def _update_value_framework(self,
                               hedonic_output: Dict[str, Any],
                               eudaimonic_output: Dict[str, Any],
                               transcendent_output: Dict[str, Any],
                               modal_weights: Dict[str, float]) -> None:
        """
        Update value framework based on modal outputs.
        
        This function updates the evaluative structures that determine what
        is considered valuable or meaningful.
        """
        # Update hedonic values
        self.value_framework["pleasure_valuation"] = (
            self.value_framework.get("pleasure_valuation", 0.5) * 0.95 +
            min(1.0, max(0.0, 0.5 + 0.1 * hedonic_output.get("net_value", 0.0))) * 0.05
        )
        
        self.value_framework["pain_avoidance"] = (
            self.value_framework.get("pain_avoidance", 0.5) * 0.95 +
            min(1.0, max(0.0, 0.5 + 0.1 * max(0, -hedonic_output.get("net_value", 0.0)))) * 0.05
        )
        
        # Update eudaimonic values
        if "growth_contribution" in eudaimonic_output:
            self.value_framework["growth_valuation"] = (
                self.value_framework.get("growth_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * eudaimonic_output.get("net_value", 0.0))) * 0.05
            )
            
            self.value_framework["skill_development"] = (
                self.value_framework.get("skill_development", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * eudaimonic_output.get("growth_contribution", 0.0))) * 0.05
            )
        
        if "purpose_contribution" in eudaimonic_output:
            self.value_framework["purpose_valuation"] = (
                self.value_framework.get("purpose_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * eudaimonic_output.get("purpose_contribution", 0.0))) * 0.05
            )
        
        if "virtue_contribution" in eudaimonic_output:
            self.value_framework["virtue_valuation"] = (
                self.value_framework.get("virtue_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * eudaimonic_output.get("virtue_contribution", 0.0))) * 0.05
            )
        
        # Update transcendent values
        if "coherence_gain" in transcendent_output:
            self.value_framework["unity_valuation"] = (
                self.value_framework.get("unity_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * transcendent_output.get("coherence_gain", 0.0))) * 0.05
            )
        
        if "integration_level" in transcendent_output:
            self.value_framework["boundary_dissolution_valuation"] = (
                self.value_framework.get("boundary_dissolution_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * transcendent_output.get("integration_level", 0.0))) * 0.05
            )
        
        if "connection_level" in transcendent_output:
            self.value_framework["connection_valuation"] = (
                self.value_framework.get("connection_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * transcendent_output.get("connection_level", 0.0))) * 0.05
            )
    
    def _update_meta_cognitive(self,
                              hedonic_output: Dict[str, Any],
                              eudaimonic_output: Dict[str, Any],
                              transcendent_output: Dict[str, Any],
                              modal_weights: Dict[str, float]) -> None:
        """
        Update meta-cognitive elements of the meaning structure.
        
        This function updates the awareness of meaning construction processes.
        """
        # Record modal awareness - recognition of current processing focus
        self.meta_cognitive["modal_awareness"] = {
            "recognized_hedonic_focus": modal_weights["hedonic"],
            "recognized_eudaimonic_focus": modal_weights["eudaimonic"],
            "recognized_transcendent_focus": modal_weights["transcendent"],
            "awareness_accuracy": 0.8  # Placeholder for actual calculation
        }
        
        # Record integration awareness - recognition of integration processes
        self.meta_cognitive["integration_awareness"] = {
            "recognized_coherence": self.coherence_score,
            "recognized_resilience": self.resilience_score,
            "awareness_accuracy": 0.7  # Placeholder for actual calculation
        }
        
        # Record narrative awareness - recognition of temporal continuity
        if len(self.narrative_elements) > 1:
            self.meta_cognitive["narrative_awareness"] = {
                "recognized_continuity": self.temporal_integration,
                "story_awareness": 0.6,  # Placeholder for narrative structure awareness
                "awareness_accuracy": 0.75  # Placeholder for actual calculation
            }
    
    def _integrate_with_previous(self, previous_structure: 'MeaningStructure') -> None:
        """
        Integrate this structure with a previous structure for temporal continuity.
        
        Args:
            previous_structure: The previous meaning structure
        """
        # Only perform if structures belong to same agent
        if previous_structure.agent_id != self.agent_id:
            return
            
        # Identity integration - preserve core components with some continuity
        for key, value in previous_structure.identity_components.items():
            if key not in self.identity_components:
                # Import from previous if not already present
                self.identity_components[key] = value
        
        # Value framework integration - preserve value continuity
        for key, value in previous_structure.value_framework.items():
            if key not in self.value_framework:
                # Import from previous if not already present
                self.value_framework[key] = value
    
    def _calculate_coherence(self) -> None:
        """
        Calculate narrative coherence score.
        
        This evaluates the internal consistency of the meaning structure.
        """
        if len(self.narrative_elements) < 2:
            self.coherence_score = 0.5  # Default for new structures
            return
        
        # Calculate several coherence factors
        
        # 1. Temporal consistency across narrative elements
        temporal_consistency = 0.0
        for i in range(1, len(self.narrative_elements)):
            prev = self.narrative_elements[i-1]
            curr = self.narrative_elements[i]
            
            # Calculate consistency between consecutive elements
            consistency = 1.0 - 0.3 * abs(prev["net_meaning"] - curr["net_meaning"])
            temporal_consistency += consistency
        
        temporal_consistency /= (len(self.narrative_elements) - 1)
        
        # 2. Identity stability - consistent self-representation
        identity_stability = min(1.0, len(self.identity_components) / 10.0)
        
        # 3. Value consistency - clear and consistent values
        value_consistency = min(1.0, len(self.value_framework) / 8.0)
        
        # 4. Meta-cognitive awareness contribution
        meta_cognitive_contribution = 0.0
        if "integration_awareness" in self.meta_cognitive:
            meta_cognitive_contribution = self.meta_cognitive["integration_awareness"].get("awareness_accuracy", 0.0)
        
        # Combine factors - weighted average
        self.coherence_score = (
            temporal_consistency * 0.4 +
            identity_stability * 0.3 +
            value_consistency * 0.2 +
            meta_cognitive_contribution * 0.1
        )
    
    def _calculate_resilience(self) -> None:
        """
        Calculate meaning resilience score.
        
        This evaluates how well the meaning structure can maintain
        integrity under perturbation.
        """
        # Factors contributing to resilience:
        # 1. Modal balance - more balanced = more resilient
        # 2. Identity stability - stronger identity = more resilient
        # 3. Value framework clarity - clearer values = more resilient
        # 4. Narrative coherence - more coherent narrative = more resilient
        # 5. Meta-cognitive awareness - greater awareness = more resilient
        
        # Calculate modal balance factor
        modal_balance = 0.0
        if len(self.narrative_elements) > 0:
            latest = self.narrative_elements[-1]
            weights = latest["modal_weights"]
            # Perfect balance would be 0.33 for each modality
            balance_deviation = sum(abs(v - 0.33) for v in weights.values()) / 3.0
            modal_balance = 1.0 - balance_deviation
        
        # Calculate identity stability
        identity_stability = min(1.0, len(self.identity_components) / 10.0)
        
        # Calculate value framework clarity
        value_clarity = min(1.0, len(self.value_framework) / 8.0)
        
        # Meta-cognitive contribution
        meta_cognitive_contribution = 0.0
        if "integration_awareness" in self.meta_cognitive:
            meta_cognitive_contribution = self.meta_cognitive["integration_awareness"].get("awareness_accuracy", 0.0)
        
        # Combine factors
        self.resilience_score = (
            modal_balance * 0.25 +
            identity_stability * 0.25 +
            value_clarity * 0.2 +
            self.coherence_score * 0.2 +
            meta_cognitive_contribution * 0.1
        )
    
    def _calculate_temporal_integration(self) -> None:
        """
        Calculate temporal integration score.
        
        This evaluates how well experiences are integrated across time.
        """
        if len(self.narrative_elements) < 3:
            self.temporal_integration = 0.5  # Default for new structures
            return
        
        # Calculate consistency across longer time spans
        long_span_consistency = 0.0
        max_span = min(10, len(self.narrative_elements) - 1)
        
        # Look at different time spans
        for span in range(2, max_span + 1):
            span_sum = 0.0
            span_count = 0
            
            for i in range(span, len(self.narrative_elements)):
                prev = self.narrative_elements[i-span]
                curr = self.narrative_elements[i]
                
                # Calculate consistency across this span
                # Weight by recency (more recent spans matter more)
                recency_weight = 1.0 - 0.05 * (span - 1)
                consistency = (1.0 - 0.2 * abs(prev["net_meaning"] - curr["net_meaning"])) * recency_weight
                span_sum += consistency
                span_count += 1
            
            if span_count > 0:
                long_span_consistency += span_sum / span_count
        
        # Normalize by number of spans
        long_span_consistency /= (max_span - 1)
        
        # Combine with other temporal factors
        narrative_continuity = 0.0
        if "narrative_awareness" in self.meta_cognitive:
            narrative_continuity = self.meta_cognitive["narrative_awareness"].get("recognized_continuity", 0.0)
        
        self.temporal_integration = long_span_consistency * 0.7 + narrative_continuity * 0.3
    
    def _calculate_modal_integration(self) -> None:
        """
        Calculate modal integration score.
        
        This evaluates how well the different modalities are integrated.
        """
        if len(self.narrative_elements) < 2:
            self.modal_integration = 0.5  # Default for new structures
            return
        
        # Get latest narrative element
        latest = self.narrative_elements[-1]
        
        # Calculate modal balance
        weights = latest["modal_weights"]
        balance_deviation = sum(abs(v - 0.33) for v in weights.values()) / 3.0
        modal_balance = 1.0 - balance_deviation
        
        # Calculate mutual information between modalities (simplified)
        # In a full implementation, would use more sophisticated measures
        hedonic_component = latest["hedonic_component"]
        eudaimonic_component = latest["eudaimonic_component"]
        transcendent_component = latest["transcendent_component"]
        
        # Simplified correlation calculation
        components = [hedonic_component, eudaimonic_component, transcendent_component]
        correlations = []
        
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                if abs(components[i]) > 0.1 and abs(components[j]) > 0.1:
                    # Simple sign alignment calculation
                    alignment = 1.0 if (components[i] > 0) == (components[j] > 0) else -1.0
                    correlations.append(alignment)
                else:
                    correlations.append(0.0)
        
        modal_correlation = sum(correlations) / len(correlations) if correlations else 0.0
        # Convert to 0-1 scale
        modal_correlation = (modal_correlation + 1.0) / 2.0
        
        # Combine factors
        self.modal_integration = modal_balance * 0.6 + modal_correlation * 0.4
    
    def apply_perturbation(self, 
                         perturbation_type: str,
                         intensity: float,
                         target_components: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Apply a perturbation to the meaning structure and record response.
        
        Args:
            perturbation_type: Type of perturbation to apply
            intensity: Intensity of the perturbation (0.0 to 1.0)
            target_components: Specific components to target, or None for general
            
        Returns:
            Dictionary with perturbation response metrics
        """
        # Save pre-perturbation state for comparison
        pre_coherence = self.coherence_score
        pre_resilience = self.resilience_score
        pre_temporal = self.temporal_integration
        pre_modal = self.modal_integration
        
        # Apply perturbation based on type
        if perturbation_type == "narrative_disruption":
            self._apply_narrative_disruption(intensity, target_components)
        elif perturbation_type == "identity_disruption":
            self._apply_identity_disruption(intensity, target_components)
        elif perturbation_type == "value_disruption":
            self._apply_value_disruption(intensity, target_components)
        elif perturbation_type == "meta_cognitive_disruption":
            self._apply_meta_cognitive_disruption(intensity)
        elif perturbation_type == "random_disruption":
            self._apply_random_disruption(intensity)
        
        # Recalculate metrics
        self._calculate_coherence()
        self._calculate_resilience()
        self._calculate_temporal_integration()
        self._calculate_modal_integration()
        
        # Calculate impact metrics
        coherence_impact = (self.coherence_score - pre_coherence) / max(0.01, pre_coherence)
        resilience_impact = (self.resilience_score - pre_resilience) / max(0.01, pre_resilience)
        temporal_impact = (self.temporal_integration - pre_temporal) / max(0.01, pre_temporal)
        modal_impact = (self.modal_integration - pre_modal) / max(0.01, pre_modal)
        
        # Record response
        response = {
            "time": self.current_age,
            "perturbation_type": perturbation_type,
            "intensity": intensity,
            "target_components": target_components,
            "pre_coherence": pre_coherence,
            "post_coherence": self.coherence_score,
            "coherence_impact": coherence_impact,
            "pre_resilience": pre_resilience,
            "post_resilience": self.resilience_score,
            "resilience_impact": resilience_impact,
            "pre_temporal": pre_temporal,
            "post_temporal": self.temporal_integration,
            "temporal_impact": temporal_impact,
            "pre_modal": pre_modal,
            "post_modal": self.modal_integration,
            "modal_impact": modal_impact,
            "overall_impact": (abs(coherence_impact) + abs(resilience_impact) + 
                             abs(temporal_impact) + abs(modal_impact)) / 4.0
        }
        
        # Add to perturbation responses
        self.perturbation_responses.append(response)
        
        # Add to modification history
        self.modification_history.append({
            "time": self.current_age,
            "type": "perturbation",
            "perturbation_type": perturbation_type,
            "intensity": intensity,
            "resulting_coherence": self.coherence_score,
            "resulting_resilience": self.resilience_score
        })
        
        return response
    
    def _apply_narrative_disruption(self, 
                                  intensity: float, 
                                  target_components: Optional[List[str]] = None) -> None:
        """Apply disruption to narrative elements."""
        if not self.narrative_elements:
            return
            
        # Number of elements to disrupt
        num_elements = max(1, int(len(self.narrative_elements) * intensity * 0.5))
        
        # Randomly select elements to disrupt
        indices = np.random.choice(len(self.narrative_elements), num_elements, replace=False)
        
        for idx in indices:
            # Disrupt net meaning
            self.narrative_elements[idx]["net_meaning"] *= (1.0 - intensity * 0.5)
            
            # Disrupt modal weights if requested
            if target_components and "modal_weights" in target_components:
                weights = self.narrative_elements[idx]["modal_weights"]
                # Randomize weights
                for mode in weights:
                    weights[mode] += (np.random.random() - 0.5) * intensity
                
                # Normalize
                total = sum(weights.values())
                for mode in weights:
                    weights[mode] /= total
    
    def _apply_identity_disruption(self, 
                                 intensity: float, 
                                 target_components: Optional[List[str]] = None) -> None:
        """Apply disruption to identity components."""
        if not self.identity_components:
            return
            
        # Number of components to disrupt
        num_components = max(1, int(len(self.identity_components) * intensity * 0.7))
        
        # Select components to disrupt
        if target_components:
            # Disrupt specific components
            components_to_disrupt = [c for c in target_components if c in self.identity_components]
            if not components_to_disrupt:
                # Fallback to random if targets not found
                components_to_disrupt = list(np.random.choice(
                    list(self.identity_components.keys()), 
                    min(num_components, len(self.identity_components)), 
                    replace=False
                ))
        else:
            # Random components
            components_to_disrupt = list(np.random.choice(
                list(self.identity_components.keys()), 
                min(num_components, len(self.identity_components)), 
                replace=False
            ))
        
        for component in components_to_disrupt:
            # Disrupt by random factor
            self.identity_components[component] *= (1.0 - intensity * np.random.random())
    
    def _apply_value_disruption(self, 
                              intensity: float, 
                              target_components: Optional[List[str]] = None) -> None:
        """Apply disruption to value framework."""
        if not self.value_framework:
            return
            
        # Number of values to disrupt
        num_values = max(1, int(len(self.value_framework) * intensity * 0.7))
        
        # Select values to disrupt
        if target_components:
            # Disrupt specific values
            values_to_disrupt = [v for v in target_components if v in self.value_framework]
            if not values_to_disrupt:
                # Fallback to random if targets not found
                values_to_disrupt = list(np.random.choice(
                    list(self.value_framework.keys()), 
                    min(num_values, len(self.value_framework)), 
                    replace=False
                ))
        else:
            # Random values
            values_to_disrupt = list(np.random.choice(
                list(self.value_framework.keys()), 
                min(num_values, len(self.value_framework)), 
                replace=False
            ))
        
        for value in values_to_disrupt:
            # Disrupt by inverting or randomizing
            if np.random.random() < 0.5:
                # Invert
                self.value_framework[value] = 1.0 - self.value_framework[value]
            else:
                # Randomize
                self.value_framework[value] = np.random.random()
    
    def _apply_meta_cognitive_disruption(self, intensity: float) -> None:
        """Apply disruption to meta-cognitive awareness."""
        if not self.meta_cognitive:
            return
            
        # Disrupt modal awareness
        if "modal_awareness" in self.meta_cognitive:
            for key in self.meta_cognitive["modal_awareness"]:
                if "accuracy" in key:
                    # Reduce awareness accuracy
                    self.meta_cognitive["modal_awareness"][key] *= (1.0 - intensity)
        
        # Disrupt integration awareness
        if "integration_awareness" in self.meta_cognitive:
            for key in self.meta_cognitive["integration_awareness"]:
                if "accuracy" in key:
                    # Reduce awareness accuracy
                    self.meta_cognitive["integration_awareness"][key] *= (1.0 - intensity)
        
        # Disrupt narrative awareness
        if "narrative_awareness" in self.meta_cognitive:
            for key in self.meta_cognitive["narrative_awareness"]:
                if "awareness" in key:
                    # Reduce awareness
                    self.meta_cognitive["narrative_awareness"][key] *= (1.0 - intensity)
    
    def _apply_random_disruption(self, intensity: float) -> None:
        """Apply random disruption across all components."""
        # Disrupt each component type with equal probability
        self._apply_narrative_disruption(intensity * np.random.random())
        self._apply_identity_disruption(intensity * np.random.random())
        self._apply_value_disruption(intensity * np.random.random())
        self._apply_meta_cognitive_disruption(intensity * np.random.random())
    
    def merge_with(self, other_structure: 'MeaningStructure', merge_weights: Dict[str, float] = None) -> None:
        """
        Merge this meaning structure with another, creating a composite.
        
        Args:
            other_structure: Another meaning structure to merge with
            merge_weights: Optional weights for merging (self vs other)
        """
        if not merge_weights:
            merge_weights = {"self": 0.5, "other": 0.5}
        
        # Ensure weights sum to 1.0
        total = sum(merge_weights.values())
        merge_weights = {k: v / total for k, v in merge_weights.items()}
        
        # Merge narrative elements (append other's elements)
        self.narrative_elements.extend(other_structure.narrative_elements)
        
        # Sort by time
        self.narrative_elements.sort(key=lambda x: x["time"])
        
        # Merge identity components
        for key, value in other_structure.identity_components.items():
            if key in self.identity_components:
                # Weighted average
                self.identity_components[key] = (
                    self.identity_components[key] * merge_weights["self"] +
                    value * merge_weights["other"]
                )
            else:
                # Copy with other's weight
                self.identity_components[key] = value * merge_weights["other"]
        
        # Merge value framework
        for key, value in other_structure.value_framework.items():
            if key in self.value_framework:
                # Weighted average
                self.value_framework[key] = (
                    self.value_framework[key] * merge_weights["self"] +
                    value * merge_weights["other"]
                )
            else:
                # Copy with other's weight
                self.value_framework[key] = value * merge_weights["other"]
        
        # Merge meta-cognitive elements (simple override)
        for key, value in other_structure.meta_cognitive.items():
            self.meta_cognitive[key] = copy.deepcopy(value)
        
        # Record merge in modification history
        self.modification_history.append({
            "time": self.current_age,
            "type": "merge",
            "other_agent_id": other_structure.agent_id,
            "merge_weights": merge_weights.copy(),
        })
        
        # Recalculate metrics
        self._calculate_coherence()
        self._calculate_resilience()
        self._calculate_temporal_integration()
        self._calculate_modal_integration()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics for this meaning structure."""
        return {
            "coherence_score": self.coherence_score,
            "resilience_score": self.resilience_score,
            "temporal_integration": self.temporal_integration,
            "modal_integration": self.modal_integration,
            "identity_components_count": len(self.identity_components),
            "value_framework_count": len(self.value_framework),
            "narrative_elements_count": len(self.narrative_elements),
            "age": self.current_age,
            "modifications_count": len(self.modification_history),
            "perturbations_count": len(self.perturbation_responses)
        }
    
    def to_json(self) -> str:
        """Convert structure to JSON for storage or transmission."""
        data = {
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "current_age": self.current_age,
            "metrics": {
                "coherence_score": self.coherence_score,
                "resilience_score": self.resilience_score,
                "temporal_integration": self.temporal_integration,
                "modal_integration": self.modal_integration
            },
            "narrative_elements": self.narrative_elements,
            "identity_components": self.identity_components,
            "value_framework": self.value_framework,
            "meta_cognitive": self.meta_cognitive,
            "modification_history": self.modification_history,
            "perturbation_responses": self.perturbation_responses
        }
        
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MeaningStructure':
        """Create a meaning structure from JSON data."""
        data = json.loads(json_str)
        
        structure = cls(agent_id=data.get("agent_id", "default"))
        structure.created_at = data.get("created_at", datetime.now().isoformat())
        structure.current_age = data.get("current_age", 0)
        
        # Load metrics
        metrics = data.get("metrics", {})
        structure.coherence_score = metrics.get("coherence_score", 0.0)
        structure.resilience_score = metrics.get("resilience_score", 0.0)
        structure.temporal_integration = metrics.get("temporal_integration", 0.0)
        structure.modal_integration = metrics.get("modal_integration", 0.0)
        
        # Load components
        structure.narrative_elements = data.get("narrative_elements", [])
        structure.identity_components = data.get("identity_components", {})
        structure.value_framework = data.get("value_framework", {})
        structure.meta_cognitive = data.get("meta_cognitive", {})
        
        # Load history
        structure.modification_history = data.get("modification_history", [])
        structure.perturbation_responses = data.get("perturbation_responses", [])
        
        return structure
