"""
Meta-Modal Agent Module

This module implements the MetaModalAgent class, which extends the standard agent architecture
with the capacity for multi-modal processing and meaning construction through the integration
of hedonic, eudaimonic, and transcendent processing domains.

The agent implements the Meta-Modal Resonance Theory (MMRT) of meaning construction through:
1. Parallel processing across three modalities
2. Dynamic oscillation between modalities
3. Temporal integration of modal outputs
4. Recursive self-modification of integration parameters
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any


class ProcessingDomain:
    """Base class for modal processing domains."""
    
    def __init__(self, name: str, grid_size: int = 50):
        """
        Initialize a processing domain.
        
        Args:
            name: Name of this processing domain
            grid_size: Size of the domain's neural grid
        """
        self.name = name
        self.grid_size = grid_size
        self.activation_field = np.zeros((grid_size, grid_size))
        self.density_field = self.initialize_density_field()
        
    def initialize_density_field(self) -> np.ndarray:
        """Initialize the neural density field for this domain."""
        # Create a base random density with higher values in the center
        density = np.random.uniform(0.1, 0.3, (self.grid_size, self.grid_size))
        
        # Increase density toward the center
        center = self.grid_size // 2
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                center_factor = np.exp(-0.01 * dist)
                density[i, j] += 0.3 * center_factor
                
        # Normalize to max of 1.0
        density = density / np.max(density)
        return density
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through this domain's algorithms.
        
        Args:
            input_data: Dictionary of input data
            
        Returns:
            Dictionary of processed outputs
        """
        # Base implementation does nothing
        return {"processed": False, "domain": self.name}


class HedonicProcessingDomain(ProcessingDomain):
    """Processes hedonic aspects of experience (pleasure/pain, approach/avoid)."""
    
    def __init__(self, grid_size: int = 50):
        """Initialize the hedonic processing domain."""
        super().__init__("hedonic", grid_size)
        self.pleasure_field = np.zeros((grid_size, grid_size))
        self.pain_field = np.zeros((grid_size, grid_size))
        self.approach_vectors = np.zeros((grid_size, grid_size, 2))
        self.avoidance_vectors = np.zeros((grid_size, grid_size, 2))
        self.reward_predictions = {}
        self.temporal_discount_rate = 0.8  # High discount rate
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through hedonic algorithms.
        
        Args:
            input_data: Dictionary of input data
            
        Returns:
            Dictionary with hedonic processing results
        """
        # Extract hedonic aspects from input
        pleasure_value = input_data.get('pleasure_value', 0.0)
        pain_value = input_data.get('pain_value', 0.0)
        location = input_data.get('location', (self.grid_size // 2, self.grid_size // 2))
        
        # Process through pleasure/pain fields
        self._update_pleasure_field(pleasure_value, location)
        self._update_pain_field(pain_value, location)
        
        # Calculate approach/avoidance vectors
        self._calculate_motivational_vectors()
        
        # Generate output
        net_hedonic_value = pleasure_value - pain_value
        dominant_vector = "approach" if net_hedonic_value > 0 else "avoidance"
        
        return {
            "domain": "hedonic",
            "net_value": net_hedonic_value,
            "dominant_vector": dominant_vector,
            "intensity": abs(net_hedonic_value),
            "prediction_error": self._calculate_prediction_error(net_hedonic_value, location)
        }
    
    def _update_pleasure_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the pleasure field with new value."""
        if value <= 0:
            return
            
        # Create gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 10:  # Activation radius
                    # Stronger at center, fading outward
                    activation = value * np.exp(-0.2 * dist) * self.density_field[i, j]
                    self.pleasure_field[i, j] += activation
    
    def _update_pain_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the pain field with new value."""
        if value <= 0:
            return
            
        # Create gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 10:  # Activation radius
                    # Stronger at center, fading outward
                    activation = value * np.exp(-0.2 * dist) * self.density_field[i, j]
                    self.pain_field[i, j] += activation
    
    def _calculate_motivational_vectors(self) -> None:
        """Calculate approach and avoidance vectors based on pleasure and pain fields."""
        # Simple gradient-based approach
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                # Calculate pleasure gradient (approach direction)
                dx_p = self.pleasure_field[i+1, j] - self.pleasure_field[i-1, j]
                dy_p = self.pleasure_field[i, j+1] - self.pleasure_field[i, j-1]
                
                # Calculate pain gradient (avoidance direction)
                dx_n = self.pain_field[i+1, j] - self.pain_field[i-1, j]
                dy_n = self.pain_field[i, j+1] - self.pain_field[i, j-1]
                
                # Set approach toward pleasure
                magnitude = np.sqrt(dx_p**2 + dy_p**2)
                if magnitude > 0:
                    self.approach_vectors[i, j, 0] = dx_p / magnitude
                    self.approach_vectors[i, j, 1] = dy_p / magnitude
                
                # Set avoidance away from pain
                magnitude = np.sqrt(dx_n**2 + dy_n**2)
                if magnitude > 0:
                    self.avoidance_vectors[i, j, 0] = -dx_n / magnitude
                    self.avoidance_vectors[i, j, 1] = -dy_n / magnitude
    
    def _calculate_prediction_error(self, actual_value: float, location: Tuple[int, int]) -> float:
        """Calculate the prediction error for this stimulus."""
        location_key = f"{location[0]},{location[1]}"
        predicted_value = self.reward_predictions.get(location_key, 0.0)
        error = actual_value - predicted_value
        
        # Update prediction for next time
        self.reward_predictions[location_key] = predicted_value + 0.1 * error
        
        return error
    
    def decay_fields(self) -> None:
        """Apply decay to emotion fields."""
        decay_rate = 0.95
        self.pleasure_field *= decay_rate
        self.pain_field *= decay_rate


class EudaimonicProcessingDomain(ProcessingDomain):
    """Processes eudaimonic aspects of experience (growth, purpose, virtue)."""
    
    def __init__(self, grid_size: int = 50):
        """Initialize the eudaimonic processing domain."""
        super().__init__("eudaimonic", grid_size)
        self.growth_field = np.zeros((grid_size, grid_size))
        self.purpose_field = np.zeros((grid_size, grid_size))
        self.virtue_field = np.zeros((grid_size, grid_size))
        self.goal_structures = {}
        self.values_matrix = np.zeros((5, 5))  # Simplified value space
        self.temporal_discount_rate = 0.5  # Medium discount rate
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through eudaimonic algorithms.
        
        Args:
            input_data: Dictionary of input data
            
        Returns:
            Dictionary with eudaimonic processing results
        """
        # Extract eudaimonic aspects from input
        growth_value = input_data.get('growth_value', 0.0)
        purpose_value = input_data.get('purpose_value', 0.0)
        virtue_value = input_data.get('virtue_value', 0.0)
        location = input_data.get('location', (self.grid_size // 2, self.grid_size // 2))
        
        # Process through eudaimonic fields
        self._update_growth_field(growth_value, location)
        self._update_purpose_field(purpose_value, location)
        self._update_virtue_field(virtue_value, location)
        
        # Check goal alignment
        goal_alignment = self._calculate_goal_alignment(input_data)
        
        # Check value alignment
        value_alignment = self._calculate_value_alignment(input_data)
        
        # Calculate net eudaimonic impact
        net_eudaimonic_value = (growth_value + purpose_value + virtue_value) / 3.0
        
        # Apply alignment modifiers
        net_eudaimonic_value *= (0.5 + 0.5 * goal_alignment)
        net_eudaimonic_value *= (0.5 + 0.5 * value_alignment)
        
        return {
            "domain": "eudaimonic",
            "net_value": net_eudaimonic_value,
            "goal_alignment": goal_alignment,
            "value_alignment": value_alignment,
            "growth_contribution": growth_value / (growth_value + purpose_value + virtue_value + 1e-10),
            "purpose_contribution": purpose_value / (growth_value + purpose_value + virtue_value + 1e-10),
            "virtue_contribution": virtue_value / (growth_value + purpose_value + virtue_value + 1e-10),
        }
    
    def _update_growth_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the growth field with new value."""
        if value <= 0:
            return
            
        # Create gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 15:  # Wider activation radius than hedonic
                    # Stronger at center, fading outward
                    activation = value * np.exp(-0.1 * dist) * self.density_field[i, j]
                    self.growth_field[i, j] += activation
    
    def _update_purpose_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the purpose field with new value."""
        if value <= 0:
            return
            
        # Create gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 15:  # Wider activation radius
                    # Stronger at center, fading outward
                    activation = value * np.exp(-0.1 * dist) * self.density_field[i, j]
                    self.purpose_field[i, j] += activation
    
    def _update_virtue_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the virtue field with new value."""
        if value <= 0:
            return
            
        # Create gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 15:  # Wider activation radius
                    # Stronger at center, fading outward
                    activation = value * np.exp(-0.1 * dist) * self.density_field[i, j]
                    self.virtue_field[i, j] += activation
    
    def _calculate_goal_alignment(self, input_data: Dict[str, Any]) -> float:
        """Calculate alignment with current goals."""
        # Simple placeholder implementation
        # In a full implementation, would check against stored goal structures
        return np.random.uniform(0.3, 1.0)
    
    def _calculate_value_alignment(self, input_data: Dict[str, Any]) -> float:
        """Calculate alignment with current values."""
        # Simple placeholder implementation
        # In a full implementation, would check against values matrix
        return np.random.uniform(0.3, 1.0)
    
    def decay_fields(self) -> None:
        """Apply decay to eudaimonic fields."""
        decay_rate = 0.97  # Slower decay than hedonic
        self.growth_field *= decay_rate
        self.purpose_field *= decay_rate
        self.virtue_field *= decay_rate


class TranscendentProcessingDomain(ProcessingDomain):
    """Processes transcendent aspects of experience (unity, connection, boundary dissolution)."""
    
    def __init__(self, grid_size: int = 50):
        """Initialize the transcendent processing domain."""
        super().__init__("transcendent", grid_size)
        self.unity_field = np.zeros((grid_size, grid_size))
        self.connection_field = np.zeros((grid_size, grid_size))
        self.boundary_dissolution_field = np.zeros((grid_size, grid_size))
        self.network_connections = np.zeros((grid_size, grid_size, grid_size, grid_size))  # Simple connectome
        self.temporal_discount_rate = 0.3  # Low discount rate - longer time horizons
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through transcendent algorithms.
        
        Args:
            input_data: Dictionary of input data
            
        Returns:
            Dictionary with transcendent processing results
        """
        # Extract transcendent aspects from input
        unity_value = input_data.get('unity_value', 0.0)
        connection_value = input_data.get('connection_value', 0.0)
        dissolution_value = input_data.get('dissolution_value', 0.0)
        location = input_data.get('location', (self.grid_size // 2, self.grid_size // 2))
        
        # Process through transcendent fields
        self._update_unity_field(unity_value, location)
        self._update_connection_field(connection_value, location)
        self._update_dissolution_field(dissolution_value, location)
        
        # Update network connectivity
        self._update_network_connectivity(location, max(unity_value, connection_value, dissolution_value))
        
        # Calculate information integration
        integration_level = self._calculate_integration()
        
        # Calculate coherence gain
        coherence_gain = self._calculate_coherence_gain()
        
        # Calculate net transcendent impact
        net_transcendent_value = unity_value + connection_value + dissolution_value + coherence_gain
        
        return {
            "domain": "transcendent",
            "net_value": net_transcendent_value,
            "integration_level": integration_level,
            "coherence_gain": coherence_gain,
            "boundary_dissolution": np.mean(self.boundary_dissolution_field)
        }
    
    def _update_unity_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the unity field with new value."""
        if value <= 0:
            return
            
        # Create wide gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 20:  # Widest activation radius
                    # Stronger at center, fading outward gradually
                    activation = value * np.exp(-0.05 * dist) * self.density_field[i, j]
                    self.unity_field[i, j] += activation
    
    def _update_connection_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the connection field with new value."""
        if value <= 0:
            return
            
        # Create wide gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 20:  # Widest activation radius
                    # Stronger at center, fading outward gradually
                    activation = value * np.exp(-0.05 * dist) * self.density_field[i, j]
                    self.connection_field[i, j] += activation
    
    def _update_dissolution_field(self, value: float, location: Tuple[int, int]) -> None:
        """Update the boundary dissolution field with new value."""
        if value <= 0:
            return
            
        # Create wide gaussian activation around the location
        x, y = location
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - x)**2 + (j - y)**2)
                if dist < 20:  # Widest activation radius
                    # Stronger at center, fading outward gradually
                    activation = value * np.exp(-0.05 * dist) * self.density_field[i, j]
                    self.boundary_dissolution_field[i, j] += activation
    
    def _update_network_connectivity(self, location: Tuple[int, int], value: float) -> None:
        """Update network connectivity based on activation."""
        if value <= 0:
            return
            
        # Simplified implementation - increase connectivity around activated location
        x, y = location
        radius = 10
        connection_strength = value * 0.1
        
        for i in range(max(0, x - radius), min(self.grid_size, x + radius)):
            for j in range(max(0, y - radius), min(self.grid_size, y + radius)):
                for k in range(max(0, x - radius), min(self.grid_size, x + radius)):
                    for l in range(max(0, y - radius), min(self.grid_size, y + radius)):
                        if i != k or j != l:  # Don't connect to self
                            # Add connection strength, capped at 1.0
                            self.network_connections[i, j, k, l] = min(
                                1.0, 
                                self.network_connections[i, j, k, l] + connection_strength
                            )
    
    def _calculate_integration(self) -> float:
        """Calculate information integration level across the network."""
        # Simplified measure - mean connection strength
        return np.mean(self.network_connections)
    
    def _calculate_coherence_gain(self) -> float:
        """Calculate coherence gain from integration."""
        # Simplified implementation
        integration = self._calculate_integration()
        return integration * np.mean(self.unity_field) * 2.0
    
    def decay_fields(self) -> None:
        """Apply decay to transcendent fields."""
        decay_rate = 0.98  # Slowest decay
        self.unity_field *= decay_rate
        self.connection_field *= decay_rate
        self.boundary_dissolution_field *= decay_rate
        self.network_connections *= 0.99  # Very slow decay of connections


class OscillatoryController:
    """Controls transitions between processing modalities."""
    
    def __init__(self, transition_rates=None, modal_residence_times=None, 
                 development_stage="novice"):
        """
        Initialize the oscillatory controller.
        
        Args:
            transition_rates: Dictionary of transition probabilities between modalities
            modal_residence_times: Dictionary of average time spent in each modality
            development_stage: Current developmental stage of integration capacity
        """
        self.modal_focus = {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.33}
        self.transition_rates = transition_rates or {
            "hedonic_to_eudaimonic": 0.2,
            "hedonic_to_transcendent": 0.1,
            "eudaimonic_to_hedonic": 0.2,
            "eudaimonic_to_transcendent": 0.2,
            "transcendent_to_hedonic": 0.1,
            "transcendent_to_eudaimonic": 0.2
        }
        self.modal_residence_times = modal_residence_times or {
            "hedonic": 5,
            "eudaimonic": 8,
            "transcendent": 3
        }
        self.current_residence_time = 0
        self.current_dominant_mode = "balanced"
        self.development_stage = development_stage
        self.development_factors = self._initialize_development_factors()
        
    def _initialize_development_factors(self) -> Dict[str, float]:
        """Initialize developmental adjustment factors based on stage."""
        if self.development_stage == "novice":
            return {
                "transition_multiplier": 0.7,  # Less likely to transition
                "integration_capacity": 0.5,   # Limited integration capacity
                "adaptive_ability": 0.5,       # Limited adaptation to context
                "modal_balance": {"hedonic": 0.5, "eudaimonic": 0.3, "transcendent": 0.2}
            }
        elif self.development_stage == "intermediate":
            return {
                "transition_multiplier": 1.0,  # Normal transition rates
                "integration_capacity": 0.8,   # Improved integration
                "adaptive_ability": 0.8,       # Better context adaptation
                "modal_balance": {"hedonic": 0.4, "eudaimonic": 0.4, "transcendent": 0.2}
            }
        elif self.development_stage == "advanced":
            return {
                "transition_multiplier": 1.2,  # More flexible transitions
                "integration_capacity": 1.0,   # Full integration capacity
                "adaptive_ability": 1.0,       # Full adaptation to context
                "modal_balance": {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34}
            }
        else:
            # Default to intermediate
            return {
                "transition_multiplier": 1.0,
                "integration_capacity": 0.8,
                "adaptive_ability": 0.8,
                "modal_balance": {"hedonic": 0.4, "eudaimonic": 0.4, "transcendent": 0.2}
            }
    
    def update(self, context: Dict[str, Any], internal_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update modal focus based on context and agent state.
        
        Args:
            context: Environmental context dictionary
            internal_state: Agent's internal state dictionary
            
        Returns:
            Updated modal focus weights
        """
        # Increment residence time in current dominant mode
        self.current_residence_time += 1
        
        # Determine current dominant mode
        self.current_dominant_mode = max(self.modal_focus, key=self.modal_focus.get)
        
        # Check for contextual triggers
        context_adjustment = self._calculate_context_adjustment(context)
        
        # Check for transition based on residence time
        should_transition = self._check_transition()
        
        if should_transition:
            # Determine new dominant mode
            new_mode = self._select_new_mode()
            
            # Execute transition
            self._execute_transition(new_mode)
            
            # Reset residence time
            self.current_residence_time = 0
        
        # Apply context adjustment
        self._apply_context_adjustment(context_adjustment)
        
        # Normalize to ensure sum = 1.0
        total = sum(self.modal_focus.values())
        for mode in self.modal_focus:
            self.modal_focus[mode] /= total
        
        return self.modal_focus
    
    def _calculate_context_adjustment(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate context-based adjustment to modal focus."""
        adjustment = {"hedonic": 0.0, "eudaimonic": 0.0, "transcendent": 0.0}
        
        # Check for threat (increases hedonic focus)
        threat_level = context.get('threat_level', 0.0)
        if threat_level > 0.5:
            adjustment["hedonic"] += 0.2 * threat_level * self.development_factors["adaptive_ability"]
            adjustment["eudaimonic"] -= 0.1 * threat_level * self.development_factors["adaptive_ability"]
            adjustment["transcendent"] -= 0.1 * threat_level * self.development_factors["adaptive_ability"]
        
        # Check for growth opportunity (increases eudaimonic focus)
        growth_opportunity = context.get('growth_opportunity', 0.0)
        if growth_opportunity > 0.5:
            adjustment["eudaimonic"] += 0.2 * growth_opportunity * self.development_factors["adaptive_ability"]
            adjustment["hedonic"] -= 0.1 * growth_opportunity * self.development_factors["adaptive_ability"]
            adjustment["transcendent"] -= 0.1 * growth_opportunity * self.development_factors["adaptive_ability"]
        
        # Check for connection opportunity (increases transcendent focus)
        connection_opportunity = context.get('connection_opportunity', 0.0)
        if connection_opportunity > 0.5:
            adjustment["transcendent"] += 0.2 * connection_opportunity * self.development_factors["adaptive_ability"]
            adjustment["hedonic"] -= 0.1 * connection_opportunity * self.development_factors["adaptive_ability"]
            adjustment["eudaimonic"] -= 0.1 * connection_opportunity * self.development_factors["adaptive_ability"]
        
        return adjustment
    
    def _check_transition(self) -> bool:
        """Check if a transition should occur based on residence time."""
        # Get residence time threshold for current mode
        threshold = self.modal_residence_times.get(self.current_dominant_mode, 5)
        
        # Apply developmental adjustment
        threshold = int(threshold * (1.0 / self.development_factors["transition_multiplier"]))
        
        # Determine transition probability based on time past threshold
        if self.current_residence_time < threshold:
            return False
        else:
            # Increasing probability after threshold
            p_transition = min(0.8, (self.current_residence_time - threshold) * 0.2)
            return np.random.random() < p_transition
    
    def _select_new_mode(self) -> str:
        """Select a new dominant mode based on transition probabilities."""
        # Get transition rate keys relevant to current mode
        transition_keys = [k for k in self.transition_rates.keys() 
                         if k.startswith(f"{self.current_dominant_mode}_to_")]
        
        if not transition_keys:
            # Fallback to random selection if no transitions defined
            modes = list(self.modal_focus.keys())
            modes.remove(self.current_dominant_mode)
            return np.random.choice(modes)
        
        # Extract destination modes
        destination_modes = [k.split("_to_")[1] for k in transition_keys]
        
        # Get transition probabilities
        probabilities = [self.transition_rates[k] for k in transition_keys]
        
        # Apply developmental adjustment
        probabilities = [p * self.development_factors["transition_multiplier"] for p in probabilities]
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            
            # Select new mode based on probabilities
            return np.random.choice(destination_modes, p=probabilities)
        else:
            # Equal probability if all zero
            return np.random.choice(destination_modes)
    
    def _execute_transition(self, new_mode: str) -> None:
        """Execute transition to new dominant mode."""
        # Calculate shift amount based on development stage
        shift_amount = 0.3 * self.development_factors["integration_capacity"]
        
        # Take weight from other modes proportionally
        total_others = sum(v for k, v in self.modal_focus.items() if k != new_mode)
        for mode in self.modal_focus:
            if mode == new_mode:
                self.modal_focus[mode] += shift_amount
            else:
                # Reduce proportionally to current weight
                reduction = (shift_amount * self.modal_focus[mode] / total_others) if total_others > 0 else 0
                self.modal_focus[mode] -= reduction
    
    def _apply_context_adjustment(self, adjustment: Dict[str, float]) -> None:
        """Apply context-based adjustment to modal focus."""
        for mode, value in adjustment.items():
            self.modal_focus[mode] += value


class MeaningStructure:
    """Represents the integrated output of multimodal processing."""
    
    def __init__(self):
        """Initialize an empty meaning structure."""
        self.narrative_elements = []  # Temporally ordered experiences
        self.identity_components = {} # Stable self-representations
        self.value_framework = {}     # Evaluative structures
        self.coherence_score = 0.0    # Measure of internal consistency
        self.resilience_score = 0.0   # Ability to maintain integrity under perturbation
        
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
            "time": len(self.narrative_elements),
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
        
        # Recalculate coherence and resilience
        self._calculate_coherence()
        self._calculate_resilience()
    
    def _update_identity_components(self,
                                   hedonic_output: Dict[str, Any],
                                   eudaimonic_output: Dict[str, Any],
                                   transcendent_output: Dict[str, Any],
                                   modal_weights: Dict[str, float]) -> None:
        """Update identity components based on modal outputs."""
        # Extract identity-relevant information from each modality
        # This is a simplified implementation - would be more sophisticated in full system
        
        # Hedonic identity components (preferences, desires)
        if "dominant_vector" in hedonic_output:
            approach_orientation = 1.0 if hedonic_output["dominant_vector"] == "approach" else -1.0
            self.identity_components["hedonic_orientation"] = (
                self.identity_components.get("hedonic_orientation", 0.0) * 0.9 +
                approach_orientation * hedonic_output.get("intensity", 0.0) * 0.1
            )
        
        # Eudaimonic identity components (goals, virtues)
        if "goal_alignment" in eudaimonic_output:
            self.identity_components["purpose_clarity"] = (
                self.identity_components.get("purpose_clarity", 0.0) * 0.9 +
                eudaimonic_output.get("goal_alignment", 0.0) * 0.1
            )
            
        # Transcendent identity components (connectedness, boundary dissolution)
        if "integration_level" in transcendent_output:
            self.identity_components["self_boundaries"] = (
                self.identity_components.get("self_boundaries", 1.0) * 0.9 +
                (1.0 - transcendent_output.get("boundary_dissolution", 0.0)) * 0.1
            )
    
    def _update_value_framework(self,
                               hedonic_output: Dict[str, Any],
                               eudaimonic_output: Dict[str, Any],
                               transcendent_output: Dict[str, Any],
                               modal_weights: Dict[str, float]) -> None:
        """Update value framework based on modal outputs."""
        # Simplified implementation - full system would have more sophisticated value structure
        
        # Update hedonic value
        self.value_framework["pleasure_valuation"] = (
            self.value_framework.get("pleasure_valuation", 0.5) * 0.95 +
            min(1.0, max(0.0, 0.5 + 0.1 * hedonic_output.get("net_value", 0.0))) * 0.05
        )
        
        # Update eudaimonic value
        if "growth_contribution" in eudaimonic_output:
            self.value_framework["growth_valuation"] = (
                self.value_framework.get("growth_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * eudaimonic_output.get("net_value", 0.0))) * 0.05
            )
        
        # Update transcendent value
        if "coherence_gain" in transcendent_output:
            self.value_framework["unity_valuation"] = (
                self.value_framework.get("unity_valuation", 0.5) * 0.95 +
                min(1.0, max(0.0, 0.5 + 0.1 * transcendent_output.get("coherence_gain", 0.0))) * 0.05
            )
    
    def _calculate_coherence(self) -> None:
        """Calculate narrative coherence score."""
        if len(self.narrative_elements) < 2:
            self.coherence_score = 0.5  # Default for new structures
            return
        
        # Simplified coherence calculation - would be more sophisticated in full system
        coherence_sum = 0.0
        
        # Calculate temporal consistency across narrative elements
        for i in range(1, len(self.narrative_elements)):
            prev = self.narrative_elements[i-1]
            curr = self.narrative_elements[i]
            
            # Calculate consistency between consecutive elements
            consistency = 1.0 - 0.3 * abs(prev["net_meaning"] - curr["net_meaning"])
            coherence_sum += consistency
        
        # Average consistency, scaled to 0-1 range
        self.coherence_score = min(1.0, max(0.0, coherence_sum / (len(self.narrative_elements) - 1)))
    
    def _calculate_resilience(self) -> None:
        """Calculate meaning resilience score."""
        # Simplified resilience calculation - would be more sophisticated in full system
        
        # Factors contributing to resilience:
        # 1. Modal balance - more balanced = more resilient
        # 2. Identity stability - stronger identity = more resilient
        # 3. Value framework clarity - clearer values = more resilient
        # 4. Narrative coherence - more coherent narrative = more resilient
        
        # Calculate modal balance factor
        modal_balance = 0.0
        if len(self.narrative_elements) > 0:
            latest = self.narrative_elements[-1]
            weights = latest["modal_weights"]
            # Perfect balance would be 0.33 for each modality
            balance_deviation = sum(abs(v - 0.33) for v in weights.values()) / 3.0
            modal_balance = 1.0 - balance_deviation
        
        # Calculate identity stability
        identity_stability = min(1.0, len(self.identity_components) / 5.0)
        
        # Calculate value framework clarity
        value_clarity = min(1.0, len(self.value_framework) / 3.0)
        
        # Combine factors
        self.resilience_score = (
            modal_balance * 0.3 +
            identity_stability * 0.3 +
            value_clarity * 0.2 +
            self.coherence_score * 0.2
        )


class MetaModalAgent:
    """Agent with meta-modal processing capacity for meaning construction."""
    
    def __init__(self, agent_id: str, grid_size: int = 50, 
                 oscillation_params: Optional[Dict[str, Any]] = None, 
                 development_stage: str = "novice"):
        """
        Initialize a meta-modal agent.
        
        Args:
            agent_id: Unique identifier for this agent
            grid_size: Size of the mental landscape grid
            oscillation_params: Parameters governing modal transitions
            development_stage: Current integration capability level
        """
        self.id = agent_id
        self.grid_size = grid_size
        self.t = 0  # Time step counter
        
        # Initialize processing domains
        self.hedonic_domain = HedonicProcessingDomain(grid_size)
        self.eudaimonic_domain = EudaimonicProcessingDomain(grid_size)
        self.transcendent_domain = TranscendentProcessingDomain(grid_size)
        
        # Initialize oscillatory controller
        self.oscillatory_controller = OscillatoryController(
            oscillation_params,
            development_stage=development_stage
        )
        
        # Initialize meaning structure
        self.meaning_structure = MeaningStructure()
        
        # Initialize state tracking
        self.internal_state = {
            "modal_focus": {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34},
            "cognitive_load": 0.0,
            "emotional_state": "neutral"
        }
        
        # Initialize metrics tracking
        self.metrics_history = {
            "modal_focus": [],
            "coherence": [],
            "resilience": [],
            "cognitive_load": []
        }
    
    def process_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an external observation through all modalities.
        
        Args:
            observation: Dictionary containing observation data
            
        Returns:
            Dictionary with processing results
        """
        # Update cognitive load
        self.internal_state["cognitive_load"] += observation.get("intensity", 0.5) * 0.1
        
        # Determine location in grid (could be based on observation characteristics)
        if "location" not in observation:
            observation["location"] = (
                np.random.randint(10, self.grid_size - 10),
                np.random.randint(10, self.grid_size - 10)
            )
        
        # Prepare modality-specific inputs
        hedonic_input = self._prepare_hedonic_input(observation)
        eudaimonic_input = self._prepare_eudaimonic_input(observation)
        transcendent_input = self._prepare_transcendent_input(observation)
        
        # Process through each domain
        hedonic_output = self.hedonic_domain.process_input(hedonic_input)
        eudaimonic_output = self.eudaimonic_domain.process_input(eudaimonic_input)
        transcendent_output = self.transcendent_domain.process_input(transcendent_input)
        
        # Update modal focus based on context and internal state
        context = self._derive_context_from_observation(observation)
        self.internal_state["modal_focus"] = self.oscillatory_controller.update(
            context, self.internal_state)
        
        # Integrate outputs into meaning structure
        self.meaning_structure.integrate_modal_outputs(
            hedonic_output, 
            eudaimonic_output, 
            transcendent_output, 
            self.internal_state["modal_focus"]
        )
        
        # Record metrics
        self._record_metrics()
        
        # Generate response
        response = self._generate_response(
            hedonic_output, eudaimonic_output, transcendent_output)
        
        # Advance time step
        self.t += 1
        
        return response
    
    def _prepare_hedonic_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation for hedonic processing."""
        # Extract hedonic aspects of observation
        input_data = {
            "pleasure_value": observation.get("pleasure_value", 0.0),
            "pain_value": observation.get("pain_value", 0.0),
            "location": observation.get("location")
        }
        
        # If no explicit hedonic values, infer from observation type
        if input_data["pleasure_value"] == 0.0 and input_data["pain_value"] == 0.0:
            obs_type = observation.get("type", "neutral")
            
            if obs_type in ["reward", "success", "connection"]:
                input_data["pleasure_value"] = observation.get("intensity", 0.5)
            elif obs_type in ["threat", "failure", "loss"]:
                input_data["pain_value"] = observation.get("intensity", 0.5)
        
        return input_data
    
    def _prepare_eudaimonic_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation for eudaimonic processing."""
        # Extract eudaimonic aspects of observation
        input_data = {
            "growth_value": observation.get("growth_value", 0.0),
            "purpose_value": observation.get("purpose_value", 0.0),
            "virtue_value": observation.get("virtue_value", 0.0),
            "location": observation.get("location")
        }
        
        # If no explicit eudaimonic values, infer from observation type
        if (input_data["growth_value"] == 0.0 and
            input_data["purpose_value"] == 0.0 and
            input_data["virtue_value"] == 0.0):
            
            obs_type = observation.get("type", "neutral")
            
            if obs_type in ["achievement", "learning", "skill"]:
                input_data["growth_value"] = observation.get("intensity", 0.5)
            elif obs_type in ["purpose", "meaning", "goal"]:
                input_data["purpose_value"] = observation.get("intensity", 0.5)
            elif obs_type in ["virtue", "morality", "ethics"]:
                input_data["virtue_value"] = observation.get("intensity", 0.5)
        
        return input_data
    
    def _prepare_transcendent_input(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare observation for transcendent processing."""
        # Extract transcendent aspects of observation
        input_data = {
            "unity_value": observation.get("unity_value", 0.0),
            "connection_value": observation.get("connection_value", 0.0),
            "dissolution_value": observation.get("dissolution_value", 0.0),
            "location": observation.get("location")
        }
        
        # If no explicit transcendent values, infer from observation type
        if (input_data["unity_value"] == 0.0 and
            input_data["connection_value"] == 0.0 and
            input_data["dissolution_value"] == 0.0):
            
            obs_type = observation.get("type", "neutral")
            
            if obs_type in ["connection", "unity", "oneness"]:
                input_data["unity_value"] = observation.get("intensity", 0.5)
                input_data["connection_value"] = observation.get("intensity", 0.5)
            elif obs_type in ["awe", "wonder", "transcendence"]:
                input_data["dissolution_value"] = observation.get("intensity", 0.5)
        
        return input_data
    
    def _derive_context_from_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Derive context information from observation."""
        context = {}
        
        obs_type = observation.get("type", "neutral")
        intensity = observation.get("intensity", 0.5)
        
        # Set context variables based on observation type
        if obs_type in ["threat", "failure", "loss"]:
            context["threat_level"] = intensity
        else:
            context["threat_level"] = 0.1
            
        if obs_type in ["achievement", "learning", "skill"]:
            context["growth_opportunity"] = intensity
        else:
            context["growth_opportunity"] = 0.1
            
        if obs_type in ["connection", "unity", "oneness"]:
            context["connection_opportunity"] = intensity
        else:
            context["connection_opportunity"] = 0.1
        
        return context
    
    def _generate_response(self, 
                         hedonic_output: Dict[str, Any],
                         eudaimonic_output: Dict[str, Any],
                         transcendent_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent response based on integrated processing."""
        # Combine modal outputs using current focus weights
        net_hedonic = hedonic_output.get("net_value", 0.0) * self.internal_state["modal_focus"]["hedonic"]
        net_eudaimonic = eudaimonic_output.get("net_value", 0.0) * self.internal_state["modal_focus"]["eudaimonic"]
        net_transcendent = transcendent_output.get("net_value", 0.0) * self.internal_state["modal_focus"]["transcendent"]
        
        net_response = net_hedonic + net_eudaimonic + net_transcendent
        
        # Determine response type based on dominant modality
        dominant_modality = max(self.internal_state["modal_focus"], 
                               key=self.internal_state["modal_focus"].get)
        
        # Determine emotional state based on valence
        if net_response > 0.3:
            emotional_state = "positive"
        elif net_response < -0.3:
            emotional_state = "negative"
        else:
            emotional_state = "neutral"
            
        self.internal_state["emotional_state"] = emotional_state
        
        # Build response
        response = {
            "agent_id": self.id,
            "time_step": self.t,
            "emotional_state": emotional_state,
            "response_intensity": abs(net_response),
            "modal_focus": self.internal_state["modal_focus"].copy(),
            "meaning_coherence": self.meaning_structure.coherence_score,
            "meaning_resilience": self.meaning_structure.resilience_score,
            "dominant_modality": dominant_modality
        }
        
        return response
    
    def _record_metrics(self) -> None:
        """Record agent metrics for analysis."""
        self.metrics_history["modal_focus"].append(self.internal_state["modal_focus"].copy())
        self.metrics_history["coherence"].append(self.meaning_structure.coherence_score)
        self.metrics_history["resilience"].append(self.meaning_structure.resilience_score)
        self.metrics_history["cognitive_load"].append(self.internal_state["cognitive_load"])
    
    def step(self) -> None:
        """Advance the agent's internal state by one time step without new input."""
        # Apply decay to processing domains
        self.hedonic_domain.decay_fields()
        self.eudaimonic_domain.decay_fields()
        self.transcendent_domain.decay_fields()
        
        # Reduce cognitive load
        self.internal_state["cognitive_load"] = max(0.0, self.internal_state["cognitive_load"] * 0.95 - 0.05)
        
        # Update modal focus (even without new input, focus can shift)
        context = {
            "threat_level": 0.1,
            "growth_opportunity": 0.1,
            "connection_opportunity": 0.1
        }
        self.internal_state["modal_focus"] = self.oscillatory_controller.update(
            context, self.internal_state)
        
        # Record metrics
        self._record_metrics()
        
        # Advance time step
        self.t += 1
    
    def interact_with_agent(self, other_agent: 'MetaModalAgent') -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Interact with another agent, exchanging meaning elements.
        
        Args:
            other_agent: Another MetaModalAgent to interact with
            
        Returns:
            Tuple of interaction results (self result, other result)
        """
        # Create interaction observation for each agent
        my_observation = {
            "type": "interaction",
            "intensity": 0.7,
            "connection_value": 0.6,
            "growth_value": 0.3
        }
        
        other_observation = {
            "type": "interaction",
            "intensity": 0.7,
            "connection_value": 0.6,
            "growth_value": 0.3
        }
        
        # Process the interaction through both agents
        my_result = self.process_observation(my_observation)
        other_result = other_agent.process_observation(other_observation)
        
        return my_result, other_result
    
    def visualize_mental_state(self) -> None:
        """Visualize the agent's current mental state."""
        # Placeholder - would create visualizations of modal fields and meaning structure
        pass


if __name__ == "__main__":
    # Simple testing code
    agent = MetaModalAgent("test_agent", grid_size=30, development_stage="intermediate")
    
    # Process a few observations
    observations = [
        {"type": "reward", "intensity": 0.8},
        {"type": "threat", "intensity": 0.6},
        {"type": "connection", "intensity": 0.9},
        {"type": "achievement", "intensity": 0.7}
    ]
    
    print(f"Testing MetaModalAgent with {len(observations)} observations...")
    for i, obs in enumerate(observations):
        print(f"\nObservation {i+1}: {obs}")
        response = agent.process_observation(obs)
        print(f"Response: {response}")
    
    print("\nFinal meaning structure metrics:")
    print(f"Coherence: {agent.meaning_structure.coherence_score:.3f}")
    print(f"Resilience: {agent.meaning_structure.resilience_score:.3f}")
