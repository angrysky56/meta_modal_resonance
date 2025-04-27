"""
Oscillatory Controller Module

This module implements the OscillatoryController class, which manages transitions
between the three processing modalities (hedonic, eudaimonic, and transcendent)
based on contextual demands, developmental stage, and internal dynamics.

The controller implements the dynamic oscillation component of the Meta-Modal 
Resonance Theory, enabling adaptive shifts in processing emphasis that respond
to environmental demands while maintaining integrative coherence.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any


class OscillatoryController:
    """
    Controls dynamic transitions between processing modalities.
    
    The oscillatory controller manages the dynamic focusing of cognitive resources
    across the three modalities based on:
    1. Current contextual demands
    2. Internal state dynamics
    3. Developmental capacity for integration
    4. Temporal residence patterns
    
    Key functions include:
    - Monitoring modal residence time
    - Calculating transition probabilities
    - Executing modal transitions
    - Adapting to contextual triggers
    - Applying developmental constraints
    """
    
    def __init__(self, 
                transition_params: Optional[Dict[str, Any]] = None, 
                development_stage: str = "novice",
                oscillatory_pattern: str = "adaptive"):
        """
        Initialize the oscillatory controller.
        
        Args:
            transition_params: Parameters governing modal transitions
            development_stage: Current developmental stage of integration capacity
            oscillatory_pattern: Pattern of oscillation (fixed, regular, adaptive, etc.)
        """
        # Modal focus weights (proportion of processing in each modality)
        self.modal_focus = {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34}
        
        # Set oscillatory pattern
        self.oscillatory_pattern = oscillatory_pattern
        
        # Initialize transition parameters
        self.transition_params = transition_params or {}
        self._set_default_transition_params()
        
        # Set developmental stage and factors
        self.development_stage = development_stage
        self.development_factors = self._initialize_development_factors()
        
        # Time tracking
        self.t = 0  # Internal time counter
        self.current_residence_time = 0
        self.modal_residence_history = {"hedonic": [], "eudaimonic": [], "transcendent": []}
        self.current_dominant_mode = "balanced"
        
        # State tracking
        self.prev_modal_focus = self.modal_focus.copy()
        self.transition_history = []
        
    def _set_default_transition_params(self) -> None:
        """Set default transition parameters if not provided."""
        # Set default transition rates if not in params
        if "transition_rates" not in self.transition_params:
            self.transition_params["transition_rates"] = {
                "hedonic_to_eudaimonic": 0.2,
                "hedonic_to_transcendent": 0.1,
                "eudaimonic_to_hedonic": 0.2,
                "eudaimonic_to_transcendent": 0.2,
                "transcendent_to_hedonic": 0.1,
                "transcendent_to_eudaimonic": 0.2
            }
            
        # Set default residence times if not in params
        if "modal_residence_times" not in self.transition_params:
            self.transition_params["modal_residence_times"] = {
                "hedonic": 5,     # Shorter for hedonic (more reactive)
                "eudaimonic": 8,  # Longer for eudaimonic (more stable)
                "transcendent": 3 # Variable for transcendent (episodic)
            }
            
        # Set default oscillatory parameters if not in params
        if "oscillatory_params" not in self.transition_params:
            self.transition_params["oscillatory_params"] = {
                "base_frequency": 0.2,  # Base oscillation frequency
                "phase_shift": {        # Phase shift between modalities
                    "hedonic_eudaimonic": np.pi / 3,
                    "hedonic_transcendent": 2 * np.pi / 3,
                    "eudaimonic_transcendent": np.pi
                },
                "amplitude_modulation": 0.3  # Amplitude of oscillation
            }
    
    def _initialize_development_factors(self) -> Dict[str, Any]:
        """Initialize developmental adjustment factors based on stage."""
        if self.development_stage == "novice":
            return {
                "transition_multiplier": 0.7,  # Less likely to transition
                "integration_capacity": 0.5,   # Limited integration capacity
                "adaptive_ability": 0.5,       # Limited adaptation to context
                "modal_balance": {"hedonic": 0.5, "eudaimonic": 0.3, "transcendent": 0.2},
                "oscillatory_stability": 0.6,  # Less stable oscillation
                "perturbation_resistance": 0.4,  # More easily disrupted
                "rigid_transition_sequences": True,  # More rigid transition patterns
                "entropy_capacity": 0.3  # Limited capacity for disorder
            }
        elif self.development_stage == "intermediate":
            return {
                "transition_multiplier": 1.0,  # Normal transition rates
                "integration_capacity": 0.8,   # Improved integration
                "adaptive_ability": 0.8,       # Better context adaptation
                "modal_balance": {"hedonic": 0.4, "eudaimonic": 0.4, "transcendent": 0.2},
                "oscillatory_stability": 0.8,  # More stable oscillation
                "perturbation_resistance": 0.7,  # Better perturbation resistance
                "rigid_transition_sequences": False,  # More flexible transitions
                "entropy_capacity": 0.6  # Better capacity for disorder
            }
        elif self.development_stage == "advanced":
            return {
                "transition_multiplier": 1.2,  # More flexible transitions
                "integration_capacity": 1.0,   # Full integration capacity
                "adaptive_ability": 1.0,       # Full adaptation to context
                "modal_balance": {"hedonic": 0.33, "eudaimonic": 0.33, "transcendent": 0.34},
                "oscillatory_stability": 1.0,  # Fully stable oscillation
                "perturbation_resistance": 0.9,  # High perturbation resistance
                "rigid_transition_sequences": False,  # Fully flexible transitions
                "entropy_capacity": 0.9  # High capacity for disorder
            }
        else:
            # Default to intermediate
            return {
                "transition_multiplier": 1.0,
                "integration_capacity": 0.8,
                "adaptive_ability": 0.8,
                "modal_balance": {"hedonic": 0.4, "eudaimonic": 0.4, "transcendent": 0.2},
                "oscillatory_stability": 0.8,
                "perturbation_resistance": 0.7,
                "rigid_transition_sequences": False,
                "entropy_capacity": 0.6
            }
    
    def update(self, context: Dict[str, Any], internal_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update modal focus based on context and agent's internal state.
        
        Args:
            context: Environmental context dictionary
            internal_state: Agent's internal state dictionary
            
        Returns:
            Updated modal focus weights
        """
        # Store previous modal focus for transition detection
        self.prev_modal_focus = self.modal_focus.copy()
        
        # Determine current dominant mode
        self.current_dominant_mode = max(self.modal_focus, key=self.modal_focus.get)
        
        # Update oscillation pattern based on specific pattern type
        if self.oscillatory_pattern == "fixed":
            # Fixed pattern maintains constant weights
            pass  # No change to modal_focus
            
        elif self.oscillatory_pattern == "regular":
            # Regular pattern uses sine wave oscillation
            self._apply_regular_oscillation()
            
        elif self.oscillatory_pattern == "random":
            # Random pattern uses probabilistic transitions
            self._apply_random_oscillation()
            
        elif self.oscillatory_pattern == "adaptive":
            # Adaptive pattern combines contextual and temporal factors
            
            # Check for contextual triggers
            context_adjustment = self._calculate_context_adjustment(context)
            
            # Increment residence time in current dominant mode
            self.current_residence_time += 1
            
            # Check for transition based on residence time
            should_transition = self._check_transition(internal_state)
            
            if should_transition:
                # Determine new dominant mode
                new_mode = self._select_new_mode(context)
                
                # Execute transition
                self._execute_transition(new_mode)
                
                # Record transition
                self.transition_history.append({
                    "time": self.t,
                    "from": self.current_dominant_mode,
                    "to": new_mode,
                    "context_trigger": max(context_adjustment.values()) > 0.2
                })
                
                # Reset residence time
                self.current_residence_time = 0
                
                # Record residence time in previous mode
                self.modal_residence_history[self.current_dominant_mode].append(self.current_residence_time)
                
                # Update current dominant mode
                self.current_dominant_mode = new_mode
            
            # Apply context adjustment
            self._apply_context_adjustment(context_adjustment)
        
        # Ensure modal focus stays within valid ranges
        self._normalize_modal_focus()
        
        # Increment time counter
        self.t += 1
        
        return self.modal_focus
    
    def _apply_regular_oscillation(self) -> None:
        """Apply regular oscillatory pattern based on sine waves."""
        # Create phase-shifted sine waves for each modality
        t_scaled = self.t * self.transition_params["oscillatory_params"]["base_frequency"]
        oscillatory_params = self.transition_params["oscillatory_params"]
        
        # Base oscillation with phase shifts
        hedonic_osc = 0.33 + oscillatory_params["amplitude_modulation"] * np.sin(t_scaled)
        eudaimonic_osc = 0.33 + oscillatory_params["amplitude_modulation"] * np.sin(
            t_scaled + oscillatory_params["phase_shift"]["hedonic_eudaimonic"])
        transcendent_osc = 0.33 + oscillatory_params["amplitude_modulation"] * np.sin(
            t_scaled + oscillatory_params["phase_shift"]["hedonic_transcendent"])
        
        # Apply developmental stability factor
        stability = self.development_factors["oscillatory_stability"]
        
        # Apply oscillation with stability factor
        self.modal_focus["hedonic"] = self.modal_focus["hedonic"] * (1 - stability) + hedonic_osc * stability
        self.modal_focus["eudaimonic"] = self.modal_focus["eudaimonic"] * (1 - stability) + eudaimonic_osc * stability
        self.modal_focus["transcendent"] = self.modal_focus["transcendent"] * (1 - stability) + transcendent_osc * stability
    
    def _apply_random_oscillation(self) -> None:
        """Apply random oscillatory pattern based on stochastic transitions."""
        # Only consider transition at certain intervals
        if self.t % 3 != 0:
            return
            
        # Determine random weight changes
        change_amount = 0.1 * random.random() * self.development_factors["entropy_capacity"]
        
        # Select random modality to increase
        increase_mode = random.choice(list(self.modal_focus.keys()))
        
        # Select random modality to decrease (different from increase)
        other_modes = [m for m in self.modal_focus.keys() if m != increase_mode]
        decrease_mode = random.choice(other_modes)
        
        # Apply changes
        self.modal_focus[increase_mode] += change_amount
        self.modal_focus[decrease_mode] -= change_amount
    
    def _calculate_context_adjustment(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate context-based adjustment to modal focus."""
        adjustment = {"hedonic": 0.0, "eudaimonic": 0.0, "transcendent": 0.0}
        
        # Environmental threat triggers hedonic (survival) processing
        threat_level = context.get('threat_level', 0.0)
        if threat_level > 0.5:
            adjustment["hedonic"] += 0.2 * threat_level * self.development_factors["adaptive_ability"]
            adjustment["eudaimonic"] -= 0.1 * threat_level * self.development_factors["adaptive_ability"]
            adjustment["transcendent"] -= 0.1 * threat_level * self.development_factors["adaptive_ability"]
        
        # Growth opportunities trigger eudaimonic processing
        growth_opportunity = context.get('growth_opportunity', 0.0)
        if growth_opportunity > 0.5:
            adjustment["eudaimonic"] += 0.2 * growth_opportunity * self.development_factors["adaptive_ability"]
            adjustment["hedonic"] -= 0.1 * growth_opportunity * self.development_factors["adaptive_ability"]
            adjustment["transcendent"] -= 0.1 * growth_opportunity * self.development_factors["adaptive_ability"]
        
        # Connection opportunities trigger transcendent processing
        connection_opportunity = context.get('connection_opportunity', 0.0)
        if connection_opportunity > 0.5:
            adjustment["transcendent"] += 0.2 * connection_opportunity * self.development_factors["adaptive_ability"]
            adjustment["hedonic"] -= 0.1 * connection_opportunity * self.development_factors["adaptive_ability"]
            adjustment["eudaimonic"] -= 0.1 * connection_opportunity * self.development_factors["adaptive_ability"]
        
        # Cognitive load affects modal distribution
        cognitive_load = context.get('cognitive_load', 0.0)
        if cognitive_load > 0.7:
            # Under high load, shift toward hedonic (more immediate processing)
            adjustment["hedonic"] += 0.1 * cognitive_load * self.development_factors["adaptive_ability"]
            adjustment["eudaimonic"] -= 0.05 * cognitive_load * self.development_factors["adaptive_ability"]
            adjustment["transcendent"] -= 0.05 * cognitive_load * self.development_factors["adaptive_ability"]
        
        return adjustment
    
    def _check_transition(self, internal_state: Dict[str, Any]) -> bool:
        """Check if a transition should occur based on residence time and internal state."""
        # Get residence time threshold for current mode
        threshold = self.transition_params["modal_residence_times"].get(self.current_dominant_mode, 5)
        
        # Apply developmental adjustment
        threshold = int(threshold * (1.0 / self.development_factors["transition_multiplier"]))
        
        # Check for perturbations that might trigger early transition
        perturbation_strength = internal_state.get('perturbation_strength', 0.0)
        perturbation_resistance = self.development_factors["perturbation_resistance"]
        
        # Perturbation can trigger early transition if strong enough
        if perturbation_strength > perturbation_resistance:
            perturbation_transition_probability = (perturbation_strength - perturbation_resistance) * 0.5
            if random.random() < perturbation_transition_probability:
                return True
        
        # Determine transition probability based on time past threshold
        if self.current_residence_time < threshold:
            # Small chance of early transition
            return random.random() < 0.05 * self.development_factors["transition_multiplier"]
        else:
            # Increasing probability after threshold
            time_past_threshold = self.current_residence_time - threshold
            p_transition = min(0.8, 0.2 + time_past_threshold * 0.1)
            return random.random() < p_transition * self.development_factors["transition_multiplier"]
    
    def _select_new_mode(self, context: Dict[str, Any]) -> str:
        """Select a new dominant mode based on transition probabilities and context."""
        # Get transition rate keys relevant to current mode
        transition_keys = [k for k in self.transition_params["transition_rates"].keys() 
                         if k.startswith(f"{self.current_dominant_mode}_to_")]
        
        if not transition_keys:
            # Fallback to random selection if no transitions defined
            modes = list(self.modal_focus.keys())
            modes.remove(self.current_dominant_mode)
            return random.choice(modes)
        
        # Extract destination modes
        destination_modes = [k.split("_to_")[1] for k in transition_keys]
        
        # Get base transition probabilities
        probabilities = [self.transition_params["transition_rates"][k] for k in transition_keys]
        
        # Modify probabilities based on context
        for i, mode in enumerate(destination_modes):
            if mode == "hedonic" and context.get("threat_level", 0.0) > 0.5:
                probabilities[i] *= 1.5  # Increase transition to hedonic under threat
            elif mode == "eudaimonic" and context.get("growth_opportunity", 0.0) > 0.5:
                probabilities[i] *= 1.5  # Increase transition to eudaimonic with growth opportunity
            elif mode == "transcendent" and context.get("connection_opportunity", 0.0) > 0.5:
                probabilities[i] *= 1.5  # Increase transition to transcendent with connection opportunity
        
        # Apply developmental adjustment
        probabilities = [p * self.development_factors["transition_multiplier"] for p in probabilities]
        
        # For novice stage with rigid transitions, enforce particular sequences
        if (self.development_factors["rigid_transition_sequences"] and 
            self.development_stage == "novice"):
            
            # Enforce hedonic -> eudaimonic -> transcendent -> hedonic cycle
            if self.current_dominant_mode == "hedonic":
                return "eudaimonic"
            elif self.current_dominant_mode == "eudaimonic":
                return "transcendent"
            else:  # transcendent
                return "hedonic"
        
        # Normalize probabilities
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
            
            # Select new mode based on probabilities
            return random.choices(destination_modes, weights=probabilities, k=1)[0]
        else:
            # Equal probability if all zero
            return random.choice(destination_modes)
    
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
    
    def _normalize_modal_focus(self) -> None:
        """Ensure modal focus weights sum to 1.0 and are non-negative."""
        # Ensure no negative values
        for mode in self.modal_focus:
            self.modal_focus[mode] = max(0.0, self.modal_focus[mode])
            
        # Normalize to sum to 1.0
        total = sum(self.modal_focus.values())
        if total > 0:
            for mode in self.modal_focus:
                self.modal_focus[mode] /= total
    
    def get_oscillation_metrics(self) -> Dict[str, Any]:
        """Calculate metrics describing the oscillatory pattern."""
        metrics = {
            "modal_time_distribution": {},
            "transition_count": len(self.transition_history),
            "avg_residence_times": {},
            "transition_matrix": {
                "hedonic": {"eudaimonic": 0, "transcendent": 0},
                "eudaimonic": {"hedonic": 0, "transcendent": 0},
                "transcendent": {"hedonic": 0, "eudaimonic": 0}
            },
            "context_triggered_transitions": 0
        }
        
        # Calculate time distribution
        time_in_mode = {"hedonic": 0, "eudaimonic": 0, "transcendent": 0}
        for t in range(self.t):
            if t < len(self.transition_history):
                dominant_at_t = self.transition_history[t]["from"]
                time_in_mode[dominant_at_t] += 1
        
        for mode, time in time_in_mode.items():
            metrics["modal_time_distribution"][mode] = time / max(1, self.t)
        
        # Calculate average residence times
        for mode, times in self.modal_residence_history.items():
            if times:
                metrics["avg_residence_times"][mode] = sum(times) / len(times)
            else:
                metrics["avg_residence_times"][mode] = 0
        
        # Calculate transition matrix
        for transition in self.transition_history:
            from_mode = transition["from"]
            to_mode = transition["to"]
            metrics["transition_matrix"][from_mode][to_mode] += 1
            
            # Count context-triggered transitions
            if transition["context_trigger"]:
                metrics["context_triggered_transitions"] += 1
        
        # Calculate percentage of context-triggered transitions
        if self.transition_history:
            metrics["context_triggered_percent"] = (
                metrics["context_triggered_transitions"] / len(self.transition_history) * 100
            )
        else:
            metrics["context_triggered_percent"] = 0
            
        return metrics
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of the oscillatory controller."""
        return {
            "modal_focus": self.modal_focus.copy(),
            "current_dominant_mode": self.current_dominant_mode,
            "residence_time": self.current_residence_time,
            "time_step": self.t,
            "development_stage": self.development_stage,
            "oscillatory_pattern": self.oscillatory_pattern
        }
    
    def set_oscillatory_pattern(self, pattern: str) -> None:
        """Change the oscillatory pattern."""
        valid_patterns = ["fixed", "regular", "random", "adaptive"]
        if pattern in valid_patterns:
            self.oscillatory_pattern = pattern
        else:
            raise ValueError(f"Invalid oscillatory pattern: {pattern}. " +
                           f"Must be one of {valid_patterns}")
    
    def apply_perturbation(self, strength: float, target_mode: Optional[str] = None) -> None:
        """
        Apply a perturbation to the oscillatory system.
        
        Args:
            strength: Strength of the perturbation (0.0 to 1.0)
            target_mode: Specific mode to target, or None for general perturbation
        """
        # Scale perturbation based on resistance
        effective_strength = strength * (1.0 - self.development_factors["perturbation_resistance"])
        
        if target_mode:
            # Target specific mode
            if target_mode in self.modal_focus:
                # Reduce target mode
                self.modal_focus[target_mode] -= effective_strength * self.modal_focus[target_mode]
                
                # Distribute to other modes
                other_modes = [m for m in self.modal_focus.keys() if m != target_mode]
                for mode in other_modes:
                    self.modal_focus[mode] += effective_strength * self.modal_focus[target_mode] / len(other_modes)
        else:
            # General perturbation - randomize weights
            for mode in self.modal_focus:
                random_adjustment = (random.random() - 0.5) * effective_strength
                self.modal_focus[mode] += random_adjustment
        
        # Ensure valid weights
        self._normalize_modal_focus()
