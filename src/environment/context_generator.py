"""
Context Generator Module

This module provides the EnvironmentalContext class that generates varying
contexts for agents to respond to during simulations. The contexts contain
different combinations of stimuli targeting hedonic, eudaimonic, and transcendent
processing modalities with varying intensities and patterns.

The context generator creates environments that test the adaptability and
integration capabilities of agents under different conditions.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Any


class EnvironmentalContext:
    """
    Generates environmental inputs with different modal demands.

    The environmental context creates a variety of stimuli patterns to test
    agent adaptability, including:
    1. Varying intensities and valences
    2. Different combinations of modal triggers
    3. Patterns of environmental stability or volatility
    4. Contextual challenges or opportunities
    """

    def __init__(self,
                complexity: float = 1.0,
                volatility: float = 0.5,
                threat_level: float = 0.3,
                modal_bias: Optional[Dict[str, float]] = None):
        """
        Initialize the environmental context generator.

        Args:
            complexity: Overall complexity of generated contexts (0.0 to 1.0)
            volatility: Rate of change in generated contexts (0.0 to 1.0)
            threat_level: Base threat level in environments (0.0 to 1.0)
            modal_bias: Optional bias toward specific modality triggers
        """
        self.complexity = min(1.0, max(0.1, complexity))
        self.volatility = min(1.0, max(0.1, volatility))
        self.threat_level = min(1.0, max(0.1, threat_level))
        self.opportunity_level = 1.0 - threat_level

        # Set modal bias (which modality gets triggered more often)
        self.modal_bias = modal_bias or {
            "hedonic": 0.33,
            "eudaimonic": 0.33,
            "transcendent": 0.34
        }

        # Normalize modal bias
        total = sum(self.modal_bias.values())
        self.modal_bias = {k: v / total for k, v in self.modal_bias.items()}

        # Initialize state
        self.current_time = 0
        self.current_state = self.generate_state()
        self.previous_states = []

        # Initialize pattern generators
        self.pattern_generators = {
            "seasonal": self._generate_seasonal_pattern,
            "cyclical": self._generate_cyclical_pattern,
            "random": self._generate_random_pattern,
            "escalating": self._generate_escalating_pattern,
            "alternating": self._generate_alternating_pattern
        }

        # Set current pattern
        self.current_pattern = "random"
        self.pattern_duration = 20
        self.pattern_time = 0

        # Initialize significant event tracking
        self.significant_events = []
        self.significant_event_probability = 0.05  # Base probability

    def generate_state(self) -> Dict[str, Any]:
        """
        Generate a new environmental state.

        Returns:
            Dictionary representing the environmental state
        """
        # Create combination of elements targeting different modalities
        state = {
            # Basic state properties
            "time": self.current_time,
            "complexity": self.complexity,
            "volatility": self.volatility,

            # Core context variables
            "threat_level": self._get_threat_level(),
            "growth_opportunity": self._get_growth_opportunity(),
            "connection_opportunity": self._get_connection_opportunity(),

            # Cognitive load factor
            "cognitive_load": random.uniform(0.2, 0.5) * self.complexity,

            # Stimulus elements (specific experiences in the environment)
            "stimulus_elements": self._generate_stimulus_elements(),

            # Whether this is a significant event
            "is_significant": False
        }

        return state

    def _get_threat_level(self) -> float:
        """Generate a threat level for the current state."""
        # Base threat with some randomness
        return min(1.0, max(0.0, self.threat_level + random.uniform(-0.2, 0.2)))

    def _get_growth_opportunity(self) -> float:
        """Generate a growth opportunity level for the current state."""
        # Inversely related to threat but not perfectly
        base = self.opportunity_level * 0.8
        return min(1.0, max(0.0, base + random.uniform(-0.1, 0.3)))

    def _get_connection_opportunity(self) -> float:
        """Generate a connection opportunity level for the current state."""
        # Somewhat independent of threat/opportunity
        base = self.opportunity_level * 0.5
        return min(1.0, max(0.0, base + random.uniform(0.0, 0.5)))

    def _generate_stimulus_elements(self) -> List[Dict[str, Any]]:
        """Generate specific stimulus elements for the environment."""
        # Number of elements based on complexity
        num_elements = max(1, int(self.complexity * 5))

        elements = []
        for _ in range(num_elements):
            # Determine which modality to target
            target_modality = self._select_target_modality()

            # Generate element based on modality
            if target_modality == "hedonic":
                elements.append(self._generate_hedonic_stimulus())
            elif target_modality == "eudaimonic":
                elements.append(self._generate_eudaimonic_stimulus())
            elif target_modality == "transcendent":
                elements.append(self._generate_transcendent_stimulus())

        return elements

    def _select_target_modality(self) -> str:
        """Select which modality to target based on modal bias."""
        modalities = list(self.modal_bias.keys())
        weights = list(self.modal_bias.values())
        return random.choices(modalities, weights=weights, k=1)[0]

    def _generate_hedonic_stimulus(self) -> Dict[str, Any]:
        """Generate a stimulus targeting hedonic processing."""
        # Determine valence (positive or negative)
        is_positive = random.random() > self.threat_level

        # Set intensity based on complexity
        intensity = random.uniform(0.3, 0.7) * self.complexity

        if is_positive:
            stimulus_types = ["reward", "pleasure", "comfort", "satisfaction"]
            valence = "positive"
            pleasure_value = intensity
            pain_value = 0.0
        else:
            stimulus_types = ["threat", "pain", "discomfort", "frustration"]
            valence = "negative"
            pleasure_value = 0.0
            pain_value = intensity

        return {
            "type": random.choice(stimulus_types),
            "modality": "hedonic",
            "valence": valence,
            "intensity": intensity,
            "pleasure_value": pleasure_value,
            "pain_value": pain_value,
            "description": f"A {valence} hedonic stimulus of intensity {intensity:.2f}"
        }

    def _generate_eudaimonic_stimulus(self) -> Dict[str, Any]:
        """Generate a stimulus targeting eudaimonic processing."""
        # Determine specific type
        stimulus_types = ["achievement", "growth", "meaning", "purpose", "virtue"]

        # Set intensity based on complexity and opportunity level
        intensity = random.uniform(0.3, 0.7) * self.complexity * self.opportunity_level

        # Distribute values across eudaimonic aspects
        growth_value = 0.0
        purpose_value = 0.0
        virtue_value = 0.0

        stimulus_type = random.choice(stimulus_types)
        if stimulus_type == "achievement" or stimulus_type == "growth":
            growth_value = intensity
        elif stimulus_type == "meaning" or stimulus_type == "purpose":
            purpose_value = intensity
        elif stimulus_type == "virtue":
            virtue_value = intensity

        return {
            "type": stimulus_type,
            "modality": "eudaimonic",
            "valence": "positive",  # Eudaimonic stimuli generally positive
            "intensity": intensity,
            "growth_value": growth_value,
            "purpose_value": purpose_value,
            "virtue_value": virtue_value,
            "description": f"An eudaimonic stimulus ({stimulus_type}) of intensity {intensity:.2f}"
        }

    def _generate_transcendent_stimulus(self) -> Dict[str, Any]:
        """Generate a stimulus targeting transcendent processing."""
        # Determine specific type
        stimulus_types = ["connection", "awe", "unity", "transcendence", "wonder"]

        # Set intensity based on complexity
        intensity = random.uniform(0.3, 0.7) * self.complexity

        # Distribute values across transcendent aspects
        unity_value = 0.0
        connection_value = 0.0
        dissolution_value = 0.0

        stimulus_type = random.choice(stimulus_types)
        if stimulus_type == "unity" or stimulus_type == "transcendence":
            unity_value = intensity
            dissolution_value = intensity * 0.5
        elif stimulus_type == "connection":
            connection_value = intensity
        elif stimulus_type == "awe" or stimulus_type == "wonder":
            dissolution_value = intensity
            unity_value = intensity * 0.5

        return {
            "type": stimulus_type,
            "modality": "transcendent",
            "valence": "positive",  # Transcendent stimuli generally positive
            "intensity": intensity,
            "unity_value": unity_value,
            "connection_value": connection_value,
            "dissolution_value": dissolution_value,
            "description": f"A transcendent stimulus ({stimulus_type}) of intensity {intensity:.2f}"
        }

    def update(self) -> Dict[str, Any]:
        """
        Update the environmental state.

        Returns:
            The new environmental state
        """
        # Save previous state
        self.previous_states.append(self.current_state)
        if len(self.previous_states) > 10:
            self.previous_states.pop(0)  # Keep limited history

        # Check for pattern updates
        self.pattern_time += 1
        if self.pattern_time >= self.pattern_duration:
            self._select_new_pattern()

        # Generate new state using current pattern
        new_state = self.pattern_generators[self.current_pattern]()

        # Check for significant event
        if self._check_significant_event():
            new_state = self._make_significant_event(new_state)

        # Update current state and time
        self.current_state = new_state
        self.current_time += 1

        return new_state

    def _select_new_pattern(self) -> None:
        """Select a new environmental pattern."""
        # Choose new pattern
        patterns = list(self.pattern_generators.keys())
        self.current_pattern = random.choice(patterns)

        # Set new duration
        self.pattern_duration = random.randint(10, 30)
        self.pattern_time = 0

        print(f"Environmental pattern changed to '{self.current_pattern}' for {self.pattern_duration} steps")

    def _generate_random_pattern(self) -> Dict[str, Any]:
        """Generate a random environmental state."""
        # Random state with volatility-based change
        if self.previous_states and random.random() > self.volatility:
            # Low volatility: modify previous state
            base_state = self.previous_states[-1].copy()

            # Apply small changes
            base_state["threat_level"] = min(1.0, max(0.0,
                base_state["threat_level"] + random.uniform(-0.1, 0.1) * self.volatility))

            base_state["growth_opportunity"] = min(1.0, max(0.0,
                base_state["growth_opportunity"] + random.uniform(-0.1, 0.1) * self.volatility))

            base_state["connection_opportunity"] = min(1.0, max(0.0,
                base_state["connection_opportunity"] + random.uniform(-0.1, 0.1) * self.volatility))

            base_state["cognitive_load"] = min(1.0, max(0.0,
                base_state["cognitive_load"] + random.uniform(-0.1, 0.1) * self.volatility))

            # Update stimulus elements
            base_state["stimulus_elements"] = self._generate_stimulus_elements()

            # Update time
            base_state["time"] = self.current_time

            return base_state
        else:
            # High volatility: generate new state
            return self.generate_state()

    def _generate_cyclical_pattern(self) -> Dict[str, Any]:
        """Generate a cyclical environmental pattern."""
        # Use sine waves for cyclical patterns
        cycle_position = (self.pattern_time / self.pattern_duration) * 2 * np.pi

        # Create new state
        state = {
            "time": self.current_time,
            "complexity": self.complexity,
            "volatility": self.volatility,

            # Cyclical threat level
            "threat_level": 0.5 + 0.4 * np.sin(cycle_position),

            # Inverse cycle for growth opportunity
            "growth_opportunity": 0.5 + 0.4 * np.sin(cycle_position + np.pi),

            # Phase-shifted cycle for connection
            "connection_opportunity": 0.5 + 0.4 * np.sin(cycle_position + np.pi/2),

            # Cognitive load follows threat
            "cognitive_load": 0.3 + 0.3 * np.sin(cycle_position),

            # Stimulus elements
            "stimulus_elements": self._generate_stimulus_elements(),

            # Not significant by default
            "is_significant": False
        }

        return state

    def _generate_seasonal_pattern(self) -> Dict[str, Any]:
        """Generate a seasonal pattern with distinct phases."""
        # Divide pattern into 4 seasons
        season = int(4 * self.pattern_time / self.pattern_duration)

        if season == 0:  # "Spring" - high growth, low threat
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": 0.2,
                "growth_opportunity": 0.8,
                "connection_opportunity": 0.6,
                "cognitive_load": 0.3,
                "stimulus_elements": self._generate_stimulus_elements(),
                "is_significant": False
            }
        elif season == 1:  # "Summer" - high growth, moderate threat
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": 0.4,
                "growth_opportunity": 0.7,
                "connection_opportunity": 0.5,
                "cognitive_load": 0.5,
                "stimulus_elements": self._generate_stimulus_elements(),
                "is_significant": False
            }
        elif season == 2:  # "Fall" - moderate growth, moderate threat
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": 0.5,
                "growth_opportunity": 0.5,
                "connection_opportunity": 0.7,
                "cognitive_load": 0.4,
                "stimulus_elements": self._generate_stimulus_elements(),
                "is_significant": False
            }
        else:  # "Winter" - low growth, high threat
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": 0.7,
                "growth_opportunity": 0.3,
                "connection_opportunity": 0.4,
                "cognitive_load": 0.6,
                "stimulus_elements": self._generate_stimulus_elements(),
                "is_significant": False
            }

        # Add some randomness
        state["threat_level"] = min(1.0, max(0.0,
            state["threat_level"] + random.uniform(-0.1, 0.1)))

        state["growth_opportunity"] = min(1.0, max(0.0,
            state["growth_opportunity"] + random.uniform(-0.1, 0.1)))

        state["connection_opportunity"] = min(1.0, max(0.0,
            state["connection_opportunity"] + random.uniform(-0.1, 0.1)))

        return state

    def _generate_escalating_pattern(self) -> Dict[str, Any]:
        """Generate an escalating pattern with increasing intensity."""
        # Calculate progress through pattern
        progress = self.pattern_time / self.pattern_duration

        # Escalating threat
        threat_level = self.threat_level + progress * (1.0 - self.threat_level)

        # Diminishing opportunity
        growth_opportunity = self.opportunity_level * (1.0 - progress * 0.7)

        # Increasing cognitive load
        cognitive_load = 0.3 + progress * 0.6

        # Create state
        state = {
            "time": self.current_time,
            "complexity": self.complexity,
            "volatility": self.volatility,
            "threat_level": threat_level,
            "growth_opportunity": growth_opportunity,
            "connection_opportunity": self.opportunity_level * (1.0 - progress * 0.5),
            "cognitive_load": cognitive_load,
            "stimulus_elements": self._generate_stimulus_elements(),
            "is_significant": False
        }

        # Add some randomness
        state["threat_level"] = min(1.0, max(0.0,
            state["threat_level"] + random.uniform(-0.05, 0.05)))

        state["growth_opportunity"] = min(1.0, max(0.0,
            state["growth_opportunity"] + random.uniform(-0.05, 0.05)))

        return state

    def _generate_alternating_pattern(self) -> Dict[str, Any]:
        """Generate an alternating pattern that switches focus between modalities."""
        # Determine which modality to focus on this step
        step_in_sequence = self.pattern_time % 3

        if step_in_sequence == 0:  # Focus on hedonic
            # Create hedonic-focused state
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": random.uniform(0.4, 0.8),  # Higher threat for hedonic focus
                "growth_opportunity": random.uniform(0.1, 0.4),
                "connection_opportunity": random.uniform(0.1, 0.3),
                "cognitive_load": random.uniform(0.5, 0.7),
                "is_significant": False
            }

            # Generate mostly hedonic stimuli
            temp_bias = self.modal_bias.copy()
            temp_bias["hedonic"] = 0.7
            temp_bias["eudaimonic"] = 0.15
            temp_bias["transcendent"] = 0.15

            # Normalize
            total = sum(temp_bias.values())
            temp_bias = {k: v / total for k, v in temp_bias.items()}

            # Save original bias
            orig_bias = self.modal_bias
            self.modal_bias = temp_bias

            # Generate stimulus elements
            state["stimulus_elements"] = self._generate_stimulus_elements()

            # Restore bias
            self.modal_bias = orig_bias

        elif step_in_sequence == 1:  # Focus on eudaimonic
            # Create eudaimonic-focused state
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": random.uniform(0.2, 0.5),
                "growth_opportunity": random.uniform(0.6, 0.9),  # Higher growth for eudaimonic
                "connection_opportunity": random.uniform(0.3, 0.6),
                "cognitive_load": random.uniform(0.3, 0.6),
                "is_significant": False
            }

            # Generate mostly eudaimonic stimuli
            temp_bias = self.modal_bias.copy()
            temp_bias["hedonic"] = 0.15
            temp_bias["eudaimonic"] = 0.7
            temp_bias["transcendent"] = 0.15

            # Normalize
            total = sum(temp_bias.values())
            temp_bias = {k: v / total for k, v in temp_bias.items()}

            # Save original bias
            orig_bias = self.modal_bias
            self.modal_bias = temp_bias

            # Generate stimulus elements
            state["stimulus_elements"] = self._generate_stimulus_elements()

            # Restore bias
            self.modal_bias = orig_bias

        else:  # Focus on transcendent
            # Create transcendent-focused state
            state = {
                "time": self.current_time,
                "complexity": self.complexity,
                "volatility": self.volatility,
                "threat_level": random.uniform(0.1, 0.4),
                "growth_opportunity": random.uniform(0.3, 0.6),
                "connection_opportunity": random.uniform(0.7, 0.9),  # Higher connection for transcendent
                "cognitive_load": random.uniform(0.2, 0.5),
                "is_significant": False
            }

            # Generate mostly transcendent stimuli
            temp_bias = self.modal_bias.copy()
            temp_bias["hedonic"] = 0.15
            temp_bias["eudaimonic"] = 0.15
            temp_bias["transcendent"] = 0.7

            # Normalize
            total = sum(temp_bias.values())
            temp_bias = {k: v / total for k, v in temp_bias.items()}

            # Save original bias
            orig_bias = self.modal_bias
            self.modal_bias = temp_bias

            # Generate stimulus elements
            state["stimulus_elements"] = self._generate_stimulus_elements()

            # Restore bias
            self.modal_bias = orig_bias

        return state

    def _check_significant_event(self) -> bool:
        """Check if a significant event should occur."""
        # Base probability modified by complexity and volatility
        probability = self.significant_event_probability * self.complexity * (1 + self.volatility)

        # Higher probability if no recent significant events
        if self.significant_events:
            time_since_last = self.current_time - self.significant_events[-1]["time"]
            if time_since_last > 10:
                probability *= 1.5

        return random.random() < probability

    def _make_significant_event(self, base_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a normal state to a significant event."""
        # Copy base state
        state = base_state.copy()

        # Mark as significant
        state["is_significant"] = True

        # Determine event type
        event_types = ["crisis", "opportunity", "revelation"]
        weights = [self.threat_level, self.opportunity_level, 0.3]  # Weights based on context
        event_type = random.choices(event_types, weights=weights, k=1)[0]

        # Modify state based on event type
        if event_type == "crisis":
            # Crisis increases threat and cognitive load
            state["threat_level"] = min(1.0, state["threat_level"] * 2)
            state["cognitive_load"] = min(1.0, state["cognitive_load"] * 1.5)
            state["growth_opportunity"] *= 0.5

            # Generate intense hedonic stimuli (negative)
            hedonic_stimulus = self._generate_hedonic_stimulus()
            hedonic_stimulus["intensity"] *= 2.0
            hedonic_stimulus["pain_value"] *= 2.0
            hedonic_stimulus["description"] = f"A severe crisis stimulus of intensity {hedonic_stimulus['intensity']:.2f}"

            # Add to stimulus elements
            state["stimulus_elements"].append(hedonic_stimulus)

        elif event_type == "opportunity":
            # Opportunity increases growth and connection
            state["growth_opportunity"] = min(1.0, state["growth_opportunity"] * 2)
            state["connection_opportunity"] = min(1.0, state["connection_opportunity"] * 1.5)
            state["threat_level"] *= 0.5

            # Generate intense eudaimonic stimulus
            eudaimonic_stimulus = self._generate_eudaimonic_stimulus()
            eudaimonic_stimulus["intensity"] *= 2.0
            eudaimonic_stimulus["growth_value"] *= 2.0
            eudaimonic_stimulus["purpose_value"] *= 2.0
            eudaimonic_stimulus["description"] = f"A significant opportunity stimulus of intensity {eudaimonic_stimulus['intensity']:.2f}"

            # Add to stimulus elements
            state["stimulus_elements"].append(eudaimonic_stimulus)

        elif event_type == "revelation":
            # Revelation increases connection and cognitive load
            state["connection_opportunity"] = min(1.0, state["connection_opportunity"] * 2)
            state["cognitive_load"] = min(1.0, state["cognitive_load"] * 1.3)

            # Generate intense transcendent stimulus
            transcendent_stimulus = self._generate_transcendent_stimulus()
            transcendent_stimulus["intensity"] *= 2.0
            transcendent_stimulus["unity_value"] *= 2.0
            transcendent_stimulus["dissolution_value"] *= 2.0
            transcendent_stimulus["description"] = f"A profound revelation stimulus of intensity {transcendent_stimulus['intensity']:.2f}"

            # Add to stimulus elements
            state["stimulus_elements"].append(transcendent_stimulus)

        # Record significant event
        event_record = {
            "time": self.current_time,
            "type": event_type,
            "intensity": 2.0,
            "description": f"Significant {event_type} event at t={self.current_time}"
        }
        self.significant_events.append(event_record)

        # Add event description to state
        state["event_type"] = event_type
        state["event_description"] = event_record["description"]

        print(f"Significant event: {event_record['description']}")

        return state

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current environmental state."""
        return self.current_state

    def set_complexity(self, complexity: float) -> None:
        """Set environment complexity level."""
        self.complexity = min(1.0, max(0.1, complexity))

    def set_volatility(self, volatility: float) -> None:
        """Set environment volatility level."""
        self.volatility = min(1.0, max(0.1, volatility))

    def set_threat_level(self, threat_level: float) -> None:
        """Set base threat level."""
        self.threat_level = min(1.0, max(0.1, threat_level))
        self.opportunity_level = 1.0 - threat_level

    def set_modal_bias(self, modal_bias: Dict[str, float]) -> None:
        """Set modal bias for stimulus generation."""
        self.modal_bias = modal_bias.copy()

        # Normalize
        total = sum(self.modal_bias.values())
        self.modal_bias = {k: v / total for k, v in self.modal_bias.items()}

    def set_pattern(self, pattern: str, duration: Optional[int] = None) -> None:
        """
        Set a specific environmental pattern.

        Args:
            pattern: The pattern to use
            duration: Optional duration for the pattern
        """
        if pattern in self.pattern_generators:
            self.current_pattern = pattern
            self.pattern_time = 0

            if duration:
                self.pattern_duration = duration
            else:
                self.pattern_duration = random.randint(10, 30)

            print(f"Environmental pattern set to '{pattern}' for {self.pattern_duration} steps")
        else:
            raise ValueError(f"Unknown pattern '{pattern}'. Available patterns: {list(self.pattern_generators.keys())}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the environment over time."""
        # Only calculate if we have history
        if not self.previous_states:
            return {
                "volatility_actual": 0.0,
                "threat_average": self.threat_level,
                "growth_average": self.opportunity_level,
                "connection_average": 0.5,
                "cognitive_load_average": 0.3,
                "significant_events_count": 0
            }

        # Calculate average values
        threat_values = [s["threat_level"] for s in self.previous_states]
        growth_values = [s["growth_opportunity"] for s in self.previous_states]
        connection_values = [s["connection_opportunity"] for s in self.previous_states]
        cognitive_values = [s["cognitive_load"] for s in self.previous_states]

        # Calculate actual volatility (average change between states)
        volatility_values = []
        for i in range(1, len(self.previous_states)):
            prev = self.previous_states[i-1]
            curr = self.previous_states[i]

            # Calculate change in key variables
            threat_change = abs(curr["threat_level"] - prev["threat_level"])
            growth_change = abs(curr["growth_opportunity"] - prev["growth_opportunity"])
            connection_change = abs(curr["connection_opportunity"] - prev["connection_opportunity"])

            # Average change
            avg_change = (threat_change + growth_change + connection_change) / 3
            volatility_values.append(avg_change)

        return {
            "volatility_actual": sum(volatility_values) / max(1, len(volatility_values)),
            "threat_average": sum(threat_values) / len(threat_values),
            "growth_average": sum(growth_values) / len(growth_values),
            "connection_average": sum(connection_values) / len(connection_values),
            "cognitive_load_average": sum(cognitive_values) / len(cognitive_values),
            "significant_events_count": len(self.significant_events)
        }
