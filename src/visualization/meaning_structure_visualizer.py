"""
Meaning Structure Visualizer Module

This module provides visualization tools for meaning structures produced by
meta-modal agents. It creates visual representations of narrative elements,
identity components, modal balances, and other aspects of meaning structures.

These visualizations help in analyzing and communicating the results of
experiments testing the Meta-Modal Resonance Theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx


class MeaningStructureVisualizer:
    """
    Creates visualizations of meaning structures and their components.
    
    This class provides various visualization functions that render
    different aspects of meaning structures, including:
    1. Narrative timelines
    2. Identity networks
    3. Modal balance distributions
    4. Meaning structure networks
    5. Perturbation responses
    """
    
    def __init__(self, style: str = 'default'):
        """
        Initialize the visualizer.
        
        Args:
            style: Visualization style to use
        """
        self.style = style
        
        # Set color schemes for different modalities
        self.color_schemes = {
            'default': {
                'hedonic': '#FF9500',  # Orange
                'eudaimonic': '#4CAF50',  # Green
                'transcendent': '#2196F3',  # Blue
                'integrated': '#9C27B0',  # Purple
                'background': '#F5F5F5',  # Light gray
                'text': '#212121',  # Dark gray
                'grid': '#E0E0E0'  # Silver
            },
            'contrast': {
                'hedonic': '#E53935',  # Red
                'eudaimonic': '#43A047',  # Green
                'transcendent': '#1E88E5',  # Blue
                'integrated': '#FFC107',  # Amber
                'background': '#FFFFFF',  # White
                'text': '#000000',  # Black
                'grid': '#BDBDBD'  # Gray
            },
            'pastel': {
                'hedonic': '#FFCCBC',  # Pastel orange
                'eudaimonic': '#C8E6C9',  # Pastel green
                'transcendent': '#BBDEFB',  # Pastel blue
                'integrated': '#E1BEE7',  # Pastel purple
                'background': '#FFFFFF',  # White
                'text': '#5D4037',  # Brown
                'grid': '#CFD8DC'  # Bluish gray
            }
        }
        
        # Use default style if specified style not found
        if style not in self.color_schemes:
            self.style = 'default'
        
        # Set font sizes
        self.font_sizes = {
            'title': 16,
            'axis_label': 12,
            'tick_label': 10,
            'legend': 10,
            'annotation': 8
        }
    
    def set_style(self, style: str) -> None:
        """Set the visualization style."""
        if style in self.color_schemes:
            self.style = style
        else:
            print(f"Style '{style}' not found. Using current style '{self.style}'.")
    
    def plot_narrative_timeline(self, meaning_structure: Any) -> Tuple[Figure, Axes]:
        """
        Plot the timeline of narrative elements.
        
        Args:
            meaning_structure: The meaning structure to visualize
            
        Returns:
            matplotlib Figure and Axes objects
        """
        if not hasattr(meaning_structure, 'narrative_elements') or not meaning_structure.narrative_elements:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No narrative elements available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_xlabel("Time")
            ax.set_ylabel("Meaning Value")
            ax.set_title("Narrative Timeline")
            return fig, ax
        
        # Extract data
        narrative_elements = meaning_structure.narrative_elements
        times = [element.get("time", i) for i, element in enumerate(narrative_elements)]
        meaning_values = [element.get("net_meaning", 0.0) for element in narrative_elements]
        
        # Extract modal components if available
        hedonic_values = [element.get("hedonic_component", 0.0) for element in narrative_elements]
        eudaimonic_values = [element.get("eudaimonic_component", 0.0) for element in narrative_elements]
        transcendent_values = [element.get("transcendent_component", 0.0) for element in narrative_elements]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        # Plot modal components as stacked areas if all are available
        if all(len(val) == len(times) for val in [hedonic_values, eudaimonic_values, transcendent_values]):
            # Plot stacked areas
            ax.fill_between(times, 0, hedonic_values, 
                           color=colors['hedonic'], alpha=0.3, label='Hedonic Component')
            
            # Add eudaimonic on top of hedonic
            ax.fill_between(times, hedonic_values,
                           [h + e for h, e in zip(hedonic_values, eudaimonic_values)], 
                           color=colors['eudaimonic'], alpha=0.3, label='Eudaimonic Component')
            
            # Add transcendent on top of both
            ax.fill_between(times, 
                           [h + e for h, e in zip(hedonic_values, eudaimonic_values)],
                           [h + e + t for h, e, t in zip(hedonic_values, eudaimonic_values, transcendent_values)], 
                           color=colors['transcendent'], alpha=0.3, label='Transcendent Component')
        
        # Plot overall meaning line
        ax.plot(times, meaning_values, 'o-', color=colors['integrated'], 
               linewidth=2.5, markersize=6, label='Net Meaning')
        
        # Add modal balance indicators at each time point
        for i, time in enumerate(times):
            if i < len(narrative_elements) and "modal_weights" in narrative_elements[i]:
                weights = narrative_elements[i]["modal_weights"]
                
                # Create pie chart inset for modal balance
                inset_ax = ax.inset_axes([time - 0.3, meaning_values[i] + 0.1, 0.6, 0.6])
                
                wedge_colors = [colors['hedonic'], colors['eudaimonic'], colors['transcendent']]
                wedge_values = [weights.get('hedonic', 1/3), weights.get('eudaimonic', 1/3), weights.get('transcendent', 1/3)]
                
                inset_ax.pie(wedge_values, colors=wedge_colors, wedgeprops=dict(alpha=0.7), radius=1)
                inset_ax.axis('equal')
        
        # Add annotations for significant events
        for i, element in enumerate(narrative_elements):
            if element.get("is_significant", False):
                ax.annotate("Significant Event", 
                           xy=(times[i], meaning_values[i]),
                           xytext=(times[i], meaning_values[i] + 0.2),
                           arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                           ha='center', fontsize=self.font_sizes['annotation'])
        
        # Add labels and legend
        ax.set_xlabel("Time", fontsize=self.font_sizes['axis_label'])
        ax.set_ylabel("Meaning Value", fontsize=self.font_sizes['axis_label'])
        ax.set_title("Narrative Timeline", fontsize=self.font_sizes['title'])
        ax.legend(fontsize=self.font_sizes['legend'])
        ax.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Set background color
        ax.set_facecolor(colors['background'])
        fig.patch.set_facecolor(colors['background'])
        
        # Adjust text colors
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color(colors['text'])
            text.set_fontsize(self.font_sizes['tick_label'])
        
        ax.xaxis.label.set_color(colors['text'])
        ax.yaxis.label.set_color(colors['text'])
        ax.title.set_color(colors['text'])
        
        return fig, ax
    
    def plot_identity_components(self, meaning_structure: Any) -> Tuple[Figure, Axes]:
        """
        Plot the identity components as a radar chart.
        
        Args:
            meaning_structure: The meaning structure to visualize
            
        Returns:
            matplotlib Figure and Axes objects
        """
        if not hasattr(meaning_structure, 'identity_components') or not meaning_structure.identity_components:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No identity components available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title("Identity Components")
            return fig, ax
        
        # Extract data
        components = list(meaning_structure.identity_components.keys())
        values = list(meaning_structure.identity_components.values())
        
        # Need at least 3 components for radar chart
        if len(components) < 3:
            # Add dummy components if needed
            while len(components) < 3:
                components.append(f"Component {len(components)+1}")
                values.append(0.0)
        
        # Normalize values to 0-1 range if not already
        min_val = min(values)
        max_val = max(values)
        if min_val < 0 or max_val > 1:
            values = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
        
        # Set up radar chart
        num_components = len(components)
        angles = np.linspace(0, 2*np.pi, num_components, endpoint=False).tolist()
        
        # Close the loop
        values.append(values[0])
        angles.append(angles[0])
        components.append(components[0])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        # Plot values
        ax.plot(angles, values, 'o-', linewidth=2, color=colors['integrated'])
        ax.fill(angles, values, alpha=0.25, color=colors['integrated'])
        
        # Categorize components by type
        hedonic_indices = [i for i, c in enumerate(components) if any(h in c.lower() for h in ['hedonic', 'pleasure', 'pain', 'preference'])]
        eudaimonic_indices = [i for i, c in enumerate(components) if any(e in c.lower() for e in ['eudaimonic', 'growth', 'purpose', 'virtue'])]
        transcendent_indices = [i for i, c in enumerate(components) if any(t in c.lower() for t in ['transcendent', 'boundary', 'unity', 'connect'])]
        
        # Highlight components by category
        for i in hedonic_indices:
            if i < len(angles):
                ax.plot([angles[i], angles[i]], [0, values[i]], '-', color=colors['hedonic'], alpha=0.7, linewidth=3)
        
        for i in eudaimonic_indices:
            if i < len(angles):
                ax.plot([angles[i], angles[i]], [0, values[i]], '-', color=colors['eudaimonic'], alpha=0.7, linewidth=3)
        
        for i in transcendent_indices:
            if i < len(angles):
                ax.plot([angles[i], angles[i]], [0, values[i]], '-', color=colors['transcendent'], alpha=0.7, linewidth=3)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", " ").title() for c in components[:-1]], 
                          fontsize=self.font_sizes['tick_label'])
        
        # Set y limits
        ax.set_ylim(0, 1)
        
        # Set title
        ax.set_title("Identity Components", fontsize=self.font_sizes['title'], color=colors['text'])
        
        # Set background color
        ax.set_facecolor(colors['background'])
        fig.patch.set_facecolor(colors['background'])
        
        return fig, ax
    
    def plot_modal_balance(self, meaning_structure: Any, 
                          time_series: bool = False) -> Tuple[Figure, Axes]:
        """
        Plot the balance between different processing modalities.
        
        Args:
            meaning_structure: The meaning structure to visualize
            time_series: Whether to plot balance over time
            
        Returns:
            matplotlib Figure and Axes objects
        """
        if not hasattr(meaning_structure, 'narrative_elements') or not meaning_structure.narrative_elements:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No modal balance data available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title("Modal Balance")
            return fig, ax
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        if time_series:
            # Plot balance over time
            
            # Extract data
            narrative_elements = meaning_structure.narrative_elements
            times = [element.get("time", i) for i, element in enumerate(narrative_elements)]
            
            # Extract modal weights if available
            hedonic_weights = []
            eudaimonic_weights = []
            transcendent_weights = []
            
            for element in narrative_elements:
                if "modal_weights" in element:
                    weights = element["modal_weights"]
                    hedonic_weights.append(weights.get('hedonic', 1/3))
                    eudaimonic_weights.append(weights.get('eudaimonic', 1/3))
                    transcendent_weights.append(weights.get('transcendent', 1/3))
                else:
                    # Default equal weights if not specified
                    hedonic_weights.append(1/3)
                    eudaimonic_weights.append(1/3)
                    transcendent_weights.append(1/3)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot weight lines
            ax.plot(times, hedonic_weights, 'o-', color=colors['hedonic'], 
                   linewidth=2, label='Hedonic')
            ax.plot(times, eudaimonic_weights, 's-', color=colors['eudaimonic'], 
                   linewidth=2, label='Eudaimonic')
            ax.plot(times, transcendent_weights, '^-', color=colors['transcendent'], 
                   linewidth=2, label='Transcendent')
            
            # Add balance threshold line
            ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.7, label='Equal Balance')
            
            # Add labels and legend
            ax.set_xlabel("Time", fontsize=self.font_sizes['axis_label'])
            ax.set_ylabel("Modal Weight", fontsize=self.font_sizes['axis_label'])
            ax.set_title("Modal Balance Over Time", fontsize=self.font_sizes['title'])
            ax.legend(fontsize=self.font_sizes['legend'])
            ax.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.5)
            
        else:
            # Plot current balance as a pie chart
            
            # Extract latest modal weights
            latest = meaning_structure.narrative_elements[-1] if meaning_structure.narrative_elements else None
            
            if not latest or "modal_weights" not in latest:
                # Use default equal weights if not specified
                weights = {'hedonic': 1/3, 'eudaimonic': 1/3, 'transcendent': 1/3}
            else:
                weights = latest["modal_weights"]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Extract values and labels
            labels = ['Hedonic', 'Eudaimonic', 'Transcendent']
            values = [weights.get('hedonic', 1/3), weights.get('eudaimonic', 1/3), weights.get('transcendent', 1/3)]
            chart_colors = [colors['hedonic'], colors['eudaimonic'], colors['transcendent']]
            
            # Add percentage labels
            autopct_values = [f'{v:.1%}' for v in values]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                           startangle=90, colors=chart_colors,
                                           wedgeprops=dict(width=0.5))
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            # Customize text
            for text in texts:
                text.set_fontsize(self.font_sizes['legend'])
                text.set_color(colors['text'])
            
            for autotext in autotexts:
                autotext.set_fontsize(self.font_sizes['legend'])
                autotext.set_color('white')
            
            # Add title
            ax.set_title("Current Modal Balance", fontsize=self.font_sizes['title'], color=colors['text'])
        
        # Set background color
        ax.set_facecolor(colors['background'])
        fig.patch.set_facecolor(colors['background'])
        
        # Adjust text colors
        if time_series:
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color(colors['text'])
                text.set_fontsize(self.font_sizes['tick_label'])
            
            ax.xaxis.label.set_color(colors['text'])
            ax.yaxis.label.set_color(colors['text'])
        
        ax.title.set_color(colors['text'])
        
        return fig, ax
    
    def plot_meaning_network(self, meaning_structure: Any) -> Tuple[Figure, Axes]:
        """
        Plot the meaning structure as a network.
        
        Args:
            meaning_structure: The meaning structure to visualize
            
        Returns:
            matplotlib Figure and Axes objects
        """
        # Create empty graph
        G = nx.Graph()
        
        # Check if we have the necessary components
        has_identities = hasattr(meaning_structure, 'identity_components') and meaning_structure.identity_components
        has_values = hasattr(meaning_structure, 'value_framework') and meaning_structure.value_framework
        has_narrative = hasattr(meaning_structure, 'narrative_elements') and meaning_structure.narrative_elements
        
        if not any([has_identities, has_values, has_narrative]):
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, "No meaning structure components available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title("Meaning Structure Network")
            return fig, ax
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        # Add center "self" node
        G.add_node("Self", type="self", size=1000)
        
        # Add identity component nodes
        if has_identities:
            for comp, value in meaning_structure.identity_components.items():
                # Determine node type based on component name
                if any(h in comp.lower() for h in ['hedonic', 'pleasure', 'pain', 'preference']):
                    node_type = "hedonic"
                elif any(e in comp.lower() for e in ['eudaimonic', 'growth', 'purpose', 'virtue']):
                    node_type = "eudaimonic"
                elif any(t in comp.lower() for e in ['transcendent', 'boundary', 'unity', 'connect']):
                    node_type = "transcendent"
                else:
                    node_type = "integrated"
                
                # Add node
                G.add_node(f"Identity: {comp}", type=node_type, size=300 * abs(value))
                
                # Connect to self
                G.add_edge("Self", f"Identity: {comp}", weight=abs(value))
        
        # Add value framework nodes
        if has_values:
            for value_name, value_strength in meaning_structure.value_framework.items():
                # Determine node type based on value name
                if any(h in value_name.lower() for h in ['pleasure', 'pain', 'hedonic']):
                    node_type = "hedonic"
                elif any(e in value_name.lower() for e in ['growth', 'purpose', 'virtue', 'eudaimonic']):
                    node_type = "eudaimonic"
                elif any(t in value_name.lower() for e in ['unity', 'connection', 'transcendent']):
                    node_type = "transcendent"
                else:
                    node_type = "integrated"
                
                # Add node
                G.add_node(f"Value: {value_name}", type=node_type, size=200 * abs(value_strength))
                
                # Connect to self
                G.add_edge("Self", f"Value: {value_name}", weight=abs(value_strength))
                
                # Connect to related identity components
                if has_identities:
                    for comp in meaning_structure.identity_components:
                        # Check for related terms
                        if any(term in comp.lower() for term in value_name.lower().split('_')):
                            G.add_edge(f"Value: {value_name}", f"Identity: {comp}", 
                                      weight=0.5 * (abs(value_strength) + abs(meaning_structure.identity_components[comp])))
        
        # Add narrative elements (simplified)
        if has_narrative:
            # Only add a few representative narrative elements to avoid clutter
            num_elements = len(meaning_structure.narrative_elements)
            
            if num_elements > 0:
                # Add first element
                first = meaning_structure.narrative_elements[0]
                G.add_node("Narrative: Begin", type="integrated", size=200)
                G.add_edge("Self", "Narrative: Begin", weight=0.7)
                
                # Add last element
                last = meaning_structure.narrative_elements[-1]
                G.add_node("Narrative: Current", type="integrated", size=300)
                G.add_edge("Self", "Narrative: Current", weight=0.9)
                G.add_edge("Narrative: Begin", "Narrative: Current", weight=0.5)
                
                # Add significant events if any
                for i, element in enumerate(meaning_structure.narrative_elements):
                    if element.get("is_significant", False):
                        event_name = f"Event: t={element.get('time', i)}"
                        G.add_node(event_name, type="integrated", size=250)
                        G.add_edge("Self", event_name, weight=0.8)
                        
                        # Connect to narrative elements
                        G.add_edge("Narrative: Begin", event_name, weight=0.3)
                        G.add_edge(event_name, "Narrative: Current", weight=0.3)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Set positions using spring layout
        pos = nx.spring_layout(G, k=0.3, seed=42)
        
        # Prepare node colors and sizes
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            node_type = G.nodes[node].get("type", "integrated")
            node_size = G.nodes[node].get("size", 300)
            
            if node_type == "hedonic":
                node_colors.append(colors['hedonic'])
            elif node_type == "eudaimonic":
                node_colors.append(colors['eudaimonic'])
            elif node_type == "transcendent":
                node_colors.append(colors['transcendent'])
            elif node_type == "self":
                node_colors.append(colors['integrated'])
            else:
                node_colors.append(colors['integrated'])
            
            node_sizes.append(node_size)
        
        # Prepare edge weights
        edge_weights = [G[u][v].get("weight", 0.5) * 5 for u, v in G.edges()]
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=self.font_sizes['annotation'], font_color=colors['text'])
        
        # Add title
        ax.set_title("Meaning Structure Network", fontsize=self.font_sizes['title'], color=colors['text'])
        
        # Set background color
        ax.set_facecolor(colors['background'])
        fig.patch.set_facecolor(colors['background'])
        
        # Turn off axis
        ax.axis('off')
        
        return fig, ax
    
    def plot_perturbation_response(self, meaning_structure: Any) -> Tuple[Figure, Axes]:
        """
        Plot the response to perturbations.
        
        Args:
            meaning_structure: The meaning structure to visualize
            
        Returns:
            matplotlib Figure and Axes objects
        """
        if not hasattr(meaning_structure, 'perturbation_responses') or not meaning_structure.perturbation_responses:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No perturbation response data available", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title("Perturbation Responses")
            return fig, ax
        
        # Extract data
        responses = meaning_structure.perturbation_responses
        
        # Group by perturbation type
        response_by_type = {}
        for response in responses:
            pert_type = response.get("perturbation_type", "unknown")
            if pert_type not in response_by_type:
                response_by_type[pert_type] = []
            response_by_type[pert_type].append(response)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        # Set bar positions
        bar_width = 0.8 / len(response_by_type)
        positions = np.arange(4)  # Four metrics to track
        
        # Create bars for each perturbation type
        for i, (pert_type, responses) in enumerate(response_by_type.items()):
            # Calculate average impacts
            coherence_impacts = [r.get("coherence_impact", 0.0) for r in responses]
            resilience_impacts = [r.get("resilience_impact", 0.0) for r in responses]
            temporal_impacts = [r.get("temporal_impact", 0.0) for r in responses]
            modal_impacts = [r.get("modal_impact", 0.0) for r in responses]
            
            avg_impacts = [
                np.mean(coherence_impacts) if coherence_impacts else 0.0,
                np.mean(resilience_impacts) if resilience_impacts else 0.0,
                np.mean(temporal_impacts) if temporal_impacts else 0.0,
                np.mean(modal_impacts) if modal_impacts else 0.0
            ]
            
            # Set bar positions
            bar_positions = positions + i * bar_width
            
            # Choose color based on type
            if "narrative" in pert_type.lower():
                bar_color = colors['hedonic']
            elif "identity" in pert_type.lower():
                bar_color = colors['eudaimonic']
            elif "value" in pert_type.lower() or "meta" in pert_type.lower():
                bar_color = colors['transcendent']
            else:
                bar_color = colors['integrated']
            
            # Create bars
            ax.bar(bar_positions, avg_impacts, bar_width, label=pert_type.title(), color=bar_color, alpha=0.7)
            
            # Add values on bars
            for j, impact in enumerate(avg_impacts):
                ax.text(bar_positions[j], impact + 0.02 * (1 if impact >= 0 else -1),
                       f"{impact:.2f}",
                       ha='center', va='bottom' if impact >= 0 else 'top',
                       fontsize=self.font_sizes['annotation'])
        
        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels and legend
        ax.set_xlabel("Impact Metric", fontsize=self.font_sizes['axis_label'])
        ax.set_ylabel("Impact Value (+ is improvement, - is degradation)", 
                     fontsize=self.font_sizes['axis_label'])
        ax.set_title("Perturbation Response Impacts", fontsize=self.font_sizes['title'])
        ax.set_xticks(positions + bar_width * (len(response_by_type) - 1) / 2)
        ax.set_xticklabels(['Coherence', 'Resilience', 'Temporal', 'Modal'])
        ax.legend(fontsize=self.font_sizes['legend'])
        ax.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.5, axis='y')
        
        # Set y limits to be symmetric around zero
        y_max = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_ylim(-y_max, y_max)
        
        # Set background color
        ax.set_facecolor(colors['background'])
        fig.patch.set_facecolor(colors['background'])
        
        # Adjust text colors
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color(colors['text'])
            text.set_fontsize(self.font_sizes['tick_label'])
        
        ax.xaxis.label.set_color(colors['text'])
        ax.yaxis.label.set_color(colors['text'])
        ax.title.set_color(colors['text'])
        
        return fig, ax
    
    def create_summary_visualization(self, meaning_structure: Any) -> Figure:
        """
        Create a comprehensive summary visualization of the meaning structure.
        
        Args:
            meaning_structure: The meaning structure to visualize
            
        Returns:
            matplotlib Figure
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        # Set subplot grid
        gs = fig.add_gridspec(2, 2)
        
        # Plot narrative timeline
        ax1 = fig.add_subplot(gs[0, 0])
        if hasattr(meaning_structure, 'narrative_elements') and meaning_structure.narrative_elements:
            # Extract data
            narrative_elements = meaning_structure.narrative_elements
            times = [element.get("time", i) for i, element in enumerate(narrative_elements)]
            meaning_values = [element.get("net_meaning", 0.0) for element in narrative_elements]
            
            # Plot overall meaning line
            ax1.plot(times, meaning_values, 'o-', color=colors['integrated'], 
                    linewidth=2.5, markersize=6, label='Net Meaning')
            
            # Add labels
            ax1.set_xlabel("Time", fontsize=self.font_sizes['axis_label'])
            ax1.set_ylabel("Meaning Value", fontsize=self.font_sizes['axis_label'])
            ax1.set_title("Narrative Timeline", fontsize=self.font_sizes['title'])
            ax1.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.5)
        else:
            ax1.text(0.5, 0.5, "No narrative elements available", 
                    horizontalalignment='center', verticalalignment='center')
            ax1.set_title("Narrative Timeline", fontsize=self.font_sizes['title'])
        
        # Plot modal balance
        ax2 = fig.add_subplot(gs[0, 1])
        if hasattr(meaning_structure, 'narrative_elements') and meaning_structure.narrative_elements:
            # Extract latest modal weights
            latest = meaning_structure.narrative_elements[-1] if meaning_structure.narrative_elements else None
            
            if not latest or "modal_weights" not in latest:
                # Use default equal weights if not specified
                weights = {'hedonic': 1/3, 'eudaimonic': 1/3, 'transcendent': 1/3}
            else:
                weights = latest["modal_weights"]
            
            # Extract values and labels
            labels = ['Hedonic', 'Eudaimonic', 'Transcendent']
            values = [weights.get('hedonic', 1/3), weights.get('eudaimonic', 1/3), weights.get('transcendent', 1/3)]
            chart_colors = [colors['hedonic'], colors['eudaimonic'], colors['transcendent']]
            
            # Create pie chart
            ax2.pie(values, labels=labels, autopct='%1.1f%%',
                   startangle=90, colors=chart_colors,
                   wedgeprops=dict(width=0.5))
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax2.axis('equal')
            
            # Add title
            ax2.set_title("Current Modal Balance", fontsize=self.font_sizes['title'])
        else:
            ax2.text(0.5, 0.5, "No modal balance data available", 
                    horizontalalignment='center', verticalalignment='center')
            ax2.set_title("Modal Balance", fontsize=self.font_sizes['title'])
        
        # Plot identity components
        ax3 = fig.add_subplot(gs[1, 0], polar=True)
        if hasattr(meaning_structure, 'identity_components') and meaning_structure.identity_components:
            # Extract data
            components = list(meaning_structure.identity_components.keys())
            values = list(meaning_structure.identity_components.values())
            
            # Need at least 3 components for radar chart
            if len(components) < 3:
                # Add dummy components if needed
                while len(components) < 3:
                    components.append(f"Component {len(components)+1}")
                    values.append(0.0)
            
            # Normalize values to 0-1 range if not already
            min_val = min(values)
            max_val = max(values)
            if min_val < 0 or max_val > 1:
                values = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in values]
            
            # Set up radar chart
            num_components = len(components)
            angles = np.linspace(0, 2*np.pi, num_components, endpoint=False).tolist()
            
            # Close the loop
            values.append(values[0])
            angles.append(angles[0])
            components.append(components[0])
            
            # Plot values
            ax3.plot(angles, values, 'o-', linewidth=2, color=colors['integrated'])
            ax3.fill(angles, values, alpha=0.25, color=colors['integrated'])
            
            # Set labels
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels([c.replace("_", " ").title() for c in components[:-1]], 
                              fontsize=self.font_sizes['tick_label'])
            
            # Set y limits
            ax3.set_ylim(0, 1)
            
            # Set title
            ax3.set_title("Identity Components", fontsize=self.font_sizes['title'])
        else:
            # Remove polar projection for text display
            fig.delaxes(ax3)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.text(0.5, 0.5, "No identity components available", 
                    horizontalalignment='center', verticalalignment='center')
            ax3.set_title("Identity Components", fontsize=self.font_sizes['title'])
            ax3.axis('off')
        
        # Plot metrics
        ax4 = fig.add_subplot(gs[1, 1])
        if hasattr(meaning_structure, 'coherence_score') and hasattr(meaning_structure, 'resilience_score'):
            # Extract metrics
            metrics = {
                'Coherence': meaning_structure.coherence_score,
                'Resilience': meaning_structure.resilience_score
            }
            
            if hasattr(meaning_structure, 'temporal_integration'):
                metrics['Temporal Integration'] = meaning_structure.temporal_integration
                
            if hasattr(meaning_structure, 'modal_integration'):
                metrics['Modal Integration'] = meaning_structure.modal_integration
            
            # Plot horizontal bars
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # Set bar colors
            bar_colors = [colors['integrated'] for _ in metric_names]
            
            # Create bars
            y_pos = np.arange(len(metric_names))
            ax4.barh(y_pos, metric_values, align='center', alpha=0.7, color=bar_colors)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(metric_names)
            
            # Add values on bars
            for i, value in enumerate(metric_values):
                ax4.text(max(0.05, value - 0.15), i, f"{value:.2f}", 
                        va='center', fontsize=self.font_sizes['annotation'],
                        color='white' if value > 0.3 else colors['text'])
            
            # Set limits and grid
            ax4.set_xlim(0, 1)
            ax4.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.5, axis='x')
            
            # Add labels
            ax4.set_xlabel("Value", fontsize=self.font_sizes['axis_label'])
            ax4.set_title("Meaning Structure Metrics", fontsize=self.font_sizes['title'])
        else:
            ax4.text(0.5, 0.5, "No metrics available", 
                    horizontalalignment='center', verticalalignment='center')
            ax4.set_title("Meaning Structure Metrics", fontsize=self.font_sizes['title'])
        
        # Add overall title
        fig.suptitle(f"Meaning Structure Summary - Agent {meaning_structure.agent_id if hasattr(meaning_structure, 'agent_id') else 'Unknown'}", 
                    fontsize=self.font_sizes['title'] + 4, color=colors['text'])
        
        # Set background color
        fig.patch.set_facecolor(colors['background'])
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(colors['background'])
            ax.title.set_color(colors['text'])
        
        # Adjust text colors for regular axes
        for ax in [ax1, ax4]:
            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color(colors['text'])
            
            if hasattr(ax, 'xaxis') and hasattr(ax.xaxis, 'label'):
                ax.xaxis.label.set_color(colors['text'])
            
            if hasattr(ax, 'yaxis') and hasattr(ax.yaxis, 'label'):
                ax.yaxis.label.set_color(colors['text'])
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        return fig
    
    def compare_meaning_structures(self, 
                                 structures: Dict[str, Any],
                                 metric: str = 'coherence_score') -> Tuple[Figure, Axes]:
        """
        Compare multiple meaning structures based on a specific metric.
        
        Args:
            structures: Dictionary mapping agent IDs to meaning structures
            metric: Metric to compare
            
        Returns:
            matplotlib Figure and Axes objects
        """
        if not structures:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No meaning structures available for comparison", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title("Meaning Structure Comparison")
            return fig, ax
        
        # Extract metric values for each structure
        agent_ids = []
        metric_values = []
        
        for agent_id, structure in structures.items():
            agent_ids.append(agent_id)
            
            if hasattr(structure, metric):
                metric_values.append(getattr(structure, metric))
            else:
                # Default value if metric not found
                metric_values.append(0.0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set colors
        colors = self.color_schemes[self.style]
        
        # Create bars
        bar_positions = np.arange(len(agent_ids))
        
        # Choose bar colors based on agent type if available
        bar_colors = []
        for agent_id in agent_ids:
            if 'hedonic' in agent_id.lower():
                bar_colors.append(colors['hedonic'])
            elif 'eudaimonic' in agent_id.lower():
                bar_colors.append(colors['eudaimonic'])
            elif 'transcendent' in agent_id.lower():
                bar_colors.append(colors['transcendent'])
            elif 'integrated' in agent_id.lower() or 'modal' in agent_id.lower():
                bar_colors.append(colors['integrated'])
            else:
                bar_colors.append(colors['integrated'])
        
        # Create bars
        ax.bar(bar_positions, metric_values, width=0.7, color=bar_colors, alpha=0.7)
        
        # Add values on bars
        for i, value in enumerate(metric_values):
            ax.text(bar_positions[i], value + 0.02, f"{value:.2f}", 
                   ha='center', va='bottom', fontsize=self.font_sizes['annotation'])
        
        # Add labels
        ax.set_xlabel("Agent", fontsize=self.font_sizes['axis_label'])
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=self.font_sizes['axis_label'])
        ax.set_title(f"Comparison of {metric.replace('_', ' ').title()}", fontsize=self.font_sizes['title'])
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(agent_ids)
        ax.grid(True, color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.5, axis='y')
        
        # Set y limit to start at 0
        ax.set_ylim(0, max(metric_values) * 1.1)
        
        # Set background color
        ax.set_facecolor(colors['background'])
        fig.patch.set_facecolor(colors['background'])
        
        # Adjust text colors
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color(colors['text'])
            text.set_fontsize(self.font_sizes['tick_label'])
        
        ax.xaxis.label.set_color(colors['text'])
        ax.yaxis.label.set_color(colors['text'])
        ax.title.set_color(colors['text'])
        
        # Rotate x labels if there are many
        if len(agent_ids) > 5:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            fig.tight_layout()
        
        return fig, ax
