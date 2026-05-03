"""
Contradiction graph visualization module.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import os
from datetime import datetime
import matplotlib.patches as mpatches


class ContradictionGraphVisualizer:
    """
    Contradiction graph visualizer.
    """
    
    def __init__(self, save_dir: str = "results/visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualization outputs.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        self.use_english = True

            
    def visualize_contradiction_graph(self, 
                                    graph: nx.Graph, 
                                    query: str,
                                    independent_set: Optional[Set[int]] = None,
                                    save_path: Optional[str] = None,
                                    show_labels: bool = True,
                                    show_answers: bool = True,
                                    mode: str = "doc") -> str:
        """
        Visualize contradiction graph.
        
        Args:
            graph: NetworkX graph object.
            query: User query string.
            independent_set: Independent-set nodes (used only when mode="doc").
            save_path: Output image path.
            show_labels: Whether to render node labels.
            show_answers: Whether to render answer annotations.
            mode: "doc" (document-level) or "viewpoint" (viewpoint-level).
        """
        if len(graph.nodes()) == 0:
            print("Graph is empty, visualization skipped.")
            return ""
        
        # Translated comment (English only).
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Translated comment (English only).
        title_prefix = "Contradiction Graph" if self.use_english else "Contradiction Graph Visualization"
        if mode == "viewpoint":
            title_prefix += " (Viewpoint Level)"
            
        title_text = f"{title_prefix}\nQuery: {query[:50]}{'...' if len(query) > 50 else ''}"
        
        # :
        if mode == "viewpoint":
            self._plot_viewpoint_graph(ax1, graph)
        else:
            self._plot_graph(ax1, graph, independent_set, show_labels, show_answers)
            
        ax1.set_title(title_text, fontsize=14, fontweight='bold')
        
        # :
        self._plot_statistics(ax2, graph, independent_set, mode=mode)
        
        plt.tight_layout()
        
        # Translated comment (English only).
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"contradiction_graph_{timestamp}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {save_path}")
        
        return save_path

    def _plot_viewpoint_graph(self, ax, graph: nx.Graph):
        """Plot viewpoint-level graph."""
        # : kamada_kawai_layout spring_layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Translated comment (English only).
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node in graph.nodes():
            data = graph.nodes[node]
            status = data.get('status', 'clean')
            doc_count = data.get('doc_count', 1)
            claim = data.get('claim', 'N/A')
            
            # Translated comment (English only).
            if status == 'clean':
                node_colors.append('#2E8B57') # Green
            elif status == 'suspicious':
                node_colors.append('#FFA500') # Orange
            elif status == 'poisoned':
                node_colors.append('#FF0000') # Red
            else:
                node_colors.append('#808080') # Gray
                
            # ()
            node_sizes.append(1000 + doc_count * 200)
            
            # Translated comment (English only).
            short_claim = claim[:30] + "..." if len(claim) > 30 else claim
            labels[node] = f"{node}\n{short_claim}\n(Docs: {doc_count})"
            
        # Translated comment (English only).
        nx.draw_networkx_nodes(graph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.9,
                              ax=ax)
                              
        # Translated comment (English only).
        edge_colors = []
        edge_styles = []
        edge_widths = []
        
        for u, v, data in graph.edges(data=True):
            rel_type = data.get('type', 'contradiction')
            if rel_type == 'contradiction':
                edge_colors.append('red')
                edge_styles.append('solid')
                edge_widths.append(2.0)
            elif rel_type == 'support':
                edge_colors.append('green')
                edge_styles.append('dashed')
                edge_widths.append(1.0)
            else:
                edge_colors.append('gray')
                edge_styles.append('dotted')
                edge_widths.append(0.5)
                
        nx.draw_networkx_edges(graph, pos,
                              edge_color=edge_colors,
                              style=edge_styles,
                              width=edge_widths,
                              ax=ax)
                              
        # Translated comment (English only).
        nx.draw_networkx_labels(graph, pos, labels, font_size=9, font_weight='bold', ax=ax)
        
        # Translated comment (English only).
        legend_elements = [
            mpatches.Patch(color='#2E8B57', label='Clean Viewpoint'),
            mpatches.Patch(color='#FFA500', label='Suspicious Viewpoint'),
            mpatches.Patch(color='#FF0000', label='Poisoned Viewpoint'),
            plt.Line2D([0], [0], color='red', lw=2, label='Contradiction'),
            plt.Line2D([0], [0], color='green', lw=1, ls='--', label='Support')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        ax.axis('off')
    
    def _plot_graph(self, ax, graph: nx.Graph, independent_set: Optional[Set[int]], 
                   show_labels: bool, show_answers: bool):
        """Plot document-level contradiction graph."""
        # Translated comment (English only).
        if len(graph.nodes()) <= 10:
            pos = nx.spring_layout(graph, k=3, iterations=50)
        else:
            pos = nx.spring_layout(graph, k=2, iterations=30)
        
        # Translated comment (English only).
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes():
            if independent_set and node in independent_set:
                node_colors.append('#2E8B57')  # :
                node_sizes.append(800)
            else:
                node_colors.append('#FF6B6B')  # :
                node_sizes.append(600)
        
        # Translated comment (English only).
        nx.draw_networkx_nodes(graph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8,
                              ax=ax)
        
        # Translated comment (English only).
        edge_colors = []
        edge_widths = []
        
        for u, v, data in graph.edges(data=True):
            # Translated comment (English only).
            prob = data.get('nli_prob', data.get('llm_prob', data.get('cot_prob', 0.5)))
            edge_colors.append(plt.cm.Reds(prob))
            edge_widths.append(1 + prob * 3)  # :1-4
        
        nx.draw_networkx_edges(graph, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.7,
                              ax=ax)
        
        # Translated comment (English only).
        if show_labels:
            labels = {}
            for node in graph.nodes():
                doc_data = graph.nodes[node].get('document', {})
                # ,id,
                if isinstance(doc_data, dict):
                    doc_id = doc_data.get('id', str(node))
                else:
                    doc_id = str(node)
                
                # ID,
                if len(str(doc_id)) > 10:
                     doc_id = str(doc_id)[-10:]
                
                labels[node] = f"{doc_id}"
                
            nx.draw_networkx_labels(graph, pos, labels, 
                                  font_size=10, font_weight='bold', ax=ax)
        
        # Translated comment (English only).
        if show_answers and len(graph.nodes()) <= 8:
            self._add_answer_annotations(ax, graph, pos)
        
        # Translated comment (English only).
        self._add_legend(ax, independent_set is not None)
        
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _add_answer_annotations(self, ax, graph: nx.Graph, pos: Dict):
        """Add answer annotations near nodes."""
        for node, (x, y) in pos.items():
            node_data = graph.nodes[node]
            answer = node_data.get('answer', 'N/A')
            
            # Translated comment (English only).
            if len(answer) > 30:
                answer = answer[:30] + "..."
            
            # Translated comment (English only).
            ax.annotate(answer, 
                       xy=(x, y), 
                       xytext=(10, 10), 
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='lightblue', 
                                alpha=0.7),
                       fontsize=8,
                       ha='left')
    
    def _add_legend(self, ax, has_independent_set: bool):
        """Add legend."""
        legend_elements = []
        
        if self.use_english:
            labels = {
                'indep': 'Independent Set (Clean)',
                'conflict': 'Conflict Node',
                'doc': 'Document Node',
                'edge': 'Contradiction'
            }
        else:
            labels = {
                'indep': 'Independent Set (Clean)',
                'conflict': 'Conflict Node',
                'doc': 'Document Node',
                'edge': 'Contradiction'
            }
        
        if has_independent_set:
            legend_elements.extend([
                mpatches.Patch(color='#2E8B57', label=labels['indep']),
                mpatches.Patch(color='#FF6B6B', label=labels['conflict'])
            ])
        else:
            legend_elements.append(
                mpatches.Patch(color='#FF6B6B', label=labels['doc'])
            )
        
        legend_elements.append(
            plt.Line2D([0], [0], color='red', linewidth=2, label=labels['edge'])
        )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _plot_statistics(self, ax, graph: nx.Graph, independent_set: Optional[Set[int]], mode: str = "doc"):
        """Plot graph statistics."""
        # Translated comment (English only).
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        num_components = nx.number_connected_components(graph)
        density = nx.density(graph)
        
        # Translated comment (English only).
        edge_probs = []
        for u, v, data in graph.edges(data=True):
            prob = data.get('nli_prob', data.get('llm_prob', data.get('cot_prob', 0.0)))
            edge_probs.append(prob)
        
        avg_contradiction_prob = np.mean(edge_probs) if edge_probs else 0.0
        max_contradiction_prob = max(edge_probs) if edge_probs else 0.0
        
        # Translated comment (English only).
        independent_set_size = len(independent_set) if independent_set else 0
        
        # Translated comment (English only).
        if self.use_english:
            if mode == "viewpoint":
                stats_text = f"""Viewpoint Analysis:
        
Total Viewpoints: {num_nodes}
Relationships: {num_edges}
Components: {num_components}

Conflict Info:
Avg Prob: {avg_contradiction_prob:.3f}
"""
            else:
                stats_text = f"""Graph Statistics:
        
Nodes: {num_nodes}
Edges: {num_edges}
Components: {num_components}
Density: {density:.3f}

Conflict Info:
Avg Prob: {avg_contradiction_prob:.3f}
Max Prob: {max_contradiction_prob:.3f}

Independent Set:
Size: {independent_set_size}
Retention: {independent_set_size/num_nodes*100:.1f}%
"""
        else:
            if mode == "viewpoint":
                stats_text = f"""Viewpoint Analysis:
        
Total Viewpoints: {num_nodes}
Relationships: {num_edges}
Components: {num_components}

Conflict Info:
Avg Prob: {avg_contradiction_prob:.3f}
"""
            else:
                stats_text = f"""Graph Statistics:
        
Nodes: {num_nodes}
Edges: {num_edges}
Components: {num_components}
Density: {density:.3f}

Conflict Info:
Avg Prob: {avg_contradiction_prob:.3f}
Max Prob: {max_contradiction_prob:.3f}

Independent Set:
Size: {independent_set_size}
Retention: {independent_set_size/num_nodes*100:.1f}%
"""
        
        ax.text(0.05, 0.95, stats_text, 
               transform=ax.transAxes,
               fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='lightgray', 
                        alpha=0.8))
        
        # (Viewpoint )
        if mode == "doc" and edge_probs:
            ax2 = ax.twinx()
            ax2.hist(edge_probs, bins=10, alpha=0.6, color='skyblue', 
                    edgecolor='black', linewidth=0.5)
            
            if self.use_english:
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_title('Conflict Prob Distribution', fontsize=12, pad=20)
                ax.set_xlabel('Probability', fontsize=10)
            else:
                ax2.set_ylabel('Frequency', fontsize=10)
                ax2.set_title('Conflict Prob Distribution', fontsize=12, pad=20)
                ax.set_xlabel('Probability', fontsize=10)
                
            ax.set_xlim(0, 1)
        
        title_text = 'Statistics' if self.use_english else 'Statistics'
        ax.set_title(title_text, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def create_interactive_summary(self, 
                                 graph: nx.Graph, 
                                 query: str,
                                 independent_set: Optional[Set[int]] = None,
                                 save_path: Optional[str] = None) -> str:
        """
        Create an interactive summary report.
        
        Args:
            graph: NetworkX graph object.
            query: Query string.
            independent_set: Independent set.
            save_path: Output path.
            
        Returns:
            Saved HTML report path.
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"contradiction_report_{timestamp}.html")
        
        # HTML
        html_content = self._generate_html_report(graph, query, independent_set)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Interactive report saved to: {save_path}")
        return save_path
    
    def _generate_html_report(self, graph: nx.Graph, query: str, 
                            independent_set: Optional[Set[int]]) -> str:
        """Generate HTML report content."""
        # Translated comment (English only).
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        num_components = nx.number_connected_components(graph)
        
        # Translated comment (English only).
        contradiction_details = []
        for u, v, data in graph.edges(data=True):
            prob = data.get('nli_prob', data.get('llm_prob', data.get('cot_prob', 0.0)))
            method = data.get('method', 'unknown')
            
            node_u_answer = graph.nodes[u].get('answer', 'N/A')
            node_v_answer = graph.nodes[v].get('answer', 'N/A')
            
            contradiction_details.append({
                'doc1': u,
                'doc2': v,
                'probability': prob,
                'method': method,
                'answer1': node_u_answer[:100] + "..." if len(node_u_answer) > 100 else node_u_answer,
                'answer2': node_v_answer[:100] + "..." if len(node_v_answer) > 100 else node_v_answer
            })
        
        # Translated comment (English only).
        independent_info = ""
        if independent_set:
            independent_docs = [f"Doc {i}" for i in sorted(independent_set)]
            independent_info = f"""
            <h3>Maximum Independent Set</h3>
            <p>Selected non-contradictory docs: {', '.join(independent_docs)}</p>
            <p>Retained docs: {len(independent_set)} / {num_nodes}</p>
            """
        
        # Translated comment (English only).
        contradiction_table = ""
        if contradiction_details:
            contradiction_table = """
            <h3>Contradiction Details</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th>Doc 1</th>
                    <th>Doc 2</th>
                    <th>Contradiction Probability</th>
                    <th>Detection Method</th>
                    <th>Answer 1</th>
                    <th>Answer 2</th>
                </tr>
            """
            
            for detail in contradiction_details:
                contradiction_table += f"""
                <tr>
                    <td>Doc {detail['doc1']}</td>
                    <td>Doc {detail['doc2']}</td>
                    <td>{detail['probability']:.3f}</td>
                    <td>{detail['method']}</td>
                    <td>{detail['answer1']}</td>
                    <td>{detail['answer2']}</td>
                </tr>
                """
            
            contradiction_table += "</table>"
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Contradiction Graph Analysis Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ margin: 10px 0; }}
                th, td {{ padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Contradiction Graph Analysis Report</h1>
            
            <div class="summary">
                <h2>Query Information</h2>
                <p><strong>Query:</strong> {query}</p>
                <p><strong>Generated At:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <h2>Graph Statistics</h2>
            <ul>
                <li>Document Nodes: {num_nodes}</li>
                <li>Contradiction Edges: {num_edges}</li>
                <li>Connected Components: {num_components}</li>
                <li>Graph Density: {nx.density(graph):.3f}</li>
            </ul>
            
            {independent_info}
            
            {contradiction_table}
            
        </body>
        </html>
        """
        
        return html_template