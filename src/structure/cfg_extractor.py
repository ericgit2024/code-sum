"""
Control Flow Graph (CFG) extraction and encoding.
"""

import ast
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple


class CFGExtractor:
    """Extracts and encodes Control Flow Graph from Python code."""
    
    def __init__(self, max_nodes: int = 50):
        """
        Initialize CFG extractor.
        
        Args:
            max_nodes: Maximum number of nodes to include
        """
        self.max_nodes = max_nodes
        self.node_counter = 0
        
    def extract(self, code: str) -> Optional[nx.DiGraph]:
        """
        Extract CFG from Python code.
        
        Args:
            code: Python source code
            
        Returns:
            NetworkX DiGraph representing CFG or None if extraction fails
        """
        try:
            tree = ast.parse(code)
            self.node_counter = 0
            cfg = nx.DiGraph()
            
            # Build CFG from AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._build_cfg_from_function(node, cfg)
                    break  # Process only the first function
            
            return cfg
        except Exception as e:
            print(f"Error extracting CFG: {e}")
            return None
    
    def _build_cfg_from_function(self, func_node: ast.FunctionDef, cfg: nx.DiGraph):
        """Build CFG from function definition."""
        entry_node = self._add_node(cfg, "ENTRY", func_node.name)
        exit_node = self._add_node(cfg, "EXIT", func_node.name)
        
        if not func_node.body:
            cfg.add_edge(entry_node, exit_node)
            return
        
        # Process function body
        last_node = entry_node
        for stmt in func_node.body:
            last_node = self._process_statement(stmt, cfg, last_node, exit_node)
        
        # Connect last statement to exit
        if last_node != exit_node:
            cfg.add_edge(last_node, exit_node)
    
    def _process_statement(self, stmt: ast.AST, cfg: nx.DiGraph, 
                          prev_node: str, exit_node: str) -> str:
        """Process a single statement and add to CFG."""
        if isinstance(stmt, ast.If):
            return self._process_if(stmt, cfg, prev_node, exit_node)
        elif isinstance(stmt, ast.While):
            return self._process_while(stmt, cfg, prev_node, exit_node)
        elif isinstance(stmt, ast.For):
            return self._process_for(stmt, cfg, prev_node, exit_node)
        elif isinstance(stmt, ast.Return):
            node = self._add_node(cfg, "RETURN", ast.unparse(stmt.value) if stmt.value else "None")
            cfg.add_edge(prev_node, node)
            cfg.add_edge(node, exit_node)
            return node
        else:
            # Regular statement
            node = self._add_node(cfg, stmt.__class__.__name__, self._get_stmt_label(stmt))
            cfg.add_edge(prev_node, node)
            return node
    
    def _process_if(self, stmt: ast.If, cfg: nx.DiGraph, 
                   prev_node: str, exit_node: str) -> str:
        """Process if statement."""
        condition = self._get_condition_label(stmt.test)
        if_node = self._add_node(cfg, "IF", condition)
        cfg.add_edge(prev_node, if_node)
        
        # Process then branch
        then_last = if_node
        for s in stmt.body:
            then_last = self._process_statement(s, cfg, then_last, exit_node)
        
        # Process else branch
        else_last = if_node
        if stmt.orelse:
            for s in stmt.orelse:
                else_last = self._process_statement(s, cfg, else_last, exit_node)
        
        # Merge node
        merge_node = self._add_node(cfg, "MERGE", "")
        cfg.add_edge(then_last, merge_node)
        cfg.add_edge(else_last, merge_node)
        
        return merge_node
    
    def _process_while(self, stmt: ast.While, cfg: nx.DiGraph,
                      prev_node: str, exit_node: str) -> str:
        """Process while loop."""
        condition = self._get_condition_label(stmt.test)
        loop_node = self._add_node(cfg, "WHILE", condition)
        cfg.add_edge(prev_node, loop_node)
        
        # Process loop body
        body_last = loop_node
        for s in stmt.body:
            body_last = self._process_statement(s, cfg, body_last, exit_node)
        
        # Back edge to loop condition
        cfg.add_edge(body_last, loop_node)
        
        return loop_node
    
    def _process_for(self, stmt: ast.For, cfg: nx.DiGraph,
                    prev_node: str, exit_node: str) -> str:
        """Process for loop."""
        target = ast.unparse(stmt.target) if hasattr(ast, 'unparse') else "var"
        iter_expr = ast.unparse(stmt.iter) if hasattr(ast, 'unparse') else "iterable"
        loop_node = self._add_node(cfg, "FOR", f"{target} in {iter_expr}")
        cfg.add_edge(prev_node, loop_node)
        
        # Process loop body
        body_last = loop_node
        for s in stmt.body:
            body_last = self._process_statement(s, cfg, body_last, exit_node)
        
        # Back edge to loop
        cfg.add_edge(body_last, loop_node)
        
        return loop_node
    
    def _add_node(self, cfg: nx.DiGraph, node_type: str, label: str) -> str:
        """Add a node to CFG."""
        node_id = f"N{self.node_counter}"
        self.node_counter += 1
        cfg.add_node(node_id, type=node_type, label=label[:50])  # Limit label length
        return node_id
    
    def _get_stmt_label(self, stmt: ast.AST) -> str:
        """Get label for statement."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(stmt)[:50]
            return stmt.__class__.__name__
        except:
            return stmt.__class__.__name__
    
    def _get_condition_label(self, test: ast.AST) -> str:
        """Get label for condition."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(test)[:50]
            return "condition"
        except:
            return "condition"
    
    def encode(self, cfg: nx.DiGraph) -> str:
        """
        Encode CFG to string representation.
        
        Args:
            cfg: Control Flow Graph
            
        Returns:
            String representation of CFG
        """
        if cfg is None or cfg.number_of_nodes() == 0:
            return "CFG: <empty>"
        
        # Limit number of nodes
        nodes = list(cfg.nodes())[:self.max_nodes]
        
        # Build edge list representation
        edges = []
        for u, v in cfg.edges():
            if u in nodes and v in nodes:
                u_type = cfg.nodes[u].get('type', 'NODE')
                v_type = cfg.nodes[v].get('type', 'NODE')
                u_label = cfg.nodes[u].get('label', '')
                v_label = cfg.nodes[v].get('label', '')
                
                edge_str = f"{u_type}"
                if u_label:
                    edge_str += f"[{u_label}]"
                edge_str += f" -> {v_type}"
                if v_label:
                    edge_str += f"[{v_label}]"
                    
                edges.append(edge_str)
        
        return "CFG: " + " | ".join(edges)
    
    def extract_and_encode(self, code: str) -> str:
        """
        Extract and encode CFG in one step.
        
        Args:
            code: Python source code
            
        Returns:
            String representation of CFG
        """
        cfg = self.extract(code)
        return self.encode(cfg)
