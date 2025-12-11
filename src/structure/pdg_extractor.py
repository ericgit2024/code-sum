"""
Program Dependence Graph (PDG) extraction and encoding.
"""

import ast
import networkx as nx
from typing import Dict, List, Optional, Set


class PDGExtractor:
    """Extracts and encodes Program Dependence Graph from Python code."""
    
    def __init__(self, max_nodes: int = 50):
        """
        Initialize PDG extractor.
        
        Args:
            max_nodes: Maximum number of nodes to include
        """
        self.max_nodes = max_nodes
        self.node_counter = 0
        
    def extract(self, code: str) -> Optional[nx.DiGraph]:
        """
        Extract PDG from Python code.
        
        Args:
            code: Python source code
            
        Returns:
            NetworkX DiGraph representing PDG or None if extraction fails
        """
        try:
            tree = ast.parse(code)
            self.node_counter = 0
            pdg = nx.DiGraph()
            
            # Build PDG from AST
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._build_pdg_from_function(node, pdg)
                    break  # Process only the first function
            
            return pdg
        except Exception as e:
            print(f"Error extracting PDG: {e}")
            return None
    
    def _build_pdg_from_function(self, func_node: ast.FunctionDef, pdg: nx.DiGraph):
        """Build PDG from function definition."""
        # Track variable definitions and uses
        var_defs = {}  # variable -> node_id where it's defined
        var_uses = {}  # variable -> list of node_ids where it's used
        
        # Process parameters
        for arg in func_node.args.args:
            param_node = self._add_node(pdg, "PARAM", arg.arg)
            var_defs[arg.arg] = param_node
        
        # Process function body
        for stmt in func_node.body:
            self._process_statement_pdg(stmt, pdg, var_defs, var_uses)
        
        # Add data dependency edges
        for var, use_nodes in var_uses.items():
            if var in var_defs:
                def_node = var_defs[var]
                for use_node in use_nodes:
                    pdg.add_edge(def_node, use_node, type="data", var=var)
    
    def _process_statement_pdg(self, stmt: ast.AST, pdg: nx.DiGraph,
                               var_defs: Dict, var_uses: Dict):
        """Process statement for PDG construction."""
        if isinstance(stmt, ast.Assign):
            # Get assigned variables
            targets = []
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    targets.append(target.id)
            
            # Get used variables in value
            used_vars = self._get_used_variables(stmt.value)
            
            # Create assignment node
            assign_node = self._add_node(pdg, "ASSIGN", 
                                        f"{', '.join(targets)} = ...")
            
            # Record definitions
            for var in targets:
                var_defs[var] = assign_node
            
            # Record uses
            for var in used_vars:
                if var not in var_uses:
                    var_uses[var] = []
                var_uses[var].append(assign_node)
        
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                var = stmt.target.id
                node = self._add_node(pdg, "AUG_ASSIGN", f"{var} op= ...")
                
                # Both use and def
                if var not in var_uses:
                    var_uses[var] = []
                var_uses[var].append(node)
                var_defs[var] = node
                
                # Check value for uses
                used_vars = self._get_used_variables(stmt.value)
                for used_var in used_vars:
                    if used_var not in var_uses:
                        var_uses[used_var] = []
                    var_uses[used_var].append(node)
        
        elif isinstance(stmt, ast.Return):
            used_vars = self._get_used_variables(stmt.value) if stmt.value else []
            return_node = self._add_node(pdg, "RETURN", "return ...")
            
            for var in used_vars:
                if var not in var_uses:
                    var_uses[var] = []
                var_uses[var].append(return_node)
        
        elif isinstance(stmt, ast.Expr):
            # Expression statement (e.g., function call)
            used_vars = self._get_used_variables(stmt.value)
            expr_node = self._add_node(pdg, "EXPR", "expression")
            
            for var in used_vars:
                if var not in var_uses:
                    var_uses[var] = []
                var_uses[var].append(expr_node)
        
        elif isinstance(stmt, (ast.If, ast.While)):
            # Control flow statements
            test_vars = self._get_used_variables(stmt.test)
            ctrl_node = self._add_node(pdg, stmt.__class__.__name__, "control")
            
            for var in test_vars:
                if var not in var_uses:
                    var_uses[var] = []
                var_uses[var].append(ctrl_node)
            
            # Process body
            for s in stmt.body:
                self._process_statement_pdg(s, pdg, var_defs, var_uses)
            
            # Process else if exists
            if hasattr(stmt, 'orelse'):
                for s in stmt.orelse:
                    self._process_statement_pdg(s, pdg, var_defs, var_uses)
        
        elif isinstance(stmt, ast.For):
            # For loop
            if isinstance(stmt.target, ast.Name):
                loop_var = stmt.target.id
                iter_vars = self._get_used_variables(stmt.iter)
                
                for_node = self._add_node(pdg, "FOR", f"for {loop_var}")
                
                # Iterator uses
                for var in iter_vars:
                    if var not in var_uses:
                        var_uses[var] = []
                    var_uses[var].append(for_node)
                
                # Loop variable definition
                var_defs[loop_var] = for_node
            
            # Process body
            for s in stmt.body:
                self._process_statement_pdg(s, pdg, var_defs, var_uses)
    
    def _get_used_variables(self, node: Optional[ast.AST]) -> Set[str]:
        """Get all variables used in an expression."""
        if node is None:
            return set()
        
        used = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                used.add(child.id)
        return used
    
    def _add_node(self, pdg: nx.DiGraph, node_type: str, label: str) -> str:
        """Add a node to PDG."""
        node_id = f"N{self.node_counter}"
        self.node_counter += 1
        pdg.add_node(node_id, type=node_type, label=label[:50])
        return node_id
    
    def encode(self, pdg: nx.DiGraph) -> str:
        """
        Encode PDG to string representation.
        
        Args:
            pdg: Program Dependence Graph
            
        Returns:
            String representation of PDG
        """
        if pdg is None or pdg.number_of_nodes() == 0:
            return "PDG: <empty>"
        
        # Limit number of nodes
        nodes = list(pdg.nodes())[:self.max_nodes]
        
        # Build dependency representation
        deps = []
        for u, v, data in pdg.edges(data=True):
            if u in nodes and v in nodes:
                u_type = pdg.nodes[u].get('type', 'NODE')
                v_type = pdg.nodes[v].get('type', 'NODE')
                u_label = pdg.nodes[u].get('label', '')
                v_label = pdg.nodes[v].get('label', '')
                dep_type = data.get('type', 'dep')
                var = data.get('var', '')
                
                dep_str = f"{u_type}"
                if u_label:
                    dep_str += f"[{u_label}]"
                dep_str += f" --{dep_type}"
                if var:
                    dep_str += f"({var})"
                dep_str += f"--> {v_type}"
                if v_label:
                    dep_str += f"[{v_label}]"
                
                deps.append(dep_str)
        
        return "PDG: " + " | ".join(deps) if deps else "PDG: <no dependencies>"
    
    def extract_and_encode(self, code: str) -> str:
        """
        Extract and encode PDG in one step.
        
        Args:
            code: Python source code
            
        Returns:
            String representation of PDG
        """
        pdg = self.extract(code)
        return self.encode(pdg)
