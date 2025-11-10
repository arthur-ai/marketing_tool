"""
Dependency graph validation for pipeline plugins.

This module provides dependency graph building and validation to ensure
plugins have correct dependencies and execution order.
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger("marketing_project.plugins.dependency_graph")


class DependencyGraph:
    """
    Builds and validates dependency graphs for pipeline plugins.

    Validates that:
    - All dependencies are satisfied by previous steps
    - No circular dependencies exist
    - Execution order is correct
    """

    def __init__(self, plugins: List[PipelineStepPlugin]):
        """
        Initialize dependency graph with plugins.

        Args:
            plugins: List of plugin instances
        """
        self.plugins = plugins
        self._graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self._step_names: Dict[int, str] = {}

    def build_graph(self) -> None:
        """
        Build the dependency graph from plugin requirements.

        Creates a graph where each node is a step_name and edges represent
        dependencies (step A depends on step B if A requires B's output).
        """
        # Map step numbers to step names
        for plugin in self.plugins:
            self._step_names[plugin.step_number] = plugin.step_name

        # Build dependency graph
        for plugin in self.plugins:
            step_name = plugin.step_name
            required_keys = plugin.get_required_context_keys()

            # Map required context keys to step names
            for key in required_keys:
                # Skip keys provided by the pipeline, not by a step
                if key in ("input_content", "content_type"):
                    continue

                # Find which step produces this key
                # Context keys typically match step names (e.g., "seo_keywords")
                producing_plugin = None
                for p in self.plugins:
                    if key == p.step_name:
                        producing_plugin = p
                        break

                if producing_plugin:
                    # Direct step dependency
                    self._graph[step_name].add(producing_plugin.step_name)
                    self._reverse_graph[producing_plugin.step_name].add(step_name)

    def check_circular_deps(self) -> Tuple[bool, Optional[List[str]]]:
        """
        Check for circular dependencies in the graph.

        Uses topological sort (Kahn's algorithm) to detect cycles.

        Returns:
            Tuple of (has_cycle, cycle_path if cycle exists)
        """
        # Calculate in-degree for each node
        # In our graph: _graph[A] = {B, C} means "A depends on B and C"
        # For topological sort: if A depends on B, then B must come before A
        # So we create edges B -> A and C -> A
        # in_degree[A] = number of nodes that must come before A = number of dependencies A has

        in_degree: Dict[str, int] = defaultdict(int)

        # Initialize all nodes
        all_nodes = set(self._graph.keys())
        for node, deps in self._graph.items():
            all_nodes.update(deps)

        # Calculate in-degree: number of dependencies each node has
        for node in all_nodes:
            # in_degree = number of things this node depends on
            in_degree[node] = len(self._graph.get(node, set()))

        # Find nodes with no dependencies (can be processed first)
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        processed = 0
        topo_order = []

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            processed += 1

            # This node is processed, so reduce in-degree of all nodes that depend on it
            # _reverse_graph[node] contains all nodes that depend on 'node'
            for dependent in self._reverse_graph.get(node, []):
                # dependent depends on node, so when node is processed,
                # dependent's in-degree decreases
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # If we didn't process all nodes, there's a cycle
        if processed != len(all_nodes):
            # Find the cycle
            cycle = self._find_cycle()
            return True, cycle

        return False, None

    def _find_cycle(self) -> List[str]:
        """
        Find a cycle in the dependency graph using DFS.

        Returns:
            List of step names forming a cycle
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in self._graph:
            if node not in visited:
                if dfs(node):
                    return path

        return []

    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Validate that all dependencies are satisfied by previous steps.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Sort plugins by step number
        sorted_plugins = sorted(self.plugins, key=lambda p: p.step_number)

        # Track what's available at each step
        available_outputs: Set[str] = {"input_content", "content_type"}

        for plugin in sorted_plugins:
            step_name = plugin.step_name
            required_keys = plugin.get_required_context_keys()

            # Check if all required keys are available
            missing = [key for key in required_keys if key not in available_outputs]

            if missing:
                errors.append(
                    f"Step {plugin.step_number} ({step_name}) requires missing context keys: {missing}. "
                    f"Available keys: {sorted(available_outputs)}"
                )

            # Add this step's output to available outputs
            available_outputs.add(step_name)

        return len(errors) == 0, errors

    def get_execution_order(self) -> List[str]:
        """
        Get the execution order based on dependencies and step numbers.

        Returns:
            List of step names in execution order
        """
        # Sort by step number (which should already respect dependencies)
        sorted_plugins = sorted(self.plugins, key=lambda p: p.step_number)
        return [plugin.step_name for plugin in sorted_plugins]

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Perform all validations: circular deps and dependency satisfaction.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Build the graph first
        self.build_graph()

        # Check for circular dependencies
        has_cycle, cycle_path = self.check_circular_deps()
        if has_cycle:
            errors.append(
                f"Circular dependency detected: {' -> '.join(cycle_path)} -> {cycle_path[0] if cycle_path else '?'}"
            )

        # Validate dependencies are satisfied
        deps_valid, deps_errors = self.validate_dependencies()
        if not deps_valid:
            errors.extend(deps_errors)

        return len(errors) == 0, errors
