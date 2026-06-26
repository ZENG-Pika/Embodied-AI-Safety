"""
Data comparators module for integration tests.
Provides functions to compare generated data with reference data.
"""

from .sequence_comparator import compare_navigation_results

__all__ = ['compare_navigation_results']
