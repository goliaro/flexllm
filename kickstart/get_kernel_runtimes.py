#!/usr/bin/env python3
"""
OPTIME Log Parser

This script reads the kickstart.log file and aggregates OPTIME entries.
For layer operations (layers.N.*), it groups them by the operation name
without the layer number. For each operation type, it displays:
- Total time across all occurrences
- Number of occurrences (count)
- Average time per occurrence
"""

import re
import os
from collections import defaultdict

def parse_optime_log(log_file_path):
    """
    Parse the log file and extract OPTIME entries.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Dictionary with operation names as keys and {'total_time': float, 'count': int} as values
    """
    # Pattern to match OPTIME lines: OPTIME[<OP_NAME>]= <TIME> ms
    optime_pattern = r'OPTIME\[([^\]]+)\]=\s*([0-9.]+)\s*ms'
    
    # Dictionary to store aggregated times and counts
    op_stats = defaultdict(lambda: {'total_time': 0.0, 'count': 0})
    
    try:
        with open(log_file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                
                # Skip lines that don't start with OPTIME
                if not line.startswith('OPTIME'):
                    continue
                
                # Match the OPTIME pattern
                match = re.search(optime_pattern, line)
                if match:
                    op_name = match.group(1)
                    time_value = float(match.group(2))
                    
                    # Check if this is a layer operation (layers.N.*)
                    layer_pattern = r'^layers\.(\d+)\.(.+)$'
                    layer_match = re.match(layer_pattern, op_name)
                    
                    if layer_match:
                        # Extract the operation name without the layer number
                        operation = layer_match.group(2)
                        aggregated_name = operation
                    else:
                        # Keep the original name for non-layer operations
                        aggregated_name = op_name
                    
                    # Add to the total time and count for this operation
                    op_stats[aggregated_name]['total_time'] += time_value
                    op_stats[aggregated_name]['count'] += 1
                else:
                    print(f"Warning: Line {line_num} starts with OPTIME but doesn't match expected format: {line}")
                    
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading log file: {e}")
        return {}
    
    return dict(op_stats)

def print_results(op_stats):
    """
    Print the aggregated results in a formatted way.
    
    Args:
        op_stats (dict): Dictionary of operation statistics
    """
    if not op_stats:
        print("No OPTIME entries found in the log file.")
        return
    
    print(f"{'Operation Name':<50} {'Total Time (ms)':<15} {'Count':<10} {'Avg Time (μs)':<15}")
    print("-" * 100)
    
    # Sort by total time (descending)
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
    
    total_time = 0
    total_count = 0
    for op_name, stats in sorted_ops:
        total_ms = stats['total_time']
        count = stats['count']
        avg_us = (total_ms * 1000) / count if count > 0 else 0  # Convert ms to μs
        print(f"{op_name:<50} {total_ms:<15.3f} {count:<10} {avg_us:<15.3f}")
        total_time += total_ms
        total_count += count
    
    print("-" * 100)
    overall_avg_us = (total_time * 1000) / total_count if total_count > 0 else 0  # Convert ms to μs
    print(f"{'TOTAL':<50} {total_time:<15.3f} {total_count:<10} {overall_avg_us:<15.3f}")

def main():
    """Main function to run the OPTIME log parser."""
    # Path to the log file
    log_file_path = "/pscratch/sd/g/goliaro/flexllm/output/kickstart/logs/kickstart.log"
    
    print("OPTIME Log Parser")
    print("=" * 50)
    print(f"Reading log file: {log_file_path}")
    print()
    
    # Parse the log file
    op_stats = parse_optime_log(log_file_path)
    
    # Print results
    print_results(op_stats)

if __name__ == "__main__":
    main()