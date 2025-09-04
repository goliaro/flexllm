#!/usr/bin/env python3
"""
OPTIME Log Parser

This script reads the kickstart.log file and aggregates OPTIME entries.
For layer operations (layers.N.*), it groups them by the operation name
without the layer number.
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
        dict: Dictionary with operation names as keys and total times as values
    """
    # Pattern to match OPTIME lines: OPTIME[<OP_NAME>]= <TIME> ms
    optime_pattern = r'OPTIME\[([^\]]+)\]=\s*([0-9.]+)\s*ms'
    
    # Dictionary to store aggregated times
    op_times = defaultdict(float)
    
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
                    
                    # Add to the total time for this operation
                    op_times[aggregated_name] += time_value
                else:
                    print(f"Warning: Line {line_num} starts with OPTIME but doesn't match expected format: {line}")
                    
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return {}
    except Exception as e:
        print(f"Error reading log file: {e}")
        return {}
    
    return dict(op_times)

def print_results(op_times):
    """
    Print the aggregated results in a formatted way.
    
    Args:
        op_times (dict): Dictionary of operation times
    """
    if not op_times:
        print("No OPTIME entries found in the log file.")
        return
    
    print(f"{'Operation Name':<50} {'Total Time (ms)':<15} {'Count':<10}")
    print("-" * 75)
    
    # Sort by total time (descending)
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)
    
    total_time = 0
    for op_name, total_ms in sorted_ops:
        print(f"{op_name:<50} {total_ms:<15.6f}")
        total_time += total_ms
    
    print("-" * 75)
    print(f"{'TOTAL':<50} {total_time:<15.6f}")

def main():
    """Main function to run the OPTIME log parser."""
    # Path to the log file
    log_file_path = "/flexllm/output/kickstart/logs/kickstart.log"
    
    print("OPTIME Log Parser")
    print("=" * 50)
    print(f"Reading log file: {log_file_path}")
    print()
    
    # Parse the log file
    op_times = parse_optime_log(log_file_path)
    
    # Print results
    print_results(op_times)

if __name__ == "__main__":
    main()