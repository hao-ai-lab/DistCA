#!/usr/bin/env python3

import argparse
import json
import sys
import os
from typing import Dict, List, Any
from attnserver import AttnServerSolver, AttnServerSolution
from wlbllm import WlbLlmSolver, WlbLlmSolution

def process_attnserver_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process an AttnServer job and return the solution."""
    solver = AttnServerSolver()
    solution = solver.solve(
        batch=job['batch'],
        num_workers=job['num_workers'],
        num_total_devices=job['num_total_devices']
    )
    return solution.dump_object()

def process_wlbllm_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process a WLB-LLM job and return the solution."""
    solver = WlbLlmSolver()
    solution = solver.solve(
        doc_lengths=job['batch'],
        max_length=job['max_length'],
        num_workers=job['num_workers']
    )
    return solution.dump_object()


def test_process_attnserver_job():
    job = {
        'batch': [100, 200, 300],
        'num_workers': 3,
        'num_total_devices': 10
    }
    solution = process_attnserver_job(job)
    print(solution)

def test_process_wlbllm_job():
    job = {
        'batch': [100, 200, 300],
        'max_length': 1000,
        'num_workers': 3
    }
    solution = process_wlbllm_job(job)  
    print(solution)


def main(args=None):
    parser = argparse.ArgumentParser(description='Process optimization jobs')
    parser.add_argument('input_file', help='Input JSONL file containing the jobs')
    parser.add_argument('output_file', help='Output JSONL file for results')
    parser.add_argument('--solver', choices=['attnserver', 'wlbllm'], required=True,
                      help='Type of solver to use')
    parser.add_argument('--test', action='store_true',
                      help='Run tests with example input files')
    
    args = parser.parse_args(args)
    
    if args.test:
        test_process_attnserver_job()
        test_process_wlbllm_job()
        return

    with open(args.input_file, 'r') as infile, open(args.output_file, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse job from JSONL
                job = json.loads(line.strip())
                
                # Process job
                if args.solver == 'attnserver':
                    result = process_attnserver_job(job)
                elif args.solver == 'wlbllm':
                    result = process_wlbllm_job(job)
                else:
                    print(f"Invalid solver: {args.solver}")
                    continue
                
                # Write result as JSONL
                outfile.write(json.dumps(result) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Error processing job at line {line_num}: {e}", file=sys.stderr)
                continue
                    
                    

if __name__ == '__main__':
    main() 