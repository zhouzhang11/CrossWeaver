#!/usr/bin/env python3
"""
Generate shell scripts for running shift experiments on 4 GPUs.
This script creates 4 independent bash scripts, each assigned to a different GPU.
"""

import os
from pathlib import Path

# Configuration
WEIGHTS = '/home/sysspace/yanght/test/new/DroneVehicle/yolov8s_new_model10/weights/best.pt'
DATA_YAML = 'drone_vehicle_m.yaml'
PROJECT_DIR = 'DroneVehicle/shift_experiments'
IMGSZ = 800
BATCH = 16

# Generate all shift combinations
shifts = []
for x in range(-30, 31, 5):
    for y in range(-30, 31, 5):
        if x == 0 and y == 0:
            continue
        shifts.append((x, y))

print(f"Total experiments: {len(shifts)}")
assert len(shifts) == 168, f"Expected 168 experiments, got {len(shifts)}"

# Distribute tasks to 4 GPUs
gpu_tasks = [[], [], [], []]
for i, shift in enumerate(shifts):
    gpu_id = i % 4
    gpu_tasks[gpu_id].append(shift)

# Print distribution
print("\nTask distribution:")
for gpu_id in range(4):
    print(f"GPU {gpu_id}: {len(gpu_tasks[gpu_id])} tasks")

# Create output directory for scripts
script_dir = Path('new')
script_dir.mkdir(exist_ok=True)

# Generate shell scripts for each GPU
for gpu_id in range(4):
    script_path = script_dir / f'run_shift_gpu{gpu_id}.sh'
    
    with open(script_path, 'w') as f:
        # Write header
        f.write('#!/bin/bash\n')
        f.write(f'# Shift experiment script for GPU {gpu_id}\n')
        f.write(f'# Total tasks: {len(gpu_tasks[gpu_id])}\n')
        f.write(f'# Generated automatically by generate_shift_scripts.py\n\n')
        
        f.write('# Change to the correct directory\n')
        f.write('cd /home/sysspace/yanght/test/new\n\n')
        
        f.write('# Log file\n')
        f.write(f'LOG_FILE="shift_gpu{gpu_id}.log"\n')
        f.write('echo "Starting shift experiments on GPU {}" > $LOG_FILE\n'.format(gpu_id))
        f.write('echo "Start time: $(date)" >> $LOG_FILE\n')
        f.write('echo "" >> $LOG_FILE\n\n')
        
        # Write commands for each shift
        for idx, (shift_x, shift_y) in enumerate(gpu_tasks[gpu_id], 1):
            f.write(f'# Task {idx}/{len(gpu_tasks[gpu_id])}: shift_x={shift_x:+d}, shift_y={shift_y:+d}\n')
            f.write(f'echo "Task {idx}/{len(gpu_tasks[gpu_id])}: shift_x={shift_x:+d}, shift_y={shift_y:+d}" >> $LOG_FILE\n')
            f.write(f'echo "Time: $(date)" >> $LOG_FILE\n')
            
            cmd = (f'python val_shift.py '
                   f'--weights {WEIGHTS} '
                   f'--data {DATA_YAML} '
                   f'--shift_x {shift_x} '
                   f'--shift_y {shift_y} '
                   f'--imgsz {IMGSZ} '
                   f'--batch {BATCH} '
                   f'--device {gpu_id} '
                   f'--project {PROJECT_DIR} '
                   f'--name shift_experiment')
            
            f.write(f'{cmd} >> $LOG_FILE 2>&1\n')
            f.write(f'echo "Completed: shift_x={shift_x:+d}, shift_y={shift_y:+d}" >> $LOG_FILE\n')
            f.write('echo "" >> $LOG_FILE\n\n')
        
        # Write footer with mAP50 summary
        f.write('echo "All tasks completed!" >> $LOG_FILE\n')
        f.write('echo "End time: $(date)" >> $LOG_FILE\n')
        f.write('echo "" >> $LOG_FILE\n')
        f.write('echo "============================================================" >> $LOG_FILE\n')
        f.write('echo "Summary of mAP50 for all shifts:" >> $LOG_FILE\n')
        f.write('echo "============================================================" >> $LOG_FILE\n')
        f.write('grep -A 4 "Validation Results:" $LOG_FILE | grep -E "Shift:|mAP50:" | paste -d " " - - | sed "s/Shift:/偏移:/g" | sed "s/mAP50:/mAP50:/g" >> $LOG_FILE\n')
        f.write('echo "============================================================" >> $LOG_FILE\n')
        f.write(f'echo "GPU {gpu_id} finished all {len(gpu_tasks[gpu_id])} tasks"\n')
    
    # Make script executable
    os.chmod(script_path, 0o755)
    print(f"Created: {script_path}")

# Create a master script to run all GPUs
master_script = script_dir / 'run_all_shifts.sh'
with open(master_script, 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('# Master script to run all shift experiments on 4 GPUs in parallel\n\n')
    f.write('cd /home/sysspace/yanght/test/new\n\n')
    f.write('echo "Starting shift experiments on all 4 GPUs..."\n')
    f.write('echo "Total experiments: 168 (42 per GPU)"\n')
    f.write('echo ""\n\n')
    
    for gpu_id in range(4):
        f.write(f'./run_shift_gpu{gpu_id}.sh &\n')
    
    f.write('\n# Wait for all background jobs to complete\n')
    f.write('wait\n\n')
    f.write('echo ""\n')
    f.write('echo "All experiments completed!"\n')
    f.write('echo "Check individual log files: shift_gpu0.log, shift_gpu1.log, shift_gpu2.log, shift_gpu3.log"\n')

os.chmod(master_script, 0o755)
print(f"Created: {master_script}")

print("\n" + "="*60)
print("Scripts generated successfully!")
print("="*60)
print("\nTo run experiments:")
print("\nOption 1 - Run all GPUs in parallel (recommended):")
print("  cd /home/sysspace/yanght/test/new")
print("  ./run_all_shifts.sh")
print("\nOption 2 - Run each GPU in separate terminals:")
print("  Terminal 1: cd /home/sysspace/yanght/test/new && ./run_shift_gpu0.sh")
print("  Terminal 2: cd /home/sysspace/yanght/test/new && ./run_shift_gpu1.sh")
print("  Terminal 3: cd /home/sysspace/yanght/test/new && ./run_shift_gpu2.sh")
print("  Terminal 4: cd /home/sysspace/yanght/test/new && ./run_shift_gpu3.sh")
print("\nLogs will be saved to:")
print("  shift_gpu0.log, shift_gpu1.log, shift_gpu2.log, shift_gpu3.log")
print("\nResults will be saved to:")
print(f"  {PROJECT_DIR}/shift_experiment_x*_y*/")
print("="*60)
