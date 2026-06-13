#!/bin/bash
# Quick test script to verify shift functionality
# Tests a single shift (x=5, y=5) to ensure everything works

cd /home/sysspace/yanght/test/new

echo "=========================================="
echo "Testing shift functionality"
echo "Running single validation with shift_x=5, shift_y=5"
echo "=========================================="
echo ""

python val_shift.py \
    --shift_x 5 \
    --shift_y 5 \
    --device 0 \
    --name test_shift

echo ""
echo "=========================================="
echo "Test completed!"
echo "Check results in: DroneVehicle/test_shift_x+5_y+5/"
echo "=========================================="
