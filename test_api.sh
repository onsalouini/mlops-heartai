#!/bin/bash

echo "=== Test API Heart Disease Prediction ===\n"

echo "Test 1: Patient 63 ans - À risque"
curl -s -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]}' | python3 -m json.tool

echo -e "\nTest 2: Patient 45 ans - Bonne santé"
curl -s -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [45,0,2,120,180,0,1,130,0,1.2,0,1,1]}' | python3 -m json.tool

echo -e "\nTest 3: Patient 55 ans - Cas limite"
curl -s -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [55,1,1,140,220,0,0,145,0,1.8,1,1,2]}' | python3 -m json.tool
