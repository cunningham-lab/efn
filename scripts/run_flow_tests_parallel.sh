nohup python3 test_flow_expressivity.py normal 1 0 0 0 2>&1 > 1.log &
nohup python3 test_flow_expressivity.py normal 0 4 0 0 2>&1 > 2.log &
nohup python3 test_flow_expressivity.py normal 0 0 1 0 2>&1 > 3.log &
nohup python3 test_flow_expressivity.py normal 0 0 2 0 2>&1 > 4.log &
nohup python3 test_flow_expressivity.py normal 0 0 0 1 2>&1 > 5.log &
nohup python3 test_flow_expressivity.py normal 0 0 0 2 2>&1 > 6.log &

nohup python3 test_flow_expressivity.py dirichlet 1 0 0 0 2>&1 > 7.log &
nohup python3 test_flow_expressivity.py dirichlet 0 4 0 0 2>&1 > 8.log &
nohup python3 test_flow_expressivity.py dirichlet 0 0 1 0 2>&1 > 9.log &
nohup python3 test_flow_expressivity.py dirichlet 0 0 2 0 2>&1 > 10.log &
nohup python3 test_flow_expressivity.py dirichlet 0 0 0 1 2>&1 > 11.log &
nohup python3 test_flow_expressivity.py dirichlet 0 0 0 2 2>&1 > 12.log &

