\ Model ControlOptimization
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 u[9] + [ 2 x1[0] ^2 + 2 x1[1] ^2 + 2 x1[2] ^2 + 2 x1[3] ^2 + 2 x1[4] ^2
   + 2 x1[5] ^2 + 2 x1[6] ^2 + 2 x1[7] ^2 + 2 x1[8] ^2 + 2 x1[9] ^2
   + 2 x2[0] ^2 + 2 x2[1] ^2 + 2 x2[2] ^2 + 2 x2[3] ^2 + 2 x2[4] ^2
   + 2 x2[5] ^2 + 2 x2[6] ^2 + 2 x2[7] ^2 + 2 x2[8] ^2 + 2 x2[9] ^2 ] / 2 
Subject To
 initial_x1: x1[0] = 8.49099e-13
 initial_x2: x2[0] = 0
 state1_0: - x1[0] + x1[1] - x2[0] = 0
 state2_0: - x2[0] + x2[1] - u[0] = 0
 state1_1: - x1[1] + x1[2] - x2[1] = 0
 state2_1: - x2[1] + x2[2] - u[1] = 0
 state1_2: - x1[2] + x1[3] - x2[2] = 0
 state2_2: - x2[2] + x2[3] - u[2] = 0
 state1_3: - x1[3] + x1[4] - x2[3] = 0
 state2_3: - x2[3] + x2[4] - u[3] = 0
 state1_4: - x1[4] + x1[5] - x2[4] = 0
 state2_4: - x2[4] + x2[5] - u[4] = 0
 state1_5: - x1[5] + x1[6] - x2[5] = 0
 state2_5: - x2[5] + x2[6] - u[5] = 0
 state1_6: - x1[6] + x1[7] - x2[6] = 0
 state2_6: - x2[6] + x2[7] - u[6] = 0
 state1_7: - x1[7] + x1[8] - x2[7] = 0
 state2_7: - x2[7] + x2[8] - u[7] = 0
 state1_8: - x1[8] + x1[9] - x2[8] = 0
 state2_8: - x2[8] + x2[9] - u[8] = 0
Bounds
 x1[0] free
 x1[1] free
 x1[2] free
 x1[3] free
 x1[4] free
 x1[5] free
 x1[6] free
 x1[7] free
 x1[8] free
 x1[9] free
 x2[0] free
 x2[1] free
 x2[2] free
 x2[3] free
 x2[4] free
 x2[5] free
 x2[6] free
 x2[7] free
 x2[8] free
 x2[9] free
 -5 <= u[0] <= 5
 -5 <= u[1] <= 5
 -5 <= u[2] <= 5
 -5 <= u[3] <= 5
 -5 <= u[4] <= 5
 -5 <= u[5] <= 5
 -5 <= u[6] <= 5
 -5 <= u[7] <= 5
 -5 <= u[8] <= 5
 -5 <= u[9] <= 5
End
