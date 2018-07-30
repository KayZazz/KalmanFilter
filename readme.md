# Kalman Filter
1. ## Introduction
	+ ### Linear Kalman filter (KF)
		+ #### Models in Kalman filter 
			In kalman filter, there are 2 kinds of models: process model and measurement model.
			
		   * Process model:
		   
		   ![process model](./Images/model_1.png)
		   
		   where  
			    - *state vector*:     
			    input vector x. x represents the states of the system you design (set of states)
			    - *state variable*: 
			    each variable inside x. Each state variable represents a state in the system(velocity, position)
			    - *trainsition matrix*: 
			    matrix phi beside input vector x . Transition matrix is the matrix transiting the states at k moment to states at k+1 moment. The transition matrix is dependent on the system you build.
			    For example: 
			    we know velocity v_k= a * t_k at k moment. 
			    In k+1 moment, v_k+1 = a* t_k+1. Then we have v_k+1 = v_k + a* (t_k+1 - t_k).  To represent v and a in matrix form:
			    [v_k+1, a_k+1]^T  = [1, delta_t]* [v_k, a_k]^T
			    [1, delta_t] is the transition matrix here
			* Measurement model:
			
			![process model](./Images/model_2.png)
		+ #### Designing a kalman filter
			
	+ ### Extended Kalman filter (EKF)
		+ 
2. ## Flowchart
3. ## Pseudo Code
4. ## My Summary
5. ## Reference
