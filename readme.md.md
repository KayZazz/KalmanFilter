---


---

<h1 id="kalman-filter">Kalman Filter</h1>
<ol>
<li>
<h2 id="introduction">Introduction</h2>
<ul>
<li>
<h3 id="linear-kalman-filter-kf">Linear Kalman filter (KF)</h3>
<ul>
<li>
<h4 id="models-in-kalman-filter">Models in Kalman filter</h4>
<p>In kalman filter, there are 2 kinds of models: process model and measurement model.</p>
<ul>
<li>
<p>Process model:</p>
<p><img src="./Images/model_1.png" alt="process model"></p>
<p>where<br>
<em>state vector</em>: input vector x. x represents the states of the system you design (set of states)<br>
<em>state variable</em>:each variable inside x. Each state variable represents a state in the system(velocity, position)<br>
<em>trainsition matrix</em>:matrix phi beside input vector x . Transition matrix is the matrix transiting the states at k moment to states at k+1 moment. The transition matrix is dependent on the system you build.<br>
For example:	<br>
we know velocity v_k= a * t_k at k moment.  In k+1 moment, v_k+1 = a* t_k+1. Then we have v_k+1 = v_k + a* (t_k+1 - t_k).  To represent v and a in matrix form:<br>
[v_k+1, a_k+1]^T  = [1, delta_t]* [v_k, a_k]^T<br>
[1, delta_t] is the transition matrix here<br>
Measurement model:</p>
</li>
</ul>
<p><img src="./Images/model_2.png" alt="process model"></p>
</li>
<li>
<h4 id="designing-a-kalman-filter">Designing a kalman filter</h4>
</li>
</ul>
</li>
<li>
<h3 id="extended-kalman-filter-ekf">Extended Kalman filter (EKF)</h3>
<ul>
<li></li>
</ul>
</li>
</ul>
</li>
<li>
<h2 id="flowchart">Flowchart</h2>
</li>
<li>
<h2 id="pseudo-code">Pseudo Code</h2>
</li>
<li>
<h2 id="my-summary">My Summary</h2>
</li>
<li>
<h2 id="reference">Reference</h2>
</li>
</ol>

