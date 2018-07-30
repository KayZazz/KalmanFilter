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
<p>In kalman filter, there are 3 kinds of models: noise model, process model and measurement model .</p>
<ul>
<li>
<p>Process model:</p>
<p><img src="./Images/model_1.png" alt="process model"></p>
<p>where<br>
<em>state vector x</em>:<br>
input vector x. x represents the states of the system you design (set of states)<br>
<em>state variable <em>:<br>
each variable inside x. Each state variable represents a state in the system(velocity, position)<br>
<em>trainsition matrix phi</em>:<br>
matrix phi beside input vector x . Transition matrix is the matrix transiting the states at k moment to states at k+1 moment. The transition matrix is dependent on the system you build.<br>
<em>noise term wk</em>:<br>
usually the noise is in Gaussain noise<br>
For example:	<br>
we know velocity v_k= a * t_k at k moment.  In k+1 moment, v_k+1 = a</em> t_k+1. Then we have v_k+1 = v_k + a</em> (t_k+1 - t_k).  To represent v and a in matrix form:<br>
[v_k+1, a_k+1]^T  = [1, delta_t]* [v_k, a_k]^T<br>
[1, delta_t] is the transition matrix here</p>
</li>
<li>
<p>Measurement model:</p>
<p><img src="./Images/model_2.png" alt="process model"><br>
where<br>
<em>zk</em>: measurement vector obtained from sensors<br>
<em>H</em>: connection matrix between measurement vector z and the state vector x<br>
<em>vk</em>:  the noise from measurement</p>
</li>
<li>
<p>Noise model:<br>
in noise model, there are 3 covariance matrices to representing the information in the model<br>
<img src="./Images/model_3.png" alt="noise model"></p>
<p><img src="./Images/model_4.png" alt="noise model"></p>
</li>
</ul>
</li>
<li>
<h4 id="designing-a-kalman-filter">Designing a kalman filter</h4>
<ul>
<li>The key idea of Kalman filter is to use maximium likehood method to minimize the error between the expected output signal and the measurement. In kalman filter, it defines <em>estimation</em> <img src="./Images/v1.png" alt="v1">  and <em>prior estimation</em> <img src="./Images/v2.png" alt="v2"><br>
Their relation is:<br>
<img src="./Images/cal_1.png" alt="relationship"><br>
where "Zk - HXk` " is called measurement residual. It is the error term between actual states and estimated states. In my prospective,  <img src="./Images/v1.png" alt="v1"> can be considered as the expected output states processed with noise term, which can have the ‘ability’ to ‘fight against’ the noise. Then it is used to predict the next output by transition matrix.<br>
What’s more, in the equation above, ‘Kk’ is called Kalman gain, which is used to control how the error term affects the estimation. Our goal is to calculate the kalman gain to predict the next output.
<ul>
<li>To calculate kalman gain<br>
To find the Kalman gain,</li>
</ul>
</li>
</ul>
</li>
</ul>
</li>
<li>
<h3 id="extended-kalman-filter-ekf">Extended Kalman filter (EKF)</h3>
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

