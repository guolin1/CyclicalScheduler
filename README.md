## OneCycleScheduler

This callback implements a cyclic scheduler for learning rate and momentum (beta_1 for Adam optimizers) according to Smith, 2018.
The method cycles through learning rate and momentum with constant frequency (step size). 

### Example: [also see notebook]
batchsize = 100<br/>
steps_per_epoch = len(x_train)/batchsize<br/>
cyclescheduler = OneCycleScheduler(<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;max_lr=5e-2,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;base_mom = 0.95,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;min_mom = 0.85,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;step_size = steps_per_epoch/2)<br/>
                        
model.fit(x_train,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y_train,<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;batch_size=batchsize,<br/> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;steps_per_epoch=steps_per_epoch,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;epochs=4,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;callbacks=[cyclescheduler])<br/>


### References
Smith, Leslie.N. (2018) A disciplined approach to neural network hyper- <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters: Part 1 -- learning rate, batch size, momentum, and <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;weight decay. (https://arxiv.org/abs/1803.09820).
