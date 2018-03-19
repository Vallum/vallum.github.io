# Hyper-parameter Tuning for Multi Gpu Deep Learning

- Environment
  - Trained for RMSProp
  - 8 GPUS
  - Images training from scratch
  - Step-down learnining rate schedule
  - L2 regularization (= weight decay)
  - extra regularization with the center loss and embedding regularization for a logit layer.
  
- Q1: I have the learned(=optimized) parameter set for single GPU, what should I do for multi gpus from this?
  - You have a strong benchmark for the multi gpus performance.
  - Keep in mind that the multi gpus model doesn't have any reason to underperform compared to a single gpu model.
  - First, try longer learning than you think which is necessary to find the optimized parameter set, and then try to reduce the learning schedule.
 
- Q2: Why not use the exponential decay for the training?
  - Between hyper-parameters like the initial learning rate, weight decay, batch sizes, epochs(=batch rounds), and the mostly learning method(=algorithms like RMSProp) 
  - If you use the exponentail decay, your learning falls too fast and it is unnecessary because the adaptive learning algorithms like RMSProp is already adjusting for the result of gradients.
  - So your training could go to the early convergence which is not optimized.
  - You could try raising the initial learning rate to compensate the early decay and convergence
  - But because the learning rate has the obvious upper bound as 0.9~1.0, you can not raise the learning rate as much as you need.
  - That you need to do is to make the learning rate last high enough till it becomes unnecessary.
  
- Q3: Why the model goes to the early convergence and seems like stop learning?
  - In many case, it is because of the effect of L2 regularization.
  - The L2 regularization called 'weight decay' because it decays the weights of your networks.
  - If you check the gradients from the L2 regularization formula, you find that it reduces the weight of network as the ratio of weight decay.
  - It means that if the weight decay is larger than your learning, your learning is of no use from any point of your learning schedule.
  - More precisely, your gradient of learning has meaning only if your gradient of learning is larger than the gradient from the weight decay.
  - So reduce the weight decay (of L2 regularization) and/or raise the initail learning rate higher and/or longer.

- Q4: Sometimes the performance get worse after convergence, what is happening?

- Q5: I have extra regularization or loss function besides the cross entropy and the standard L2 regularization. What should I do?
  - Most of all, draw a chart for each loss and monitor it.
  - Sometimes they rise with L2 loss and sometimes they go down with the entropy loss.
  - But I find that usually L2 loss is the precursor of the other losses.
  - It is because that the weight decay is unconditional for the performance of networks.
  - The other loss have their own learning curve, so find the convergence pattern and make them converge.
  
- Q6: What is the effect of batch size in a multi gpus model?
  - Not because of multi gpus, but because of the increased batch size followed the multi gpus, the higher learning rate is acceptable and necessary simultaneously.
  - If you can find the best mix of the learning rate schedule, a proper weight decay ratio, and enough round of epochs to converge with relate to your maximum batch size which is allowed in your GPU memory limit, then all done! 
  
- In summary: try this
  - start with 
     - a maximum batch size which your GPU memory can contain
     - a learning epochs as long as you can be patient with
     - a highest initial learning rate which is not unstable (sometimes it is aroung 0.8 or more but not more than 1.0 perhaps)
  - then
     - last your initail learning rate until your learning becomes unstable. Stick with it! 
     - reduce your learning rate with log scale. (divide it 1/10 or 1/8)
