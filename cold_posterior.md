## About the Cold Posterior

```
Cold Posteriors: among all temperized posteriors the best posterior predictive performance on holdout data is achieved at temperature T < 1.
from arXiv 2002.02405, Wenzel et al.
```

### papers
- Wenzel et al., How good is the bayes posterior in deep neural networks really?, arXiv preprint arXiv:2002.02405, 2020
- Laurence Aitchison, A statistical theory of cold posteriors in deep neural networks, arXiv preprint, 2020.
- Laurence Aitchison, A statistical theory of semi-supervised learning. arXiv preprint, 2020.

### Summary
- BNN(Baysian Neural Networks) are not strong as expected. But why?
- In Wenzel et al.(2020) the authors say that the temperature should be "cold" which means that T << 1.
- Technically, it means the dataset replecated for N(= 1/T) times for each parameter set.
- Two papers of Aitchinson(2020) explain it theoretically with looking into the manuvered process of making sample dataset like Imagenet or CIFAR-10.
- Labelers' consensus(i.e. unanimous agreement) based choice twist the distribution of dataset.
- So the beginning of the claim is that labels should be counted as times with the number of consensus participants.
```
This likelihood is equivalent to labelling each datapoint S times with the same label, and therefore
has the effect of setting λ = S in a tempered posterior
 - from arXiv 2008.05912 Aitchinson
```

### Related Work
- Bayesian Deep Learning (T=1) vs. MAP(T=0)
- Entropy miminization and pseudo-labelling in Semi-supervised learning (arXiv 2008.05913 Aitchinson)
- Data augmentation (arXiv 2008.05913 Aitchinson)
- MixMatch in semi-supervised learning (arXiv 2008.05913 Aitchinson)
- They are all lower-bounds on principled log-likelihood objectives
