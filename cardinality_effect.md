Maybe I could be wrong, but I am this idea.

In CVPR 2017, they chose the next paper as a best paper.

Densely Connected Convolutional Networks by Gao Huang, Zhuang Liu, Laurens van der Maaten, & Kilian Q. Weinberger
https://arxiv.org/abs/1608.06993
(Zukerbuck even proudly posted that this is from Facebook AI in his own timeline.)

But, even before I read this paper, it is so accustomed to me from the very first page.

Actually, the below research commonly point the same thing.

Semi-Supervised Learning with Ladder Networks
Antti Rasmus, Harri Valpola, Mikko Honkala, Mathias Berglund, Tapani Raiko
https://arxiv.org/abs/1507.02672

Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections
Xiao-Jiao Mao, Chunhua Shen, Yu-Bin Yang
https://arxiv.org/abs/1606.08921

U-Net: Convolutional Networks for Biomedical Image Segmentation
Olaf Ronneberger, Philipp Fischer, Thomas Brox
https://arxiv.org/abs/1505.04597

Aggregated Residual Transformations for Deep Neural Networks
Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
https://arxiv.org/abs/1611.05431
(이건 Res-Next)

Shortcut connection somethings are used for promoting cardinality effect
like Google Inception + Inception-Resnet-V2

In short, 4-depth network e.g. A-B-C-D can be added shortcuts like A-C, A-D, B-D.
It is similar to the network which has a double-cardinality such as A-(B|C)-D.

Of counce, they are differenct in the point that the later doesn't have a parallel connection of B-C,
and a skip connection of A-D,
but we can compensate that by increasing depth without difficulty.

If from the beginning we provide E as a size of sum of B+C channels, 
is A-(B|C)-D different from A-E-D?
So to speak, why is B+C better than E?

They say B+C has more numbers of transformation than E.
The numbers of transformation, they call it cardinality.
(The idea is from Google Inception and from 1611.05431.)

If we assume B, C, E each as a transformation,
B+C has less connections but has more transformation cases.

The optimization which increases cardinality but reduces cardinality has been already discovered in Neuroscience 
https://en.wikipedia.org/wiki/Synaptic_pruning
(Synaptic pruning, which includes both axon and dendrite pruning, is the process of synapse elimination that occurs between early childhood and the onset of puberty in many mammals, including humans.

It is believed that the purpose of synaptic pruning is to remove unnecessary neuronal structures from the brain; as the human brain develops, the need to understand more complex structures becomes much more pertinent, and simpler associations formed at childhood are thought to be replaced by complex structures.)

I speculate that the path of deep learning discovery follows from reducing inefficient fully connected layer to adding findings of meaningful dimmensions.

I will discuss more of such issues in this blog from now on.
Deep learning의 발전과정은 fully connected network에서 노드간 무의미한 connection 숫자는 줄이고, 의미를 가진 dimmension을 발견해나가는 과정으로 발전하고 있다고 보면 될 것 같습니다.