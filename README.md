Loss does not change per epoch as the training works by trying out every node for next node
always and choosing the least loss one. So, for every epoch, given the same input-output
pairs, no parameters are used to find path during training and hence, it always takes the same
paths for the same pairs during training, causing the same outputs and losses for every epoch.