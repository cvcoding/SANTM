Sparse Access Neural Turing Machine (SANTM)
Overview
This repository contains the implementation of the Sparse Access Neural Turing Machine (SANTM) for memorizing long-term information in sequence learning tasks. The SANTM is a neural network architecture that combines a three-level neural controller with an external memory. The model breaks input information into variable-length segments, forms short-term memory, and stores it in the external memory as long-term information.

Architecture
The SANTM architecture comprises:

Bottom Level: Input information is segmented into variable-length pieces.
Middle Level: The segmented information is collected and cascaded to form short-term memory.
Top Level: The short-term memory is stored in the external memory for long-term recall.
For efficient information retrieval from the external memory, a multi-head self-attention module based on ChebNet spectral graph convolution is employed. An inductive bias of locality is introduced to address the data-hungry issue of global self-attention.

Implementation Details
Neural Controller: Implemented using deep neural networks.
External Memory: Managed using sparse data structures for efficient memory access.
Multi-head Self-Attention: Based on ChebNet spectral graph convolution for localized attention.
Sparsity Constraints: An optimization scheme is proposed to impose sparsity constraints on the attention mask.
Optimization
To train the model, an optimization scheme is implemented that enforces sparsity constraints on the attention mask. This helps reduce the computational complexity and improve the memory capacity of the model.

Memory Capacity Analysis
A theoretical analysis of the network's memory capacity is conducted, and an optimal memory access rate is deduced based on the analysis of the mask's sparsity.

Experiments
The SANTM is evaluated on various sequence learning tasks with both fixed- and variable-length data. The experimental results demonstrate that the SANTM outperforms other state-of-the-art models.

References
SANTM: A Sparse Access Neural Turing Machine with Local Multi-head Self-attention for Long-term Memorization， Dongjing Shan and Jing Zhu.

Contributions
Contributions are welcome! Please feel free to open issues, submit pull requests, or contact the authors for more information.

The implementation of recurrent networks, encompassing Skip LSTM, Dilated LSTM, DRRLSTM, and Neural Turing Machines, has been executed utilizing Tensorflow. Meanwhile, the Transformer and Mamba models have been realized in Pytorch.

During training, the models were executed on a single NVIDIA RTX 3060 GPU.

