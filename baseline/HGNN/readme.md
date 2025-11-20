origin/ 是原始代码，包含原始的HGNN模型和数据集
论文链接https://arxiv.org/pdf/2401.08696
仓库链接https://github.com/sjtu-zhao-lab/hierarchical-gnn-for-hls

e2e是基于原始代码的e2e训练代码，包括训练和推理

/home/user/zedongpeng/workspace/Huggingface/forgehls_lite_100designs_polybenchpart 有已经运行过vitis_hls -f *.tcl 的designs，可以用做训练
我希望在这个数据集上测试HGNN的效果 该怎么做？你先在baseline/HGNN/e2e/readme.md中列出需要做的步骤，然后我判断是否可以