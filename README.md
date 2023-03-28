# MoireDet
cuda:10.0;
pytorch 1.4;
torchvision 0.5;


# MoireScape*

Our proposed MoireScape dataset contains two subsets:
 - MoireScape-real: It contains 500 real image pairs for evaluating moiré edge map estimation. Each pair includes a real camera-captured screen image and its moiré layer with the setup in Fig. 3. To extract moire edge map from a moire layer, the more_layer_segmentation tool can be used.
 - MoireScape-synthetic: It contains 18,147 different moiré layers and 4,000 natural images. After varying moiré layers and the their combinations, 50,000 synthetic triplets are collected for the purpose of training (90%) and testing(10%). Each triplet contains a natural image, a moiré layer, and their synthetic mixture.

*: Since each subset surpass 25M, please download them via the online disc link in each text file. 
 
## Citation

If you benefit from this work, please cite the mentioned and our paper:

	@article{Yang2023Moire,
		author = {Cong Yang and Zhenyu Yang and Yan Ke and Tao Chen and Marcin Grzegorzek and John See},
		title = {Doing More With Moiré Pattern Detection in Digital Photos},
		journal = {IEEE Transactions on Image Processing},
            volume = {32},
            pages = {694-708},
            year = {2023}
	}
