=====================================================================
Large-scale Fashion Recognition and Retrieval (DeepFashion) Dataset
=====================================================================

=====================================
In-shop Clothes Retrieval Benchmark
=====================================

--------------------------------------------------------
By Multimedia Lab, The Chinese University of Hong Kong
--------------------------------------------------------

For more information about the dataset, visit the project website:

  http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

If you use the dataset in a publication, please cite the paper below:

  @inproceedings{liu2016deepfashion,
 	author = {Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang},
 	title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 	booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 	month = June,
 	year = {2016} 
  }

Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes.



========================
Change Log
========================

Version 1.0, released on 18/07/2016
Version 1.1, released on 29/08/2016, add item description annotations
Version 1.2, released on 22/12/2016, add landmarks annotations



========================
File Information
========================

- Images (Img/img.zip)
    52,712 in-shop clothes images (~200,000 cross-pose/scale pairs). See IMAGE section below for more info.

- Bounding Box Annotations (Anno/list_bbox_inshop.txt)
    bounding box labels. See BBOX LABELS section below for more info.

- Fashion Landmark Annotations (Anno/list_landmarks_inshop.txt)
	fashion landmark labels. See LANDMARK LABELS section below for more info.

- Item Annotations (Anno/list_item_inshop.txt)
	item labels. See ITEM LABELS section below for more info.

- Description Annotations (Anno/list_description_inshop.json)
	item descriptions. See DESCRIPTION LABELS section below for more info.

- Evaluation Partitions (Eval/list_eval_partition.txt)
	image names for training, validation and testing set respectively. See EVALUATION PARTITIONS section below for more info.



=========================
IMAGE
=========================

------------ img.zip ------------

format: JPG

---------------------------------------------------

Notes:
1. Images are centered and resized to 256*256;
2. The aspect ratios of original images are kept unchanged.

---------------------------------------------------



=========================
BBOX LABELS
=========================

-------------- list_bbox_inshop.txt ---------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <clothes type> <pose type> <bbox location>

---------------------------------------------------

Notes:
1. The order of bbox labels accords with the order of entry names;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes;
3. In pose type, "1" represents frontal view, "2" represents side view, "3" represents back view, "4" represents zoom-out view, "5" represents zoom-in view, "6" represents stand-alone view;
4. In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "x_2" and "y_2" represent the lower right point coordinate of bounding box. Bounding box locations are listed in the order of [x_1, y_1, x_2, y_2].

---------------------------------------------------



=========================
LANDMARK LABELS
=========================

------------ list_landmarks_inshop.txt ------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <clothes type> <variation type> [<landmark visibility 1> <landmark location x_1> <landmark location y_1>, ... <landmark visibility 8> <landmark location x_8> <landmark location y_8>]

---------------------------------------------------

Notes:
1. The order of landmark labels accords with the order of entry names;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes. Upper-body clothes possess six fahsion landmarks, lower-body clothes possess four fashion landmarks, full-body clothes possess eight fashion landmarks;
3. In variation type, "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in;
4. In landmark visibility state, "0" represents visible, "1" represents invisible/occluded, "2" represents truncated/cut-off;
5. For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]; For lower-body clothes, landmark annotations are listed in the order of ["left waistline", "right waistline", "left hem", "right hem"]; For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].

---------------------------------------------------



=========================
ITEM LABELS
=========================

--------------- list_item_inshop.txt --------------

First Row: number of items

Rest of the Rows: <item id>

---------------------------------------------------

Notes:
1. Please refer to the paper "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations" for more details.

---------------------------------------------------



=========================
DESCRIPTION LABELS
=========================

----------- list_description_inshop.json ----------

Each Row: <item id> <item color> <item description>

---------------------------------------------------

Notes:
1. Please refer to the paper "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations" for more details.

---------------------------------------------------



=========================
EVALUATION PARTITIONS
=========================

------------- list_eval_partition.txt -------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <item id> <evaluation status>

---------------------------------------------------

Notes:
1. In evaluation status, "train" represents training image, "query" represents query image, "gallery" represents gallery image;
2. Items of clothes images are NOT overlapped within this dataset partition;
3. Please refer to the paper "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations" for more details.

---------------------------------------------------



=========================
Contact
=========================

Please contact Ziwei Liu (lz013@ie.cuhk.edu.hk) for questions about the dataset.