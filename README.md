<p align="center">
<h1 align="center"><strong> OpenObj: Open-Vocabulary Object-Level Neural Radiance Fields with Fine-Grained Understanding</strong></h1>
</p>



<p align="center">
  <a href="https://openobj.github.io/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-👔-green?">
  </a> 
  
  <a href="https://arxiv.org/pdf/2406.08009412" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-📖-blue?">
  </a> 
  
  <a href="https://youtu.be/BeUdxrjItDE" target='_blank'>
    <img src="https://img.shields.io/badge/Video-📹-red?">
  </a> 
</p>


 ## 🏠  Abstract
In recent years, there has been a surge of interest in open-vocabulary 3D scene reconstruction facilitated by visual language models (VLMs), which showcase remarkable capabilities in open-set retrieval. However, existing methods face some limitations: they either focus on learning point-wise features, resulting in blurry semantic understanding, or solely tackle object-level reconstruction, thereby overlooking the intricate details of the object's interior. To address these challenges, we introduce OpenObj, an innovative approach to build open-vocabulary object-level Neural Radiance Fields (NeRF) with fine-grained understanding. In essence, OpenObj establishes a robust framework for efficient and watertight scene modeling and comprehension at the object-level. Moreover, we incorporate part-level features into the neural fields, enabling a nuanced representation of object interiors. This approach captures object-level instances while maintaining a fine-grained understanding. The results on multiple datasets demonstrate that OpenObj achieves superior performance in zero-shot semantic segmentation and retrieval tasks. Additionally, OpenObj supports real-world robotics tasks at multiple scales, including global movement and local manipulation. connectivity to construct a hierarchical graph. Validation results from public dataset SemanticKITTI demonstrate that, OpenGraph achieves the highest segmentation and query accuracy.
 
<img src="https://github.com/BIT-DYN/OpenObj/blob/main/poster.jpg">


## 🛠  Install

### Install the required libraries
Use conda to install the required environment. To avoid problems, it is recommended to follow the instructions below to set up the environment.


```bash
conda env create -f environment.yml
```

###  Install CropFormer Model
Follow the [instructions](https://github.com/qqlu/Entity/blob/main/Entityv2/README.md) to install the CropFormer model and download the pretrained weights [CropFormer_hornet_3x](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x).

###  Install TAP Model
Follow the [instructions](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#installation) to install the TAP model and download the pretrained weights [here](https://github.com/baaivision/tokenize-anything?tab=readme-ov-file#models).


###  Install SBERT ModelD
```bash
pip install -U sentence-transformers
```
Download pretrained weights
```bash
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
```


### Clone this repo

```bash
git clone https://github.com/BIT-DYN/OpenObj
cd OpenObj
```


## 📊 Prepare dataset
OpenGraph has completed validation on Replica (as same with [vMap](https://github.com/kxhit/vMAP)) and Scannet. 
Please download the following datasets.

* [Replica Demo](https://huggingface.co/datasets/kxic/vMAP/resolve/main/demo_replica_room_0.zip) - Replica Room 0 only for faster experimentation.
* [Replica](https://huggingface.co/datasets/kxic/vMAP/resolve/main/vmap.zip) - All Pre-generated Replica sequences.
* [ScanNet](https://github.com/ScanNet/ScanNet) - Official ScanNet sequences.


## 🏃 Run

### Object Segmentation and Understanding
Run the following command to identifie and comprehend object instances from color images.
```bash
cd maskclustering
python3 mask_gen.py  --input /data/dyn/object/vmap/room_0/imap/00/rgb/*.png --input_depth /data/dyn/object/vmap/room_0/imap/00/depth/*.png --output results/room_0/mask/ --opts MODEL.WEIGHTS CropFormer_hornet_3x_03823a.pth 
```
You can see a visualization of the results in the ```results/vis``` folder.

### Mask Clustering
Run the following command to ensure consistent object association across frames.
```bash
python3 mask_graph.py --config_file ./configs/room_0.yaml --input_mask results/room_0/mask/mask_init_all.pkl --input_depth /data/dyn/object/vmap/room_0/imap/00/depth/*.png --input_pose  /data/dyn/object/vmap/room_0/imap/00/traj_w_c.txt --output_graph results/room_0/mask/graph/ --input_rgb /data/dyn/object/vmap/room_0/imap/00/rgb/*.png --output_dir /data/dyn/object/vmap/room_0/imap/00/ --input_semantic /data/dyn/object/vmap/room_0/imap/00/semantic_class/*.png 
```
You can see a visualization of the results in the ```results/graph``` folder.
And this will generate some folders (```class_our/```  ```instance_our/```) and documents (```object_clipfeat.pkl``` ```object_capfeat.pkl``` ```object_caption.pkl```) in the data directory, which are necessary for the follow-up process. 


### Part-level Fine-Grained Feature Extraction
Run the following command to distinguish parts and extracts their visual features.
```bash
cd ../partlevel
python sam_clip_dir.py --input_image /data/dyn/object/vmap/room_0/imap/00/rgb/*.png --output_dir /data/dyn/object/vmap/room_0/imap/00/partlevel --down_sample 5
```
This will generate a folder (```partlevel/```) in the data directory, which is necessary for the follow-up process. 


### NeRF Rendering and Training
Run the following command to vectorize the training of NeRFs for all objects.
```bash
cd ../nerf
python train.py --config ./configs/Replica/room_0.json --logdir results/room_0
```
This will generate a folder (```ckpt/```) in the result directory containing the network parameters for all objects.

###  visulization
Run the following command to generate the vis documents.
```bash
cd ../nerf
python gen_map_vis.py --scene_name room_0 --dataset_name Replica
```
Interactions can be made using our visualization files.
```bash
cd ../nerf
python vis_interaction.py --scene_name room_0 --dataset_name Replica --is_partcolor
```


Then in the open3d visualizer window, you can use the following key callbacks to change the visualization.

Press ```C``` to toggle the ceiling.

Press ```S``` to color the meshes by the object class. 

Press ```R``` to color the meshes by RGB.

Press ```I``` to color the meshes by object instance ID.

Press ```O``` to color the meshes by part-level feature.

Press ```F``` and type object text and num in the terminal, and the meshes will be colored by the similarity.

Press ```P``` and type object text and num and part text in the terminal, and the meshes will be colored by the similarity.


## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@article{openobj,
  title={OpenObj: Open-Vocabulary Object-Level Neural Radiance Fields with Fine-Grained Understanding},
  author={Deng, Yinan and Wang, Jiahui and Zhao, Jingyu and Dou, Jianyu and Yang, Yi and Yue, Yufeng},
  journal={arXiv preprint arXiv:2406.08009},
  year={2024}
}
```

## 👏 Acknowledgements
We would like to express our gratitude to the open-source projects and their contributors [vMap](https://github.com/kxhit/vMAP). 
Their valuable work has greatly contributed to the development of our codebase.
