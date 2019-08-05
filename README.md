![JIGEN](https://github.com/fmcarlucci/JigenDG/blob/master/jigsaw2-1.png)

# JigenDG
Repository for the CVPR19 oral paper "[Domain Generalization by Solving Jigsaw Puzzles](https://arxiv.org/pdf/1903.06864.pdf)".
To cite, please use:
```
@inproceedings{carlucci2019domain,
  title={Domain Generalization by Solving Jigsaw Puzzles},
  author={Carlucci, Fabio Maria and D'Innocente, Antonio and Bucci, Silvia and Caputo, Barbara and Tommasi, Tatiana},
  booktitle={CVPR},
  year={2019}
}
```


## SETUP
Pytorch models will automatically download if needed. You can download the caffemodel we used for AlexNet from here https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing

Once downloaded, move it into models/pretrained/alexnet_caffe.pth.tar

Once you have download the data for the different experiments, you must update the files in data/txt_list to match the actual location of your files.
For example, if you saved your data into /home/user/data/images/ you have to change these lines:
```
/home/fmc/data/PACS/kfold/art_painting/dog/pic_001.jpg 0
/home/fmc/data/PACS/kfold/art_painting/dog/pic_002.jpg 0
/home/fmc/data/PACS/kfold/art_painting/dog/pic_003.jpg 0
```
into:

```
/home/user/data/images/PACS/kfold/art_painting/dog/pic_001.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_002.jpg 0
/home/user/data/images/PACS/kfold/art_painting/dog/pic_003.jpg 0
```

A quick way is to use sed:
`for i in *.txt; do sed -i "s@/home/fmc/data/@/home/user/data/images/@g" $i; done`


## Running experiments

Run *run_PACS_photo.sh* to run the DG experiment on PACS, with photo as target (using AlexNet).

**Note** that when using ResNet you should set the image_size to **222**
An example on how to get ResNet18 results on PACS, art_painting as target:
```
python train_jigsaw.py --batch_size 128 --n_classes 7 --learning_rate 0.001 --network resnet18 --val_size 0.1 --folder_name test --jigsaw_n_classes 30 --train_all True --TTA False --nesterov False --min_scale 0.8 --max_scale 1.0 --random_horiz_flip 0.5 --jitter 0.4 --tile_random_grayscale 0.1 --source photo cartoon sketch --target art_painting --jig_weight 0.7 --bias_whole_image 0.9 --image_size 222
```


### VLCS
Some people are having issues accessing the VLCS dataset; I have uploaded a copy here: http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file
