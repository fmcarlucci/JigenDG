# JigenDG
Repository for the CVPR19 oral paper "Domain Generalization by Solving Jigsaw Puzzles"


## SETUP
Pytorch models will automatically download. You can download the caffemodel we used for AlexNet from here: https://www.filehosting.org/file/details/803193/alexnet_caffe.pth.tar
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

Run *run_PACS_photo.sh* to run the DG experiment on PACS, with photo as target.
