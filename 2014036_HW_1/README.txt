There is K_CLUS and stt( specifying file name to be scanned ) in main.py which you can change for different cases and see the answer on console.
On each run you get nmi, ami, ri, and ari averaged over 10 independent trials of k-means. There would also be three plots having ground truth plot having no of clusters specified ground truth, k means plot made by us according to K_CLUS and objective function vs iteration for our k-means function run when k = ground truth as specified in the question.

SO u have to set K_CLUS according to the no of clusters u want and stt with the file name u want to perform k means on. U will be asked this on console.

After setting these just run on the console and u will get result of nmi, ami, ri, and ari of 10 independent runs of k means.

Seeds Dataset:
stt = seeds_dataset.txt
Iris Dataset:
stt = iris_data.txt
Segmentation Dataset:
stt = segmentation.txt
Vertebral Dataset:
stt = column_3C.dat

K_CLUS can be of ur choice.


For bonus too same procedure follows.

main.py is for first part of assignment.
gmm_bonus.py is for bonus.