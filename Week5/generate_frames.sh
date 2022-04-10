#Sequence 01
for i in `seq -f "%03g" 1 5`;
do
  mkdir "dataset/train/S01/c$i/img"
  ffmpeg -ss 0 -i "dataset/train/S01/c$i/vdo.avi" "dataset/train/S01/c$i/img/%04d.jpeg"
done

#Sequence 03
for i in `seq -f "%03g" 10 15`;
do
  mkdir "dataset/train/S03/c$i/img"
  ffmpeg -ss 0 -i "dataset/train/S03/c$i/vdo.avi" "dataset/train/S03/c$i/img/%04d.jpeg"
done

#Sequence 04
for i in `seq -f "%03g" 16 40`;
do
  rm -R -f "dataset/train/S04/c$i/img"
  mkdir "dataset/train/S04/c$i/img"
  ffmpeg -ss 0 -i "dataset/train/S04/c$i/vdo.avi" "dataset/train/S04/c$i/img/%04d.jpeg"
done
