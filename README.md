## Kaggle - [TensorFlow - Help Protect the Great Barrier Reef](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)
### コンペ概要
海中動画から、サンゴをいっぱい食べてしまうヒトデを検出するコンペ

### Notebook
- yolov5-with-akaze.ipynb  
  - yolov5で検出を行った後、TrackingでAKAZEを使用した。
  - このNotebookで銅メダルを獲得。  
- yolov5-with-bytetrack.ipynb  
  - yolov5で検出を行った後、TrackingでByteTrackを使用した。  
  - [ByteTrack](https://github.com/ifzhang/ByteTrack)はコンペ期間時点でSOTAのTrackerであったが、今回のタスクでは機能しなかった。（ByteTrackの内部で走るKalman Filterが適切な予測を出せていなかった）
