# Prototype of Pose Estimation and Braid Groups

## Overview 
While working with Pose Estimation and learning Braid Groups on the side. I've thought about the correlation between Computer Vision (Pose Estimation) and Braid Groups. With the notes I've taken, I thought I experiment with it. 

# Resources 
* [A journey through the mathematical theory of braids by Ester Dalvit](http://matematita.science.unitn.it/braids/)
* Code credited by: [Braid Group Visualization by Denbox](https://github.com/Denbox/Braid-Group-Visualization) 
* [Pose Estimation Tutorial by Mark Jay](https://www.youtube.com/watch?v=nUjGLjOmF7o&list=PLX-LrBk6h3wQ17z1axCOAS1QVS1dvTEvR)
* [My Pose Estimation GUI Project](https://github.com/BnZel/pose_estimation_gui)
* PyQT5 and [PyQTGraph](https://www.pyqtgraph.org)

## Project In Detail
Embedding MatPlotLib's graph onto the UI

Using dictionaries to save keypoints (flatten keypoints will be explained later)
```python
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.save_keypoints = defaultdict(list)
        self.save_flatten_keypoints = defaultdict(list)
```
## Layout
```python
 # ----------left layout----------
        self.leftDisplay = QWidget()
        self.windowLeft = gl.GLViewWidget()
        self.windowLeft.setCameraPosition(distance=47, elevation=12)

        g_x = gl.GLGridItem()
        g_y = gl.GLGridItem()
        g_z = gl.GLGridItem()
        g_x.rotate(90,0,1,0)
        g_y.rotate(90,1,0,0)
        g_x.translate(-10,0,0)
        g_y.translate(0,-10,0)
        g_z.translate(0,0,-10)

        self.windowLeft.addItem(g_x)
        self.windowLeft.addItem(g_y)
        self.windowLeft.addItem(g_z)

        self.layoutLeft = QVBoxLayout(self)
        self.layoutLeft.addWidget(self.windowLeft)

        self.leftDisplay.setLayout(self.layoutLeft)
        self.leftItems = QWidget(self)
        self.layoutLeftItems = QGridLayout(self)

        # widgets 
        self.btnUpload = QPushButton("Upload Image",self)
        self.btnUpload.clicked.connect(self.uploadFile)  
        self.btnUpload.setDisabled(False)

        self.btnClearCrossings = QPushButton("Clear crossings",self)
        self.btnClearCrossings.clicked.connect(self.clearCrossings)
        self.btnClearCrossings.setDisabled(True)

        # keypoints list
        self.lblKeypoints = QLabel(self)
        self.lblKeypoints.setFont(QFont('Bold',15))
        self.lblKeypoints.setText("Keypoints")
        self.listKeypoints = QListWidget(self)
        self.listKeypoints.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

        # add widgets
        self.layoutLeftItems.addWidget(self.btnUpload,1,0)
        self.layoutLeftItems.addWidget(self.btnClearCrossings,1,1)
        self.layoutLeftItems.addWidget(self.lblKeypoints,2,0)
        self.layoutLeftItems.addWidget(self.listKeypoints,3,0)

        self.layoutLeftItems.setRowStretch(4,1)
        self.leftItems.setLayout(self.layoutLeftItems)
```
![Left Layout](https://github.com/BnZel/poses_and_braids/blob/master/demo/left-layout.PNG)

```python
        # ----------right layout----------
        self.rightLayout = QVBoxLayout(self)
        self.rightDisplay = QWidget(self)

        # flatten keypoints list 
        self.lblFlattenKeypoints = QLabel(self)
        self.lblFlattenKeypoints.setFont(QFont('Bold',15))
        self.lblFlattenKeypoints.setText("Flattened Keypoints")
        self.listFlattenKeypoints = QListWidget(self)
        self.listFlattenKeypoints.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)

        self.rightItems = QWidget(self)
        self.layoutRightItems = QGridLayout(self)
        self.layoutRightItems.addWidget(self.lblFlattenKeypoints,2,0)
        self.layoutRightItems.addWidget(self.listFlattenKeypoints,3,0)
        self.rightItems.setLayout(self.layoutRightItems)

        self.rightLayout.addWidget(self.canvas)
        self.rightLayout.addWidget(self.rightItems)
        self.rightDisplay.setLayout(self.rightLayout)

        self.layoutRightItems.setRowStretch(4,1)

        model = 'mobilenet_thin'
        self.w, self.h = model_wh('432x368')
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w,self.h))
        self.poseLifting = Prob3dPose('./lifting/models/prob_model_params.mat')
```
![Right Layout](https://github.com/BnZel/poses_and_braids/blob/master/demo/right-layout.PNG)

## Utilizing The Braids Data
*Note: Given the difficulty I had to fully understand the code, I encourage the reader to view the source code and read the comments*

## From Pose Estimation Keypoints to Braids
In theory, I have noted that a set of matrices either from the pose or braids could be translated bi-directionally. However with the limited knowledge I have in this subject, in a practical sense...:
```python
        # in plotPose() function
        # as the visualization only accepts integers (labeled as sigmas) as integers
        # I decided to take the keypoints and converted it
        # as well as turning it into a single list 
        # from there I use the avaliable methods written by Denbox
        # NOTE: length of connections are [0:17] in relation the fixed length of the represented sigma's
        int_key = np.array(kp,dtype=np.int)
        flatten_keypoints = np.concatenate(int_key).ravel().tolist() 

        self.crossings = self.valid_crossings(flatten_keypoints)
        self.n = self.valid_number_of_strands(None)
        self.strand_positions = self.compute_strand_paths()
        self.draw()
``` 

## Displaying Pose and Braid Points
```python
    def saveKeypoints(self, kp):
        print("saving keypoints...")

        self.save_keypoints.update({self.k:kp})
        
        # update to list
        for key, values in self.save_keypoints.items():
                self.listKeypoints.clear()
                print(key, values)
                self.listKeypoints.addItem(str(values))

        self.k += 1

    def saveFlattenedKeypoints(self, kp):
        print("saving flattened keypoints...")

        self.save_flatten_keypoints.update({self.k:kp})

        for key, values in self.save_flatten_keypoints.items():
            self.listFlattenKeypoints.clear()
            print(key, values)
            self.listFlattenKeypoints.addItem(str(values))
        
        self.kf += 1
```
![Pose To Braid Data](https://github.com/BnZel/poses_and_braids/blob/master/demo/pose-to-braid-data.PNG)

## Removing All Data
```python
    # this function will look like the final layout image
  
    def clearCrossings(self):
        print("clear crossings and pose")

        self.ax.clear()
        self.canvas.draw()
        self.listKeypoints.clear()
        self.listFlattenKeypoints.clear()

        pts = self.windowLeft.removeItem(self.points)
        lns = self.lines
        for i in list(self.lines):
            lns = self.windowLeft.removeItem(self.lines[i])

        return pts, lns, self.btnClearCrossings.setDisabled(True), self.btnUpload.setDisabled(False)
```        

## Final Layout
![Final Layout](https://github.com/BnZel/poses_and_braids/blob/master/demo/ui.PNG)

## Demo
*Note: Due to the speed of the program, images will provide the detail*
![Demo 1](https://github.com/BnZel/poses_and_braids/blob/master/demo/demo1.PNG)
![Demo 2](https://github.com/BnZel/poses_and_braids/blob/master/demo/demo2.png)
![Demo 3](https://github.com/BnZel/poses_and_braids/blob/master/demo/demo3.PNG)

## Conclusion
* With the latest progress, perhaps if anyone comes across this repo. It would be appreciated to continue adding and fixing this project
* I realize that in theory, it will be difficult to put over practially due to certain limitations whether it may be knowledge, hardware, or software.
* Overall a rather interesting concept that I've been thinking about
