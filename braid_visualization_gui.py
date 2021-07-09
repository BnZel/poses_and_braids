from __init__ import *

class Braid(QWidget):
    def __init__(self):
        super(Braid, self).__init__()
        self.setGeometry(10,50, 1980, 1080)
        self.setWindowTitle('UI')
        self.initUI()

    def initUI(self):
        self.mainLayout = QGridLayout()
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # from set pose estimation connections
        self.connection = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

        self.lines = {}
        self.crossings = []
        self.save_keypoints = defaultdict(list)
        self.save_flatten_keypoints = defaultdict(list)
        self.k = 0
        self.kf = 0

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

        self.mainLayout.addWidget(self.leftDisplay,0,0)
        self.mainLayout.addWidget(self.leftItems,1,0)
        self.mainLayout.addWidget(self.rightDisplay,0,1)
        self.mainLayout.addWidget(self.rightItems,1,1)
        self.setLayout(self.mainLayout)

    def uploadFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Upload Image", "","PNG (*.PNG);; JPEG (*.JPEG) (*.JPG)", options=options)

        if Path(fileName).suffix == ".PNG".casefold() or (Path(fileName).suffix == ".JPEG".casefold() or Path(fileName).suffix == ".JPG".casefold()):
            img = common.read_imgfile(fileName)
            self.plotPose(img)
            self.btnUpload.setDisabled(True)

    def plotPose(self, filename):
        kp = self.mesh(filename)

        self.points = gl.GLScatterPlotItem(
            pos = kp,
            color = pg.glColor((0,255,0)),
            size = 15
        )
        self.windowLeft.addItem(self.points) 

        for n, pts in enumerate(self.connection):
            self.lines[n] = gl.GLLinePlotItem(
                pos = np.array([kp[p] for p in pts]),
                color = pg.glColor((0,0,255)),
                width = 3,
                antialias=True
            )
            self.windowLeft.addItem(self.lines[n])

        self.saveKeypoints(kp)

        int_key = np.array(kp,dtype=np.int)
        flatten_keypoints = np.concatenate(int_key).ravel().tolist() 
        print(flatten_keypoints)

        self.saveFlattenedKeypoints(flatten_keypoints)

        self.crossings = self.valid_crossings(flatten_keypoints)
        self.n = self.valid_number_of_strands(None)
        self.strand_positions = self.compute_strand_paths()
        self.draw()

        self.btnClearCrossings.setDisabled(False)

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

    def mesh(self, image):
        image_h, image_w = image.shape[:2]
        width = 640
        height = 480 
        pose_2d_mpiis = []
        visibilities = []
        humans = self.e.inference(image, resize_to_default=False, upsample_size=4.0)
        img = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append([(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii])
            visibilities.append(visibility)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        
        print("number of dimensions: ",pose_2d_mpiis.ndim)

        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)

        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        keypoints = pose_3d[0].transpose()

        print(keypoints)

        # large coords will extend outside frame
        # divide to display within frame
        return keypoints / 80     

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
    
    def valid_number_of_strands(self, n):
        largest_sigma = max([abs(i) for i in self.crossings])
        if n == None:
            n = largest_sigma + 1
        elif n < largest_sigma:
            raise ValueError('It is impossible to have {} strands with an S-{} crossing!'.format(n, largest_sigma))
        return n

    def valid_crossings(self, crossings):
        if not all(isinstance(i, int) for i in crossings):
            raise ValueError('All crossings must be integers!\n{}'.format(crossings))
        return crossings

    def draw(self, length = 8, line_width = 4.0, save_name='', show=True):
        print("drawing...")

        plt.figure(figsize=(length, length / 12 * len(self.strand_positions)))
        plt.axis('off')        

        self.colors = [cm.jet(i) for i in np.linspace(0.0, 1.0, len(self.strand_positions))]
        for x in range(len(self.crossings)):

            #separate between crossings and non-crossings
            if self.crossings[x] == 0:
                non_crossing_indices = list(range(self.n))
            else:
                crossing_indices = [abs(self.crossings[x])-1, abs(self.crossings[x])]
                non_crossing_indices = list(range(crossing_indices[0])) + list(range(crossing_indices[1]+1, self.n))
                over  = crossing_indices[0] if np.sign(self.crossings[x]) == -1 else crossing_indices[1]
                under = crossing_indices[0] if np.sign(self.crossings[x]) ==  1 else crossing_indices[1]
                self.over_data  = self.segment_drawing_data(x, over , -1 * np.sign(self.crossings[x]))
                self.under_data = self.segment_drawing_data(x, under,      np.sign(self.crossings[x]))
                strand_at_over  = self.pos_of_strand_at(over,  strand_paths=self.strand_positions, x=x)
                strand_at_under = self.pos_of_strand_at(under, strand_paths=self.strand_positions, x=x)

                self.ax.plot(*self.over_data, color=self.colors[strand_at_over] , linewidth=line_width)
                self.ax.plot(*self.under_data, color=self.colors[strand_at_under] , linewidth=line_width)
                self.canvas.draw()

            for i in non_crossing_indices:
                strand_at_i = self.pos_of_strand_at(i, strand_paths=self.strand_positions, x=x)
                self.data = self.segment_drawing_data(x, i, 0)

                self.ax.plot(*self.data, color=self.colors[strand_at_i], linewidth=line_width)
                self.canvas.draw()
        
        print("done")

    #given an x (crossing) and y (height on plot), find the index of the strand in this position
    def pos_of_strand_at(self, y, strand_paths, x=-1): #x defaults to most recent strand position
        if not y in range(self.n):
            raise ValueError('It is impossible to have a strand at position {}, when there are only {} positions.'.format(y,self.n))
        #now we don't need to check if there is a strand at the desired position, because there are n strands in n different positions
        #this means all positions must have a strand
        strand_positions_at_x = [i[x] for i in strand_paths]
        return strand_positions_at_x.index(y)

    def compute_strand_paths(self):
        strand_paths = [[i] for i in range(self.n)] #each list contains coordinates of each strand at each time step
        for c in self.crossings:
            #add coordinates for new timestep
            #assume no changes are made and add previous coordinates as the next ones
            for s in strand_paths:
                s.append(s[-1])

            #if there is a crossing, then update the position of the two strands that crossed
            if c != 0:
                lower_strand = self.pos_of_strand_at(abs(c)-1, strand_paths=strand_paths)
                upper_strand = self.pos_of_strand_at(abs(c)  , strand_paths=strand_paths)
                strand_paths[lower_strand][-1] += 1
                strand_paths[upper_strand][-1] -= 1

        return strand_paths

    #x_start should be an integer corresponding to a crossing
    #y_start should be an integer corresponding to the height of a strand at position x
    #direction should be -1, 0, or 1, indicating the end height of y_data
    def segment_drawing_data(self, x_start, y_start, direction, n = 100):
        x_data = np.linspace(x_start, x_start+1, num = n)
        cos_range = np.linspace(0, np.pi, num = n)
        y_data = list(map(lambda x: direction * (1 - np.cos(x)) / 2 + y_start, cos_range))
        return [x_data, y_data]

#this is outside of the class on purpose!
def mult_braids(braids):
    if len(braids) == 0:
        return None
    if any([i.n != braids[0].n for i in braids]):
        raise ValueError('Braids have differing numbers of strands! {}'.format([i.n for i in braids]))
    return Braid(sum([i.crossings for i in braids], []), n=braids[0].n)

if __name__ == "__main__":
    app = QApplication([])
    win = Braid()
    win.show()
    sys.exit(app.exec_())