using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

using System.Threading;
using OpenCVForUnity;
using PositionsVector = System.Collections.Generic.List<OpenCVForUnity.Rect>;

namespace OpenCVForUnityExample
{
    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class CameraInputTest : MonoBehaviour
    {
        //public Image m_srcImage;
        string haarcascade_frontalface_default_xml_filepath;
        string haarcascade_eye_xml_filepath;
        string haarcascade_mcs_mouth_xml_filepath;
        string haarcascade_mcs_nose_xml_filepath;
        string lbpcascade_frontalface_xml_filepath;

        OpenCVForUnity.Rect[] face_rects;
        OpenCVForUnity.Rect[] originalEye_rects;
        OpenCVForUnity.Rect[] originalNose_rects;
        OpenCVForUnity.Rect[] originalMouths_rects;
        OpenCVForUnity.Rect[] filterEye_rects = new OpenCVForUnity.Rect[2];
        OpenCVForUnity.Rect filterNose_rects = new OpenCVForUnity.Rect();
        OpenCVForUnity.Rect filterMouths_rects = new OpenCVForUnity.Rect();

        OpenCVForUnity.Rect[] posEye_rects = new OpenCVForUnity.Rect[2];
        OpenCVForUnity.Rect posNose_rects = new OpenCVForUnity.Rect();
        OpenCVForUnity.Rect posMouths_rects = new OpenCVForUnity.Rect();
        OpenCVForUnity.Rect posForehead_rects = new OpenCVForUnity.Rect();
        OpenCVForUnity.Rect[] posCheek_rects = new OpenCVForUnity.Rect[2];
        //OpenCVForUnity.Rect posChin_rects = new OpenCVForUnity.Rect();

        const int eyeImgSize_width = 400;
        const int eyeImgSize_height = 400;
        const int noseImgSize_width = 500;
        const int noseImgSize_height = 300;
        const int mouseImgSize_width = 500;
        const int mouseImgSize_height = 400;
        const int foreheadImgSize_width = 500;

        WebCamTextureToMatHelper webCamTextureToMatHelper;
        InnerParameters innerParameters;
        Parameters parameters;
        List<TrackedObject> trackedObjects = new List<TrackedObject>();
        List<float> weightsPositionsSmoothing = new List<float>();
        List<float> weightsSizesSmoothing = new List<float>();

        private string _SavePath;
        private string _SaveLatestPath;
        private int _CaptureCounter = 0;
        private int _SignalID = 0;

        Mat grayMat;
        Mat grayMat4Thread;
        bool shouldDetectInMultiThread = false;
        bool didUpdateTheDetectionResult = false;
        OpenCVForUnity.Rect[] rectsWhereRegions;
        MatOfRect detectionResult;
        CascadeClassifier cascade;
        CascadeClassifier cascade4Thread;
        Texture2D texture;
        List<OpenCVForUnity.Rect> resultObjects = new List<OpenCVForUnity.Rect>();
        List<OpenCVForUnity.Rect> detectedObjectsInRegions = new List<OpenCVForUnity.Rect>();
        bool isThreadRunning = false;
        bool shouldStopThread = false;

        FpsMonitor fpsMonitor;

        private enum TrackedState : int
        {
            NEW_RECTANGLE = -1,
            INTERSECTED_RECTANGLE = -2
        }

        private enum CameraState: int
        {
            WAITFORFACE = 0,
            WAITFORSECONDS,
            FINDEYENOSTMOUTH,
            TIMELAPSESNAPSHOT,
            ENDSHOT
        }

        CameraState camerastate;
        int curSnapShotIndex = 0;
        int totalSnapShot = 5;
        float curCountDown = 0f;
        float totalCountDown = 1.0f; //seconds

        float curEndingCountDown = 0f;
        float totalEndingCountDown = 1f; //seconds

        float curWaitingCountDown = 0f;
        float totalWaitingCountDown = 2f; //seconds

        public GameObject SnapshotUIObj;
        public GameObject EndUIObj;
        public GameObject FlashUIObj;
        public GameObject CanvasObj;
        public GameObject SnapshotUIProgressObj;
        public GameObject TipsUIObj;
        public GameObject TipsStartPicturingUIObj;

        List<String> snapShotsName = new List<String>();
        List<String> snapShotsRename = new List<String>();

        // Start is called before the first frame update
        void Start()
        {
            Screen.orientation = ScreenOrientation.Portrait;

            fpsMonitor = GetComponent<FpsMonitor>();
            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();

            // Cascade path 训练集路径
            haarcascade_frontalface_default_xml_filepath = Utils.getFilePath("haarcascade_frontalface_alt.xml");
            haarcascade_eye_xml_filepath = Utils.getFilePath("haarcascade_eye.xml");
            haarcascade_mcs_nose_xml_filepath = Utils.getFilePath("Nariz.xml");
            haarcascade_mcs_mouth_xml_filepath = Utils.getFilePath("Mouth.xml");
            lbpcascade_frontalface_xml_filepath = Utils.getFilePath("lbpcascade_frontalface.xml");

            String tempdir = Path.GetFullPath(".");
            _SavePath = Path.Combine(tempdir, "_ExportSnapshot\\");
            _SaveLatestPath = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop) + "\\Pictures\\";//Path.Combine(tempdir, "_ExportSnapshot\\latest\\");

            Run();

            camerastate = CameraState.WAITFORFACE;

            EndUIObj.SetActive(false);
            SnapshotUIObj.SetActive(false);
            SnapshotUIProgressObj.SetActive(false);
            TipsUIObj.SetActive(true);
            TipsStartPicturingUIObj.SetActive(false);

            //set 0 in the start
            _SignalID = 0;
            File.WriteAllText(_SaveLatestPath + "signal.txt", _SignalID.ToString());
            // Load image 读取原图
            //Mat srcMat = Imgcodecs.imread(Utils.getFilePath("0_2k.jpeg"), 1);
            //Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_BGR2RGB);
        }

        void Update()
        {
            if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
            {
                switch(camerastate)
                {
                    case CameraState.WAITFORFACE:
                        {
                            UpdateDetected(true);
                        }
                        break;
                    case CameraState.WAITFORSECONDS:
                        {
                            UpdateDetected(false);
                            DrawOrgansPose();
                            TipsStartPicturingUIObj.GetComponent<UnityEngine.UI.Text>().text = "Say Cheeze~ : " + curWaitingCountDown.ToString("f2");
                            if (curWaitingCountDown <= 0)
                            {
                                TipsStartPicturingUIObj.SetActive(false);
                                camerastate = CameraState.FINDEYENOSTMOUTH;
                                curWaitingCountDown = totalWaitingCountDown;
                            }
                            curWaitingCountDown -= Time.deltaTime;
                        }
                        break;
                    case CameraState.FINDEYENOSTMOUTH:
                        {
                            UpdateDetected(false);
                            DrawOrgansPose();

                            curCountDown = totalCountDown;
                            curSnapShotIndex = 0;

                            SnapshotUIObj.SetActive(true);
                            SnapshotUIProgressObj.SetActive(true);
                            SnapshotUIObj.GetComponent<Image>().fillAmount = 0;
                            camerastate = CameraState.TIMELAPSESNAPSHOT;
                        }
                        break;
                    case CameraState.TIMELAPSESNAPSHOT:
                        {
                            UpdateDetected(false);
                            DrawOrgansPose();

                            SnapshotUIObj.GetComponent<Image>().fillAmount = (curSnapShotIndex * totalCountDown + curCountDown) / (totalSnapShot * totalCountDown);

                            if (curCountDown >= totalCountDown)
                            {
                                OnTakeSnapshotButtonClick();
                                StartFlashUI();
                                curCountDown = 0;
                                curSnapShotIndex++;

                                //Debug.Log(" SnapshotUIObj.GetComponent<Image>().fillAmount " + SnapshotUIObj.GetComponent<Image>().fillAmount);

                            }
                            curCountDown += Time.deltaTime;

                            if(curSnapShotIndex >= totalSnapShot)
                            {
                                camerastate = CameraState.ENDSHOT;
                                SnapshotUIObj.SetActive(false);
                                SnapshotUIProgressObj.SetActive(false);

                                EndUIObj.SetActive(true);

                                curEndingCountDown = totalEndingCountDown;
                                _CaptureCounter = 0;
                                //write down ID to signal file
                                _SignalID++;
                                Debug.Log("Write Signal!!!!!!!!!!!!!!!!!!!!!!!!! " + _SignalID);
                                File.WriteAllText(_SaveLatestPath + "signal.txt", _SignalID.ToString());
                                //string text = System.IO.File.ReadAllText(_SaveLatestPath + "signal.txt");
                                //int value = Convert.ToInt32(text);
                            }
                        }
                        break;
                    case CameraState.ENDSHOT:
                        {
                            UpdateDetected(false);

                            if(curEndingCountDown < 0)
                            {
                                camerastate = CameraState.WAITFORFACE;
                                EndUIObj.SetActive(false);
                                TipsUIObj.SetActive(true);
                                TipsStartPicturingUIObj.SetActive(false);
                            }
                            curEndingCountDown -= Time.deltaTime;
                        }
                        break;
                }
            }
        }

        private void UpdateDetected(bool detectFiveSense)
        {

            Mat rgbaMat = webCamTextureToMatHelper.GetMat();

            Imgproc.cvtColor(rgbaMat, grayMat, Imgproc.COLOR_RGBA2GRAY);
            //Imgproc.equalizeHist(grayMat, grayMat);

            if (!shouldDetectInMultiThread)
            {
                grayMat.copyTo(grayMat4Thread);

                shouldDetectInMultiThread = true;
            }

            OpenCVForUnity.Rect[] rects;

            if (didUpdateTheDetectionResult)
            {
                didUpdateTheDetectionResult = false;

                //Debug.Log("DetectionBasedTracker::process: get _rectsWhereRegions were got from resultDetect");
                rectsWhereRegions = detectionResult.toArray();

                rects = rectsWhereRegions;
                for (int i = 0; i < rects.Length; i++)
                {
                    Imgproc.rectangle(rgbaMat, new Point(rects[i].x, rects[i].y), new Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), new Scalar(0, 0, 255, 255), 2);
                }
            }
            else
            {
                //Debug.Log("DetectionBasedTracker::process: get _rectsWhereRegions from previous positions");
                rectsWhereRegions = new OpenCVForUnity.Rect[trackedObjects.Count];

                for (int i = 0; i < trackedObjects.Count; i++)
                {
                    int n = trackedObjects[i].lastPositions.Count;
                    //if (n > 0) UnityEngine.Debug.LogError("n > 0 is false");

                    OpenCVForUnity.Rect r = trackedObjects[i].lastPositions[n - 1].clone();
                    if (r.area() == 0)
                    {
                        Debug.Log("DetectionBasedTracker::process: ERROR: ATTENTION: strange algorithm's behavior: trackedObjects[i].rect() is empty");
                        continue;
                    }

                    //correction by speed of rectangle
                    if (n > 1)
                    {
                        Point center = CenterRect(r);
                        Point center_prev = CenterRect(trackedObjects[i].lastPositions[n - 2]);
                        Point shift = new Point((center.x - center_prev.x) * innerParameters.coeffObjectSpeedUsingInPrediction,
                                          (center.y - center_prev.y) * innerParameters.coeffObjectSpeedUsingInPrediction);

                        r.x += (int)Math.Round(shift.x);
                        r.y += (int)Math.Round(shift.y);
                    }
                    rectsWhereRegions[i] = r;
                }

                rects = rectsWhereRegions;
                for (int i = 0; i < rects.Length; i++)
                {
                    Imgproc.rectangle(rgbaMat, new Point(rects[i].x, rects[i].y), new Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), new Scalar(0, 255, 0, 255), 2);
                }
            }

            detectedObjectsInRegions.Clear();
            if (rectsWhereRegions.Length > 0)
            {

                int len = rectsWhereRegions.Length;
                for (int i = 0; i < len; i++)
                {
                    DetectInRegion(grayMat, rectsWhereRegions[i], detectedObjectsInRegions);
                }
            }

            UpdateTrackedObjects(detectedObjectsInRegions);
            GetObjects(resultObjects);

            face_rects = resultObjects.ToArray();
            for (int i = 0; i < face_rects.Length; i++)
            {
                //Debug.Log("detect faces " + face_rects[i]);
                Imgproc.rectangle(rgbaMat, new Point(face_rects[i].x, face_rects[i].y), new Point(face_rects[i].x + face_rects[i].width, face_rects[i].y + face_rects[i].height), new Scalar(255, 0, 0, 255), 2);

                Mat roi_gray_img = new Mat(grayMat, new OpenCVForUnity.Rect(0, 0, face_rects[i].x + face_rects[i].width, face_rects[i].y + face_rects[i].height));
                Mat roi_img = new Mat(rgbaMat, new OpenCVForUnity.Rect(0, 0, face_rects[i].x + face_rects[i].width, face_rects[i].y + face_rects[i].height));

                //when the face closes to the center detect five sense organs
                int face_centerX = (int)(face_rects[i].x + face_rects[i].width * 0.5);
                int face_centerY = (int)(face_rects[i].y + face_rects[i].height * 0.5);
                int screen_centerX = (int)(webCamTextureToMatHelper.requestedWidth * 0.5);
                int screen_centerY = (int)(webCamTextureToMatHelper.requestedHeight * 0.5);

                //Debug.Log(" face_rects[i].y + face_rects[i].height " + (face_rects[i].y + face_rects[i].height));

                if (detectFiveSense 
                    //&& Mathf.Abs(face_centerX - screen_centerX) < webCamTextureToMatHelper.requestedWidth * 0.05
                    //&& Mathf.Abs(face_centerY - screen_centerY) < webCamTextureToMatHelper.requestedHeight * 0.05
                    && face_rects[i].width > webCamTextureToMatHelper.requestedWidth * 0.8//0.88
                    && face_rects[i].y + face_rects[i].height < webCamTextureToMatHelper.requestedHeight * 0.8/*0.86*/)
                {
                    //detect eye, nose month
                    if (!FindEyes(roi_gray_img, roi_img, i))
                    {
                        Debug.Log("No eyes has been found!");
                    }

                    FindNoses(roi_gray_img, roi_img, i);
                    FindMouth(roi_gray_img, roi_img, i);

                    if(isFindSense())
                    {
                        SetPosRects();
                        camerastate = CameraState.WAITFORSECONDS;
                        curWaitingCountDown = totalWaitingCountDown;
                        TipsUIObj.SetActive(false);
                        TipsStartPicturingUIObj.SetActive(true);
                    }
                }

            }
            Utils.fastMatToTexture2D(rgbaMat, texture);
        }

        bool isFindSense()
        {
            if (filterEye_rects[0] == null
                        || filterEye_rects[1] == null
                        || filterNose_rects == null
                        || filterMouths_rects == null
                        || filterNose_rects.y < filterEye_rects[0].y + filterEye_rects[0].height * 0.5
                        || filterMouths_rects.y < filterNose_rects.y + filterNose_rects.height * 0.5)

                return false;

            float con = (float)(filterNose_rects.x + filterNose_rects.width * 0.5); // center of nose

            if ((filterEye_rects[0].x < filterEye_rects[1].x
                && (con < filterEye_rects[0].x + filterEye_rects[0].width || con > filterEye_rects[1].x))
                || (filterEye_rects[0].x > filterEye_rects[1].x && (con < filterEye_rects[1].x + filterEye_rects[1].width || con > filterEye_rects[0].x)))
                return false;

            float com = (float)(filterMouths_rects.x + filterMouths_rects.width * 0.5); // center of mouth

            if ((filterEye_rects[0].x < filterEye_rects[1].x
                && (com < filterEye_rects[0].x + filterEye_rects[0].width || com > filterEye_rects[1].x))
                || (filterEye_rects[0].x > filterEye_rects[1].x && (com < filterEye_rects[1].x + filterEye_rects[1].width || com > filterEye_rects[0].x)))
                return false;

            return true;
        }

        void SetPosRects()
        {

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();
            int s_w = webCamTextureMat.width();
            int s_h = webCamTextureMat.height();

            for (int k = 0; k < filterEye_rects.Length; k++)
            {
                posEye_rects[k] = filterEye_rects[k];

                if(posEye_rects[k].width < eyeImgSize_width)
                {
                    posEye_rects[k].x -= (int)((eyeImgSize_width - posEye_rects[k].width) * 0.5); 
                    posEye_rects[k].width = eyeImgSize_width;
                }

                if (posEye_rects[k].height < eyeImgSize_height)
                {
                    posEye_rects[k].y -= (int)((eyeImgSize_height - posEye_rects[k].height) * 0.5);
                    posEye_rects[k].height = eyeImgSize_height;
                }
            }

            //make sure 0 is the left eye
            OpenCVForUnity.Rect tmpRect = posEye_rects[0];
            if (posEye_rects[0].x > posEye_rects[1].x)
            {
                posEye_rects[0] = posEye_rects[1];
                posEye_rects[1] = tmpRect;
            }

            posNose_rects = filterNose_rects;

            if (posNose_rects.width < noseImgSize_width)
            {
                posNose_rects.x -= (int)((noseImgSize_width - posNose_rects.width) * 0.5);
                posNose_rects.width = noseImgSize_width;
            }

            if (posNose_rects.height < noseImgSize_height)
            {
                posNose_rects.y -= noseImgSize_height - posNose_rects.height;
                posNose_rects.height = noseImgSize_height;
            }

            posMouths_rects = filterMouths_rects;

            if (posMouths_rects.width < mouseImgSize_width)
            {
                posMouths_rects.x -= (int)((mouseImgSize_width - posMouths_rects.width) * 0.5);
                posMouths_rects.width = mouseImgSize_width;
            }

            if (posMouths_rects.height < mouseImgSize_height)
            {
                posMouths_rects.height = mouseImgSize_height;
                posMouths_rects.y = posNose_rects.y + posNose_rects.height;

                if (posMouths_rects.y + posMouths_rects.height > s_h)
                    posMouths_rects.height = s_h - posMouths_rects.y;
            }

            //if (posMouths_rects.height < 400)
            //{
            //    posMouths_rects.y -= 400 - posMouths_rects.height;
            //    posMouths_rects.height = 400;
            //}

            CalculateForehead();
            CalculateCheeks(s_w, s_h);
            //CalculateChin();
        }

        void CalculateForehead()
        {

            if(posEye_rects == null || posEye_rects[0] == null || posEye_rects[1] == null)
            {
                Debug.Log("no eyes no forehead T-T");
                return;
            }

            // assme foreheadw = zone between two eyes
            int x = (int)(posEye_rects[0].x + posEye_rects[0].width * 0.5);
            int y = posEye_rects[0].y - posEye_rects[0].height;
            int w = Mathf.Abs(posEye_rects[0].x - posEye_rects[1].x);
            int h = posEye_rects[0].height;

            posForehead_rects = new OpenCVForUnity.Rect(x, y, w, h);

            if (posForehead_rects.width < mouseImgSize_width)
            {
                posForehead_rects.x -= (int)((foreheadImgSize_width - posForehead_rects.width) * 0.5);
                posForehead_rects.width = foreheadImgSize_width;
            }
        }

        void CalculateCheeks(int screenWidth, int screenHeight)
        {
            if (posEye_rects == null || posEye_rects[0] == null || posEye_rects[1] == null)
            {
                Debug.Log("no eyes no cheeks T-T");
                return;
            }

            int noseGap = Mathf.Abs((posEye_rects[0].x + posEye_rects[0].width) - filterNose_rects.x);
            if (filterNose_rects.x == 0)
                noseGap = 0;
            int x_L = posEye_rects[0].x;
            int x_R = posEye_rects[1].x ;
            int y = posEye_rects[0].y + posEye_rects[0].height;
            int w = posEye_rects[0].width;
            int h = posEye_rects[0].height;

            if (x_R + w > screenWidth)
                w = screenWidth - x_R;

            posCheek_rects[0] = new OpenCVForUnity.Rect(x_L, y, w, h);
            posCheek_rects[1] = new OpenCVForUnity.Rect(x_R, y, w, h);

  
        }

        //void CalculateChin()
        //{
        //    if (posMouths_rects == null)
        //    {
        //        Debug.Log("no mouth no chin T-T");
        //        return;
        //    }

        //    int x = posMouths_rects.x;
        //    int y = posMouths_rects.y + posMouths_rects.height;
        //    int w = posMouths_rects.width;
        //    int h = Mathf.Abs(face_rects[0].y + face_rects[0].height - y);

        //    posChin_rects = new OpenCVForUnity.Rect(x, y, w, h);
        //}

        private void DrawOrgansPose()
        {
            Mat rgbaMat = webCamTextureToMatHelper.GetMat();

            int s_w = rgbaMat.width();
            int s_h = rgbaMat.height();

            // eyes
            //Imgproc.rectangle(rgbaMat, new Point(posEye_rects[0].x, posEye_rects[0].y), new Point(posEye_rects[0].x + posEye_rects[0].width, posEye_rects[0].y + posEye_rects[0].height), new Scalar(0, 0, 255, 255), 2);
            //Imgproc.rectangle(rgbaMat, new Point(posEye_rects[1].x, posEye_rects[1].y), new Point(posEye_rects[1].x + posEye_rects[1].width, posEye_rects[1].y + posEye_rects[1].height), new Scalar(0, 0, 255, 255), 2);
            Point center1 = new Point((posEye_rects[0].x + posEye_rects[0].x + posEye_rects[0].width) / 2, (posEye_rects[0].y + posEye_rects[0].y + posEye_rects[0].height) / 2);
            Imgproc.circle(rgbaMat, center1, posEye_rects[0].width / 2, new Scalar(0, 0, 255, 255), 2);

            Point center2 = new Point((posEye_rects[1].x + posEye_rects[1].x + posEye_rects[1].width) / 2, (posEye_rects[1].y + posEye_rects[1].y + posEye_rects[1].height) / 2);
            Imgproc.circle(rgbaMat, center2, posEye_rects[1].width / 2, new Scalar(0, 0, 255, 255), 2);

            // nose
            Imgproc.rectangle(rgbaMat, new Point(posNose_rects.x, posNose_rects.y), new Point(posNose_rects.x + posNose_rects.width, posNose_rects.y + posNose_rects.height), new Scalar(0, 255, 0, 255), 2);

            // mouth
            Imgproc.rectangle(rgbaMat, new Point(posMouths_rects.x, posMouths_rects.y), new Point(posMouths_rects.x + posMouths_rects.width, posMouths_rects.y + posMouths_rects.height), new Scalar(255, 0, 0, 255), 2);

            // cheeks
            Imgproc.rectangle(rgbaMat, new Point(posCheek_rects[0].x, posCheek_rects[0].y), new Point(posCheek_rects[0].x + posCheek_rects[0].width, posCheek_rects[0].y + posCheek_rects[0].height), new Scalar(0, 255, 255, 255), 2);
            Imgproc.rectangle(rgbaMat, new Point(posCheek_rects[1].x, posCheek_rects[1].y), new Point(posCheek_rects[1].x + posCheek_rects[1].width, posCheek_rects[1].y + posCheek_rects[1].height), new Scalar(0, 255, 255, 255), 2);

            // forehead
            Imgproc.rectangle(rgbaMat, new Point(posForehead_rects.x, posForehead_rects.y), new Point(posForehead_rects.x + posForehead_rects.width, posForehead_rects.y + posForehead_rects.height), new Scalar(255, 255, 0, 255), 2);

            // chin
            //Imgproc.rectangle(rgbaMat, new Point(posChin_rects.x, posChin_rects.y), new Point(posChin_rects.x + posChin_rects.width, posChin_rects.y + posChin_rects.height), new Scalar(255, 255, 255, 255), 2);

            Utils.fastMatToTexture2D(rgbaMat, texture);

        }

        private void Run()
        {
            weightsPositionsSmoothing.Add(1);
            weightsSizesSmoothing.Add(0.5f);
            weightsSizesSmoothing.Add(0.3f);
            weightsSizesSmoothing.Add(0.2f);

            parameters.maxTrackLifetime = 5;

            innerParameters.numLastPositionsToTrack = 4;
            innerParameters.numStepsToWaitBeforeFirstShow = 6;
            innerParameters.numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown = 3;
            innerParameters.numStepsToShowWithoutDetecting = 3;
            innerParameters.coeffTrackingWindowSize = 2.0f;
            innerParameters.coeffObjectSizeToTrack = 0.85f;
            innerParameters.coeffObjectSpeedUsingInPrediction = 0.8f;

            webCamTextureToMatHelper.Initialize();
        }

        void FindFiveSenseOrgans(Mat srcMat)
        {
            // Convert into grayMat with RGB2GRAY 转为灰度图像
            Mat grayMat = new Mat();
            Imgproc.cvtColor(srcMat, grayMat, Imgproc.COLOR_RGB2GRAY);

            // Detect all the faces in the image 检测图像中的所有脸
            MatOfRect faces = new MatOfRect();
            CascadeClassifier cascade = new CascadeClassifier(haarcascade_frontalface_default_xml_filepath);
            cascade.detectMultiScale(grayMat, faces, 1.1d, 2, 2, new Size(20, 20), new Size());
            //Debug.Log(faces); //检测到多少个脸 [ elemSize*1*CV_32SC4, isCont=True, isSubmat=False, nativeObj=0x1128611568, dataAddr=0x0 ]

            face_rects = faces.toArray();
            for (int i = 0; i < face_rects.Length; i++)
            {
                //Debug.Log("detect faces " + rects[i]);

                Scalar color = new Scalar(0, 255, 0, 255); //green

                Imgproc.rectangle(srcMat, new Point(face_rects[i].x, face_rects[i].y), new Point(face_rects[i].x + face_rects[i].width, face_rects[i].y + face_rects[i].height), color, 2);
                Mat roi_gray_img = new Mat(grayMat, new OpenCVForUnity.Rect(0, 0, face_rects[i].x + face_rects[i].width, face_rects[i].y + face_rects[i].height));
                Mat roi_img = new Mat(srcMat, new OpenCVForUnity.Rect(0, 0, face_rects[i].x + face_rects[i].width, face_rects[i].y + face_rects[i].height));

                if (!FindEyes(roi_gray_img, roi_img, i))
                {
                    Debug.Log("No eyes has been found!");
                }

                FindNoses(roi_gray_img, roi_img, i);

                FindMouth(roi_gray_img, roi_img, i);
            }

            // create img
            //Texture2D t2d = new Texture2D(srcMat.width(), srcMat.height());
            //Utils.matToTexture2D(srcMat, t2d);
            //Sprite sp = Sprite.Create(t2d, new UnityEngine.Rect(0, 0, t2d.width, t2d.height), Vector2.zero);
            //m_srcImage.sprite = sp;
            //m_srcImage.preserveAspect = true;
        }

        bool FindEyes(Mat grayimg, Mat img, int curfaceId)
        {
            // Detect the eyes on the face 眼长在脸上
            MatOfRect eyes = new MatOfRect();
            CascadeClassifier eyecascade = new CascadeClassifier(haarcascade_eye_xml_filepath);
            eyecascade.detectMultiScale(grayimg, eyes, 1.3d, 5, 2, new Size(20, 20), new Size());

            OpenCVForUnity.Rect[] _rect;
            int eyeNum = 0;

            if (eyes.elemSize() > 0)
            {
                originalEye_rects = eyes.toArray();

                int rectLen = originalEye_rects.Length;

                _rect = new OpenCVForUnity.Rect[rectLen];

                for (int t = 0; t < rectLen; t++)
                {
                    Point center = new Point((originalEye_rects[t].x + originalEye_rects[t].x + originalEye_rects[t].width) / 2, (originalEye_rects[t].y + originalEye_rects[t].y + originalEye_rects[t].height) / 2);

                    if (IsPointInRect(center, face_rects[curfaceId]))
                    {
                        _rect[t] = originalEye_rects[t];
                        eyeNum++;
                    }
                }

                if (eyeNum == 0)
                    return false;


                // loop testing distance between eyes to find the Y closest pair of eyes
                int rangeY = 65535;
                for (int i = 0; i < rectLen; i++)
                {
                    if (_rect[i] != null)
                    {
                        for (int j = 0; j < rectLen; j++)
                        {
                            if (j != i && _rect[j] != null)
                            {
                                int range = Mathf.Abs(_rect[i].y - _rect[j].y);
                                if (range < rangeY)
                                {
                                    rangeY = range;
                                    filterEye_rects[0] = _rect[i];
                                    filterEye_rects[1] = _rect[j];
                                }
                            }
                        }
                    }
                }

                // only detected 1 eye
                if (rangeY == 65535)
                {
                    for (int i = 0; i < rectLen; i++)
                    {
                        if (_rect[i] != null)
                        {
                            filterEye_rects[0] = _rect[i];

                            Point center = new Point((filterEye_rects[0].x + filterEye_rects[0].x + filterEye_rects[0].width) / 2, (filterEye_rects[0].y + filterEye_rects[0].y + filterEye_rects[0].height) / 2);
                            Imgproc.circle(img, center, filterEye_rects[0].width / 2, new Scalar(255, 255, 0, 255), 2);

                            filterEye_rects[1] = null;

                            return true;
                        }
                    }
                }

                // detected more than 1 eyes
                for (int i = 0; i < 2; i++)
                {
                    Point center = new Point((filterEye_rects[i].x + filterEye_rects[i].x + filterEye_rects[i].width) / 2, (filterEye_rects[i].y + filterEye_rects[i].y + filterEye_rects[i].height) / 2);
                    Imgproc.circle(img, center, filterEye_rects[i].width / 2, new Scalar(255, 255, 0, 255), 2);
                }
                return true;
            }

            return false;

        }


        void FindNoses(Mat grayimg, Mat img, int curfaceId)
        {
            MatOfRect noses = new MatOfRect();
            CascadeClassifier nosecascade = new CascadeClassifier(haarcascade_mcs_nose_xml_filepath);
            nosecascade.detectMultiScale(grayimg, noses, 1.3d, 5, 2, new Size(20, 20), new Size());
            Scalar color = new Scalar(255, 0, 0, 255); //red
            int nose_thickness = 2;
            Point p1 = new Point(0, 0);
            Point p2 = new Point(0, 0);

            if (noses.elemSize() > 0)
            {
                originalNose_rects = noses.toArray();
                for (int t = 0; t < originalNose_rects.Length; t++)
                {
                    p1 = new Point(originalNose_rects[t].x, originalNose_rects[t].y);
                    p2 = new Point(originalNose_rects[t].x + originalNose_rects[t].width, originalNose_rects[t].y + originalNose_rects[t].height);

                    if (IsPointInRect(p1, face_rects[curfaceId])
                        && (filterEye_rects != null || p1.y > (filterEye_rects[0].y + filterEye_rects[0].height * 0.5)))
                    {
                        filterNose_rects = originalNose_rects[t];
                        Imgproc.rectangle(img, p1, p2, color, nose_thickness);
                    }
                }
            }
        }

        void FindMouth(Mat grayimg, Mat img, int curfaceId)
        {
            Scalar color = new Scalar(0, 0, 255, 255); //blue
            int mouth_thickness = 2;
            MatOfRect mouths = new MatOfRect();
            CascadeClassifier mouthcascade = new CascadeClassifier(haarcascade_mcs_mouth_xml_filepath);
            mouthcascade.detectMultiScale(grayimg, mouths, 1.3d, 5, 2, new Size(20, 20), new Size());

            if (mouths.elemSize() > 0)
            {
                originalMouths_rects = mouths.toArray();
                for (int t = 0; t < originalMouths_rects.Length; t++)
                {
                    Point p1 = new Point(originalMouths_rects[t].x, originalMouths_rects[t].y);
                    Point p2 = new Point(originalMouths_rects[t].x + originalMouths_rects[t].width, originalMouths_rects[t].y + originalMouths_rects[t].height);

                    if (IsPointInRect(p1, face_rects[curfaceId]))
                    {
                        if (originalNose_rects == null || p1.y > (originalNose_rects[0].y + originalNose_rects[0].height * 0.5))
                        {
                            if (filterEye_rects == null || (filterEye_rects[0] != null && p1.y > (filterEye_rects[0].y + filterEye_rects[0].height)))
                            {
                                filterMouths_rects = originalMouths_rects[t];
                                Imgproc.rectangle(img, p1, p2, color, mouth_thickness);
                            }
                        }
                    }
                }
            }
        }

        bool IsPointInRect(Point p, OpenCVForUnity.Rect rect)
        {
            if (p.x < rect.x
                || p.y < rect.y
                || p.x > rect.x + rect.width
                || p.y > rect.y + rect.height)
                return false;

            return true;
        }

        private void DetectInRegion(Mat img, OpenCVForUnity.Rect r, List<OpenCVForUnity.Rect> detectedObjectsInRegions)
        {
            OpenCVForUnity.Rect r0 = new OpenCVForUnity.Rect(new Point(), img.size());
            OpenCVForUnity.Rect r1 = new OpenCVForUnity.Rect(r.x, r.y, r.width, r.height);
            OpenCVForUnity.Rect.inflate(r1, (int)((r1.width * innerParameters.coeffTrackingWindowSize) - r1.width) / 2,
                (int)((r1.height * innerParameters.coeffTrackingWindowSize) - r1.height) / 2);
            r1 = OpenCVForUnity.Rect.intersect(r0, r1);

            if (r1 != null && (r1.width <= 0) || (r1.height <= 0))
            {
                Debug.Log("DetectionBasedTracker::detectInRegion: Empty intersection");
                return;
            }


            int d = Math.Min(r.width, r.height);
            d = (int)Math.Round(d * innerParameters.coeffObjectSizeToTrack);


            MatOfRect tmpobjects = new MatOfRect();

            Mat img1 = new Mat(img, r1);//subimage for rectangle -- without data copying

            cascade.detectMultiScale(img1, tmpobjects, 1.1, 2, 0 | Objdetect.CASCADE_DO_CANNY_PRUNING | Objdetect.CASCADE_SCALE_IMAGE | Objdetect.CASCADE_FIND_BIGGEST_OBJECT, new Size(d, d), new Size());


            OpenCVForUnity.Rect[] tmpobjectsArray = tmpobjects.toArray();
            int len = tmpobjectsArray.Length;
            for (int i = 0; i < len; i++)
            {
                OpenCVForUnity.Rect tmp = tmpobjectsArray[i];
                OpenCVForUnity.Rect curres = new OpenCVForUnity.Rect(new Point(tmp.x + r1.x, tmp.y + r1.y), tmp.size());
                detectedObjectsInRegions.Add(curres);
            }
        }

        private void GetObjects(List<OpenCVForUnity.Rect> result)
        {
            result.Clear();

            for (int i = 0; i < trackedObjects.Count; i++)
            {
                OpenCVForUnity.Rect r = CalcTrackedObjectPositionToShow(i);
                if (r.area() == 0)
                {
                    continue;
                }
                result.Add(r);
                //LOGD("DetectionBasedTracker::process: found a object with SIZE %d x %d, rect={%d, %d, %d x %d}", r.width, r.height, r.x, r.y, r.width, r.height);
            }
        }

        private void UpdateTrackedObjects(List<OpenCVForUnity.Rect> detectedObjects)
        {
            int N1 = (int)trackedObjects.Count;
            int N2 = (int)detectedObjects.Count;

            for (int i = 0; i < N1; i++)
            {
                trackedObjects[i].numDetectedFrames++;
            }

            int[] correspondence = new int[N2];
            for (int i = 0; i < N2; i++)
            {
                correspondence[i] = (int)TrackedState.NEW_RECTANGLE;
            }


            for (int i = 0; i < N1; i++)
            {
                TrackedObject curObject = trackedObjects[i];

                int bestIndex = -1;
                int bestArea = -1;

                int numpositions = (int)curObject.lastPositions.Count;

                //if (numpositions > 0) UnityEngine.Debug.LogError("numpositions > 0 is false");

                OpenCVForUnity.Rect prevRect = curObject.lastPositions[numpositions - 1];

                for (int j = 0; j < N2; j++)
                {
                    if (correspondence[j] >= 0)
                    {
                        //Debug.Log("DetectionBasedTracker::updateTrackedObjects: j=" + i + " is rejected, because it has correspondence=" + correspondence[j]);
                        continue;
                    }
                    if (correspondence[j] != (int)TrackedState.NEW_RECTANGLE)
                    {
                        //Debug.Log("DetectionBasedTracker::updateTrackedObjects: j=" + j + " is rejected, because it is intersected with another rectangle");
                        continue;
                    }

                    OpenCVForUnity.Rect r = OpenCVForUnity.Rect.intersect(prevRect, detectedObjects[j]);
                    if (r != null && (r.width > 0) && (r.height > 0))
                    {
                        //LOGD("DetectionBasedTracker::updateTrackedObjects: There is intersection between prevRect and detectedRect, r={%d, %d, %d x %d}",
                        //        r.x, r.y, r.width, r.height);
                        correspondence[j] = (int)TrackedState.INTERSECTED_RECTANGLE;

                        if (r.area() > bestArea)
                        {
                            //LOGD("DetectionBasedTracker::updateTrackedObjects: The area of intersection is %d, it is better than bestArea=%d", r.area(), bestArea);
                            bestIndex = j;
                            bestArea = (int)r.area();
                        }
                    }
                }

                if (bestIndex >= 0)
                {
                    //LOGD("DetectionBasedTracker::updateTrackedObjects: The best correspondence for i=%d is j=%d", i, bestIndex);
                    correspondence[bestIndex] = i;

                    for (int j = 0; j < N2; j++)
                    {
                        if (correspondence[j] >= 0)
                            continue;

                        OpenCVForUnity.Rect r = OpenCVForUnity.Rect.intersect(detectedObjects[j], detectedObjects[bestIndex]);
                        if (r != null && (r.width > 0) && (r.height > 0))
                        {
                            //LOGD("DetectionBasedTracker::updateTrackedObjects: Found intersection between "
                            //    "rectangles j=%d and bestIndex=%d, rectangle j=%d is marked as intersected", j, bestIndex, j);
                            correspondence[j] = (int)TrackedState.INTERSECTED_RECTANGLE;
                        }
                    }
                }
                else
                {
                    //LOGD("DetectionBasedTracker::updateTrackedObjects: There is no correspondence for i=%d ", i);
                    curObject.numFramesNotDetected++;
                }
            }

            //LOGD("DetectionBasedTracker::updateTrackedObjects: start second cycle");
            for (int j = 0; j < N2; j++)
            {
                int i = correspondence[j];
                if (i >= 0)
                {//add position
                 //Debug.Log("DetectionBasedTracker::updateTrackedObjects: add position");
                    trackedObjects[i].lastPositions.Add(detectedObjects[j]);
                    while ((int)trackedObjects[i].lastPositions.Count > (int)innerParameters.numLastPositionsToTrack)
                    {
                        trackedObjects[i].lastPositions.Remove(trackedObjects[i].lastPositions[0]);
                    }
                    trackedObjects[i].numFramesNotDetected = 0;
                }
                else if (i == (int)TrackedState.NEW_RECTANGLE)
                { //new object
                  //Debug.Log("DetectionBasedTracker::updateTrackedObjects: new object");
                    trackedObjects.Add(new TrackedObject(detectedObjects[j]));
                }
                else
                {
                    //Debug.Log ("DetectionBasedTracker::updateTrackedObjects: was auxiliary intersection");
                }
            }

            int t = 0;
            TrackedObject it;
            while (t < trackedObjects.Count)
            {
                it = trackedObjects[t];

                if ((it.numFramesNotDetected > parameters.maxTrackLifetime)
                    ||
                    ((it.numDetectedFrames <= innerParameters.numStepsToWaitBeforeFirstShow)
                    &&
                    (it.numFramesNotDetected > innerParameters.numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown)))
                {
                    //int numpos = (int)it.lastPositions.Count;
                    //if (numpos > 0) UnityEngine.Debug.LogError("numpos > 0 is false");
                    //Rect r = it.lastPositions [numpos - 1];
                    //Debug.Log("DetectionBasedTracker::updateTrackedObjects: deleted object " + r.x + " " + r.y + " " + r.width + " " + r.height);

                    trackedObjects.Remove(it);

                }
                else
                {
                    t++;
                }
            }
        }

        private OpenCVForUnity.Rect CalcTrackedObjectPositionToShow(int i)
        {
            if ((i < 0) || (i >= trackedObjects.Count))
            {
                Debug.Log("DetectionBasedTracker::calcTrackedObjectPositionToShow: ERROR: wrong i=" + i);
                return new OpenCVForUnity.Rect();
            }
            if (trackedObjects[i].numDetectedFrames <= innerParameters.numStepsToWaitBeforeFirstShow)
            {
                //Debug.Log("DetectionBasedTracker::calcTrackedObjectPositionToShow: " + "trackedObjects[" + i + "].numDetectedFrames=" + trackedObjects[i].numDetectedFrames + " <= numStepsToWaitBeforeFirstShow=" + innerParameters.numStepsToWaitBeforeFirstShow + " --- return empty Rect()");
                return new OpenCVForUnity.Rect();
            }
            if (trackedObjects[i].numFramesNotDetected > innerParameters.numStepsToShowWithoutDetecting)
            {
                return new OpenCVForUnity.Rect();
            }

            List<OpenCVForUnity.Rect> lastPositions = trackedObjects[i].lastPositions;

            int N = lastPositions.Count;
            if (N <= 0)
            {
                Debug.Log("DetectionBasedTracker::calcTrackedObjectPositionToShow: ERROR: no positions for i=" + i);
                return new OpenCVForUnity.Rect();
            }

            int Nsize = Math.Min(N, (int)weightsSizesSmoothing.Count);
            int Ncenter = Math.Min(N, (int)weightsPositionsSmoothing.Count);

            Point center = new Point();
            double w = 0, h = 0;
            if (Nsize > 0)
            {
                double sum = 0;
                for (int j = 0; j < Nsize; j++)
                {
                    int k = N - j - 1;
                    w += lastPositions[k].width * weightsSizesSmoothing[j];
                    h += lastPositions[k].height * weightsSizesSmoothing[j];
                    sum += weightsSizesSmoothing[j];
                }
                w /= sum;
                h /= sum;
            }
            else
            {
                w = lastPositions[N - 1].width;
                h = lastPositions[N - 1].height;
            }

            if (Ncenter > 0)
            {
                double sum = 0;
                for (int j = 0; j < Ncenter; j++)
                {
                    int k = N - j - 1;
                    Point tl = lastPositions[k].tl();
                    Point br = lastPositions[k].br();
                    Point c1;
                    //c1=tl;
                    //c1=c1* 0.5f;//
                    c1 = new Point(tl.x * 0.5f, tl.y * 0.5f);
                    Point c2;
                    //c2=br;
                    //c2=c2*0.5f;
                    c2 = new Point(br.x * 0.5f, br.y * 0.5f);
                    //c1=c1+c2;
                    c1 = new Point(c1.x + c2.x, c1.y + c2.y);

                    //center=center+  (c1  * weightsPositionsSmoothing[j]);
                    center = new Point(center.x + (c1.x * weightsPositionsSmoothing[j]), center.y + (c1.y * weightsPositionsSmoothing[j]));
                    sum += weightsPositionsSmoothing[j];
                }
                //center *= (float)(1 / sum);
                center = new Point(center.x * (1 / sum), center.y * (1 / sum));
            }
            else
            {
                int k = N - 1;
                Point tl = lastPositions[k].tl();
                Point br = lastPositions[k].br();
                Point c1;
                //c1=tl;
                //c1=c1* 0.5f;
                c1 = new Point(tl.x * 0.5f, tl.y * 0.5f);
                Point c2;
                //c2=br;
                //c2=c2*0.5f;
                c2 = new Point(br.x * 0.5f, br.y * 0.5f);

                //center=c1+c2;
                center = new Point(c1.x + c2.x, c1.y + c2.y);
            }
            //Point2f tl=center-(Point2f(w,h)*0.5);
            Point tl2 = new Point(center.x - (w * 0.5f), center.y - (h * 0.5f));
            //Rect res(cvRound(tl.x), cvRound(tl.y), cvRound(w), cvRound(h));
            OpenCVForUnity.Rect res = new OpenCVForUnity.Rect((int)Math.Round(tl2.x), (int)Math.Round(tl2.y), (int)Math.Round(w), (int)Math.Round(h));
            //LOGD("DetectionBasedTracker::calcTrackedObjectPositionToShow: Result for i=%d: {%d, %d, %d x %d}", i, res.x, res.y, res.width, res.height);

            return res;
        }

        public Point CenterRect(OpenCVForUnity.Rect r)
        {
            return new Point(r.x + (r.width / 2), r.y + (r.height / 2));
        }

        private struct Parameters
        {
            //public int minObjectSize;
            //public int maxObjectSize;
            //public float scaleFactor;
            //public int minNeighbors;

            public int maxTrackLifetime;
            //public int minDetectionPeriod; //the minimal time between run of the big object detector (on the whole frame) in ms (1000 mean 1 sec), default=0
        };

        private struct InnerParameters
        {
            public int numLastPositionsToTrack;
            public int numStepsToWaitBeforeFirstShow;
            public int numStepsToTrackWithoutDetectingIfObjectHasNotBeenShown;
            public int numStepsToShowWithoutDetecting;
            public float coeffTrackingWindowSize;
            public float coeffObjectSizeToTrack;
            public float coeffObjectSpeedUsingInPrediction;
        };

        private class TrackedObject
        {
            public PositionsVector lastPositions;
            public int numDetectedFrames;
            public int numFramesNotDetected;
            public int id;
            static private int _id = 0;

            public TrackedObject(OpenCVForUnity.Rect rect)
            {
                lastPositions = new PositionsVector();

                numDetectedFrames = 1;
                numFramesNotDetected = 0;

                lastPositions.Add(rect.clone());

                _id = GetNextId();
                id = _id;
            }

            static int GetNextId()
            {
                _id++;
                return _id;
            }
        }

        public void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");

            StopThread();

            if (grayMat4Thread != null)
                grayMat4Thread.Dispose();

            if (cascade4Thread != null)
                cascade4Thread.Dispose();

            if (grayMat != null)
                grayMat.Dispose();

            if (texture != null)
            {
                Texture2D.Destroy(texture);
                texture = null;
            }

            if (cascade != null)
                cascade.Dispose();

            trackedObjects.Clear();
        }

        private void Detect()
        {
            MatOfRect objects = new MatOfRect();
            if (cascade4Thread != null)
                cascade4Thread.detectMultiScale(grayMat4Thread, objects, 1.1, 2, Objdetect.CASCADE_SCALE_IMAGE, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(grayMat4Thread.height() * 0.2, grayMat4Thread.height() * 0.2), new Size());

            //Thread.Sleep(200);

            detectionResult = objects;
        }

        private void ThreadWorker()
        {
            isThreadRunning = true;

            while (!shouldStopThread)
            {
                if (!shouldDetectInMultiThread)
                    continue;

                Detect();

                shouldDetectInMultiThread = false;
                didUpdateTheDetectionResult = true;
            }

            isThreadRunning = false;
        }

        private void InitThread()
        {
            StopThread();

            grayMat4Thread = new Mat();

            cascade4Thread = new CascadeClassifier();
            cascade4Thread.load(haarcascade_frontalface_default_xml_filepath);

            if (cascade4Thread.empty())
            {
                Debug.LogError("cascade4Thread file is not loaded.Please copy from “OpenCVForUnity/StreamingAssets/” to “Assets/StreamingAssets/” folder. ");
            }

            shouldDetectInMultiThread = false;

            StartThread(ThreadWorker);

        }

        private void StartThread(Action action)
        {
            shouldStopThread = false;

            ThreadPool.QueueUserWorkItem(_ => action());

            Debug.Log("Thread Start");
        }

        private void StopThread()
        {
            if (!isThreadRunning)
                return;

            shouldStopThread = true;

            while (isThreadRunning)
            {
                //Wait threading stop
            }
            Debug.Log("Thread Stop");
        }

        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }

        public void OnWebCamTextureToMatHelperInitialized()
        {
            Debug.Log("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

            texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);

            gameObject.GetComponent<Renderer>().material.mainTexture = texture;

            gameObject.transform.localScale = new Vector3(webCamTextureMat.cols(), webCamTextureMat.rows(), 1);

            Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", webCamTextureMat.width().ToString());
                fpsMonitor.Add("height", webCamTextureMat.height().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }


            float width = webCamTextureMat.width();
            float height = webCamTextureMat.height();

            float widthScale = (float)Screen.width / width;
            float heightScale = (float)Screen.height / height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = height / 2;
            }


            grayMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC1);
            cascade = new CascadeClassifier();
            cascade.load(lbpcascade_frontalface_xml_filepath);

            if (cascade.empty())
            {
                Debug.LogError("cascade file is not loaded.Please copy from “OpenCVForUnity/StreamingAssets/” to “Assets/StreamingAssets/” folder. ");
            }
            InitThread();
        }

        public void OnTakeSnapshotButtonClick()
        {
            //remove abandoned code
            //if (detectedObjectsInRegions.Count <= 0)
            //{
            //    Debug.Log("No face has been detected!");
            //    return;
            //}

            String s_year = System.DateTime.Now.Year.ToString();
            String s_month = System.DateTime.Now.Month.ToString();
            String s_day = System.DateTime.Now.Day.ToString();
            String s_hour = System.DateTime.Now.Hour.ToString();
            String s_min = System.DateTime.Now.Minute.ToString();
            String s_sec = System.DateTime.Now.Second.ToString();

            String s_name = s_year + s_month + s_day + s_hour + s_min + s_sec + "_" + _CaptureCounter.ToString();

            ++_CaptureCounter;

            StartCoroutine(StartSaveSnapshot(0.1f, s_name));
        }

        IEnumerator StartSaveSnapshot(float waitTime,  String s_name)
        {
            WebCamTexture webCamTexture = webCamTextureToMatHelper.GetWebCamTexture();

            Texture2D rottex = RotateNeg90Snap(webCamTexture, webCamTexture.width, webCamTexture.height);

            String name;
            String rename;
            String curID = (_SignalID + 1).ToString() + "-";

            //save eyes
            for (int i = 0; i < filterEye_rects.Length; i++)
            {
                name = s_name + "_eye_" + i.ToString();
                snapShotsName.Add(name); 

                saveSnaps(rottex, posEye_rects[i], name);

                if( i == 0)
                    rename = /*curID +*/ "3-" + _CaptureCounter.ToString();
                else
                    rename = /*curID +*/ "1-" + _CaptureCounter.ToString();
                snapShotsRename.Add(rename);
            }

            if(filterEye_rects.Length <= 0)
            {
                Debug.Log("filterEye_rects.Length <= 0 ");
            }
            if (filterEye_rects[0] == null)
            {
                Debug.Log("filterEye_rects[0] == null");
            }
            if (filterEye_rects[1] == null)
            {
                Debug.Log("filterEye_rects[1] == null");
            }

            //save nose
            if (filterNose_rects != null)
            {
                name = s_name + "_nose";
                snapShotsName.Add(name);
                saveSnaps(rottex, posNose_rects, name);
                rename = /*curID +*/ "2-" + _CaptureCounter.ToString();
                snapShotsRename.Add(rename);
            }
            else
            {
                Debug.Log("filterNose_rects == null");
            }
               

            //save mouth
            if (filterMouths_rects != null)
            {
                name = s_name + "_mouth";
                saveSnaps(rottex, posMouths_rects, name);
                rename = /*curID +*/ "6-" + _CaptureCounter.ToString();
                snapShotsName.Add(name);
                snapShotsRename.Add(rename);
            }
            else
            {
                Debug.Log("filterMouths_rects == null");
            }

            //save cheeks
            for (int i = 0; i < posCheek_rects.Length; i++)
            {
                name = s_name + "_cheek_" + i.ToString();
                if (i == 0)
                    rename = /*curID + */"5-" + _CaptureCounter.ToString();
                else
                    rename = /*curID +*/ "4-" + _CaptureCounter.ToString();
                saveSnaps(rottex, posCheek_rects[i], s_name + "_cheek_" + i.ToString());

                snapShotsName.Add(name);
                snapShotsRename.Add(rename);
            }

            //save chin
            //if (posChin_rects != null)
            //    saveSnaps(webCamTexture, posChin_rects, s_name + "_chin");

            //save forehead
            if (posForehead_rects != null)
            {
                name = s_name + "_forehead";
                rename = /*curID +*/ "0-" + _CaptureCounter.ToString();
                saveSnaps(rottex, posForehead_rects, s_name + "_forehead");

                snapShotsName.Add(name);
                snapShotsRename.Add(rename);
            }
            yield return new WaitForSeconds(5 * waitTime);

            String targetCopyfile;

            //write to latest folder
            for (int i = 0; i < snapShotsName.Count; i++)
            {
                targetCopyfile = Path.Combine(_SavePath, snapShotsName[i] + ".jpeg");
                if (!File.Exists(targetCopyfile))
                    yield return new WaitForSeconds(5 * waitTime);
                renameSaveLatestFile(targetCopyfile, Path.Combine(_SaveLatestPath, snapShotsRename[i] + ".jpeg"));
            }
            yield return new WaitForSeconds(waitTime);
        }

        // save pictures 
        Texture2D rotSnap;
        void saveSnaps(Texture2D input, OpenCVForUnity.Rect _rect, String _name)
        {
            //Debug.Log("SaveSnaps " + _name + "_rect.x " + _rect.x + "_rect.y " + _rect.y +  " _rect.width " + _rect.width + " _rect.height " + _rect.height);

            Texture2D snap;

            int x = _rect.x;
            int y = input.height - _rect.y - _rect.height;
            int w =  _rect.width;
            int h =  _rect.height;

            snap = new Texture2D(w, h);
            snap.SetPixels(input.GetPixels(x, y, w, h));

            snap.Apply();

            // Note: different coordinate system, rectsWhereRegions's 0,0 is left top, when getting pixel it is left down.
            System.IO.File.WriteAllBytes(_SavePath + _name.ToString() + ".jpeg", snap.EncodeToJPG());
        }

        void renameSaveLatestFile(String sourcename, String name)
        {
            System.IO.File.Copy(sourcename, name, true);
        }

        Texture2D RotateNeg90Snap(WebCamTexture tex, int w, int h)
        {
            Texture2D rottex = new Texture2D(h, w);
            Color c;

            for (int i = 0; i < w - 1; i++)
            {
                for (int j = 0; j < h - 1; j++)
                {
                    c = tex.GetPixel(i, j);
                    rottex.SetPixel(j, i, c);
                }
            }
            rottex.Apply();
            return rottex;
        }

        void StartFlashUI()
        {
            GameObject g = Instantiate(FlashUIObj, new Vector3(0, 0, 0), Quaternion.identity);
            g.transform.SetParent(CanvasObj.transform, false);
            g.SetActive(true);
        }

    }
}
