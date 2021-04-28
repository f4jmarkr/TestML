//
//  ViewController.swift
//  TestML
//
//  Created by Louis Lanctôt on 2020-05-23.
//  Copyright © 2020 Louis. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import Vision
import PencilKit
// ??Why is PencilKit used??


class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {

    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }

    /// Concurrent queue to be used for model predictions
    let predictionQueue = DispatchQueue(label: "predictionQueue",
                                        qos: .userInitiated,
                                        attributes: [],
                                        autoreleaseFrequency: .inherit,
                                        target: nil)

    /// The ARSceneView
    @IBOutlet var sceneView: ARSCNView!

    /// Label used to display any relevant information to the user
    /// This can be the model name, the predicted dice, the recognized digit, ...
    @IBOutlet weak var infoLabel: UILabel!

    /// Layer used to host detectionOverlay layer
    var rootLayer: CALayer!
    
    /// The detection overlay layer used to render bounding boxes
    var detectionOverlay: CALayer!

    /// Whether the current frame should be skipped (in terms of model predictions)
    var shouldSkipFrame = 0
    
    /// How often (in terms of camera frames) should the app run predictions
    let predictEvery = 3

    /// Vision request for the detection model
    var diceDetectionRequest: VNCoreMLRequest!

    /// Flag used to decide whether to draw bounding boxes for detected objects
    var showBoxes = true {
        didSet {
            if !showBoxes {
                removeBoxes()
            }
        }
    }

    /// Size of the camera image buffer (used for overlaying boxes)
    var bufferSize: CGSize! {
        didSet {
            if bufferSize != nil {
                if oldValue == nil {
                    setupLayers()
                } else if oldValue != bufferSize {
                    updateDetectionOverlaySize()
                }
            }

        }
    }

    /// The last known image orientation
    /// When the image orientation changes, the buffer size used for rendering boxes needs to be adjusted
    var lastOrientation: CGImagePropertyOrientation = .right

    /// Last known dice values
    var lastDiceValues = [Int]()
    /// last observed dice
    var lastObservations = [VNRecognizedObjectObservation]()

    enum RollState {
        case other
        case started
        case ended
    }

    /// Current state of the dice roll
    var rollState = RollState.other

    override func viewDidLoad() {
        super.viewDidLoad()

        // Set the view's delegate
        sceneView.delegate = self

        // Set the session's delegate
        sceneView.session.delegate = self

        // Create a new scene
        let scene = SCNScene()

        // Set the scene to the view
        sceneView.scene = scene

        // Get the root layer so in order to draw rectangles
        rootLayer = sceneView.layer

        // Load the detection models
        /// - Tag: SetupVisionRequest

        
        guard let detector = try? VNCoreMLModel(for: Sign().model) else {
            print("Failed to load detector!")
            return
        }

        // Use a threshold provider to specify custom thresholds for the object detector.
        detector.featureProvider = ThresholdProvider()

        diceDetectionRequest = VNCoreMLRequest(model: detector) { [weak self] request, error in
            self?.detectionRequestHandler(request: request, error: error)
        }
        // .scaleFill results in a slight skew but the model was trained accordingly
        // see https://developer.apple.com/documentation/vision/vnimagecropandscaleoption for more information
        diceDetectionRequest.imageCropAndScaleOption = .scaleFill
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)

        // Disable dimming for demo
        UIApplication.shared.isIdleTimerDisabled = true

        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()

        // Run the view's session
        sceneView.session.run(configuration)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)

        // Pause the view's session
        sceneView.session.pause()
    }

    func bounds(for observation: VNRecognizedObjectObservation) -> CGRect {
        let boundingBox = observation.boundingBox
        // Coordinate system is like macOS, origin is on bottom-left and not top-left

        // The resulting bounding box from the prediction is a normalized bounding box with coordinates from bottom left
        // It needs to be flipped along the y axis
        let fixedBoundingBox = CGRect(x: boundingBox.origin.x,
                                      y: 1.0 - boundingBox.origin.y - boundingBox.height,
                                      width: boundingBox.width,
                                      height: boundingBox.height)

        // Return a flipped and scaled rectangle corresponding to the coordinates in the sceneView
        return VNImageRectForNormalizedRect(fixedBoundingBox, Int(sceneView.frame.width), Int(sceneView.frame.height))
    }

    // MARK: - Vision Callbacks

    /// Handles results from the detection requests
    ///
    /// - parameters:
    ///     - request: The VNRequest that has been processed
    ///     - error: A potential error that may have occurred
    func detectionRequestHandler(request: VNRequest, error: Error?) {
        // Perform several error checks before proceeding
        if let error = error {
            print("An error occurred with the vision request: \(error.localizedDescription)")
            return
        }
        guard let request = request as? VNCoreMLRequest else {
            print("Vision request is not a VNCoreMLRequest")
            return
        }
        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            print("Request did not return recognized objects: \(request.results?.debugDescription ?? "[No results]")")
            return
		}

        guard !observations.isEmpty else {
            removeBoxes()
            if !lastObservations.isEmpty {
                DispatchQueue.main.async {
                    self.infoLabel.text = ""
                }
            }
            lastObservations = []
            lastDiceValues = []
            // Since there are no detected dice, the roll is in .other state
            rollState = .other
            return
        }

        if showBoxes && rollState != .ended {
            drawBoxes(observations: observations)
        }

		for observation in observations {
			for label in observation.labels {
				if label.confidence > 0.95, !label.identifier.isEmpty {
					DispatchQueue.main.async {
						self.infoLabel.text = label.identifier
					}
				}
			}
		}

    }

}


extension ViewController {
    // MARK: - ARSCNViewDelegate
    func session(_ session: ARSession, didFailWithError error: Error) {
        infoLabel.text = "ARKit Error:\(error.localizedDescription)"
    }

    func sessionWasInterrupted(_ session: ARSession) {
        infoLabel.text = "ARKit session was interrupted"
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
    }

    // MARK: - ARSessionDelegate

    /// Method called when the ARSession produces a new frame
    ///
    /// - parameter frame: The ARFrame containing information about the last
    ///                    captured video frame and the state of the world
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // When rate-limiting predicitons, skip frames to predict every x
        if shouldSkipFrame > 0 {
            shouldSkipFrame = (shouldSkipFrame + 1) % predictEvery
        }

        predictionQueue.async {
            /// - Tag: MappingOrientation
            // The frame is always oriented based on the camera sensor,
            // so in most cases Vision needs to rotate it for the model to work as expected.
            let orientation = UIDevice.current.orientation

            // The image captured by the camera
            let image = frame.capturedImage

            let imageOrientation: CGImagePropertyOrientation
            switch orientation {
            case .portrait:
                imageOrientation = .right
            case .portraitUpsideDown:
                imageOrientation = .left
            case .landscapeLeft:
                imageOrientation = .up
            case .landscapeRight:
                imageOrientation = .down
            case .unknown:
               // print("The device orientation is unknown, the predictions may be affected")
                fallthrough
            default:
                // By default keep the last orientation
                // This applies for faceUp and faceDown
                imageOrientation = self.lastOrientation
            }

            // For object detection, keeping track of the image buffer size
            // to know how to draw bounding boxes based on relative values.
            if self.bufferSize == nil || self.lastOrientation != imageOrientation {
                self.lastOrientation = imageOrientation
                let pixelBufferWidth = CVPixelBufferGetWidth(image)
                let pixelBufferHeight = CVPixelBufferGetHeight(image)
                if [.up, .down].contains(imageOrientation) {
                    self.bufferSize = CGSize(width: pixelBufferWidth,
                                             height: pixelBufferHeight)
                } else {
                    self.bufferSize = CGSize(width: pixelBufferHeight,
                                             height: pixelBufferWidth)
                }
            }


            /// - Tag: PassingFramesToVision

            // Invoke a VNRequestHandler with that image
            let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: imageOrientation, options: [:])

            do {
                try handler.perform([self.diceDetectionRequest])
            } catch {
                print("CoreML request failed with error: \(error.localizedDescription)")
            }

        }
    }
}

extension ViewController {
    /// Sets up CALayers for rendering bounding boxes
    func setupLayers() {
        DispatchQueue.main.async {
            self.detectionOverlay = CALayer() // container layer that has all the renderings of the observations
            self.detectionOverlay.name = "DetectionOverlay"
            self.detectionOverlay.bounds = CGRect(x: 0.0,
                                                  y: 0.0,
                                                  width: self.sceneView.frame.width,
                                                  height: self.sceneView.frame.height)
            self.detectionOverlay.position = CGPoint(x: self.rootLayer.bounds.midX,
                                                     y: self.rootLayer.bounds.midY)
            self.rootLayer.addSublayer(self.detectionOverlay)
        }
    }

    /// Update the size of the overlay layer if the sceneView size changed
    func updateDetectionOverlaySize() {
        DispatchQueue.main.async {
            self.detectionOverlay.bounds = CGRect(x: 0.0,
                                                  y: 0.0,
                                                  width: self.sceneView.frame.width,
                                                  height: self.sceneView.frame.height)
        }
    }

    /// Update layer geometry when needed
    func updateLayerGeometry() {
        DispatchQueue.main.async {
            let bounds = self.rootLayer.bounds
            var scale: CGFloat

            let xScale: CGFloat = bounds.size.width / self.sceneView.frame.height
            let yScale: CGFloat = bounds.size.height / self.sceneView.frame.width

            scale = fmax(xScale, yScale)
            if scale.isInfinite {
                scale = 1.0
            }
            CATransaction.begin()
            CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)

            self.detectionOverlay.position = CGPoint(x: bounds.midX, y: bounds.midY)

            CATransaction.commit()
        }
    }

    /// Creates a text layer to display the label for the given box
    ///
    /// - parameters:
    ///     - bounds: Bounds of the detected object
    ///     - identifier: Class label for the detected object
    ///     - confidence: Confidence in the prediction
    /// - returns: A newly created CATextLayer
    func createTextSubLayerInBounds(_ bounds: CGRect, identifier: String) -> CATextLayer {
        let textLayer = CATextLayer()
        textLayer.name = "Object Label"
        let attributedString = NSMutableAttributedString(string: "\(identifier)")
        let largeFont = UIFont(name: "Menlo", size: bounds.height * 0.7)!
        let attributes = [NSAttributedString.Key.font: largeFont,
                          NSAttributedString.Key.foregroundColor: UIColor.white]
        attributedString.addAttributes(attributes,
                                       range: NSRange(location: 0, length: identifier.count))
        textLayer.string = attributedString
        textLayer.bounds = CGRect(x: 0, y: 0, width: bounds.size.height, height: bounds.size.width)
        textLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        textLayer.shadowOpacity = 0.0
        textLayer.foregroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [1.0, 1.0, 1.0, 1.0])
        textLayer.contentsScale = 2.0 // retina rendering
        return textLayer
    }

    /// Creates a reounded rectangle layer with the given bounds
    /// - parameter bounds: The bounds of the rectangle
    /// - returns: A newly created CALayer
    func createRoundedRectLayerWithBounds(_ bounds: CGRect) -> CALayer {
        let shapeLayer = CALayer()
        shapeLayer.bounds = bounds
        shapeLayer.position = CGPoint(x: bounds.midX, y: bounds.midY)
        shapeLayer.name = "Found Object"
        shapeLayer.backgroundColor = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [0.0, 0.8, 1.0, 0.6])
        shapeLayer.cornerRadius = 14
        return shapeLayer
    }

    /// Removes all bounding boxes from the screen
    func removeBoxes() {
        drawBoxes(observations: [])
    }

    /// Draws bounding boxes based on the object observations
    ///
    /// - parameter observations: The list of object observations from the object detector
    func drawBoxes(observations: [VNRecognizedObjectObservation]) {
        DispatchQueue.main.async {
            CATransaction.begin()
            CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
            self.detectionOverlay.sublayers = nil // remove all the old recognized objects

            for observation in observations {

                // Select only the label with the highest confidence.
                guard let topLabel = observation.labels.first?.identifier else {
                    print("Object observation has no labels")
                    continue
                }

                let objectBounds = self.bounds(for: observation)

                let shapeLayer = self.createRoundedRectLayerWithBounds(objectBounds)
                let textLayer = self.createTextSubLayerInBounds(objectBounds,
                                                                identifier: topLabel)
                shapeLayer.addSublayer(textLayer)
                self.detectionOverlay.addSublayer(shapeLayer)
            }

            self.updateLayerGeometry()
            CATransaction.commit()
        }
    }
}


/// - Tag: ThresholdProvider
/// Class providing customized thresholds for object detection model
class ThresholdProvider: MLFeatureProvider {
    /// The actual values to provide as input
    ///
    /// Create ML Defaults are 0.45 for IOU and 0.25 for confidence.
    /// Here the IOU threshold is relaxed a little bit because there are
    /// sometimes multiple overlapping boxes per die.
    /// Technically, relaxing the IOU threshold means
    /// non-maximum-suppression (NMS) becomes stricter (fewer boxes are shown).
    /// The confidence threshold can also be relaxed slightly because
    /// objects look very consistent and are easily detected on a homogeneous
    /// background.
    open var values = [
        "iouThreshold": MLFeatureValue(double: 0.3),
        "confidenceThreshold": MLFeatureValue(double: 0.2)
    ]

    /// The feature names the provider has, per the MLFeatureProvider protocol
    var featureNames: Set<String> {
        return Set(values.keys)
    }

    /// The actual values for the features the provider can provide
    func featureValue(for featureName: String) -> MLFeatureValue? {
        return values[featureName]
    }
}

