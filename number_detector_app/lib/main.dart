import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv2;

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: DigitRecognizer(),
    );
  }
}

class DigitRecognizer extends StatefulWidget {
  @override
  _DigitRecognizerState createState() => _DigitRecognizerState();
}

class _DigitRecognizerState extends State<DigitRecognizer> {
  late Interpreter interpreter;
  String prediction = "No Prediction";

  @override
  void initState() {
    super.initState();
    loadModel();
    recognizeDigit();
  }

  /// Load TFLite model
  Future<void> loadModel() async {
    try {
      interpreter = await Interpreter.fromAsset('assets/best_model.tflite');
      print("Model loaded successfully");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  /// Recognize digit from asset image
  Future<void> recognizeDigit() async {
    try {
      // Load and preprocess the image
      ByteData imageData = await rootBundle.load('assets/zero.jpg');
      Uint8List imageBytes = imageData.buffer.asUint8List();
      List<List<List<double>>> input =
          await preprocessImage(imageBytes, 28, 28);

      // Allocate output tensor
      var output = List.filled(10, 0.0).reshape([1, 10]);

      // Run inference
      interpreter.run(input, output);

      // Find the index of the highest probability
      int predictedDigit = output[0].indexWhere(
          (value) => value == output[0].reduce((a, b) => a > b ? a : b));

      setState(() {
        prediction = "Digit: $predictedDigit";
      });
    } catch (e) {
      print("Error recognizing digit: $e");
    }
  }

  /// Preprocess the image: resize, grayscale, normalize
  Future<List<List<List<double>>>> preprocessImage(
      Uint8List imageBytes, int targetWidth, int targetHeight) async {
    ui.Codec codec = await ui.instantiateImageCodec(imageBytes,
        targetWidth: targetWidth, targetHeight: targetHeight);
    ui.FrameInfo frameInfo = await codec.getNextFrame();
    ui.Image image = frameInfo.image;

    ByteData? byteData =
        await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null)
      throw Exception("Unable to convert image to byte data");

    Uint8List rgbaBytes = byteData.buffer.asUint8List();
    List<List<List<double>>> input = List.generate(
        targetHeight, (_) => List.generate(targetWidth, (_) => [0.0]));

    for (int y = 0; y < targetHeight; y++) {
      for (int x = 0; x < targetWidth; x++) {
        int index = (y * targetWidth + x) * 4;
        int r = rgbaBytes[index];
        int g = rgbaBytes[index + 1];
        int b = rgbaBytes[index + 2];

        // Grayscale value = (0.3 * R + 0.59 * G + 0.11 * B)
        double grayscale = (0.3 * r + 0.59 * g + 0.11 * b) / 255.0;
        input[y][x][0] = grayscale; // Model expects normalized grayscale values
      }
    }

    return input;
  }

  @override
  void dispose() {
    interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Digit Recognizer'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('Prediction Result:'),
            const SizedBox(height: 16),
            Text(prediction, style: const TextStyle(fontSize: 20)),
          ],
        ),
      ),
    );
  }
}
