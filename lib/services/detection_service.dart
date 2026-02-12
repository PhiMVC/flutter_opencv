import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection.dart';

class DetectionResult {
  const DetectionResult({
    required this.detections,
    required this.inferenceMs,
    required this.imageSize,
  });

  final List<Detection> detections;
  final int inferenceMs;
  final Size imageSize;
}

class DetectionService {
  DetectionService({
    required this.modelAsset,
    this.threads = 2,
    this.scoreThreshold = 0.4,
    this.nmsThreshold = 0.45,
    this.maxDetections = 20,
    this.enableLogs = true,
    this.logEveryN = 1,
    this.logSampleSize = 8,
  });

  final String modelAsset;
  final int threads;
  final double scoreThreshold;
  final double nmsThreshold;
  final int maxDetections;
  final bool enableLogs;
  final int logEveryN;
  final int logSampleSize;

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  List<Tensor> _outputTensors = const [];
  Map<int, Object>? _outputBuffers;
  TensorType? _inputType;
  int _inputWidth = 0;
  int _inputHeight = 0;
  int _inputChannels = 3;
  Float32List? _inputBufferF32;
  Uint8List? _inputBufferU8;
  Uint8List? _rgbBuffer;
  bool _isBusy = false;
  String? _modelError;
  bool _loggedOutputSample = false;
  int _inferenceCount = 0;

  bool get isBusy => _isBusy;
  bool get isReady =>
      _interpreter != null &&
      _outputBuffers != null &&
      _inputWidth > 0 &&
      _inputHeight > 0 &&
      _inputChannels == 3;
  String? get modelError => _modelError;

  Future<void> load() async {
    _log('Loading model: $modelAsset');
    final options = InterpreterOptions()..threads = threads;
    final interpreter = await Interpreter.fromAsset(
      modelAsset,
      options: options,
    );
    IsolateInterpreter? isolate;
    try {
      isolate = await IsolateInterpreter.create(address: interpreter.address);
    } catch (_) {
      isolate = null;
    }

    final inputTensor = _getInputTensor(interpreter);
    final inputShape = List<int>.from(inputTensor.shape);
    final (inputW, inputH, inputC) = _parseInputShape(inputShape);
    final outputTensors = _getOutputTensors(interpreter);
    _log(
      'Input tensor shape=$inputShape type=${inputTensor.type} '
      'params=${_formatParams(inputTensor)}',
    );
    for (var i = 0; i < outputTensors.length; i++) {
      final t = outputTensors[i];
      _log(
        'Output[$i] shape=${t.shape} type=${t.type} '
        'params=${_formatParams(t)}',
      );
    }

    final previousIsolate = _isolateInterpreter;
    _isolateInterpreter = isolate;
    _interpreter?.close();
    _interpreter = interpreter;
    _inputType = inputTensor.type;
    _inputWidth = inputW;
    _inputHeight = inputH;
    _inputChannels = inputC;
    _outputTensors = outputTensors;
    _outputBuffers = _allocateOutputBuffers(outputTensors);
    _inputBufferF32 = null;
    _inputBufferU8 = null;
    _modelError = inputC == 3 ? null : 'Model input expects $inputC channels';
    _loggedOutputSample = false;
    if (_modelError != null) {
      _log('Model error: $_modelError');
    }

    await previousIsolate?.close();
  }

  Future<DetectionResult?> run(CameraImage image) async {
    if (!isReady || _isBusy) {
      return null;
    }

    _isBusy = true;
    cv.Mat? srcMat;
    cv.Mat? inputMat;
    try {
      final inputW = _inputWidth;
      final inputH = _inputHeight;
      if (inputW <= 0 || inputH <= 0) {
        return null;
      }

      final start = DateTime.now();
      final rgb = _convertCameraImageToRgb(image);
      srcMat = cv.Mat.fromList(
        image.height,
        image.width,
        cv.MatType.CV_8UC3,
        rgb,
      );

      final (letterbox, mat) = _letterboxAndResize(srcMat, inputW, inputH);
      inputMat = mat;
      final inputBytes = inputMat.data;
      final inputTensor = _buildInputTensor(inputBytes, inputW, inputH);

      final outputBuffers = _outputBuffers;
      if (outputBuffers == null) {
        _log('Output buffers not allocated.');
        return null;
      }

      _inferenceCount++;
      if (_shouldLogSample()) {
        _log(
          'Run #$_inferenceCount input type=$_inputType '
          'shape=[1,$inputH,$inputW,3] bytes=${inputBytes.length}',
        );
        final headSample = _sampleOutput(inputTensor, logSampleSize);
        _log('Input head sample=${headSample.isEmpty ? "[]" : headSample}');
        final centerSample = _sampleInputCenter(inputW, inputH, logSampleSize);
        _log('Input center sample=${centerSample.isEmpty ? "[]" : centerSample}');
      }

      final isolate = _isolateInterpreter;
      if (isolate != null) {
        await isolate.runForMultipleInputs([inputTensor], outputBuffers);
      } else {
        _interpreter!.runForMultipleInputs([inputTensor], outputBuffers);
      }

      final detections = _parseDetections(outputBuffers, letterbox);
      if (_shouldLogSample()) {
        _loggedOutputSample = true;
        for (var i = 0; i < _outputTensors.length; i++) {
          final output = outputBuffers[i];
          if (output == null) continue;
          final sample = _sampleOutput(output, 8);
          _log('Output[$i] sample=${sample.isEmpty ? "[]" : sample}');
        }
      }
      final elapsedMs = DateTime.now().difference(start).inMilliseconds;
      _modelError = null;

      return DetectionResult(
        detections: detections,
        inferenceMs: elapsedMs,
        imageSize: Size(image.width.toDouble(), image.height.toDouble()),
      );
    } catch (e) {
      _modelError = 'Inference error: $e';
      return null;
    } finally {
      inputMat?.dispose();
      srcMat?.dispose();
      _isBusy = false;
    }
  }

  void dispose() {
    _isolateInterpreter?.close();
    _interpreter?.close();
    _isolateInterpreter = null;
    _interpreter = null;
  }

  Tensor _getInputTensor(Interpreter interpreter) {
    final dynamic dyn = interpreter;
    return dyn.getInputTensor(0) as Tensor;
  }

  void _log(String message) {
    if (!enableLogs) return;
    debugPrint('[DetectionService] $message');
  }

  String _formatParams(Tensor tensor) {
    final params = tensor.params;
    return 'scale=${params.scale}, zeroPoint=${params.zeroPoint}';
  }

  bool _shouldLogSample() {
    if (!enableLogs) return false;
    if (logEveryN <= 0) {
      return !_loggedOutputSample;
    }
    return _inferenceCount % logEveryN == 0;
  }

  List<num> _sampleOutput(Object output, int maxCount) {
    final result = <num>[];
    void visit(dynamic value) {
      if (result.length >= maxCount) return;
      if (value is num) {
        result.add(value);
        return;
      }
      if (value is List) {
        for (final entry in value) {
          if (result.length >= maxCount) break;
          visit(entry);
        }
      }
    }

    visit(output);
    return result;
  }

  List<num> _sampleInputCenter(int inputW, int inputH, int maxCount) {
    final channels = _inputChannels;
    if (channels <= 0 || inputW <= 0 || inputH <= 0) {
      return const [];
    }
    final centerOffset =
        ((inputH ~/ 2) * inputW + (inputW ~/ 2)) * channels;
    final inputType = _inputType;
    if (inputType == TensorType.uint8) {
      final buffer = _inputBufferU8;
      return _sampleFlatBuffer(buffer, centerOffset, maxCount);
    }
    final buffer = _inputBufferF32;
    return _sampleFlatBuffer(buffer, centerOffset, maxCount);
  }

  List<num> _sampleFlatBuffer(List<num>? buffer, int start, int maxCount) {
    if (buffer == null || buffer.isEmpty || start < 0) {
      return const [];
    }
    final result = <num>[];
    final end = (start + maxCount).clamp(0, buffer.length);
    for (var i = start; i < end; i++) {
      result.add(buffer[i]);
    }
    return result;
  }

  List<Tensor> _getOutputTensors(Interpreter interpreter) {
    final dynamic dyn = interpreter;
    try {
      final tensors = dyn.getOutputTensors() as List;
      return tensors.cast<Tensor>();
    } catch (_) {
      final count = _getOutputTensorCount(dyn);
      return List.generate(
        count,
        (index) => dyn.getOutputTensor(index) as Tensor,
      );
    }
  }

  int _getOutputTensorCount(dynamic interpreter) {
    try {
      return interpreter.getOutputTensorCount() as int;
    } catch (_) {
      try {
        return (interpreter.getOutputTensors() as List).length;
      } catch (_) {
        return 1;
      }
    }
  }

  Map<int, Object> _allocateOutputBuffers(List<Tensor> outputs) {
    final buffers = <int, Object>{};
    for (var i = 0; i < outputs.length; i++) {
      buffers[i] = _allocateTensorBuffer(outputs[i]);
    }
    return buffers;
  }

  Object _allocateTensorBuffer(Tensor tensor) {
    final shape = tensor.shape;
    final total = shape.fold<int>(1, (a, b) => a * b);
    switch (tensor.type) {
      case TensorType.float32:
        return Float32List(total).reshape(shape);
      case TensorType.uint8:
        return Uint8List(total).reshape(shape);
      case TensorType.int8:
        return Int8List(total).reshape(shape);
      case TensorType.int32:
        return Int32List(total).reshape(shape);
      default:
        return List<double>.filled(total, 0.0).reshape(shape);
    }
  }

  (int, int, int) _parseInputShape(List<int> shape) {
    if (shape.length == 4) {
      if (shape[3] == 3) {
        return (shape[2], shape[1], shape[3]);
      }
      if (shape[1] == 3) {
        return (shape[3], shape[2], shape[1]);
      }
      return (shape[2], shape[1], shape[3]);
    }
    if (shape.length == 3) {
      if (shape[2] == 3) {
        return (shape[1], shape[0], shape[2]);
      }
      return (shape[1], shape[0], shape[2]);
    }
    return (0, 0, 0);
  }

  Object _buildInputTensor(Uint8List inputBytes, int inputW, int inputH) {
    final inputType = _inputType;
    final inputSize = inputW * inputH * _inputChannels;
    if (inputType == TensorType.uint8) {
      var buffer = _inputBufferU8;
      if (buffer == null || buffer.length != inputSize) {
        buffer = Uint8List(inputSize);
        _inputBufferU8 = buffer;
      }
      final copyLength =
          inputBytes.length < inputSize ? inputBytes.length : inputSize;
      buffer.setRange(0, copyLength, inputBytes);
      if (copyLength < inputSize) {
        buffer.fillRange(copyLength, inputSize, 0);
      }
      return buffer.reshape([1, inputH, inputW, 3]);
    }

    var buffer = _inputBufferF32;
    if (buffer == null || buffer.length != inputSize) {
      buffer = Float32List(inputSize);
      _inputBufferF32 = buffer;
    }
    final scale = 1.0 / 255.0;
    final copyLength =
        inputBytes.length < inputSize ? inputBytes.length : inputSize;
    for (var i = 0; i < copyLength; i++) {
      buffer[i] = inputBytes[i] * scale;
    }
    if (copyLength < inputSize) {
      for (var i = copyLength; i < inputSize; i++) {
        buffer[i] = 0;
      }
    }
    return buffer.reshape([1, inputH, inputW, 3]);
  }

  (_LetterboxInfo, cv.Mat) _letterboxAndResize(
    cv.Mat src,
    int inputW,
    int inputH,
  ) {
    final srcW = src.width;
    final srcH = src.height;
    if (srcW == 0 || srcH == 0) {
      final empty = cv.Mat.zeros(inputH, inputW, src.type);
      return (
        _LetterboxInfo(
          scale: 1,
          dx: 0,
          dy: 0,
          inputWidth: inputW,
          inputHeight: inputH,
          srcWidth: srcW,
          srcHeight: srcH,
        ),
        empty,
      );
    }

    final scale = math.min(inputW / srcW, inputH / srcH);
    final resizedW = (srcW * scale).round();
    final resizedH = (srcH * scale).round();
    final resized = cv.resize(src, (resizedW, resizedH));

    final canvas = cv.Mat.zeros(inputH, inputW, src.type);
    final dx = ((inputW - resizedW) / 2).round();
    final dy = ((inputH - resizedH) / 2).round();
    final roiRect = cv.Rect(dx, dy, resizedW, resizedH);
    final roi = canvas.region(roiRect);
    resized.copyTo(roi);
    roi.dispose();
    roiRect.dispose();
    resized.dispose();

    return (
      _LetterboxInfo(
        scale: scale,
        dx: dx.toDouble(),
        dy: dy.toDouble(),
        inputWidth: inputW,
        inputHeight: inputH,
        srcWidth: srcW,
        srcHeight: srcH,
      ),
      canvas,
    );
  }

  Uint8List _convertCameraImageToRgb(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final needed = width * height * 3;
    var rgb = _rgbBuffer;
    if (rgb == null || rgb.length != needed) {
      rgb = Uint8List(needed);
      _rgbBuffer = rgb;
    }
    final yPlane = image.planes[0];
    final uPlane = image.planes.length > 1 ? image.planes[1] : null;
    final vPlane = image.planes.length > 2 ? image.planes[2] : null;

    final yRowStride = yPlane.bytesPerRow;
    final uvRowStride = uPlane?.bytesPerRow ?? 0;
    final uvPixelStride = uPlane?.bytesPerPixel ?? 1;

    var index = 0;
    for (var y = 0; y < height; y++) {
      final yRow = yRowStride * y;
      final uvRow = uvRowStride * (y >> 1);
      for (var x = 0; x < width; x++) {
        final yValue = yPlane.bytes[yRow + x];
        int uValue = 0;
        int vValue = 0;
        if (uPlane != null) {
          final uvIndex = uvRow + (x >> 1) * uvPixelStride;
          if (image.planes.length == 2) {
            uValue = uPlane.bytes[uvIndex];
            vValue = uPlane.bytes[uvIndex + 1];
          } else if (vPlane != null) {
            uValue = uPlane.bytes[uvIndex];
            vValue = vPlane.bytes[uvIndex];
          }
        }

        final c = yValue - 16;
        final d = uValue - 128;
        final e = vValue - 128;
        var r = (298 * c + 409 * e + 128) >> 8;
        var g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        var b = (298 * c + 516 * d + 128) >> 8;

        r = _clampInt(r, 0, 255);
        g = _clampInt(g, 0, 255);
        b = _clampInt(b, 0, 255);

        rgb[index++] = r;
        rgb[index++] = g;
        rgb[index++] = b;
      }
    }
    return rgb;
  }

  List<Detection> _parseDetections(
    Map<int, Object> outputs,
    _LetterboxInfo letterbox,
  ) {
    final ssd = _tryParseSsd(outputs, letterbox);
    if (ssd.isNotEmpty) {
      return _nms(ssd);
    }
    final yolo = _tryParseYolo(outputs, letterbox);
    if (yolo.isNotEmpty) {
      return _nms(yolo);
    }
    return const [];
  }

  List<Detection> _tryParseSsd(
    Map<int, Object> outputs,
    _LetterboxInfo letterbox,
  ) {
    int? boxesIdx;
    final twoD = <int>[];
    for (var i = 0; i < _outputTensors.length; i++) {
      final shape = _outputTensors[i].shape;
      if (shape.length == 3 && shape.last == 4) {
        boxesIdx = i;
      } else if (shape.length == 2) {
        twoD.add(i);
      }
    }
    if (boxesIdx == null || twoD.length < 2) {
      return const [];
    }

    final classesIdx = twoD.first;
    final scoresIdx = twoD.length > 1 ? twoD[1] : twoD.first;
    final boxes = outputs[boxesIdx] as List;
    final classes = outputs[classesIdx] as List;
    final scores = outputs[scoresIdx] as List;
    if (boxes.isEmpty || classes.isEmpty || scores.isEmpty) {
      return const [];
    }

    final boxesList = boxes[0] as List;
    final classesList = classes[0] as List;
    final scoresList = scores[0] as List;

    final detections = <Detection>[];
    final count = math.min(
      boxesList.length,
      math.min(scoresList.length, classesList.length),
    );
    for (var i = 0; i < count; i++) {
      final score = (scoresList[i] as num).toDouble();
      if (score < scoreThreshold) {
        continue;
      }
      final box = boxesList[i] as List;
      final ymin = (box[0] as num).toDouble();
      final xmin = (box[1] as num).toDouble();
      final ymax = (box[2] as num).toDouble();
      final xmax = (box[3] as num).toDouble();

      final rectInput = Rect.fromLTRB(
        xmin * letterbox.inputWidth,
        ymin * letterbox.inputHeight,
        xmax * letterbox.inputWidth,
        ymax * letterbox.inputHeight,
      );
      final rect = _mapToOriginal(rectInput, letterbox);
      final classId = (classesList[i] as num).round();
      detections.add(Detection(rect: rect, score: score, classId: classId));
    }
    return detections;
  }

  List<Detection> _tryParseYolo(
    Map<int, Object> outputs,
    _LetterboxInfo letterbox,
  ) {
    final outputIdx = _selectYoloOutputIndex();
    if (outputIdx == null) {
      return const [];
    }
    _log('YOLO output index=$outputIdx shape=${_outputTensors[outputIdx].shape}');

    final output = outputs[outputIdx];
    if (output is! List || output.isEmpty) {
      return const [];
    }

    final rows = output[0] as List;
    if (rows.isEmpty) {
      return const [];
    }

    final firstRow = rows[0] as List;
    if (firstRow.isEmpty) {
      return const [];
    }

    final isChannelFirst = rows.length < firstRow.length && rows.length <= 256;
    if (isChannelFirst) {
      _log('YOLO output detected as channel-first (C,N). Transposing.');
      return _parseYoloChannelFirst(rows, letterbox);
    }

    final cols = firstRow.length;
    if (cols < 5) {
      return const [];
    }

    final format = _detectYoloBoxFormatRows(
      rows,
      letterbox.inputWidth,
      letterbox.inputHeight,
    );
    _log('YOLO box format: ${format == _YoloBoxFormat.xyxy ? "xyxy" : "xywh"}');

    final detections = <Detection>[];
    for (final row in rows) {
      final values = row as List;
      if (values.length < 5) {
        continue;
      }
      final x = (values[0] as num).toDouble();
      final y = (values[1] as num).toDouble();
      final w = (values[2] as num).toDouble();
      final h = (values[3] as num).toDouble();

      double score;
      int classId;
      if (values.length > 6) {
        final objectness = (values[4] as num).toDouble();
        var maxClass = 0.0;
        var maxIdx = 0;
        for (var i = 5; i < values.length; i++) {
          final prob = (values[i] as num).toDouble();
          if (prob > maxClass) {
            maxClass = prob;
            maxIdx = i - 5;
          }
        }
        score = objectness * maxClass;
        classId = maxIdx;
      } else if (values.length == 6) {
        score = (values[4] as num).toDouble();
        classId = (values[5] as num).round();
      } else {
        score = (values[4] as num).toDouble();
        classId = 0;
      }

      if (score < scoreThreshold) {
        continue;
      }

      final normalized = x <= 1.5 && y <= 1.5 && w <= 1.5 && h <= 1.5;
      final scaleX = normalized ? letterbox.inputWidth : 1.0;
      final scaleY = normalized ? letterbox.inputHeight : 1.0;
      final rectInput =
          format == _YoloBoxFormat.xyxy
              ? Rect.fromLTRB(
                x * scaleX,
                y * scaleY,
                w * scaleX,
                h * scaleY,
              )
              : Rect.fromLTRB(
                x * scaleX - (w * scaleX) / 2,
                y * scaleY - (h * scaleY) / 2,
                x * scaleX + (w * scaleX) / 2,
                y * scaleY + (h * scaleY) / 2,
              );
      final rect = _mapToOriginal(rectInput, letterbox);
      detections.add(Detection(rect: rect, score: score, classId: classId));
    }

    return detections;
  }

  List<Detection> _parseYoloChannelFirst(
    List rows,
    _LetterboxInfo letterbox,
  ) {
    final channels = rows.length;
    if (channels < 5) {
      return const [];
    }
    final firstRow = rows[0] as List;
    if (firstRow.isEmpty) {
      return const [];
    }
    final count = firstRow.length;
    final format = _detectYoloBoxFormatChannelFirst(
      rows,
      letterbox.inputWidth,
      letterbox.inputHeight,
    );
    _log('YOLO box format: ${format == _YoloBoxFormat.xyxy ? "xyxy" : "xywh"}');

    final detections = <Detection>[];

    for (var i = 0; i < count; i++) {
      final x = (rows[0][i] as num).toDouble();
      final y = (rows[1][i] as num).toDouble();
      final w = (rows[2][i] as num).toDouble();
      final h = (rows[3][i] as num).toDouble();

      double score;
      int classId;
      if (channels > 6) {
        final objectness = (rows[4][i] as num).toDouble();
        var maxClass = 0.0;
        var maxIdx = 0;
        for (var c = 5; c < channels; c++) {
          final prob = (rows[c][i] as num).toDouble();
          if (prob > maxClass) {
            maxClass = prob;
            maxIdx = c - 5;
          }
        }
        score = objectness * maxClass;
        classId = maxIdx;
      } else if (channels == 6) {
        score = (rows[4][i] as num).toDouble();
        classId = (rows[5][i] as num).round();
      } else {
        score = (rows[4][i] as num).toDouble();
        classId = 0;
      }

      if (score < scoreThreshold) {
        continue;
      }

      final normalized = x <= 1.5 && y <= 1.5 && w <= 1.5 && h <= 1.5;
      final scaleX = normalized ? letterbox.inputWidth : 1.0;
      final scaleY = normalized ? letterbox.inputHeight : 1.0;
      final rectInput =
          format == _YoloBoxFormat.xyxy
              ? Rect.fromLTRB(
                x * scaleX,
                y * scaleY,
                w * scaleX,
                h * scaleY,
              )
              : Rect.fromLTRB(
                x * scaleX - (w * scaleX) / 2,
                y * scaleY - (h * scaleY) / 2,
                x * scaleX + (w * scaleX) / 2,
                y * scaleY + (h * scaleY) / 2,
              );
      final rect = _mapToOriginal(rectInput, letterbox);
      detections.add(Detection(rect: rect, score: score, classId: classId));
    }

    return detections;
  }

  int? _selectYoloOutputIndex() {
    var bestScore = -1.0;
    int? bestIdx;
    for (var i = 0; i < _outputTensors.length; i++) {
      final shape = _outputTensors[i].shape;
      if (shape.length != 3) {
        continue;
      }
      final score = _scoreYoloShape(shape);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  double _scoreYoloShape(List<int> shape) {
    if (shape.length != 3) {
      return -1;
    }
    final a = shape[1];
    final b = shape[2];
    var score = -1.0;
    if (b >= 5 && b <= 512 && a >= 10) {
      score = math.max(score, a + b / 1000.0);
    }
    if (a >= 5 && a <= 512 && b >= 10) {
      score = math.max(score, b + a / 1000.0);
    }
    return score;
  }

  _YoloBoxFormat _detectYoloBoxFormatRows(
    List rows,
    int inputW,
    int inputH,
  ) {
    var xyxyScore = 0;
    var xywhScore = 0;
    final sampleCount = math.min(rows.length, 20);
    for (var i = 0; i < sampleCount; i++) {
      final row = rows[i] as List;
      if (row.length < 4) continue;
      final x = (row[0] as num).toDouble();
      final y = (row[1] as num).toDouble();
      final a = (row[2] as num).toDouble();
      final b = (row[3] as num).toDouble();
      final normalized = x <= 1.5 && y <= 1.5 && a <= 1.5 && b <= 1.5;
      final scaleX = normalized ? inputW : 1.0;
      final scaleY = normalized ? inputH : 1.0;

      final xyxyValid = _validBox(
        x * scaleX,
        y * scaleY,
        a * scaleX,
        b * scaleY,
        inputW,
        inputH,
      );
      if (xyxyValid) xyxyScore++;

      final left = x * scaleX - (a * scaleX) / 2;
      final top = y * scaleY - (b * scaleY) / 2;
      final right = x * scaleX + (a * scaleX) / 2;
      final bottom = y * scaleY + (b * scaleY) / 2;
      final xywhValid = _validBox(left, top, right, bottom, inputW, inputH);
      if (xywhValid) xywhScore++;
    }
    return xyxyScore >= xywhScore
        ? _YoloBoxFormat.xyxy
        : _YoloBoxFormat.xywh;
  }

  _YoloBoxFormat _detectYoloBoxFormatChannelFirst(
    List rows,
    int inputW,
    int inputH,
  ) {
    if (rows.length < 4) {
      return _YoloBoxFormat.xywh;
    }
    final firstRow = rows[0] as List;
    if (firstRow.isEmpty) {
      return _YoloBoxFormat.xywh;
    }
    var xyxyScore = 0;
    var xywhScore = 0;
    final sampleCount = math.min(firstRow.length, 20);
    for (var i = 0; i < sampleCount; i++) {
      final x = (rows[0][i] as num).toDouble();
      final y = (rows[1][i] as num).toDouble();
      final a = (rows[2][i] as num).toDouble();
      final b = (rows[3][i] as num).toDouble();
      final normalized = x <= 1.5 && y <= 1.5 && a <= 1.5 && b <= 1.5;
      final scaleX = normalized ? inputW : 1.0;
      final scaleY = normalized ? inputH : 1.0;

      final xyxyValid = _validBox(
        x * scaleX,
        y * scaleY,
        a * scaleX,
        b * scaleY,
        inputW,
        inputH,
      );
      if (xyxyValid) xyxyScore++;

      final left = x * scaleX - (a * scaleX) / 2;
      final top = y * scaleY - (b * scaleY) / 2;
      final right = x * scaleX + (a * scaleX) / 2;
      final bottom = y * scaleY + (b * scaleY) / 2;
      final xywhValid = _validBox(left, top, right, bottom, inputW, inputH);
      if (xywhValid) xywhScore++;
    }
    return xyxyScore >= xywhScore
        ? _YoloBoxFormat.xyxy
        : _YoloBoxFormat.xywh;
  }

  bool _validBox(
    double left,
    double top,
    double right,
    double bottom,
    int inputW,
    int inputH,
  ) {
    if (left.isNaN || top.isNaN || right.isNaN || bottom.isNaN) {
      return false;
    }
    if (right <= left || bottom <= top) {
      return false;
    }
    if (right < 0 || bottom < 0) {
      return false;
    }
    if (left > inputW * 2 || top > inputH * 2) {
      return false;
    }
    return true;
  }

  Rect _mapToOriginal(Rect rect, _LetterboxInfo info) {
    if (info.scale == 0) {
      return Rect.zero;
    }
    final left = (rect.left - info.dx) / info.scale;
    final top = (rect.top - info.dy) / info.scale;
    final right = (rect.right - info.dx) / info.scale;
    final bottom = (rect.bottom - info.dy) / info.scale;

    return Rect.fromLTRB(
      _clampDouble(left, 0, info.srcWidth.toDouble()),
      _clampDouble(top, 0, info.srcHeight.toDouble()),
      _clampDouble(right, 0, info.srcWidth.toDouble()),
      _clampDouble(bottom, 0, info.srcHeight.toDouble()),
    );
  }

  List<Detection> _nms(List<Detection> detections) {
    if (detections.length <= 1) {
      return detections;
    }
    detections.sort((a, b) => b.score.compareTo(a.score));
    final selected = <Detection>[];
    final active = List<bool>.filled(detections.length, true);

    for (var i = 0; i < detections.length; i++) {
      if (!active[i]) {
        continue;
      }
      final det = detections[i];
      selected.add(det);
      if (selected.length >= maxDetections) {
        break;
      }
      for (var j = i + 1; j < detections.length; j++) {
        if (!active[j]) {
          continue;
        }
        final other = detections[j];
        if (_iou(det.rect, other.rect) > nmsThreshold) {
          active[j] = false;
        }
      }
    }
    return selected;
  }

  double _iou(Rect a, Rect b) {
    final left = math.max(a.left, b.left);
    final top = math.max(a.top, b.top);
    final right = math.min(a.right, b.right);
    final bottom = math.min(a.bottom, b.bottom);
    final intersection = (right - left) * (bottom - top);
    if (intersection <= 0) {
      return 0;
    }
    final union = a.width * a.height + b.width * b.height - intersection;
    if (union <= 0) {
      return 0;
    }
    return intersection / union;
  }

  int _clampInt(int value, int min, int max) {
    if (value < min) {
      return min;
    }
    if (value > max) {
      return max;
    }
    return value;
  }

  double _clampDouble(double value, double min, double max) {
    if (value.isNaN || value.isInfinite) {
      return min;
    }
    if (value < min) {
      return min;
    }
    if (value > max) {
      return max;
    }
    return value;
  }
}

class _LetterboxInfo {
  const _LetterboxInfo({
    required this.scale,
    required this.dx,
    required this.dy,
    required this.inputWidth,
    required this.inputHeight,
    required this.srcWidth,
    required this.srcHeight,
  });

  final double scale;
  final double dx;
  final double dy;
  final int inputWidth;
  final int inputHeight;
  final int srcWidth;
  final int srcHeight;
}

enum _YoloBoxFormat { xywh, xyxy }
