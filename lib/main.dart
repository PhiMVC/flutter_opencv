import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'models/detection.dart';
import 'models/metrics.dart';
import 'services/detection_service.dart';
import 'services/image_embedding_service.dart';
import 'services/image_iou_service.dart';
import 'services/metrics_service.dart';
import 'services/temp_gallery_service.dart';
import 'widgets/detection_painter.dart';
import 'widgets/axes_overlay.dart';
import 'widgets/metrics_panel.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const CameraDemoApp());
}

class CameraDemoApp extends StatefulWidget {
  const CameraDemoApp({super.key});

  @override
  State<CameraDemoApp> createState() => _CameraDemoAppState();
}

class _CameraDemoAppState extends State<CameraDemoApp> {
  List<CameraDescription> _cameras = const [];
  String? _error;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    if (_cameras.isEmpty) {
      _loadCameras();
    }
  }

  Future<void> _loadCameras() async {
    try {
      final cameras = await availableCameras();
      if (!mounted) {
        return;
      }
      setState(() {
        _cameras = cameras;
        _loading = false;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = 'Camera error: $e';
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    Widget home;
    if (_loading) {
      home = const Scaffold(body: Center(child: CircularProgressIndicator()));
    } else if (_error != null) {
      home = Scaffold(
        body: Center(
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.white),
            textAlign: TextAlign.center,
          ),
        ),
      );
    } else {
      home = CameraHome(cameras: _cameras);
    }

    return MaterialApp(
      title: 'Camera Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(scaffoldBackgroundColor: Colors.black),
      home: home,
    );
  }
}

class CameraHome extends StatefulWidget {
  const CameraHome({super.key, required this.cameras});

  final List<CameraDescription> cameras;

  @override
  State<CameraHome> createState() => _CameraHomeState();
}

class _CameraHomeState extends State<CameraHome> {
  static const bool _enableProfiling = false;
  static const int _profileEveryN = 30;
  static const bool _enableVerboseLogs = false;
  static const bool _useBackgroundPreprocess = true;
  static const bool _useBackgroundMetrics = true;
  static const ResolutionPreset _cameraResolution = ResolutionPreset.low;
  static const int _minInferenceIntervalMs = 100;
  static const int _metricsMaxDim = 240;
  static const Duration _metricsInterval = Duration(milliseconds: 250);
  static const Duration _detectionHold = Duration(milliseconds: 250);
  static const double _scoreThreshold = 0.2;
  static const double _nmsThreshold = 0.9;
  static const int _maxDetections = 100;
  static const int _embeddingInputSize = 112;
  static const double _embeddingThreshold = 0.6;
  static const EmbeddingMetric _embeddingMetric = EmbeddingMetric.cosine;

  CameraController? _controller;
  Metrics _metrics = Metrics.initial();
  String? _error;
  bool _isStreaming = false;

  late final MetricsService _metricsService;
  late final DetectionService _detectionService;
  late final TempGalleryService _galleryService;
  late final ImageIouService _iouService;
  late final ImageEmbeddingService _embeddingService;
  List<Detection> _detections = const [];
  DateTime _lastInference = DateTime.fromMillisecondsSinceEpoch(0);
  int _lastInferenceMs = 0;
  DateTime _lastMetrics = DateTime.fromMillisecondsSinceEpoch(0);
  DateTime _lastNonEmptyDetection = DateTime.fromMillisecondsSinceEpoch(0);
  List<Detection> _lastNonEmptyDetections = const [];
  Size? _lastNonEmptyImageSize;
  Size? _analysisImageSize;
  String? _modelError;
  bool _isCapturing = false;
  List<File> _tempImages = const [];
  double? _lastIou;
  bool _metricsBusy = false;
  int _profileCount = 0;
  @override
  void initState() {
    super.initState();
    _metricsService = MetricsService(
      maxDim: _metricsMaxDim,
      useBackgroundMeanStdDev: _useBackgroundMetrics,
      profilingEnabled: _enableProfiling,
      profilingEveryN: _profileEveryN,
    );
    _detectionService = DetectionService(
      modelAsset: 'assets/best_float32.tflite',
      scoreThreshold: _scoreThreshold,
      nmsThreshold: _nmsThreshold,
      maxDetections: _maxDetections,
      enableLogs: false,
      logEveryN: 30,
      profilingEnabled: _enableProfiling,
      profilingEveryN: _profileEveryN,
      useBackgroundPreprocess: _useBackgroundPreprocess,
    );
    _galleryService = TempGalleryService();
    _iouService = ImageIouService();
    _embeddingService = ImageEmbeddingService(targetSize: _embeddingInputSize);
    _loadTempImages();
    _initializeCamera();
    _loadModel();
  }

  @override
  void dispose() {
    _stopImageStream();
    _detectionService.dispose();
    _metricsService.dispose();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) {
      setState(() {
        _error = 'No camera found on this device.';
      });
      return;
    }

    final camera = widget.cameras.first;

    final controller = CameraController(
      camera,
      _cameraResolution,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await controller.initialize();
      _controller = controller;
      if (mounted) {
        setState(() {});
      }
      await _startImageStream();
    } on CameraException catch (e) {
      setState(() {
        _error = 'Camera error: ${e.code}';
      });
    }
  }

  Future<void> _loadModel() async {
    try {
      if (!mounted) {
        return;
      }

      await _detectionService.load();
      setState(() {
        _modelError = _detectionService.modelError;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _modelError = 'Model error: $e';
      });
    }
  }

  Future<void> _loadTempImages() async {
    try {
      final images = await _galleryService.listImages();
      if (!mounted) {
        return;
      }
      setState(() {
        _tempImages = images;
      });
    } catch (_) {
      // Ignore gallery errors for now.
    }
  }

  Future<void> _captureImage() async {
    if (_isCapturing) {
      return;
    }
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      await _stopImageStream();
      final file = await controller.takePicture();
      await _galleryService.saveFromXFile(file);
      await _loadTempImages();
    } catch (_) {
      // Ignore capture errors to keep preview responsive.
    } finally {
      if (mounted) {
        await _startImageStream();
        setState(() {
          _isCapturing = false;
        });
      }
    }
  }

  Future<void> _openGallery() async {
    if (!mounted) {
      return;
    }
    await _loadTempImages();
    if (!mounted) {
      return;
    }

    await showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) {
        var sheetImages = List<File>.from(_tempImages);
        var selected = <File>[];
        double? iou;
        double? embeddingScore;
        bool? embeddingSimilar;
        var iouBusy = false;
        var embeddingBusy = false;
        var deleteBusy = false;
        var deleteMode = false;
        final deleteSelected = <File>{};

        return StatefulBuilder(
          builder: (context, setSheetState) {
            Future<void> compareIoU() async {
              if (selected.length != 2) {
                return;
              }
              setSheetState(() {
                iouBusy = true;
              });
              final result = await _iouService.computeIoU(
                selected[0],
                selected[1],
              );
              if (!mounted) {
                return;
              }
              setSheetState(() {
                iouBusy = false;
                iou = result;
              });
              setState(() {
                _lastIou = result;
              });
            }

            Future<void> compareEmbedding() async {
              if (selected.length != 2) {
                return;
              }
              setSheetState(() {
                embeddingBusy = true;
              });
              final embeddingA = await _embeddingService.computeEmbedding(
                selected[0],
              );
              final embeddingB = await _embeddingService.computeEmbedding(
                selected[1],
              );
              if (!mounted) {
                return;
              }
              if (embeddingA == null || embeddingB == null) {
                setSheetState(() {
                  embeddingBusy = false;
                  embeddingScore = null;
                  embeddingSimilar = null;
                });
                return;
              }
              final result = _metricsService.compareEmbeddings(
                embeddingA: embeddingA,
                embeddingB: embeddingB,
                threshold: _embeddingThreshold,
                metric: _embeddingMetric,
              );
              setSheetState(() {
                embeddingBusy = false;
                embeddingScore = result.score;
                embeddingSimilar = result.isSimilar;
              });
            }

            Future<void> deleteImages() async {
              if (deleteSelected.isEmpty) {
                return;
              }
              setSheetState(() {
                deleteBusy = true;
              });
              final toDelete = deleteSelected.toList();
              await _galleryService.deleteImages(toDelete);
              if (!mounted) {
                return;
              }
              setSheetState(() {
                deleteBusy = false;
                deleteMode = false;
                deleteSelected.clear();
                selected.clear();
                iou = null;
                embeddingScore = null;
                embeddingSimilar = null;
                sheetImages.removeWhere(
                  (file) => toDelete.any((d) => d.path == file.path),
                );
              });
              await _loadTempImages();
            }

            void enterDeleteMode(File file) {
              setSheetState(() {
                deleteMode = true;
                deleteSelected
                  ..clear()
                  ..add(file);
                selected.clear();
                iou = null;
                embeddingScore = null;
                embeddingSimilar = null;
              });
            }

            return SafeArea(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.white24,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Text(
                          'Kho ảnh tạm (${sheetImages.length})',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const Spacer(),
                        if (deleteMode)
                          Row(
                            children: [
                              TextButton(
                                onPressed:
                                    deleteBusy
                                        ? null
                                        : () {
                                          setSheetState(() {
                                            deleteSelected
                                              ..clear()
                                              ..addAll(sheetImages);
                                          });
                                        },
                                child: const Text('Chọn tất cả'),
                              ),
                              TextButton(
                                onPressed:
                                    deleteBusy
                                        ? null
                                        : () {
                                          setSheetState(() {
                                            deleteMode = false;
                                            deleteSelected.clear();
                                          });
                                        },
                                child: const Text('Hủy'),
                              ),
                            ],
                          )
                        else
                          Column(
                            children: [
                              if (iou != null) ...[
                                const SizedBox(height: 6),
                                Text(
                                  'IoU ${(iou! * 100).toStringAsFixed(1)}%',
                                  style: const TextStyle(
                                    color: Colors.lightGreenAccent,
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],

                              if (embeddingScore != null) ...[
                                const SizedBox(height: 6),
                                Text(
                                  _embeddingMetric == EmbeddingMetric.cosine
                                      ? 'Embedding ${(embeddingScore! * 100).toStringAsFixed(1)}% '
                                          '${embeddingSimilar == true ? '(giống)' : '(khác)'}'
                                      : 'Embedding L2 ${embeddingScore!.toStringAsFixed(3)} '
                                          '${embeddingSimilar == true ? '(giống)' : '(khác)'}',
                                  style: TextStyle(
                                    color:
                                        embeddingSimilar == true
                                            ? Colors.lightGreenAccent
                                            : Colors.orangeAccent,
                                    fontSize: 13,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ],
                          ),
                      ],
                    ),

                    const SizedBox(height: 12),
                    SizedBox(
                      height: 320,
                      child: GridView.builder(
                        gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                              crossAxisCount: 3,
                              crossAxisSpacing: 8,
                              mainAxisSpacing: 8,
                            ),
                        itemCount: sheetImages.length,
                        itemBuilder: (context, index) {
                          final file = sheetImages[index];
                          final isSelected =
                              deleteMode
                                  ? deleteSelected.contains(file)
                                  : selected.contains(file);
                          return GestureDetector(
                            onTap: () {
                              setSheetState(() {
                                if (deleteMode) {
                                  if (isSelected) {
                                    deleteSelected.remove(file);
                                  } else {
                                    deleteSelected.add(file);
                                  }
                                } else {
                                  if (isSelected) {
                                    selected.remove(file);
                                  } else {
                                    if (selected.length < 2) {
                                      selected.add(file);
                                    } else {
                                      selected
                                        ..removeAt(0)
                                        ..add(file);
                                    }
                                  }
                                  iou = null;
                                  embeddingScore = null;
                                  embeddingSimilar = null;
                                }
                              });
                            },
                            onLongPress: () {
                              if (deleteBusy) {
                                return;
                              }
                              if (!deleteMode) {
                                enterDeleteMode(file);
                              } else {
                                setSheetState(() {
                                  if (isSelected) {
                                    deleteSelected.remove(file);
                                  } else {
                                    deleteSelected.add(file);
                                  }
                                });
                              }
                            },
                            child: Container(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(
                                  color:
                                      isSelected
                                          ? (deleteMode
                                              ? Colors.redAccent
                                              : Colors.lightGreenAccent)
                                          : Colors.white24,
                                  width: isSelected ? 2 : 1,
                                ),
                              ),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(7),
                                child: Image.file(file, fit: BoxFit.cover),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        if (deleteMode) ...[
                          Expanded(
                            child: OutlinedButton(
                              onPressed:
                                  deleteSelected.isNotEmpty && !deleteBusy
                                      ? deleteImages
                                      : null,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: Colors.white,
                                alignment: Alignment.center,
                              ),
                              child:
                                  deleteBusy
                                      ? const SizedBox(
                                        height: 16,
                                        width: 16,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Colors.white,
                                        ),
                                      )
                                      : Text(
                                        'Xóa ảnh (${deleteSelected.length})',
                                        textAlign: TextAlign.center,
                                      ),
                            ),
                          ),
                        ] else ...[
                          Expanded(
                            child: OutlinedButton(
                              onPressed:
                                  selected.length == 2 &&
                                          !iouBusy &&
                                          !embeddingBusy
                                      ? compareIoU
                                      : null,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: Colors.white,
                                alignment: Alignment.center,
                              ),
                              child:
                                  iouBusy
                                      ? const SizedBox(
                                        height: 16,
                                        width: 16,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Colors.white,
                                        ),
                                      )
                                      : const Text(
                                        'So sánh IoU',
                                        textAlign: TextAlign.center,
                                      ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: OutlinedButton(
                              onPressed:
                                  selected.length == 2 &&
                                          !iouBusy &&
                                          !embeddingBusy
                                      ? compareEmbedding
                                      : null,
                              style: OutlinedButton.styleFrom(
                                foregroundColor: Colors.white,
                                alignment: Alignment.center,
                              ),
                              child:
                                  embeddingBusy
                                      ? const SizedBox(
                                        height: 16,
                                        width: 16,
                                        child: CircularProgressIndicator(
                                          strokeWidth: 2,
                                          color: Colors.white,
                                        ),
                                      )
                                      : const Text(
                                        'So sánh embedding',
                                        textAlign: TextAlign.center,
                                      ),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Future<void> _startImageStream() async {
    final controller = _controller;
    if (controller == null || _isStreaming) {
      return;
    }

    try {
      await controller.startImageStream(_onImage);
      if (mounted) {
        setState(() {
          _isStreaming = true;
        });
      } else {
        _isStreaming = true;
      }
    } on CameraException catch (_) {
      // Ignore stream errors for the demo; preview still works.
    }
  }

  Future<void> _stopImageStream() async {
    final controller = _controller;
    if (controller == null || !_isStreaming) {
      return;
    }

    try {
      await controller.stopImageStream();
    } on CameraException catch (_) {
      // Ignore.
    } finally {
      if (mounted) {
        setState(() {
          _isStreaming = false;
        });
      } else {
        _isStreaming = false;
      }
    }
  }

  void _onImage(CameraImage image) {
    final now = DateTime.now();
    if (!mounted) {
      return;
    }

    final shouldProfile = _shouldProfileFrame();
    final frameSw = shouldProfile ? Stopwatch() : null;
    frameSw?.start();
    if (!_metricsBusy && now.difference(_lastMetrics) >= _metricsInterval) {
      _lastMetrics = now;
      _metricsBusy = true;
      _processMetrics(image);
    }
    _maybeRunDetection(image, now);

    if (shouldProfile) {
      frameSw?.stop();
      debugPrint('[Profile] Frame sync=${frameSw?.elapsedMilliseconds ?? 0}ms');
    }
  }

  bool _shouldProfileFrame() {
    if (!_enableProfiling) return false;
    _profileCount++;
    if (_profileEveryN <= 0) {
      return _profileCount == 1;
    }
    return _profileCount % _profileEveryN == 0;
  }

  Future<void> _processMetrics(CameraImage image) async {
    try {
      final nextMetrics =
          _useBackgroundMetrics
              ? await _metricsService.processAsync(
                current: _metrics,
                image: image,
              )
              : _metricsService.process(current: _metrics, image: image);
      if (mounted) {
        setState(() {
          _metrics = nextMetrics;
        });
      }
    } catch (_) {
      // Ignore metric errors to avoid blocking the preview.
    } finally {
      _metricsBusy = false;
    }
  }

  void _maybeRunDetection(CameraImage image, DateTime now) {
    if (!_detectionService.isReady || _detectionService.isBusy) {
      return;
    }
    final minIntervalMs =
        _lastInferenceMs <= 0
            ? _minInferenceIntervalMs
            : math.max(_minInferenceIntervalMs, _lastInferenceMs);
    if (now.difference(_lastInference).inMilliseconds < minIntervalMs) {
      return;
    }
    _lastInference = now;
    _runDetection(image);
  }

  Future<void> _runDetection(CameraImage image) async {
    try {
      final start = DateTime.now();
      final result = await _detectionService.run(image);
      if (!mounted) {
        return;
      }
      if (result == null) {
        setState(() {
          _modelError = _detectionService.modelError;
        });
        return;
      }
      var nextDetections = result.detections;
      var nextImageSize = result.imageSize;
      if (nextDetections.isNotEmpty) {
        _lastNonEmptyDetection = start;
        _lastNonEmptyDetections = nextDetections;
        _lastNonEmptyImageSize = nextImageSize;
        if (_enableVerboseLogs) {
          final top = nextDetections.first;
          debugPrint(
            'TF detect: ${nextDetections.length} objects. '
            'Top id=${top.classId}, '
            'score=${(top.score * 100).toStringAsFixed(1)}%',
          );
        }
      } else if (start.difference(_lastNonEmptyDetection) <= _detectionHold &&
          _lastNonEmptyDetections.isNotEmpty) {
        nextDetections = _lastNonEmptyDetections;
        if (_lastNonEmptyImageSize != null) {
          nextImageSize = _lastNonEmptyImageSize!;
        }
      }
      setState(() {
        _analysisImageSize = nextImageSize;
        _detections = nextDetections;
        _lastInferenceMs = result.inferenceMs;
        _modelError = _detectionService.modelError;
      });
    } catch (e) {
      if (!mounted) {
        return;
      }
      setState(() {
        _modelError = 'Inference error: $e';
      });
    }
  }

  DeviceOrientation _getApplicableOrientation(CameraController controller) {
    return controller.value.isRecordingVideo
        ? controller.value.recordingOrientation!
        : (controller.value.previewPauseOrientation ??
            controller.value.lockedCaptureOrientation ??
            controller.value.deviceOrientation);
  }

  int _deviceOrientationToDegrees(DeviceOrientation orientation) {
    switch (orientation) {
      case DeviceOrientation.portraitUp:
        return 0;
      case DeviceOrientation.landscapeLeft:
        return 90;
      case DeviceOrientation.portraitDown:
        return 180;
      case DeviceOrientation.landscapeRight:
        return 270;
    }
  }

  int _overlayQuarterTurns(CameraController controller) {
    final orientation = _getApplicableOrientation(controller);
    final deviceDegrees = _deviceOrientationToDegrees(orientation);
    final sensorOrientation = controller.description.sensorOrientation;
    final isFront =
        controller.description.lensDirection == CameraLensDirection.front;
    final rotationDegrees =
        isFront
            ? (sensorOrientation + deviceDegrees) % 360
            : (sensorOrientation - deviceDegrees + 360) % 360;
    return (rotationDegrees ~/ 90) % 4;
  }

  Widget _buildCaptureBar() {
    final galleryLabel = 'Kho ảnh (${_tempImages.length})';
    final captureLabel = _isCapturing ? 'Đang chụp...' : 'Chụp ảnh';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.12)),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _isCapturing ? null : _captureImage,
                  icon: const Icon(Icons.camera_alt_outlined, size: 18),
                  label: Text(captureLabel),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _openGallery,
                  icon: const Icon(Icons.photo_library_outlined, size: 18),
                  label: Text(galleryLabel),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: Colors.white,
                  ),
                ),
              ),
            ],
          ),
          if (_lastIou != null) ...[
            const SizedBox(height: 6),
            Text(
              'IoU ${(100 * _lastIou!).toStringAsFixed(1)}%',
              style: const TextStyle(
                color: Colors.lightGreenAccent,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        body: Center(
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.white),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final rotateOverlay =
        !kIsWeb && defaultTargetPlatform == TargetPlatform.android;
    final quarterTurns = rotateOverlay ? _overlayQuarterTurns(controller) : 0;

    return Material(
      child: Stack(
        alignment: Alignment.topCenter,
        children: [
          SafeArea(
            child: CameraPreview(
              controller,
              child: CustomPaint(
                painter: DetectionPainter(
                  detections: _detections,
                  imageSize: _analysisImageSize,
                  quarterTurns: rotateOverlay ? quarterTurns : 0,
                  isFrontCamera:
                      controller.description.lensDirection ==
                      CameraLensDirection.front,
                  inferenceMs: _lastInferenceMs,
                  modelError: _modelError,
                ),
              ),
            ),
          ),
          Positioned.fill(
            child: SafeArea(
              child: IgnorePointer(
                child: ValueListenableBuilder<Offset>(
                  valueListenable: _metricsService.sensorUpListenable,
                  builder: (context, sensorUp, _) {
                    return AxesOverlay(sensorUp: sensorUp);
                  },
                ),
              ),
            ),
          ),
          Positioned(
            left: 12,
            right: 12,
            bottom: 12,
            child: SafeArea(
              top: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  _buildCaptureBar(),
                  const SizedBox(height: 8),
                  MetricsPanel(metrics: _metrics),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
