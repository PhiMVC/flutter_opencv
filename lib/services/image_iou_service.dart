import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

class ImageIouService {
  ImageIouService({
    this.targetSize = 224,
  });

  final int targetSize;

  Future<double> computeIoU(File first, File second) async {
    final bytesA = await _decodeToRgba(first);
    final bytesB = await _decodeToRgba(second);
    if (bytesA == null || bytesB == null) {
      return 0;
    }

    final meanA = _meanLuma(bytesA);
    final meanB = _meanLuma(bytesB);

    var intersection = 0;
    var union = 0;
    for (var i = 0; i < bytesA.length && i < bytesB.length; i += 4) {
      final lumaA = _lumaAt(bytesA, i);
      final lumaB = _lumaAt(bytesB, i);
      final maskA = lumaA >= meanA;
      final maskB = lumaB >= meanB;
      if (maskA && maskB) {
        intersection++;
      }
      if (maskA || maskB) {
        union++;
      }
    }

    if (union == 0) {
      return 1.0;
    }
    return intersection / union;
  }

  Future<Uint8List?> _decodeToRgba(File file) async {
    final bytes = await file.readAsBytes();
    final codec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: targetSize,
      targetHeight: targetSize,
    );
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    codec.dispose();
    return data?.buffer.asUint8List();
  }

  double _meanLuma(Uint8List bytes) {
    var sum = 0.0;
    var count = 0;
    for (var i = 0; i < bytes.length; i += 4) {
      sum += _lumaAt(bytes, i);
      count++;
    }
    return count == 0 ? 0 : sum / count;
  }

  double _lumaAt(Uint8List bytes, int index) {
    final r = bytes[index].toDouble();
    final g = bytes[index + 1].toDouble();
    final b = bytes[index + 2].toDouble();
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }
}
