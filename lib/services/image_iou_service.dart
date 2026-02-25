import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

/// Computes a simple "IoU-like" similarity between two images.
/// Tính độ giống nhau kiểu "IoU" đơn giản giữa hai ảnh.
///
/// Mechanism:
/// Cơ chế:
/// 1) Resize both images to a fixed square size.
/// 1) Thu nhỏ cả hai ảnh về cùng kích thước vuông cố định.
/// 2) Convert each pixel to luma (grayscale intensity).
/// 2) Chuyển từng pixel sang luma (cường độ xám).
/// 3) Build a binary mask per image by thresholding with mean luma.
/// 3) Tạo mask nhị phân theo ngưỡng luma trung bình của mỗi ảnh.
/// 4) Compute Intersection / Union of the two masks.
/// 4) Tính giao / hợp giữa hai mask.
///
/// Note: This is NOT bounding-box IoU. It is a quick
/// luminance-mask overlap score in [0..1].
/// Lưu ý: Đây KHÔNG phải IoU bounding box, mà là điểm
/// chồng lấp mask theo độ sáng trong [0..1].
class ImageIouService {
  /// [targetSize] controls the resize resolution used for comparison.
  /// [targetSize] quy định kích thước ảnh sau khi resize để so sánh.
  ImageIouService({this.targetSize = 224});

  final int targetSize;

  /// Returns a similarity score in [0..1] based on mask overlap.
  /// Trả về điểm tương đồng [0..1] dựa trên độ chồng lấp mask.
  Future<double> computeIoU(File first, File second) async {
    // Decode and resize images to same size, then read RGBA bytes.
    // Giải mã và resize ảnh về cùng kích thước, rồi đọc RGBA bytes.
    final bytesA = await _decodeToRgba(first);
    final bytesB = await _decodeToRgba(second);
    if (bytesA == null || bytesB == null) {
      return 0;
    }

    // Use mean luma as a cheap global threshold for each image.
    // Dùng luma trung bình làm ngưỡng toàn cục đơn giản cho từng ảnh.
    final meanA = _meanLuma(bytesA);
    final meanB = _meanLuma(bytesB);

    // Compute intersection / union over thresholded masks.
    // Tính giao / hợp trên các mask đã được ngưỡng hóa.
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

    // If both masks are empty, treat as fully similar.
    // Nếu cả hai mask đều rỗng, xem như giống nhau hoàn toàn.
    if (union == 0) {
      return 1.0;
    }
    return intersection / union;
  }

  /// Decode image file to RGBA bytes at [targetSize] x [targetSize].
  /// Giải mã ảnh sang RGBA bytes ở kích thước [targetSize] x [targetSize].
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

  /// Average grayscale intensity across all pixels.
  /// Trung bình cường độ xám của toàn bộ pixel.
  double _meanLuma(Uint8List bytes) {
    var sum = 0.0;
    var count = 0;
    for (var i = 0; i < bytes.length; i += 4) {
      sum += _lumaAt(bytes, i);
      count++;
    }
    return count == 0 ? 0 : sum / count;
  }

  /// Per-pixel luma from RGBA bytes (BT.601).
  /// Luma cho từng pixel từ RGBA bytes (BT.601).
  double _lumaAt(Uint8List bytes, int index) {
    final r = bytes[index].toDouble();
    final g = bytes[index + 1].toDouble();
    final b = bytes[index + 2].toDouble();
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }
}
