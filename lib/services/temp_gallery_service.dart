import 'dart:io';

import 'package:camera/camera.dart';
import 'package:path_provider/path_provider.dart';

class TempGalleryService {
  TempGalleryService({this.folderName = 'temp_gallery'});

  final String folderName;

  Future<Directory> _getGalleryDir() async {
    final root = await getTemporaryDirectory();
    final dir = Directory('${root.path}/$folderName');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir;
  }

  Future<File> saveFromXFile(XFile source) async {
    final dir = await _getGalleryDir();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final target = File('${dir.path}/capture_$timestamp.jpg');
    await source.saveTo(target.path);
    return target;
  }

  Future<List<File>> listImages() async {
    final dir = await _getGalleryDir();
    final entities = await dir.list().toList();
    final files =
        entities.whereType<File>().where((file) {
          final lower = file.path.toLowerCase();
          return lower.endsWith('.jpg') ||
              lower.endsWith('.jpeg') ||
              lower.endsWith('.png');
        }).toList();
    files.sort((a, b) {
      final aTime = a.statSync().modified;
      final bTime = b.statSync().modified;
      return bTime.compareTo(aTime);
    });
    return files;
  }
}
