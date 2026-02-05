import 'dart:ffi' as ffi;
import 'dart:io' show Platform;

final class Result extends ffi.Struct {
  @ffi.Float()
  external double a;

  @ffi.Float()
  external double b;
}

typedef _ProcessNative =
    Result Function(
      ffi.Pointer<ffi.Uint8> data,
      ffi.Int32 width,
      ffi.Int32 height,
      ffi.Int32 rowStride,
    );

typedef ProcessDart =
    Result Function(
      ffi.Pointer<ffi.Uint8> data,
      int width,
      int height,
      int rowStride,
    );

class MyLib {
  late final ffi.DynamicLibrary _lib;
  late final ProcessDart processGray;

  MyLib() {
    if (Platform.isAndroid) {
      _lib = ffi.DynamicLibrary.open("libmylib.so");
    } else if (Platform.isIOS) {
      _lib = ffi.DynamicLibrary.process();
    } else {
      throw UnsupportedError("Only Android/iOS supported");
    }

    processGray =
        _lib
            .lookup<ffi.NativeFunction<_ProcessNative>>("process_gray")
            .asFunction();
  }
}
