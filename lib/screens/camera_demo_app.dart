import 'package:flutter/material.dart';

import 'camera_home.dart';

class CameraDemoApp extends StatelessWidget {
  const CameraDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Camera Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(scaffoldBackgroundColor: Colors.black),
      home: const CameraHome(),
    );
  }
}
